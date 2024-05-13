import torch
import torch.nn.functional as F

from .utils import FreeMatchThresholingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


@ALGORITHMS.register('freematch')
class FreeMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio

    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_lb, idx_ulb, x_ulb):

        # inference and calculate sup/unsup losses

        # aug_1 : data augmentation to get pseudolabels "id" or "weak", "strong"
        # aug_1 is weak for vanilla fixmatch

        # aug_2:  data augmentation in consistency loss "id" or "weak", "strong"
        # aug_2 is strong for vanilla fixmatch

        if(self.aug_1=="id" and self.aug_2=="id"):
            x_ulb_w = x_ulb
            x_ulb_s = x_ulb 

        elif(self.aug_1=="weak" and self.aug_2=="weak"):
            x_ulb_s =  x_ulb_w 

        elif(self.aug_1=="id" and self.aug_2 =="weak"):
            x_ulb_w = x_ulb 
            x_ulb_s = x_ulb_w 

        elif(self.aug_1=="weak" and self.aug_2=="strong"):
            pass 

        else:
            self.print_fn('XXXXXXXX Combination not supported XXXXXXXX') 
            # strong, strong 
            # strong id
            # id     strong
            # strong weak 
            # weak   id 

        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                rep, logits, Lkl, logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, feats_x_lb, feats_x_ulb_w, feats_x_ulb_s = self.use_cat_func(x_lb, x_ulb_w, x_ulb_s, num_lb)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}


            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            # if using BaM, the function run bam related stuff
            self.check_if_use_bam(Lkl, logits, num_lb, rep)
            if(self.batch_pl_flag):
                # calculate mask
                mask_batch = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)


                # generate unlabeled targets using pseudo label hook
                pseudo_labels_batch = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=logits_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T)
            else:
                # either global_pl_flag is True or post_hoc_calib is true. 
                pseudo_labels_batch = self.pseudo_labels[idx_ulb]
                mask_batch          = self.mask[idx_ulb]
                
            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_labels_batch,
                                          'ce',
                                          mask=mask_batch)
            
            # calculate entropy loss
            if mask_batch.sum() > 0:
               ent_loss, _ = entropy_loss(mask_batch, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            # ent_loss = 0.0
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss
        
        self.log_batch_pseudo_labeling_stats(mask_batch,pseudo_labels_batch,idx_ulb)
        self.log_full_pseudo_labeling_stats()

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask_batch.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
        ]
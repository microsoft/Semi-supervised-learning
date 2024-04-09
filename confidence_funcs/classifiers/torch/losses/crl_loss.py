
import sys 
sys.path.append('../../')
sys.path.append('../../../')

import torch
import torch.nn.functional as F  
import src.utils.crl_history as crl_history
import src.utils.metrics as metrics 
from  .abstract_loss import * 


#NOTE: Potential problem with this method and its solution:
# There are more correct points then incorrect points, so class-imbalance is 
# there and the estimation of correctness probs might be inaccurate 


class CRLLoss(AbstractLoss):
    def __init__(self, loss_conf=None):

        super(CRLLoss, self).__init__()
        self.device = loss_conf['device']
        self.state = {}
        
        self.state['history'] = crl_history.History(loss_conf['num_train_pts'])

        self.cls_criterion = nn.CrossEntropyLoss().to(self.device)
        self.ranking_criterion = nn.MarginRankingLoss(margin=0.0).to(self.device)

        self.rank_target = loss_conf["rank_target"]
        self.rank_weight = loss_conf["rank_weight"]

        self.result = {}

    def forward(self, input, target, idx=None):
     
        # compute ranking target value normalize (0 ~ 1) range
        # max(softmax)
        if self.rank_target == 'softmax':
            conf = F.softmax(input, dim=1)
            confidence, _ = conf.max(dim=1)
        # entropy
        elif self.rank_target == 'entropy':
            #value_for_normalizing = log(num_classes)
            value_for_normalizing = 2.3
            confidence = metrics.negative_entropy(input,
                                                    normalize=True,
                                                    max_value=value_for_normalizing)
        # margin
        elif self.rank_target == 'margin':
            conf, _ = torch.topk(F.softmax(input), 2, dim=1)
            conf[:,0] = conf[:,0] - conf[:,1]
            confidence = conf[:,0]

        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)
        
        #rank_input2 = confidence
        
        idx_ = idx.detach().numpy() 
        idx2_ = idx2.detach().numpy()
        # calc target, margin
        rank_target_2, rank_margin = self.state["history"].get_target_margin(idx_, idx2_)
        
        rank_target_2, rank_margin = rank_target_2.to(self.device), rank_margin.to(self.device)
        

        rank_target_nonzero = rank_target_2.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        
        
        #print((rank_target_2==1).sum(),(rank_target_2==-1).sum())
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero
        
        # ranking loss
        ranking_loss = self.ranking_criterion(rank_input1,
                                        rank_input2,
                                        rank_target_2)

        # total loss

        ranking_loss = self.rank_weight * ranking_loss

        
        
        
        cls_loss = self.cls_criterion(input, target)
        loss = cls_loss + ranking_loss
        
        self.result['loss'] = loss.item() 
        self.result['xent'] = cls_loss.item() 
        self.result['ranking_loss'] = ranking_loss.item()

        return loss 
    
    
    def batch_closure_callback(self, batch_state):
        idx     = batch_state['idx']
        correct = batch_state['correct']
        logits  = batch_state['logits']
        self.state["history"].correctness_update(idx.detach(), correct.detach(), logits.detach())
        
    
    def epoch_closure_callback(self, epoch_state):
        # max correctness update
        self.state["history"].max_correctness_update(epoch_state['epoch_num'])
        pass


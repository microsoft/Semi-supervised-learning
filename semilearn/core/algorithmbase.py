# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from time import time

import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score,
)

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from semilearn.core.hooks import (
    Hook,
    get_priority,
    CheckpointHook,
    TimerHook,
    LoggingHook,
    DistSamplerSeedHook,
    ParamUpdateHook,
    EvaluationHook,
    EMAHook,
    WANDBHook,
    AimHook,
)
from semilearn.core.utils import (
    get_dataset,
    get_data_loader,
    get_optimizer,
    get_cosine_schedule_with_warmup,
    Bn_Controller,
)
from semilearn.core.criterions import CELoss, ConsistencyLoss

from semilearn.datasets.utils import randomly_split_labeled_basic_dataset

from confidence_funcs.calibration.calibrators import get_calibrator
from confidence_funcs.classifiers.torch.pytorch_clf import PyTorchClassifier

from confidence_funcs.core.threshold_estimation import *


class ThresholdScheduler:
    def __init__(self, warmup_epochs, init_thres, final_thres, total_steps, steps_per_epoch):
        warmup_iter = steps_per_epoch * warmup_epochs
        warmup_schedule = np.linspace(init_thres, final_thres, warmup_iter)
        decay_iter = total_steps - warmup_iter
        constant_schedule = np.ones(decay_iter)*final_thres
        self.thres_schedule = np.concatenate((warmup_schedule, constant_schedule))
        self.iter = int(-1)

    def get_threshold(self):
        self.iter += 1
        return self.thres_schedule[self.iter]


class AlgorithmBase:
    """
    Base class for algorithms
    init algorithm specific parameters and common parameters

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        # common arguments
        self.args = args
        self.post_hoc_frequency = args.post_hoc_frequency
        self.cur_clf = None
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        # commaon utils arguments
        self.tb_log = tb_log
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # common model related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None

        # build dataset
        self.dataset_dict = self.set_dataset()

        # build data loader
        self.loader_dict = self.set_data_loader()

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()
        if self.args.bayes:
            self.bayes_optimizer, self.bayes_scheduler, self.quantile_sch = self.set_bayes_optimizer()
        else:
            self.bayes_optimizer, self.bayes_scheduler, self.quantile_sch = None, None, None

        # build supervised loss and unsupervised loss
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks
        self.hooks_dict = OrderedDict()  # actual object to be used to call hooks
        self.set_hooks()

        self.post_hoc_calib_conf = None

        self.aug_1 = args.aug_1
        self.aug_2 = args.aug_2

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(
            self.args,
            self.algorithm,
            self.args.dataset,
            self.args.num_labels,
            self.args.num_classes,
            self.args.data_dir,
            self.args.include_lb_to_ulb,
        )
        # N_v, nu #N_cal, N_th

        # take_from_train_lb ?
        # take_from_eval ?
        # self.args.take_from "train_lb", "eval"

        need_d_cal = self.args.n_cal > 0
        need_d_th = self.args.n_th > 0

        if need_d_cal or need_d_th:
            take_d_cal_th_from = self.args.take_d_cal_th_from

            n = self.args.n_cal + self.args.n_th
            # if(self.args.take_from_eval):
            ds1, ds2 = randomly_split_labeled_basic_dataset(
                dataset_dict[take_d_cal_th_from],
                self.num_classes,
                size_1=n,
                fixed_seed=True,
                class_balance=True,
            )

            """
            dataset_dict[take_d_cal_th_from] = (
                ds2  # remaining n_lb (or n_eval) - n samples
            )
            """

            self.print_fn(
                f"len(d_train) = {len(dataset_dict['train_lb'])} and len(d_eval) = {len(dataset_dict['eval'])}"
            )

            if need_d_cal and need_d_th:
                ds11, ds12 = randomly_split_labeled_basic_dataset(
                    ds1,
                    self.num_classes,
                    size_1=self.args.n_cal,
                    fixed_seed=True,
                    class_balance=True,
                )
                ds11.Y = torch.Tensor(ds11.targets)
                ds12.Y = torch.Tensor(ds12.targets)
                # print(ds11.Y)
                dataset_dict["d_cal"] = ds11
                dataset_dict["d_th"] = ds12
                self.print_fn(f"len(d_cal) = {len(ds11)} and len(d_th) = {len(ds12)}")

            elif need_d_cal and not need_d_th:
                # need only post-hoc calib data.  [for fixed threshold or heuristic thresholds]
                dataset_dict["d_cal"] = ds1
                self.print_fn(f"len(d_cal) = {len(ds1)}")

            else:
                # not doing post-hoc calibration but using threshold-estimation
                dataset_dict["d_th"] = ds1
                self.print_fn(f"len(d_th) = {len(ds1)}")

        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = (
            len(dataset_dict["train_ulb"])
            if dataset_dict["train_ulb"] is not None
            else 0
        )
        self.args.lb_dest_len = len(dataset_dict["train_lb"])
        self.print_fn(
            "unlabeled data number: {}, labeled data number {}".format(
                self.args.ulb_dest_len, self.args.lb_dest_len
            )
        )
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return

        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict["train_lb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_lb"],
            self.args.batch_size,
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=self.args.num_workers,
            distributed=self.distributed,
        )

        loader_dict["train_ulb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_ulb"],
            self.args.batch_size * self.args.uratio,
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=2 * self.args.num_workers,
            distributed=self.distributed,
        )

        self.print_fn(f"size of eval dataset = {len(self.dataset_dict['eval'].data)}")

        loader_dict["eval"] = get_data_loader(
            self.args,
            self.dataset_dict["eval"],
            self.args.eval_batch_size,
            # make sure data_sampler is None for evaluation
            data_sampler=None,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        if self.dataset_dict["test"] is not None:
            loader_dict["test"] = get_data_loader(
                self.args,
                self.dataset_dict["test"],
                self.args.eval_batch_size,
                # make sure data_sampler is None for evaluation
                data_sampler=None,
                num_workers=self.args.num_workers,
                drop_last=False,
            )
        self.print_fn(f"[!] data loader keys: {loader_dict.keys()}")
        return loader_dict

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(
            self.model,
            self.args.optim,
            self.args.lr,
            self.args.momentum,
            self.args.weight_decay,
            self.args.layer_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.num_train_iter, num_warmup_steps=self.args.num_warmup_iter
        )
        return optimizer, scheduler
    
    def set_bayes_optimizer(self):
        from torch import optim
        bayes_optimizer = optim.Adam(
            self.model.classifier.parameters(),
            lr=self.args.bayes_args.bayes_lr
        )
        bayes_scheduler = get_cosine_schedule_with_warmup(
            bayes_optimizer,
            self.num_train_iter,
            num_warmup_steps=self.args.num_warmup_iter
        )
        # we have a quantile scheduler for the std threshold
        quantile_sch = ThresholdScheduler(
            self.args.bayes_args.quansch_warmup,
            self.args.bayes_args.init_quan,
            self.args.bayes_args.final_quan,
            self.num_train_iter,
            self.num_eval_iter
        )
        return bayes_optimizer, bayes_scheduler, quantile_sch

    def set_model(self):
        """
        initialize model
        """
        if self.args.bayes:
            model = self.net_builder(
                num_classes=self.num_classes,
                pretrained=self.args.use_pretrain,
                pretrained_path=self.args.pretrain_path,
                bayes=self.args.bayes,
                prior_mu=self.args.bayes_args.prior_mu,
                prior_sigma=self.args.bayes_args.prior_sig
            )
        else:
            model = self.net_builder(
                num_classes=self.num_classes,
                pretrained=self.args.use_pretrain,
                pretrained_path=self.args.pretrain_path
            )
        
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        if self.args.bayes:
            ema_model = self.net_builder(
                num_classes=self.num_classes,
                pretrained=self.args.use_pretrain,
                pretrained_path=self.args.pretrain_path,
                bayes=self.args.bayes,
                prior_mu=self.args.bayes_args.prior_mu,
                prior_sigma=self.args.bayes_args.prior_sig
            )
        else:
            ema_model = self.net_builder(
                num_classes=self.num_classes,
                pretrained=self.args.use_pretrain,
                pretrained_path=self.args.pretrain_path
            )
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue

            if var is None:
                continue

            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
            
        if self.args.bayes:
            self.args.bayes_args.quantile = float(self.quantile_sch.get_threshold())

        return input_dict

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var

        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict

    def process_log_dict(self, log_dict=None, prefix="train", **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f"{prefix}/" + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def accumulate_pseudo_labels(self,idx_ulb, mask, pseudo_labels):
        # expected sizes len(mask) = len(idx_ulb) = len(pseudo_labels)

        # the points that have got new pseudo labels, should have their previous pseudo label overwritten.
        # the points that did not get pseudo labels, will maintain their previous pseudo label.

        idx_ulb = idx_ulb.to(self.device)

        mask_bool = mask.ge(1.0)
        
        idx_ulb_pl = idx_ulb[mask_bool] # indices that got new pseudo label
        
        self.pseudo_labels[idx_ulb_pl] = pseudo_labels[mask_bool]  # overwrite the pseudo label for these points.

        self.mask[idx_ulb] = torch.clamp( mask + self.mask[idx_ulb], min=0.0, max=1.0) # update the mask accordingly.
        
        self.print_fn(f"{torch.sum(mask).item()},  {torch.sum( self.mask[idx_ulb] ).item()},{torch.sum(self.mask).item()}")
        
        pseudo_label = self.pseudo_labels[idx_ulb]
        mask         = self.mask[idx_ulb]

        return pseudo_label, mask


    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model
        # record log_dict
        # return log_dict
        raise NotImplementedError

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")
        self.agg_pl_cov = 0.0

        device = str(next(self.model.parameters()).device)
        n_u = len(self.dataset_dict["train_ulb"].targets)
        self.n_u = n_u
        self.n_l = len(self.dataset_dict["train_lb"].targets)

        self.y_true_ulb = torch.tensor(self.dataset_dict["train_ulb"].targets).to(
            device
        )

        self.device = device

        self.pseudo_labels = torch.zeros(n_u).long().to(device)
        self.mask = torch.zeros(n_u).to(device)
        
        self.batch_pl_flag = True
        self.full_pl_flag  = False

        if(self.post_hoc_calib_conf):
            self.batch_pl_flag = False 

        # accumulate_pseudo_labels = True
        self.acc_pseudo_labels_flag = self.args.accumulate_pseudo_labels

        if self.args.use_true_labels:
            self.pseudo_labels = torch.tensor(
                self.dataset_dict["train_ulb"].targets
            ).to(device)
            self.mask = torch.ones(len(self.pseudo_labels)).to(device)

        # self.X_ulb = torch.tensor(self.dataset_dict['train_ulb'].data).to(self.device)

        for epoch in range(self.start_epoch, self.epochs):

            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            # print(len(self.loader_dict['train_lb']), len(self.loader_dict['train_ulb']))
            idcs = []
            for data_lb, data_ulb in zip(
                self.loader_dict["train_lb"], self.loader_dict["train_ulb"]
            ):
                # prevent the training iterations exceed args.num_train_iter

                
                Freq = 100 #self.post_hoc_frequency
                

                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")

                # for every iter>0, if post-hoc-calib is not none, then learn new g.
                # validation data ??

                # <<<<<<<<<<<<<<<<<<<<<<<<< BEGIN CALIBRATION BLOCK <<<<<<<<<<<<<<<<<<<<<<<<<

                if self.post_hoc_calib_conf and self.it % Freq == 0 and self.it >= Freq:
                    
                    self.post_hoc_calib_pseudo_labeling()

                else:
                    # self.print_fn('=========================    No Post-hoc Calibration     =========================')
                    self.cur_calibrator = None

                if(self.post_hoc_calib_conf is None and self.full_pl_flag and self.it % Freq == 0 and self.it >= Freq):

                    self.full_pseudo_labeling()
                # >>>>>>>>>>>>>>>>>>>>>>>>>>> END CALIBRATION BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>

                # this step only computes the loss
                self.out_dict, self.log_dict = self.train_step(
                    **self.process_batch(**data_lb, **data_ulb)
                )

                # parameters are updated in the call backs.
                self.call_hook("after_train_step")
                # the parameters are updated once the above hooks are called.

                self.it += 1

            # print('EPOCH DONE')
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")


    def post_hoc_calib_pseudo_labeling(self):

        device = self.device 

        self.cur_clf = PyTorchClassifier(logger=self.logger)
        self.cur_clf.model = self.model
        self.post_hoc_calib_conf['device'] = self.device 

        # self.print_fn('========================= Training Post-hoc Calibrator   =========================')
        self.cur_calibrator = get_calibrator(
            self.cur_clf, self.post_hoc_calib_conf, self.logger
        )

        # randomly split the current available validation points into two parts.
        # one part will be used for training the calibrator and other part for finding
        # the auto-labeling thresholds.

        # self.print_fn(f"Number of points for training calibrator : {len(self.dataset_dict['d_cal'])}")
        self.cur_calibrator.fit(
            self.dataset_dict["d_cal"], ds_val_nc=self.dataset_dict["d_th"]
        )

        # get pseudo labels and mask here.
        # estimate threshold, pseudo label etc.
        inf_conf = {'feature_key':'x_lb', 'idx_key':'idx_lb'}
        val_inf_out_th = self.cur_calibrator.predict(
            self.dataset_dict["d_th"], inference_conf=inf_conf
        )

        lst_classes = np.arange(0, self.num_classes, 1)
        auto_lbl_conf = self.post_hoc_calib_conf.auto_lbl_conf
        val_idcs = np.arange(0, len(self.dataset_dict["d_th"].targets), 1)
        eps = auto_lbl_conf.auto_label_err_threshold

        tic  = time()
        lst_t_val, val_idcs_to_rm, val_err, cov = determine_threshold(
            lst_classes,
            val_inf_out_th,
            auto_lbl_conf,
            self.dataset_dict["d_th"],
            val_idcs,
            self.logger,
            err_threshold=eps,
        )
        toc = time()
        self.print_fn(f"Total time in threshold estimation : {toc-tic}")

        # pseudo label and mask
        tic = time()
        inf_conf = {'feature_key':'x_ulb_w', 'idx_key':'idx_ulb', 'batch_size':500}
        unlbld_inf_out = self.cur_calibrator.predict(
            self.dataset_dict["train_ulb"], inference_conf=inf_conf
        )
        toc = time()
        self.print_fn(f"Total time in inference on unlabeled data : {toc-tic}")

        scores = unlbld_inf_out[auto_lbl_conf["score_type"]]

        y_hat = unlbld_inf_out["labels"].to(device)

        tt = torch.tensor([lst_t_val[y_hat[i]] for i in range(self.n_u)]).to(
            device
        )
        scores = torch.tensor(scores).to(device)
        mask_full = scores.ge(tt).to(dtype=scores.dtype)
        pseudo_labels_full = y_hat
        
        idx_ulb = torch.arange(0,self.n_u).to(self.device)

        self.accumulate_pseudo_labels(idx_ulb,mask_full,pseudo_labels_full)

        #self.pseudo_labels = self.pseudo_labels.to(device)
        #self.mask = self.mask.to(device)

        self.model.train() 
        

    def full_pseudo_labeling(self):

        #self.cur_clf = PyTorchClassifier(logger=self.logger)
        #self.cur_clf.model = self.model
        inf_conf = {'feature_key':'x_ulb_w', 'idx_key':'idx_ulb'}
        inf_out = self.cur_clf.predict(self.dataset_dict["train_ulb"],inference_conf=inf_conf)
        
        probs_x_ulb_w = inf_out['probs']

        # compute mask
        mask_full = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
        
        # generate unlabeled targets using pseudo label hook
        pseudo_labels_full = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                        logits=probs_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T,
                                        softmax=False)

        idx_ulb = torch.arange(0,self.n_u).to(self.device)

        self.accumulate_pseudo_labels(idx_ulb,mask_full,pseudo_labels_full)


    def log_batch_pseudo_labeling_stats(self,mask_batch,pseudo_labels_batch,idx_ulb_batch):

        n_a_batch = torch.sum(mask_batch)
        batch_cov = (n_a_batch/ len(mask_batch)).item() 
        batch_acc = 0.0 
        if(n_a_batch>0):
            batch_acc_mask = self.y_true_ulb[idx_ulb_batch]==pseudo_labels_batch
            batch_acc      = torch.sum(batch_acc_mask[mask_batch.ge(1.0)])/n_a_batch 
        
        if not self.tb_log is None:
            self.tb_log.update({"batch_pl_cov":batch_cov, "batch_pl_acc":batch_acc}, self.it)
    
    def log_full_pseudo_labeling_stats(self):

        n_a = torch.sum(self.mask).detach()
        
        self.n_a = n_a 

        cov = n_a/self.n_u 
        acc = 0.0 
        if(n_a>0):
            acc_mask = self.y_true_ulb==self.pseudo_labels
            acc      = torch.sum(acc_mask[self.mask.ge(1.0)])/n_a
        
        if not self.tb_log is None:
            self.tb_log.update({"agg_pl_cov":cov, "agg_pl_acc":acc}, self.it)
        
        self.agg_pl_cov = cov
            

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        y_logits = []

        with torch.no_grad():
            for data in eval_loader:
                x = data["x_lb"]
                y = data["y_lb"]
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)
                # print(x.keys())
                # print(y.shape)
                num_batch = y.shape[0]
                total_num += num_batch

                if self.args.bayes and out_key == "logits":
                    logits, Lkl = self.model(x)[out_key]
                    Lkl = Lkl / logits[:len(y)].shape[0] * self.args.bayes_args.kl
                else:
                    logits = self.model(x)[out_key]
                    Lkl = 0
                # TODO: verify the types of y_true and y_pred! and what's there inside them
                loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1) + Lkl 
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += loss.item() * num_batch

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_probs, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        F1 = f1_score(y_true, y_pred, average="macro")

        # cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        # self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {
            eval_dest + "/loss": total_loss / total_num,
            eval_dest + "/top-1-acc": top1,
            eval_dest + "/top-5-acc": top5,
            eval_dest + "/balanced_acc": balanced_top1,
            eval_dest + "/precision": precision,
            eval_dest + "/recall": recall,
            eval_dest + "/F1": F1,
        }
        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_logits
        return eval_dict

    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        # Added Pseudo-labels and masks in the keys!
        save_dict = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_scaler": self.loss_scaler.state_dict(),
            "it": self.it + 1,
            "epoch": self.epoch + 1,
            "best_it": self.best_it,
            "best_eval_acc": self.best_eval_acc,
            "pseudo_labels": self.pseudo_labels,
            "mask": self.mask
        }
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()
        return save_dict

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        checkpoint = torch.load(load_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        self.it = checkpoint["it"]
        self.start_epoch = checkpoint["epoch"]
        self.epoch = self.start_epoch
        self.best_it = checkpoint["best_it"]
        self.best_eval_acc = checkpoint["best_eval_acc"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.print_fn("Model loaded")
        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith("module"):
                new_key = ".".join(key.split(".")[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority="NORMAL"):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)
    
    def use_cat_func(self, x_lb, x_ulb_w, x_ulb_s, num_lb):
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outputs = self.model(inputs)
        Lkl = 0
        rep = None
        if self.args.bayes:
            rep = outputs["pre_logits"]
            logits, Lkl = outputs["logits"]
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        else:
            logits_x_lb = outputs['logits'][:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
        feats_x_lb = outputs['feat'][:num_lb]
        feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
        return rep, logits, Lkl, logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, feats_x_lb, feats_x_ulb_w, feats_x_ulb_s
                
    def check_if_use_bam(self, Lkl, logits, num_lb, rep):
        if self.args.bayes:    
            self.sup_loss += Lkl / logits[:num_lb].shape[0] * self.args.bayes_args.kl
            
            rep_u = rep.detach()[num_lb:]
            rep_u_w, rep_u_s = rep_u.chunk(2)
            
            mean_output_u_w, std_output_u_w = self.bayes_predict(self.args.bayes_args, self.model.classifier, rep_u_w)
            
            max_probs_u_w, targets_u = torch.max(mean_output_u_w,dim=-1)
            pred_std = torch.gather(std_output_u_w,1,targets_u.view(-1,1)).squeeze(1)
            
            mask = pred_std.le(self.args.bayes_args.std_threshold) 
            mask = mask.float()

            # Update threshold based on the quantile
            if self.args.bayes_args.quantile != -1 and self.epoch > self.args.bayes_args.q_warmup:
                new_threshold = torch.quantile(pred_std, self.args.bayes_args.quantile).item()
                if self.args.bayes_args.q_queue:
                    self.quan_queue.append(new_threshold)
                    if len(self.quan_queue) > 50: self.quan_queue.pop(0) # maintain last 50 values
                    new_threshold = np.mean(self.quan_queue)
                    self.args.bayes_args.args.std_threshold = new_threshold

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict

    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        return {}
    
    @staticmethod
    def bayes_predict(args, bayeslayer, reps, T=1):
        ''' creates args.bayes_samples models and get mean and std of softmax(output) '''
        with torch.no_grad():
            with autocast():
                outputs = [torch.softmax(bayeslayer(reps)[0]/T,dim=-1) for _ in range(args.bayes_samples)]
            outputs = torch.stack(outputs)
            mean_output = torch.mean(outputs, 0)
            std_output = torch.std(outputs,0)
            # max_prob, preds = torch.max(mean_preds, dim=-1)
        return mean_output, std_output


class ImbAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)

        # imbalanced arguments
        self.lb_imb_ratio = self.args.lb_imb_ratio
        self.ulb_imb_ratio = self.args.ulb_imb_ratio
        self.imb_algorithm = self.args.imb_algorithm

    def imb_init(self, *args, **kwargs):
        """
        intiialize imbalanced algorithm parameters
        """
        pass

    def set_optimizer(self):
        if "vit" in self.args.net and self.args.dataset in [
            "cifar100",
            "food101",
            "semi_aves",
            "semi_aves_out",
        ]:
            return super().set_optimizer()
        elif self.args.dataset in ["imagenet", "imagenet127"]:
            return super().set_optimizer()
        else:
            self.print_fn("Create optimizer and scheduler")
            optimizer = get_optimizer(
                self.model,
                self.args.optim,
                self.args.lr,
                self.args.momentum,
                self.args.weight_decay,
                self.args.layer_decay,
                bn_wd_skip=False,
            )
            scheduler = None
            return optimizer, scheduler

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/bbuing9/DARP/blob/64a94b46b5fcf3307178c2cb2f97fe051c43e500/common.py#L107

import math
import torch
import numpy as np
from scipy import optimize
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import smooth_targets


class DARPPseudoLabelingHook(PseudoLabelingHook):
    def __init__(self, warmup_epochs, alpha, iter_T, num_refine_iter, dataset_len, num_classes, target_disb):
        super(DARPPseudoLabelingHook, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.iter_T = iter_T
        self.num_refine_iter = num_refine_iter
        self.num_classes = num_classes
        self.dataset_len = dataset_len
        self.target_disb = torch.tensor(target_disb)

        self.pseudo_orig = torch.ones(self.dataset_len, self.num_classes) / self.num_classes
        self.pseudo_refine = torch.ones(self.dataset_len, self.num_classes) / self.num_classes

    @torch.no_grad()
    def refine_pseudo_labels(self, idx_u, targets_u, batch_idx, epoch):
        # Update the saved predictions with current one
        self.pseudo_orig[idx_u, :] = targets_u.data.cpu().to(self.pseudo_orig.dtype)
        self.pseudo_orig_backup = self.pseudo_orig.clone()

        if epoch > self.warmup_epochs:
            if batch_idx % self.num_refine_iter == 0:
                # Iterative normalization
                targets_u, weights_u = self.estimate_pseudo(self.target_disb, self.pseudo_orig)
                scale_term = targets_u * weights_u.reshape(1, -1)
                self.pseudo_orig = (self.pseudo_orig * scale_term + 1e-6) \
                                   / (self.pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                opt_res = self.opt_solver(self.pseudo_orig, self.target_disb, self.iter_T)

                # Updated pseudo-labels are saved
                self.pseudo_refine = opt_res

                # Select
                targets_u = opt_res[idx_u.to(opt_res.device)].detach().cuda()
                self.pseudo_orig = self.pseudo_orig_backup
            else:
                # Using previously saved pseudo-labels
                targets_u = self.pseudo_refine[idx_u.to(self.pseudo_refine.device)].cuda()

        return targets_u

    @torch.no_grad()
    def estimate_pseudo(self, q_y, saved_q):
        pseudo_labels = torch.zeros(len(saved_q), self.num_classes)
        k_probs = torch.zeros(self.num_classes)

        for i in range(1, self.num_classes + 1):
            i = self.num_classes - i
            num_i = int(self.alpha * q_y[i])
            sorted_probs, idx = saved_q[:, i].sort(dim=0, descending=True)
            pseudo_labels[idx[: num_i], i] = 1
            k_probs[i] = sorted_probs[:num_i].sum()

        return pseudo_labels, (q_y + 1e-6) / (k_probs + 1e-6)

    @torch.no_grad()
    def f(self, x, a, b, c, d):
        return np.sum(a * b * np.exp(-1 * x / c)) - d

    @torch.no_grad()
    def opt_solver(self, probs, target_distb, num_iter, num_newton=30):
        entropy = (-1 * probs * torch.log(probs + 1e-6)).sum(1)
        weights = (1 / entropy)
        N, K = probs.size(0), probs.size(1)

        A, w, lam, nu, r, c = probs.numpy(), weights.numpy(), np.ones(N), np.ones(K), np.ones(N), target_distb.numpy()
        A_e = A / math.e
        X = np.exp(-1 * lam / w)
        Y = np.exp(-1 * nu.reshape(1, -1) / w.reshape(-1, 1))
        prev_Y = np.zeros(K)
        X_t, Y_t = X, Y

        for n in range(num_iter):
            # Normalization
            denom = np.sum(A_e * Y_t, 1)
            X_t = r / denom

            # Newton method
            Y_t = np.zeros(K)
            for i in range(K):
                Y_t[i] = optimize.newton(self.f, prev_Y[i], maxiter=num_newton, args=(A_e[:, i], X_t, w, c[i]), tol=1.0e-01)
            prev_Y = Y_t
            Y_t = np.exp(-1 * Y_t.reshape(1, -1) / w.reshape(-1, 1))

        denom = np.sum(A_e * Y_t, 1)
        X_t = r / denom
        M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)

        return M

    @torch.no_grad()
    def gen_ulb_targets(self, algorithm, logits, use_hard_label=True, T=1.0,
                        softmax=True,  # whether to compute softmax for logits, input must be logits
                        label_smoothing=0.0):
        logits = logits.detach()
        if softmax:
            # probs = torch.softmax(logits / T, dim=1)
            probs = algorithm.compute_prob(logits / T)
        else:
            probs = logits

        targets_u = self.refine_pseudo_labels(algorithm.idx_ulb, probs,
                                              algorithm.it, algorithm.epoch)

        if use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(targets_u, dim=-1)

            if label_smoothing:
                pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
            return pseudo_label
        else:  # return soft label
            return targets_u
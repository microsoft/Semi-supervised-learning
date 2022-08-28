
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelHook
from semilearn.algorithms.utils import ce_loss, consistency_loss,  SSL_Argument, str2bool

import semilearn.algorithms as algs


class DARP(AlgorithmBase):
    def __init__(self, algorithm, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_alg = algs.__dict__[algorithm](*args, **kwargs)

        self.init(warm=args.darp_warm_epochs, alpha=args.darp_alpha,
                  iter_T=args.darp_iter_T, num_iter=args.darp_num_iter)

    def init(self, warm, alpha, iter_T, num_iter):
        self.warm = warm
        self.alpha = alpha
        self.iter_T = iter_T
        self.num_iter = num_iter

        self.pseudo_orig = torch.ones(len(train_unlabeled_set.targets), num_class) / num_class
        self.pseudo_refine = torch.ones(len(train_unlabeled_set.targets), num_class) / num_class

    def refine_pseudo_labels(self, idx_u, targets_u, batch_idx, epoch, target_disb):
        # Update the saved predictions with current one
        self.pseudo_orig[idx_u, :] = targets_u.data.cpu()
        self.pseudo_orig_backup = self.pseudo_orig.clone()

        if epoch > self.warm:
            if batch_idx % self.num_iter == 0:
                # Iterative normalization
                targets_u, weights_u = self.estimate_pseudo(target_disb, self.pseudo_orig)
                scale_term = targets_u * weights_u.reshape(1, -1)
                self.pseudo_orig = (self.pseudo_orig * scale_term + 1e-6) \
                                   / (self.pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                opt_res = self.opt_solver(self.pseudo_orig, target_disb)

                # Updated pseudo-labels are saved
                self.pseudo_refine = opt_res

                # Select
                targets_u = opt_res[idx_u].detach().cuda()
                self.pseudo_orig = self.pseudo_orig_backup
            else:
                # Using previously saved pseudo-labels
                targets_u = self.pseudo_refine[idx_u].cuda()

        return targets_u

    def estimate_pseudo(self, q_y, saved_q):
        pseudo_labels = torch.zeros(len(saved_q), num_class)
        k_probs = torch.zeros(num_class)

        for i in range(1, num_class + 1):
            i = num_class - i
            num_i = int(args.alpha * q_y[i])
            sorted_probs, idx = saved_q[:, i].sort(dim=0, descending=True)
            pseudo_labels[idx[: num_i], i] = 1
            k_probs[i] = sorted_probs[:num_i].sum()

        return pseudo_labels, (q_y + 1e-6) / (k_probs + 1e-6)

    def f(self, x, a, b, c, d):
        return np.sum(a * b * np.exp(-1 * x / c)) - d

    def opt_solver(self, probs, target_distb, num_iter=args.iter_T, num_newton=30):
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
                Y_t[i] = optimize.newton(f, prev_Y[i], maxiter=num_newton, args=(A_e[:, i], X_t, w, c[i]), tol=1.0e-01)
            prev_Y = Y_t
            Y_t = np.exp(-1 * Y_t.reshape(1, -1) / w.reshape(-1, 1))

        denom = np.sum(A_e * Y_t, 1)
        X_t = r / denom
        M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)

        return M

    def train_step(self, *args, **kwargs):
        # todo: how to call refine_pseudo_labels()
        self.base_alg.train_step(*args, **kwargs)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--darp_warm_epochs', int, 200),
            SSL_Argument('--darp_alpha', float, 2.0),
            SSL_Argument('--darp_iter_T', int, 10),
            SSL_Argument('--darp_num_iter', int, 10),
        ]

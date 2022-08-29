
import torch
from semilearn.algorithms.hooks import DistAlignEMAHook


class ProgressiveDistAlignEMAHook(DistAlignEMAHook):
    """
    Progressive Distribution Alignment in CRest
    """
    @torch.no_grad()
    def dist_align(self, algorithm, ulb_probs, lb_probs=None):
        # update queue
        self.update_p(algorithm, ulb_probs, lb_probs)

        # dist align
        p_target = self.p_target
        if algorithm.cur_dist_align_t != 1:
            p_target = p_target ** algorithm.cur_dist_align_t
            p_target = p_target / p_target.sum()
        ulb_probs_aligned = ulb_probs * (p_target + 1e-6) / (self.p_model + 1e-6)
        ulb_probs_aligned = ulb_probs_aligned / ulb_probs_aligned.sum(dim=-1, keepdim=True)
        return ulb_probs_aligned
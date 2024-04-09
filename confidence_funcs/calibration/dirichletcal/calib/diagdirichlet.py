import numpy as np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator
from ..utils import clip_for_log


class DiagonalDirichletCalibrator(FullDirichletCalibrator):

     def fit(self, X, y, *args, **kwargs):
        X_ = np.log(clip_for_log(X))
        self.calibrator_ = MultinomialRegression(method='Diag', l2=self.l2).fit(X_, y, *args, **kwargs)
        return self

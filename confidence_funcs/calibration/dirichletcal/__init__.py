from .version import __version__
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from .calib.fulldirichlet import FullDirichletCalibrator
from .calib.diagdirichlet import DiagonalDirichletCalibrator
from .calib.fixeddirichlet import FixedDiagonalDirichletCalibrator


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, matrix_type='full', l2=0.0, comp_l2=False,
                 initializer='identity'):
        print(('WARNING: DirichletCalibrator class is legacy code and needs' +
               'to be tested. Use the classes defined in dirichletcal.calib' +
               'instead.'))
        if matrix_type not in ['full', 'diagonal', 'fixed_diagonal']:
            raise ValueError
        self.matrix_type = matrix_type
        self.l2 = l2
        self.comp_l2 = comp_l2
        self.initializer = initializer


    def __setup(self):
        if isinstance(self.l2, list):
            self.l2_grid = self.l2
        else:
            self.l2_grid = [self.l2]
        if isinstance(self.comp_l2, list):
            self.comp_l2 = self.comp_l2
        else:
            self.comp_l2 = [self.comp_l2]
        self.calibrator_ = None


    def fit(self, x, y, x_val=None, y_val=None, **kwargs):
        self.__setup()

        if self.matrix_type == 'diagonal':
            self.calibrator_ = DiagonalDirichletCalibrator(
                l2=self.l2, initializer=self.initializer)
        elif self.matrix_type == 'fixed_diagonal':
            self.calibrator_ = FixedDiagonalDirichletCalibrator(
                l2=self.l2, initializer=self.initializer)
        elif self.matrix_type == 'full':
            self.calibrator_ = FullDirichletCalibrator(self.l2_grid, self.comp_l2,
                initializer=self.initializer)
        else:
            raise ValueError

        _X = np.copy(x)
        if len(x.shape) == 1:
            _X = np.vstack(((1 - _X), _X)).T

        _X_val = x_val
        if x_val is not None:
            _X_val = np.copy(x_val)
            if len(x_val.shape) == 1:
                _X_val = np.vstack(((1 - _X_val), _X_val)).T

        self.calibrator_ = self.calibrator_.fit(_X, y, X_val=_X_val,
                                                y_val=y_val, **kwargs)
        return self

    @property
    def l2_(self):
        if (self.calibrator_ is not None) and (hasattr(self.calibrator_, 'l2')):
            return self.calibrator_.l2
        return None

    @property
    def weights_(self):
        if (self.calibrator_ is not None) and (hasattr(self.calibrator_, 'weights_')):
            return self.calibrator_.weights_
        return None

    @property
    def coef_(self):
        if (self.calibrator_ is not None) and (hasattr(self.calibrator_, 'coef_')):
            return self.calibrator_.coef_
        return None

    @property
    def intercept_(self):
        if (self.calibrator_ is not None) and (hasattr(self.calibrator_, 'intercept_')):
            return self.calibrator_.intercept_
        return None

    @property
    def cannonical_weights(self):
        b = self.weights_[:, -1]
        w = self.weights_[:, :-1]
        col_min = np.min(w, axis=0)
        a = w - col_min

        def softmax(z):
            return np.divide(np.exp(z), np.sum(np.exp(z)))

        c = softmax(np.matmul(w, np.log(np.ones(len(b)) / len(b))) + b)
        return np.hstack((a, c.reshape(-1, 1)))

    def predict_proba(self, s):

        _s = np.copy(s)
        if len(s.shape) == 1:
            _s = np.vstack(((1 - _s), _s)).T
            return self.calibrator_.predict_proba(_s)[:, 1]

        return self.calibrator_.predict_proba(_s)

    def predict(self, s):

        _s = np.copy(s)
        if len(s.shape) == 1:
            _s = np.vstack(((1 - _s), _s)).T
            return self.calibrator_.predict(_s)[:, 1]

        return self.calibrator_.predict(_s)

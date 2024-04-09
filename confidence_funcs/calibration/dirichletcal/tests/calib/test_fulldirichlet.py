import unittest
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from . import get_simple_binary_example
from . import get_extreme_binary_example
from . import get_simple_ternary_example

from sklearn.metrics import accuracy_score


class TestFullDirichlet(unittest.TestCase):
    def setUp(self):
        self.cal = FullDirichletCalibrator()

    def test_fit_predict(self):
        for S, y in (get_simple_binary_example(),
                     get_simple_ternary_example()):
            self.cal = FullDirichletCalibrator()
            self.cal.fit(S, y)
            predictions = self.cal.predict_proba(S).argmax(axis=1)
            acc = accuracy_score(y, predictions)
            self.assertGreater(acc, 0.97,
                               "accuracy must be superior to 97 percent")

    def test_extreme_values(self):
        S, y = get_extreme_binary_example()
        self.cal = FullDirichletCalibrator()
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertGreater(acc, 0.99,
                           "accuracy must be superior to 99 percent")

    def test_lambda(self):
        S, y = get_simple_ternary_example()
        l2_odir = 1e-2
        cal = FullDirichletCalibrator(reg_lambda=l2_odir,
                                      reg_mu=l2_odir, reg_norm=False)
        cal.fit(S, y)
        predictions = cal.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertGreater(acc, 0.98,
                           "accuracy must be superior to 99 percent")

        l2_odir = 1e2
        cal = FullDirichletCalibrator(reg_lambda=l2_odir,
                                      reg_mu=l2_odir, reg_norm=False)
        cal.fit(S, y)
        predictions = cal.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertLess(acc, 0.6,
                        "accuracy must be smaller than 60 percent")


def main():
    unittest.main()


if __name__ == '__main__':
    main()

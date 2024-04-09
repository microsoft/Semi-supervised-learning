import unittest
import numpy as np
from dirichletcal.calib.vectorscaling import VectorScaling
from . import get_simple_binary_example
from . import get_extreme_binary_example
from . import get_simple_ternary_example

from sklearn.metrics import accuracy_score


class TestVectorScaling(unittest.TestCase):
    def setUp(self):
        self.cal = VectorScaling(logit_constant=0.0)

    def test_fit_predict(self):
        S, y = get_simple_binary_example()
        self.cal = VectorScaling(logit_constant=0.0)
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertGreater(acc, 0.99, "accuracy must be superior to 99 percent")

        S, y = get_simple_ternary_example()
        self.cal = VectorScaling(logit_constant=0.0)
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertGreater(acc, 0.98, "accuracy must be superior to 99 percent")

    def test_extreme_values(self):
        S, y = get_extreme_binary_example()
        self.cal = VectorScaling(logit_constant=0.0)
        self.cal.fit(S, y)
        predictions = self.cal.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertGreater(acc, 0.99, "accuracy must be superior to 99 percent")


def main():
    unittest.main()


if __name__ == '__main__':
    main()

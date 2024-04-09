import unittest
import numpy as np
from dirichletcal.calib.multinomial import MultinomialRegression
from dirichletcal.calib.multinomial import _get_weights
from . import get_simple_binary_example
from . import get_simple_ternary_example

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


class TestMultinomial(unittest.TestCase):
    def test_fit_predict(self):
        S, y = make_classification(n_samples=1000, n_classes=2, n_features=2,
                                   n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1, scale=100,
                                   class_sep=100.0, random_state=42)

        mlr = MultinomialRegression()
        mlr.fit(S, y)
        predictions = mlr.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertGreater(acc, 0.99, "accuracy must be superior to 99 percent")

        S, y = make_classification(n_samples=1000, n_classes=3, n_features=3,
                                   n_informative=3, n_redundant=0,
                                   n_clusters_per_class=1, scale=100,
                                   class_sep=100.0, random_state=243)
        mlr = MultinomialRegression()
        mlr.fit(S, y)
        predictions = mlr.predict_proba(S).argmax(axis=1)
        acc = accuracy_score(y, predictions)
        self.assertGreater(acc, 0.9, "accuracy must be superior to 9 percent")

    def test_get_weights_no_ref_row(self):
        k = 3
        params = np.arange(k*(k+1)) + 1
        full_matrix = _get_weights(params, k=k, ref_row=False, method='Full')
        expected = params.reshape(k, k+1)
        np.testing.assert_array_equal(full_matrix, expected)

        k = 3
        params = np.arange(k) + 1
        full_matrix = _get_weights(params, k=k, ref_row=False, method='FixDiag')
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], 'float')
        np.testing.assert_array_equal(full_matrix, expected)

    def test_get_weights_ref_row(self):
        k = 5
        params = np.arange(k*(k+1)) + 1
        full_matrix = _get_weights(params, k=k, ref_row=True, method='Full')
        expected = params.reshape(k, k+1)
        expected -= expected[-1, :]
        np.testing.assert_array_equal(full_matrix, expected)

        k = 3
        params = np.arange(k) + 1
        full_matrix = _get_weights(params, k=k, ref_row=True, method='FixDiag')
        expected = np.hstack((np.eye(k), np.zeros((k, 1))))
        expected -= expected[-1, :]
        np.testing.assert_array_equal(full_matrix, expected)

    def test_optimizers(self):
        S, y = make_classification(n_samples=20, n_classes=2, n_features=2,
                                   n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1, scale=10,
                                   class_sep=100.0, random_state=42)

        mlr = MultinomialRegression(optimizer='newton')
        mlr.fit(S, y)
        # TODO: add some test here

        mlr = MultinomialRegression(optimizer='fmin_l_bfgs_b')
        mlr.fit(S, y)
        # TODO: add some test here


def main():
    unittest.main()


if __name__ == '__main__':
    main()

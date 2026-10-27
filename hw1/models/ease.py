import numpy as np
from scipy import sparse, linalg


class EASE:
    def __init__(self, regularization=300):
        self.regularization = regularization

    def fit(self, events, n_users, n_items, **kwargs):
        likes = events[events.rating >= 4]
        self.history = sparse.csr_matrix(
            (np.ones(len(likes)), (likes.u, likes.i)), shape=(n_users, n_items)
        )
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::n_items + 1] += self.regularization
        factor = linalg.cho_factor(gram, lower=True, overwrite_a=True)
        precision = linalg.cho_solve(factor, np.eye(n_items))
        coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(coefficients, 0)
        self.coefficients = coefficients.astype(np.float32)
        return self

    def score(self, users, **kwargs):
        return np.asarray(self.history[users] @ self.coefficients)

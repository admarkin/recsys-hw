import numpy as np
from scipy import sparse


class ImplicitALS:
    def __init__(self, factors=32, regularization=1., alpha=40., iterations=20, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def fit(self, events, n_users, n_items, **kwargs):
        rng = np.random.default_rng(self.seed)
        positive = events[events.rating >= 4]
        matrix = sparse.csr_matrix((np.ones(len(positive)), (positive.u, positive.i)),
                                  shape=(n_users, n_items))
        matrix.data[:] = 1.
        self.users = rng.normal(0, .01, (n_users, self.factors))
        self.items = rng.normal(0, .01, (n_items, self.factors))
        self.loss_history = []
        eye = np.eye(self.factors)

        def update(history, fixed, output):
            gram = fixed.T @ fixed + self.regularization * eye
            for row in range(history.shape[0]):
                ids = history.indices[history.indptr[row]:history.indptr[row + 1]]
                vectors = fixed[ids]
                lhs = gram + self.alpha * (vectors.T @ vectors)
                rhs = (1. + self.alpha) * vectors.sum(axis=0)
                output[row] = np.linalg.solve(lhs, rhs)

        transposed = matrix.T.tocsr()
        rows, cols = matrix.nonzero()
        for _ in range(self.iterations):
            update(matrix, self.items, self.users)
            update(transposed, self.users, self.items)
            observed = np.sum(self.users[rows] * self.items[cols], axis=1)
            all_squared = np.sum((self.users.T @ self.users) * (self.items.T @ self.items))
            loss = all_squared - np.sum(observed ** 2)
            loss += (1. + self.alpha) * np.sum((1. - observed) ** 2)
            loss += self.regularization * (np.sum(self.users ** 2) + np.sum(self.items ** 2))
            self.loss_history.append(float(loss))
        return self

    def score(self, users, **kwargs):
        return self.users[users] @ self.items.T

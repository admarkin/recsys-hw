import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=100, shrinkage=10):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, events, n_users, n_items, **kwargs):
        likes = events[events.rating >= 4]
        self.history = sparse.csr_matrix(
            (np.ones(len(likes)), (likes.u, likes.i)), shape=(n_users, n_items)
        )
        cooccurrence = (self.history.T @ self.history).tocsc()
        norms = np.sqrt(np.asarray(self.history.sum(axis=0)).ravel())
        rows, cols, values = [], [], []
        for item in range(n_items):
            start, end = cooccurrence.indptr[item:item + 2]
            other = cooccurrence.indices[start:end]
            overlap = cooccurrence.data[start:end]
            similarity = overlap / np.maximum(norms[other] * norms[item], 1e-12)
            similarity *= overlap / (overlap + self.shrinkage)
            similarity[other == item] = 0
            order = np.lexsort((other, -similarity))[:self.neighbors]
            order = order[similarity[order] > 0]
            rows.extend(other[order])
            cols.extend(np.full(len(order), item))
            values.extend(similarity[order])
        self.similarity = sparse.csc_matrix(
            (values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32
        )
        return self

    def score(self, users, **kwargs):
        return (self.history[users] @ self.similarity).toarray()

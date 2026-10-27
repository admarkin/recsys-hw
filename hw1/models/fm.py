import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _FM(nn.Module):
    def __init__(self, n_users, n_features, factors):
        super().__init__()
        self.user_vectors = nn.Embedding(n_users, factors)
        self.user_linear = nn.Embedding(n_users, 1)
        self.feature_vectors = nn.Embedding(n_features, factors, padding_idx=0)
        self.feature_linear = nn.Embedding(n_features, 1, padding_idx=0)
        nn.init.normal_(self.user_vectors.weight, std=.03)
        nn.init.normal_(self.feature_vectors.weight, std=.03)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.feature_linear.weight)
        with torch.no_grad():
            self.feature_vectors.weight[0].zero_()

    def item_terms(self, ids, values):
        nonzero = values != 0
        lengths = nonzero.sum(dim=1)
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)[:-1]])
        flat_ids, flat_values = ids[nonzero], values[nonzero]
        summed = F.embedding_bag(flat_ids, self.feature_vectors.weight, offsets,
                                 mode='sum', per_sample_weights=flat_values)
        squared = F.embedding_bag(flat_ids, self.feature_vectors.weight.square(), offsets,
                                  mode='sum', per_sample_weights=flat_values.square())
        interactions = .5 * (summed.square() - squared).sum(dim=1)
        linear = F.embedding_bag(flat_ids, self.feature_linear.weight, offsets,
                                 mode='sum', per_sample_weights=flat_values).squeeze(-1)
        return summed, linear + interactions

    def forward(self, users, ids, values):
        item, bias = self.item_terms(ids, values)
        return (self.user_vectors(users) * item).sum(dim=1) + bias + self.user_linear(users).squeeze(-1)


class FactorizationMachine:
    def __init__(self, factors=32, regularization=.0001, epochs=20,
                 learning_rate=.01, metadata=True, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.metadata = metadata
        self.seed = seed

    def fit(self, events, n_users, n_items, features, **kwargs):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ids, values, feature_count = features
        if not self.metadata:
            ids, values, feature_count = ids[:, :1], values[:, :1], n_items + 1
        ids = torch.tensor(ids, dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)
        self.model = _FM(n_users, feature_count, self.factors)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        positive = events[events.rating >= 4][['u', 'i']].to_numpy(np.int64, copy=True)
        seen_keys = np.sort(events.u.to_numpy(np.int64) * n_items + events.i.to_numpy(np.int64))

        def is_known(users, items):
            keys = users * n_items + items
            positions = np.searchsorted(seen_keys, keys)
            return seen_keys[np.minimum(positions, len(seen_keys) - 1)] == keys

        negative_by_user = {
            int(u): group.i.to_numpy(np.int64)
            for u, group in events[events.rating < 4].groupby('u')
        }
        self.loss_history = []
        for _ in range(self.epochs):
            total_loss, total_count = 0., 0
            for batch in np.array_split(rng.permutation(len(positive)), max(1, int(np.ceil(len(positive) / 2048)))):
                pairs = positive[batch]
                u, pos = pairs[:, 0], pairs[:, 1]
                neg = rng.integers(n_items, size=len(batch))
                invalid = is_known(u, neg)
                while invalid.any():
                    neg[invalid] = rng.integers(n_items, size=invalid.sum())
                    invalid = is_known(u, neg)
                for j in np.flatnonzero(rng.random(len(batch)) < .5):
                    explicit_negative = negative_by_user.get(int(u[j]))
                    if explicit_negative is not None:
                        neg[j] = rng.choice(explicit_negative)
                users = torch.from_numpy(u.copy())
                score_pos = self.model(users, ids[pos], values[pos])
                score_neg = self.model(users, ids[neg], values[neg])
                loss = F.softplus(score_neg - score_pos).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
                total_count += len(batch)
            self.loss_history.append(total_loss / total_count)
        with torch.no_grad():
            self.item_vectors, self.item_bias = [x.numpy().copy() for x in self.model.item_terms(ids, values)]
            self.user_vectors = self.model.user_vectors.weight.detach().numpy().copy()
            self.user_bias = self.model.user_linear.weight.detach().numpy().ravel().copy()
        return self

    def score(self, users, **kwargs):
        return self.user_vectors[users] @ self.item_vectors.T + self.item_bias + self.user_bias[users, None]

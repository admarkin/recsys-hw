import numpy as np
import torch
from torch import nn
from scipy import sparse


class _TimeModel(nn.Module):
    def __init__(self, n_users, n_items, factors, n_bins):
        super().__init__()
        self.p = nn.Embedding(n_users, factors)
        self.q = nn.Embedding(n_items, factors)
        self.y = nn.EmbeddingBag(n_items, factors, mode='sum')
        self.drift = nn.Embedding(n_users, factors)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.alpha = nn.Embedding(n_users, 1)
        self.bit = nn.Embedding(n_items * n_bins, 1)
        self.n_bins = n_bins
        for embedding in [self.p, self.q, self.y]:
            nn.init.normal_(embedding.weight, std=.03)
        for embedding in [self.drift, self.bu, self.bi, self.alpha, self.bit]:
            nn.init.zeros_(embedding.weight)


class TimeSVDpp:
    def __init__(self, factors=32, regularization=.001, epochs=25,
                 learning_rate=.01, bin_days=365, beta=.4, temporal=True, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.bin_days = bin_days
        self.beta = beta
        self.temporal = temporal
        self.seed = seed

    def deviation(self, users, days):
        delta = (np.asarray(days) - self.mean_day[users]) / 365.25
        return np.sign(delta) * np.abs(delta) ** self.beta

    def fit(self, events, n_users, n_items, **kwargs):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ordered = events.sort_values(['u', 'i']).reset_index(drop=True)
        days = ordered.day.to_numpy(float)
        u, i = ordered.u.to_numpy(np.int64), ordered.i.to_numpy(np.int64)
        self.mean = float(ordered.rating.mean())
        self.start_day = float(days.min())
        self.n_bins = int((days.max() - self.start_day) // self.bin_days) + 1
        self.mean_day = ordered.groupby('u').day.mean().reindex(range(n_users), fill_value=days.mean()).to_numpy()
        history = sparse.csr_matrix((np.ones(len(ordered)), (u, i)), shape=(n_users, n_items))
        self.history = history
        self.model = _TimeModel(n_users, n_items, self.factors, self.n_bins)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        row_offsets = np.r_[0, np.cumsum(np.bincount(u, minlength=n_users))]
        dev = torch.tensor(self.deviation(u, days), dtype=torch.float32)
        bins = torch.tensor(((days - self.start_day) // self.bin_days).astype(np.int64))
        ratings = torch.tensor(ordered.rating.to_numpy(), dtype=torch.float32)
        self.loss_history = []
        for _ in range(self.epochs):
            total, count = 0., 0
            for batch_users in np.array_split(rng.permutation(n_users), max(1, int(np.ceil(n_users / 128)))):
                row_ids = np.concatenate([np.arange(row_offsets[user], row_offsets[user + 1]) for user in batch_users])
                if len(row_ids) == 0:
                    continue
                histories = [history.indices[history.indptr[user]:history.indptr[user + 1]] for user in batch_users]
                lengths = np.array([len(h) for h in histories])
                flat = torch.tensor(np.concatenate(histories), dtype=torch.long)
                offsets = torch.tensor(np.r_[0, np.cumsum(lengths)[:-1]], dtype=torch.long)
                implicit = self.model.y(flat, offsets) / torch.tensor(np.sqrt(np.maximum(lengths, 1)), dtype=torch.float32)[:, None]
                local_users = torch.repeat_interleave(torch.arange(len(batch_users)), torch.tensor(np.diff(row_offsets)[batch_users]))
                users = torch.tensor(u[row_ids], dtype=torch.long)
                items = torch.tensor(i[row_ids], dtype=torch.long)
                d = dev[row_ids] if self.temporal else torch.zeros(len(row_ids))
                vectors = self.model.p(users) + implicit[local_users] + self.model.drift(users) * d[:, None]
                prediction = self.mean + self.model.bu(users).squeeze(-1) + self.model.bi(items).squeeze(-1)
                prediction = prediction + self.model.alpha(users).squeeze(-1) * d
                if self.temporal:
                    prediction = prediction + self.model.bit(items * self.n_bins + bins[row_ids]).squeeze(-1)
                prediction = prediction + (vectors * self.model.q(items)).sum(dim=1)
                loss = (prediction - ratings[row_ids]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(row_ids)
                count += len(row_ids)
            self.loss_history.append(total / count)
        with torch.no_grad():
            flat = torch.tensor(history.indices.astype(np.int64))
            offsets = torch.tensor(history.indptr[:-1].astype(np.int64))
            lengths = np.diff(history.indptr)
            implicit = self.model.y(flat, offsets) / torch.tensor(np.sqrt(np.maximum(lengths, 1)), dtype=torch.float32)[:, None]
            self.user_vectors = (self.model.p.weight + implicit).detach().numpy().copy()
            self.item_vectors = self.model.q.weight.detach().numpy().copy()
            self.drift_vectors = self.model.drift.weight.detach().numpy().copy()
            self.user_bias = self.model.bu.weight.detach().numpy().ravel().copy()
            self.user_drift = self.model.alpha.weight.detach().numpy().ravel().copy()
            self.item_bias = self.model.bi.weight.detach().numpy().ravel().copy()
            self.item_time_bias = self.model.bit.weight.detach().numpy().reshape(n_items, self.n_bins).copy()
        return self

    def score(self, users, day, **kwargs):
        d = self.deviation(users, day) if self.temporal else np.zeros(len(users))
        vectors = self.user_vectors[users] + self.drift_vectors[users] * d[:, None]
        prediction = vectors @ self.item_vectors.T + self.mean + self.item_bias
        prediction += (self.user_bias[users] + self.user_drift[users] * d)[:, None]
        if self.temporal:
            time_bin = int(np.clip((day - self.start_day) // self.bin_days, 0, self.n_bins - 1))
            prediction += self.item_time_bias[:, time_bin]
        return prediction

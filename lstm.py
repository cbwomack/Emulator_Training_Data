# lstm.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict

jax.config.update("jax_enable_x64", True)

class TinyLSTM(nn.Module):
    hidden_size: int = 16
    dense_size: int = 32
    dropout_rate: float = 0.0  # default 0.0 for determinism in inverse runs
    train: bool = True

    @nn.compact
    def __call__(self, x, *, rng=None, train: bool | None = None):
        """
        x: (B, 1, D) single-step sequence
        Returns: (B, 1) prediction
        """
        if train is None:
            train = self.train

        B, T, D = x.shape
        assert T == 1, "This module is currently set up for seq_len=1"

        # LSTM over the single step
        lstm = nn.LSTMCell(self.hidden_size)
        h0 = jnp.zeros((B, self.hidden_size), dtype=jnp.float64)
        c0 = jnp.zeros((B, self.hidden_size), dtype=jnp.float64)
        x0 = x[:, 0, :]  # (B, D)

        (carry_h, carry_c), y_lstm = lstm((h0, c0), x0)
        h = carry_h

        if self.dropout_rate > 0.0 and train:
            if rng is None:
                raise ValueError("rng key required when dropout is enabled")
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train, rng=rng)

        h = nn.Dense(self.dense_size)(h)
        h = nn.relu(h)
        if self.dropout_rate > 0.0 and train:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train, rng=rng)

        y = nn.Dense(1)(h)       # (B, 1)
        return y.squeeze(-1)[:, None]  # (B, 1)

def init_lstm(rng, input_dim: int, hidden=16, dense=32, dropout=0.0, train=True):
    model = TinyLSTM(hidden_size=hidden, dense_size=dense, dropout_rate=dropout, train=train)
    dummy = jnp.zeros((2, 1, input_dim), dtype=jnp.float64)
    variables = model.init({"params": rng, "dropout": rng}, dummy, rng=rng, train=train)
    params = variables["params"]
    return model, params

def apply_lstm(params: FrozenDict, model: TinyLSTM, x, rng, train: bool):
    return model.apply({"params": params}, x, rng=rng, train=train)

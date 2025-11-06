# features.py
from __future__ import annotations
from typing import Dict, Tuple, Any, NamedTuple, Optional
import jax
import jax.numpy as jnp
from physics import emissions_to_temperature, PhysicsParams, PhysicsOut

jax.config.update("jax_enable_x64", True)

class ScalerState(NamedTuple):
    mean: jnp.ndarray
    std: jnp.ndarray
    eps: jnp.float64

def standardize_fit(X: jnp.ndarray, eps: float = 1e-9) -> ScalerState:
    """
    X: (N, D) → fit mean/std over samples axis.
    """
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    std = jnp.sqrt(jnp.maximum(var, jnp.float64(eps)))
    return ScalerState(mean=mean, std=std, eps=jnp.float64(eps))

def standardize_apply(X: jnp.ndarray, st: ScalerState) -> jnp.ndarray:
    return (X - st.mean) / st.std

def ppm_from_GtC(E_GtC_per_year: jnp.ndarray) -> jnp.ndarray:
    # match your internal convention: ppm = GtC / 2.12
    return E_GtC_per_year / jnp.float64(2.12)

def make_training_data_from_U(
    U_ppm_per_year: jnp.ndarray,
    gen_cfg: Dict[str, Any],
    physics_params: PhysicsParams,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Build features/labels from optimal-training emissions series U(t).
    To mirror your current pipeline (seq len = 1), we expose:
      - features: [U, Catm, RF] by default (configurable)
      - label: ΔT(t)
    Shapes:
      X: (T, 1, D)   1-step sequences
      y: (T,)        scalar target per step
    Returns (X, y, aux) with aux carrying raw series for optional use.
    """
    feats = gen_cfg.get("features", ("U", "Catm", "RF"))
    # Run physics once for this U:
    phys: PhysicsOut = emissions_to_temperature(U_ppm_per_year, physics_params)

    feat_cols = []
    for name in feats:
        if name.lower() == "u":
            feat_cols.append(U_ppm_per_year)
        elif name.lower() == "catm":
            feat_cols.append(phys.Catm)
        elif name.lower() == "rf":
            feat_cols.append(phys.RF)
        elif name.lower() in ("t", "temp", "delta_t"):
            feat_cols.append(phys.T)
        else:
            raise ValueError(f"Unknown feature name '{name}'")

    X2d = jnp.stack(feat_cols, axis=1)  # (T, D)
    X = X2d[:, None, :]                 # (T, 1, D)
    y = phys.T                          # (T,)
    aux = {"T": phys.T, "Catm": phys.Catm, "RF": phys.RF, "U": U_ppm_per_year}
    return X, y, aux

def make_eval_data_from_targets(
    W_list: Tuple[jnp.ndarray, ...],
    y_list: Tuple[jnp.ndarray, ...],
    scaler: ScalerState,
    n_features: int
) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
    """
    Apply a train-fitted scaler to evaluation features.
    Each W is shape (Te, 1, D). This function just standardizes across feature dim.
    """
    def _prep(W):
        Te = W.shape[0]
        W2 = W.reshape(Te, n_features)
        W2s = standardize_apply(W2, scaler).reshape(Te, 1, n_features)
        return W2s

    Xte = tuple(_prep(W) for W in W_list)
    return Xte, y_list

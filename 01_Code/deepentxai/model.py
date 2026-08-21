"""DEEPENTXAI — multimodal feature-fusion CNN-LSTM (Keras Functional API).

Four molecular views (ChemBERTa | Morgan | RDKit | MACCS) are encoded separately,
fused, reshaped into a sequence, then passed through Residual CNN blocks, Channel
Attention (squeeze-excite), a Bidirectional LSTM, Residual Dense blocks, Dropout
and fully-connected layers to a single Sigmoid activity neuron.

The CNN-LSTM identity of DEEPENTXAI is preserved and enhanced — not replaced.
"""
from __future__ import annotations

from typing import Dict

import keras
from keras import layers

# canonical input order used everywhere (model + training + prediction)
INPUT_ORDER = ["morgan", "rdkit", "maccs", "chemberta"]


# --------------------------------------------------------------------------- #
# Reusable blocks
# --------------------------------------------------------------------------- #
def _branch_encoder(inp, embed_dim: int, act: str, dropout: float, name: str):
    y = layers.Dense(256, name=f"{name}_dense1")(inp)
    y = layers.BatchNormalization(name=f"{name}_bn")(y)
    y = layers.Activation(act, name=f"{name}_act")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop")(y)
    return layers.Dense(embed_dim, activation=act, name=f"{name}_embed")(y)


def _residual_cnn_block(x, filters: int, kernel: int, act: str, i: int):
    sc = x
    y = layers.Conv1D(filters, kernel, padding="same", name=f"cnn{i}_c1")(x)
    y = layers.BatchNormalization(name=f"cnn{i}_bn1")(y)
    y = layers.Activation(act, name=f"cnn{i}_a1")(y)
    y = layers.Conv1D(filters, kernel, padding="same", name=f"cnn{i}_c2")(y)
    y = layers.BatchNormalization(name=f"cnn{i}_bn2")(y)
    if sc.shape[-1] != filters:
        sc = layers.Conv1D(filters, 1, padding="same", name=f"cnn{i}_proj")(sc)
    y = layers.Add(name=f"cnn{i}_add")([sc, y])
    return layers.Activation(act, name=f"cnn{i}_out")(y)


def _channel_attention(x, ratio: int = 8, name: str = "se"):
    c = x.shape[-1]
    s = layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)
    s = layers.Dense(max(c // ratio, 4), activation="relu", name=f"{name}_fc1")(s)
    s = layers.Dense(c, activation="sigmoid", name=f"{name}_fc2")(s)
    s = layers.Reshape((1, c), name=f"{name}_rs")(s)
    return layers.Multiply(name=f"{name}_scale")([x, s])


def _residual_dense_block(x, units: int, act: str, dropout: float, i: int):
    sc = x
    y = layers.Dense(units, name=f"rd{i}_d1")(x)
    y = layers.BatchNormalization(name=f"rd{i}_bn1")(y)
    y = layers.Activation(act, name=f"rd{i}_a1")(y)
    y = layers.Dropout(dropout, name=f"rd{i}_drop")(y)
    y = layers.Dense(units, name=f"rd{i}_d2")(y)
    y = layers.BatchNormalization(name=f"rd{i}_bn2")(y)
    if sc.shape[-1] != units:
        sc = layers.Dense(units, name=f"rd{i}_proj")(sc)
    y = layers.Add(name=f"rd{i}_add")([sc, y])
    return layers.Activation(act, name=f"rd{i}_out")(y)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
def build_deepentxai(dims: Dict[str, int], hp: Dict) -> keras.Model:
    """Assemble the DEEPENTXAI multimodal fusion CNN-LSTM.

    dims : {'morgan':2048,'rdkit':100,'maccs':167,'chemberta':768}
    hp   : embed_dim, cnn_blocks, cnn_filters, cnn_kernel, lstm_units,
           dense_units, dropout, activation
    """
    e = int(hp["embed_dim"]); act = hp["activation"]; dr = float(hp["dropout"])
    seq_channels = 8

    inputs = {m: keras.Input((dims[m],), name=m) for m in INPUT_ORDER}
    encoded = [_branch_encoder(inputs[m], e, act, dr, m) for m in INPUT_ORDER]

    fused = layers.Concatenate(name="fusion")(encoded)          # (4*e,)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)

    # project + reshape the fused vector into a (T, C) pseudo-sequence for CNN-LSTM
    x = layers.Dense(64 * seq_channels, activation=act, name="to_seq")(fused)
    x = layers.Reshape((64, seq_channels), name="reshape_seq")(x)

    for i in range(int(hp["cnn_blocks"])):
        x = _residual_cnn_block(x, int(hp["cnn_filters"]), int(hp["cnn_kernel"]), act, i)
    x = _channel_attention(x)
    x = layers.MaxPooling1D(2, name="pool")(x)
    x = layers.Bidirectional(layers.LSTM(int(hp["lstm_units"])), name="bilstm")(x)

    x = _residual_dense_block(x, int(hp["dense_units"]), act, dr, 0)
    x = layers.Dropout(dr, name="head_drop")(x)
    x = layers.Dense(int(hp["dense_units"]), activation=act, name="head_dense")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="activity")(x)

    return keras.Model(inputs=[inputs[m] for m in INPUT_ORDER], outputs=out, name="DEEPENTXAI")

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import ctypes
import hashlib
import json
import math
import random
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    _RL_AVAILABLE = True
    _RL_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    gym = None
    spaces = None
    PPO = None
    BaseCallback = object
    Monitor = None
    DummyVecEnv = None
    _RL_AVAILABLE = False
    _RL_IMPORT_ERROR = exc

WordDiff = Tuple[int, int]
PolytopeDiff = Tuple[WordDiff, ...]
PolytopePair = Tuple[PolytopeDiff, PolytopeDiff]
UINT16_MODULUS = 1 << 16
SPECK32_BLOCK_BITS = 32
BIT_SHIFTS = np.arange(15, -1, -1, dtype=np.uint16)

def hamming_weight(diff: Sequence[int]) -> int:
    return int(int(diff[0]).bit_count() + int(diff[1]).bit_count())
def normalize_difference(diff: Sequence[int]) -> WordDiff:
    if len(diff) != 2:
        raise ValueError(f"差分必须包含两个 16-bit 字，实际为 {diff!r}")
    left, right = int(diff[0]), int(diff[1])
    if not (0 <= left < UINT16_MODULUS and 0 <= right < UINT16_MODULUS):
        raise ValueError(f"差分超出 uint16 范围: {(left, right)}")
    return left, right
def canonical_polytope(polytope: Sequence[Sequence[int]]) -> PolytopeDiff:
    normalized = tuple(sorted(normalize_difference(d) for d in polytope))
    if not normalized:
        raise ValueError("多胞体至少需要一个非零输入差分")
    if any(d == (0, 0) for d in normalized):
        raise ValueError("差分元组不能包含零差分；P0 已是锚点")
    if len(set(normalized)) != len(normalized):
        raise ValueError("同一多胞体内不能出现重复差分")
    return normalized
def canonical_pair(polytope_0: Sequence[Sequence[int]], polytope_1: Sequence[Sequence[int]]) -> PolytopePair:
    p0 = canonical_polytope(polytope_0)
    p1 = canonical_polytope(polytope_1)
    if len(p0) != len(p1):
        raise ValueError("两个类别必须具有相同数量的非锚点差分")
    if p0 == p1:
        raise ValueError("两个类别的多胞体差分元组不能完全相同")
    return (p0, p1) if p0 < p1 else (p1, p0)

def pair_to_jsonable(pair: PolytopePair) -> Dict[str, List[List[int]]]:
    return {
        "polytope_label_0": [[int(a), int(b)] for a, b in pair[0]],
        "polytope_label_1": [[int(a), int(b)] for a, b in pair[1]],
    }

def pair_cache_key(pair: PolytopePair) -> str:
    payload = json.dumps(pair_to_jsonable(pair), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class CryptoPanel:
    keys: np.ndarray
    plaintext0: np.ndarray
    seed: int

    @property
    def sample_size(self) -> int:
        return int(self.plaintext0.shape[0])


class RL_SearchBestDiff:

    def __init__(self, config_path: str, nr: int):
        self.config_path = str(config_path)
        self.config = self.load_config(config_path)
        self.nr = int(nr)
        if not 1 <= self.nr <= 22:
            raise ValueError(f"SPECK32/64 轮数必须位于 [1,22]，当前为 {self.nr}")

        self.block_size = int(self.config["ciphertext_size"])
        self.master_key_size = int(self.config["master_key_size"])
        self.plaintext_size = int(self.config["plaintext_size"])
        data_type_name = self.config["data_type"]

        dtype_map = {
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
        }
        ctypes_map = {
            "uint8": ctypes.c_uint8,
            "uint16": ctypes.c_uint16,
            "uint32": ctypes.c_uint32,
            "uint64": ctypes.c_uint64,
        }
        if data_type_name not in dtype_map:
            raise ValueError(f"不支持的数据类型: {data_type_name}")

        self.data_type = dtype_map[data_type_name]
        self.ctypes_type = ctypes_map[data_type_name]
        self.word_size = np.dtype(self.data_type).itemsize * 8
        self.cipher_path = str(self.config["saved_path"])

        # The revised pipeline is intentionally specific to SPECK32/64.
        if (
            self.block_size != 32
            or self.plaintext_size != 32
            or self.master_key_size != 64
            or self.word_size != 16
        ):
            raise ValueError(
                "本实现针对 SPECK32/64：ciphertext/plaintext=32 bits, key=64 bits, word=uint16"
            )

        self._cipher_dll = None
        self._encrypt_batch = None
        self._encrypt_polytope_batch = None

    @staticmethod
    def load_config(config_path: str) -> Mapping[str, object]:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_encrypt_function(self):
        """Load the DLL/SO. Return signature remains compatible with the old code."""
        dll = ctypes.CDLL(self.cipher_path)
        encrypt_batch = getattr(dll, "encrypt_batch")
        encrypt_batch.argtypes = (
            ctypes.POINTER(self.ctypes_type),
            ctypes.POINTER(self.ctypes_type),
            ctypes.c_int,
            ctypes.POINTER(self.ctypes_type),
            ctypes.c_int,
        )
        encrypt_batch.restype = None

        encrypt_polytope = getattr(dll, "encrypt_polytope_batch", None)
        if encrypt_polytope is not None:
            encrypt_polytope.argtypes = (
                ctypes.POINTER(self.ctypes_type),  # plaintext0
                ctypes.POINTER(self.ctypes_type),  # diffs
                ctypes.c_int,                     # num_diffs
                ctypes.POINTER(self.ctypes_type),  # keys
                ctypes.c_int,                     # rounds
                ctypes.POINTER(self.ctypes_type),  # ciphertexts
                ctypes.c_int,                     # batch_size
            )
            encrypt_polytope.restype = ctypes.c_int

        self._cipher_dll = dll
        self._encrypt_batch = encrypt_batch
        self._encrypt_polytope_batch = encrypt_polytope
        return dll, encrypt_batch

    def _require_loaded(self, encrypt_batch=None):
        function = encrypt_batch if encrypt_batch is not None else self._encrypt_batch
        if function is None:
            raise RuntimeError("尚未加载加密库，请先调用 load_encrypt_function()")
        return function

    def make_panel(self, sample_size: int, seed: int) -> CryptoPanel:
        if sample_size <= 0:
            raise ValueError("sample_size 必须为正整数")
        rng = np.random.default_rng(int(seed))
        keys = rng.integers(
            0,
            UINT16_MODULUS,
            size=(sample_size, 4),
            dtype=np.uint16,
        )
        plaintext0 = rng.integers(
            0,
            UINT16_MODULUS,
            size=(sample_size, 2),
            dtype=np.uint16,
        )
        return CryptoPanel(
            keys=np.ascontiguousarray(keys),
            plaintext0=np.ascontiguousarray(plaintext0),
            seed=int(seed),
        )

    def encrypt(self, plaintext, keys, ciphertext, sample_size, encrypt_batch=None) -> None:
        function = self._require_loaded(encrypt_batch)
        plaintext = np.require(plaintext, dtype=np.uint16, requirements=["C", "A"])
        keys = np.require(keys, dtype=np.uint16, requirements=["C", "A"])
        ciphertext = np.require(ciphertext, dtype=np.uint16, requirements=["C", "A", "W"])
        function(
            plaintext.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
            keys.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
            self.nr,
            ciphertext.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
            int(sample_size),
        )

    def encrypt_polytope(
        self,
        polytope: Sequence[Sequence[int]],
        panel: CryptoPanel,
        encrypt_batch=None,
    ) -> np.ndarray:
        """Return ciphertext vertices with shape (d+1, N, 2), vertex-major."""
        normalized = canonical_polytope(polytope)
        n = panel.sample_size
        diffs = np.ascontiguousarray(normalized, dtype=np.uint16)
        output = np.empty((len(normalized) + 1, n, 2), dtype=np.uint16)

        if self._encrypt_polytope_batch is not None:
            status = self._encrypt_polytope_batch(
                panel.plaintext0.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
                diffs.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
                len(normalized),
                panel.keys.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
                self.nr,
                output.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
                n,
            )
            if status != 0:
                raise RuntimeError(f"encrypt_polytope_batch 返回错误码 {status}")
            return output

        function = self._require_loaded(encrypt_batch)
        all_diffs: Tuple[WordDiff, ...] = ((0, 0),) + normalized
        for vertex, diff in enumerate(all_diffs):
            if diff == (0, 0):
                plaintext = panel.plaintext0
            else:
                plaintext = np.ascontiguousarray(
                    panel.plaintext0 ^ np.asarray(diff, dtype=np.uint16)
                )
            self.encrypt(plaintext, panel.keys, output[vertex], n, function)
        return output

    def make_xor_outputs(
        self,
        diffs: Sequence[Sequence[int]],
        panel: CryptoPanel,
        encrypt_batch=None,
    ) -> List[np.ndarray]:
        normalized = canonical_polytope(diffs)
        vertices = self.encrypt_polytope(normalized, panel, encrypt_batch)
        anchor = vertices[0]
        return [np.bitwise_xor(anchor, vertices[i + 1]) for i in range(len(normalized))]

    def make_train_data(self, diffs, sample_size, encrypt_batch=None):
        seed = int.from_bytes(os.urandom(8), "little", signed=False)
        panel = self.make_panel(sample_size, seed)
        return self.make_xor_outputs(diffs, panel, encrypt_batch)

    @staticmethod
    def convert_to_binary(arr: np.ndarray, word_size: int = 16) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"arr 必须为二维数组，实际形状 {arr.shape}")
        words = arr.T.astype(np.uint64, copy=False)
        shifts = np.arange(word_size - 1, -1, -1, dtype=np.uint64)
        bits = ((words[..., None] >> shifts) & 1).astype(np.uint8)
        return bits.reshape(words.shape[0], words.shape[1] * word_size)

def words_to_bits(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint16)
    if words.ndim != 2 or words.shape[1] != 2:
        raise ValueError(f"期望 words 形状为 (N,2)，实际为 {words.shape}")
    return ((words[..., None] >> BIT_SHIFTS) & 1).astype(np.uint8, copy=False)


def one_word_to_bits(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint16)
    if words.ndim != 1:
        raise ValueError(f"期望一维 word 数组，实际为 {words.shape}")
    return ((words[:, None] >> BIT_SHIFTS) & 1).astype(np.uint8, copy=False)


def ror16(x: np.ndarray, r: int) -> np.ndarray:
    r %= 16
    value = x.astype(np.uint32, copy=False)
    return (((value >> r) | (value << (16 - r))) & 0xFFFF).astype(np.uint16)


def speck_previous_right_word(ciphertext: np.ndarray) -> np.ndarray:
    return ror16(np.bitwise_xor(ciphertext[:, 1], ciphertext[:, 0]), 2)


def feature_channel_count(
    num_diffs: int,
    include_anchor_xor: bool = True,
    include_previous_right: bool = True,
) -> int:
    channels = 2 * (num_diffs + 1)  # raw x/y words for every vertex
    if include_anchor_xor:
        channels += 2 * num_diffs
    if include_previous_right:
        channels += num_diffs + 1
        channels += num_diffs
    return channels


def vertices_to_bit_features(
    vertices: np.ndarray,
    include_anchor_xor: bool = True,
    include_previous_right: bool = True,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.uint16)
    if vertices.ndim != 3 or vertices.shape[2] != 2:
        raise ValueError(f"vertices 必须具有形状 (V,N,2)，实际为 {vertices.shape}")
    if vertices.shape[0] < 2:
        raise ValueError("至少需要锚点和一个非锚点顶点")

    channels: List[np.ndarray] = []
    for vertex in vertices:
        bits = words_to_bits(vertex)  # (N,2,16)
        channels.extend([bits[:, 0, :], bits[:, 1, :]])

    if include_anchor_xor:
        anchor = vertices[0]
        for vertex in vertices[1:]:
            bits = words_to_bits(np.bitwise_xor(anchor, vertex))
            channels.extend([bits[:, 0, :], bits[:, 1, :]])

    if include_previous_right:
        previous_right = [speck_previous_right_word(vertex) for vertex in vertices]
        channels.extend(one_word_to_bits(word) for word in previous_right)
        anchor_previous = previous_right[0]
        channels.extend(
            one_word_to_bits(np.bitwise_xor(anchor_previous, word))
            for word in previous_right[1:]
        )

    return np.stack(channels, axis=1).astype(np.uint8, copy=False)


def flatten_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.uint8)
    if features.ndim != 3:
        raise ValueError(f"features 必须是 (N,C,16)，实际为 {features.shape}")
    return np.ascontiguousarray(features.reshape(features.shape[0], -1))

@dataclass(frozen=True)
class ProjectionBank:
    hash_indices: np.ndarray
    pair_indices: np.ndarray
    seed: int

    @classmethod
    def build(
        cls,
        feature_dim: int,
        seed: int,
        num_hashes: int = 48,
        hash_width: int = 8,
        num_pairs: int = 128,
    ) -> "ProjectionBank":
        if feature_dim < hash_width:
            raise ValueError("feature_dim 小于 hash_width")
        rng = np.random.default_rng(seed)
        hash_indices = np.stack(
            [rng.choice(feature_dim, size=hash_width, replace=False) for _ in range(num_hashes)],
            axis=0,
        ).astype(np.int32)
        pair_indices = np.stack(
            [rng.choice(feature_dim, size=2, replace=False) for _ in range(num_pairs)],
            axis=0,
        ).astype(np.int32)
        return cls(hash_indices=hash_indices, pair_indices=pair_indices, seed=int(seed))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def projected_js_score(
    class_0: np.ndarray,
    class_1: np.ndarray,
    projection_bank: ProjectionBank,
    smoothing: float = 0.5,
) -> float:
    """Average JSD of fixed 8-bit projections of the COMPLETE feature vectors."""
    if class_0.shape[1] != class_1.shape[1]:
        raise ValueError("两个类别的特征维度不一致")
    width = projection_bank.hash_indices.shape[1]
    if width > 16:
        raise ValueError("当前离散投影仅支持 width <= 16")
    weights = (1 << np.arange(width, dtype=np.uint32))[None, :]
    bins = 1 << width
    scores = []

    for indices in projection_bank.hash_indices:
        code_0 = np.sum(class_0[:, indices].astype(np.uint32) * weights, axis=1)
        code_1 = np.sum(class_1[:, indices].astype(np.uint32) * weights, axis=1)
        count_0 = np.bincount(code_0.astype(np.int64), minlength=bins).astype(np.float64)
        count_1 = np.bincount(code_1.astype(np.int64), minlength=bins).astype(np.float64)
        count_0 += smoothing
        count_1 += smoothing
        p = count_0 / count_0.sum()
        q = count_1 / count_1.sum()
        scores.append(_js_divergence(p, q))
    return float(np.mean(scores))


def marginal_effect_score(class_0: np.ndarray, class_1: np.ndarray) -> float:
    """Mean absolute bit-probability gap, in [0,1]."""
    return float(np.mean(np.abs(class_0.mean(axis=0) - class_1.mean(axis=0))))


def correlation_gap_score(
    class_0: np.ndarray,
    class_1: np.ndarray,
    projection_bank: ProjectionBank,
) -> float:
    """Average difference of selected second-order ±1 correlations, normalized to [0,1]."""
    idx_a = projection_bank.pair_indices[:, 0]
    idx_b = projection_bank.pair_indices[:, 1]

    x0a = class_0[:, idx_a].astype(np.float32) * 2.0 - 1.0
    x0b = class_0[:, idx_b].astype(np.float32) * 2.0 - 1.0
    x1a = class_1[:, idx_a].astype(np.float32) * 2.0 - 1.0
    x1b = class_1[:, idx_b].astype(np.float32) * 2.0 - 1.0

    corr_0 = np.mean(x0a * x0b, axis=0)
    corr_1 = np.mean(x1a * x1b, axis=0)
    return float(np.mean(np.abs(corr_0 - corr_1)) / 2.0)


def input_overlap_ratio(pair: PolytopePair) -> float:
    shared = len(set(pair[0]).intersection(pair[1]))
    return float(shared / len(pair[0]))


@dataclass(frozen=True)
class SurrogateConfig:
    include_anchor_xor: bool = True
    include_previous_right: bool = True
    js_weight: float = 0.65
    marginal_weight: float = 0.15
    correlation_weight: float = 0.20
    overlap_penalty: float = 0.05


def evaluate_pair_on_panels(
    searcher: RL_SearchBestDiff,
    pair: PolytopePair,
    panels: Sequence[CryptoPanel],
    encrypt_batch,
    projection_bank: ProjectionBank,
    surrogate: SurrogateConfig,
) -> Dict[str, float]:
    pair = canonical_pair(pair[0], pair[1])
    panel_records: List[Dict[str, float]] = []

    for panel in panels:
        vertices_0 = searcher.encrypt_polytope(pair[0], panel, encrypt_batch)
        vertices_1 = searcher.encrypt_polytope(pair[1], panel, encrypt_batch)
        flat_0 = flatten_features(
            vertices_to_bit_features(
                vertices_0,
                include_anchor_xor=surrogate.include_anchor_xor,
                include_previous_right=surrogate.include_previous_right,
            )
        )
        flat_1 = flatten_features(
            vertices_to_bit_features(
                vertices_1,
                include_anchor_xor=surrogate.include_anchor_xor,
                include_previous_right=surrogate.include_previous_right,
            )
        )

        js = projected_js_score(flat_0, flat_1, projection_bank)
        marginal = marginal_effect_score(flat_0, flat_1)
        correlation = correlation_gap_score(flat_0, flat_1, projection_bank)
        raw = (
            surrogate.js_weight * js
            + surrogate.marginal_weight * marginal
            + surrogate.correlation_weight * correlation
        )
        panel_records.append(
            {
                "projected_js": js,
                "marginal_gap": marginal,
                "correlation_gap": correlation,
                "raw_score": raw,
            }
        )

    raw_scores = np.asarray([record["raw_score"] for record in panel_records], dtype=np.float64)
    overlap = input_overlap_ratio(pair)
    stability = float(1.0 / (1.0 + np.std(raw_scores) / (np.mean(raw_scores) + 1e-12)))
    mean_raw = float(np.mean(raw_scores))
    score = mean_raw * (0.8 + 0.2 * stability) - surrogate.overlap_penalty * overlap

    return {
        "score": float(score),
        "projected_js": float(np.mean([r["projected_js"] for r in panel_records])),
        "marginal_gap": float(np.mean([r["marginal_gap"] for r in panel_records])),
        "correlation_gap": float(np.mean([r["correlation_gap"] for r in panel_records])),
        "stability": stability,
        "input_overlap": overlap,
    }



def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[positives].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def proxy_mlp_auc(
    searcher: RL_SearchBestDiff,
    pair: PolytopePair,
    encrypt_batch,
    sample_size_per_class: int,
    seed: int,
    surrogate: SurrogateConfig,
    epochs: int = 3,
    batch_size: int = 1024,
) -> Dict[str, float]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("proxy_mlp_auc 需要 PyTorch") from exc

    if sample_size_per_class < 256:
        raise ValueError("proxy 样本量过小")
    pair = canonical_pair(pair[0], pair[1])

    def make_split(panel_seed: int):
        panel_0 = searcher.make_panel(sample_size_per_class, panel_seed)
        panel_1 = searcher.make_panel(sample_size_per_class, panel_seed + 1)
        features_0 = flatten_features(
            vertices_to_bit_features(
                searcher.encrypt_polytope(pair[0], panel_0, encrypt_batch),
                surrogate.include_anchor_xor,
                surrogate.include_previous_right,
            )
        )
        features_1 = flatten_features(
            vertices_to_bit_features(
                searcher.encrypt_polytope(pair[1], panel_1, encrypt_batch),
                surrogate.include_anchor_xor,
                surrogate.include_previous_right,
            )
        )
        x = np.concatenate([features_0, features_1], axis=0).astype(np.float32)
        x = x * 2.0 - 1.0
        y = np.concatenate(
            [np.zeros(sample_size_per_class), np.ones(sample_size_per_class)]
        ).astype(np.float32)
        rng = np.random.default_rng(panel_seed + 2)
        permutation = rng.permutation(y.size)
        return x[permutation], y[permutation]

    x_train, y_train = make_split(seed)
    x_val, y_val = make_split(seed + 10_000_019)

    torch.manual_seed(seed & 0xFFFFFFFF)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_val)).squeeze(1).numpy()
    auc = _binary_auc(y_val.astype(np.int8), logits)
    accuracy = float(np.mean((logits >= 0.0) == (y_val >= 0.5)))
    return {
        "proxy_auc": float(auc),
        "proxy_accuracy": accuracy,
        "proxy_advantage": float(max(0.0, 2.0 * auc - 1.0)),
    }


@dataclass
class SearchConfig:
    num_diffs: int = 3
    min_hw: int = 1
    max_hw: int = 6
    max_steps: int = 64

    coarse_samples: int = 4096
    coarse_panel_count: int = 2
    projection_hashes: int = 48
    projection_width: int = 8
    projection_pairs: int = 128

    total_timesteps: int = 20_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.95
    gae_lambda: float = 0.90
    ent_coef: float = 0.01
    clip_range: float = 0.20

    reward_scale: float = 50.0
    invalid_penalty: float = -0.10
    no_op_penalty: float = -0.002
    step_penalty: float = -0.001
    archive_size: int = 64

    refine_top_k: int = 12
    refine_samples: int = 16_384
    refine_panel_count: int = 3
    proxy_top_k: int = 3
    proxy_samples_per_class: int = 4096
    proxy_epochs: int = 3

    base_seed: int = 20260712
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    def validate(self) -> None:
        if self.num_diffs < 1:
            raise ValueError("num_diffs 必须 >= 1")
        if not 1 <= self.min_hw <= self.max_hw <= SPECK32_BLOCK_BITS:
            raise ValueError("汉明重量约束非法")
        if self.max_steps < 1:
            raise ValueError("max_steps 必须为正")
        if self.coarse_panel_count < 1 or self.refine_panel_count < 1:
            raise ValueError("panel_count 必须为正")
        if self.refine_top_k < 1:
            raise ValueError("refine_top_k 必须为正")


def _diffs_to_bits(diffs: Sequence[WordDiff]) -> np.ndarray:
    output = []
    for left, right in diffs:
        for value in (left, right):
            output.extend((value >> shift) & 1 for shift in range(15, -1, -1))
    return np.asarray(output, dtype=np.int8)


def _bits_to_diffs(bits: np.ndarray, num_diffs: int) -> PolytopeDiff:
    bits = np.asarray(bits, dtype=np.int8)
    expected = num_diffs * SPECK32_BLOCK_BITS
    if bits.size != expected:
        raise ValueError(f"状态位数错误: {bits.size} != {expected}")
    diffs: List[WordDiff] = []
    for index in range(num_diffs):
        block = bits[index * 32 : (index + 1) * 32]
        left = 0
        right = 0
        for bit in block[:16]:
            left = (left << 1) | int(bit)
        for bit in block[16:]:
            right = (right << 1) | int(bit)
        diffs.append((left, right))
    return tuple(diffs)


def pair_to_state(pair: PolytopePair) -> np.ndarray:
    return np.concatenate([_diffs_to_bits(pair[0]), _diffs_to_bits(pair[1])]).astype(np.int8)


def state_to_pair(state: np.ndarray, num_diffs: int) -> PolytopePair:
    half = num_diffs * SPECK32_BLOCK_BITS
    p0 = _bits_to_diffs(state[:half], num_diffs)
    p1 = _bits_to_diffs(state[half:], num_diffs)
    return canonical_pair(p0, p1)


def validate_pair_constraints(pair: PolytopePair, config: SearchConfig) -> bool:
    try:
        pair = canonical_pair(pair[0], pair[1])
    except ValueError:
        return False
    if len(pair[0]) != config.num_diffs:
        return False
    for polytope in pair:
        for diff in polytope:
            hw = hamming_weight(diff)
            if hw < config.min_hw or hw > config.max_hw:
                return False
    return True


def _input_jaccard_distance(a: WordDiff, b: WordDiff) -> float:
    x = (a[0] << 16) | a[1]
    y = (b[0] << 16) | b[1]
    union = (x | y).bit_count()
    if union == 0:
        return 0.0
    return float((x ^ y).bit_count() / union)


def build_initial_pair(
    seed_pool: Sequence[Mapping[str, object]],
    anchor_index: int,
    config: SearchConfig,
    rng: np.random.Generator,
) -> PolytopePair:
    candidates: List[WordDiff] = []
    qualities: Dict[WordDiff, float] = {}
    for record in seed_pool:
        try:
            diff = normalize_difference(record["diff"])
        except Exception:
            continue
        hw = hamming_weight(diff)
        if config.min_hw <= hw <= config.max_hw and diff != (0, 0):
            if diff not in qualities:
                candidates.append(diff)
            qualities[diff] = max(float(record.get("quality", record.get("score", 0.0))), qualities.get(diff, -np.inf))

    if len(candidates) < 2 * config.num_diffs:
        raise ValueError(
            f"有效种子数量不足：至少需要 {2 * config.num_diffs} 个，实际为 {len(candidates)}"
        )

    anchor = normalize_difference(seed_pool[anchor_index % len(seed_pool)]["diff"])
    if anchor not in candidates:
        anchor = candidates[anchor_index % len(candidates)]

    # Deterministic quality order with a small RNG tie-break.
    random_tie = {d: float(rng.random()) for d in candidates}
    ordered = sorted(candidates, key=lambda d: (qualities[d], random_tie[d]), reverse=True)

    def choose_diverse(selected: List[WordDiff], forbidden: set, count: int) -> List[WordDiff]:
        result = list(selected)
        while len(result) < count:
            available = [d for d in ordered if d not in forbidden and d not in result]
            if not available:
                raise RuntimeError("无法构造满足唯一性要求的初始多胞体")
            if not result:
                choice = available[0]
            else:
                choice = max(
                    available,
                    key=lambda d: (
                        min(_input_jaccard_distance(d, s) for s in result),
                        qualities[d],
                    ),
                )
            result.append(choice)
        return result

    poly_0 = choose_diverse([anchor], set(), config.num_diffs)
    poly_1 = choose_diverse([], set(poly_0), config.num_diffs)
    pair = canonical_pair(poly_0, poly_1)
    if not validate_pair_constraints(pair, config):
        raise RuntimeError("内部错误：生成的初始 pair 不满足约束")
    return pair


if _RL_AVAILABLE:  # pragma: no branch

    class PairPolytopicEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            searcher: RL_SearchBestDiff,
            encrypt_batch,
            initial_pair: PolytopePair,
            config: SearchConfig,
            worker_seed: int,
        ):
            super().__init__()
            config.validate()
            self.searcher = searcher
            self.encrypt_batch = encrypt_batch
            self.config = config
            self.worker_seed = int(worker_seed)
            self.rng = np.random.default_rng(self.worker_seed)

            self.initial_pair = canonical_pair(initial_pair[0], initial_pair[1])
            if not validate_pair_constraints(self.initial_pair, config):
                raise ValueError("initial_pair 不满足搜索约束")

            self.state_bits = pair_to_state(self.initial_pair)
            self.total_state_bits = int(self.state_bits.size)
            self.no_op_action = self.total_state_bits
            self.action_space = spaces.Discrete(self.total_state_bits + 1)
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.total_state_bits + 2,),
                dtype=np.float32,
            )

            self.panels = [
                self.searcher.make_panel(
                    config.coarse_samples,
                    self.worker_seed + 1_000_003 * panel_index,
                )
                for panel_index in range(config.coarse_panel_count)
            ]
            feature_dim = feature_channel_count(
                config.num_diffs,
                config.surrogate.include_anchor_xor,
                config.surrogate.include_previous_right,
            ) * 16
            self.projection_bank = ProjectionBank.build(
                feature_dim=feature_dim,
                seed=self.worker_seed + 97,
                num_hashes=config.projection_hashes,
                hash_width=config.projection_width,
                num_pairs=config.projection_pairs,
            )

            self.cache: Dict[str, Dict[str, float]] = {}
            self.archive: Dict[str, Dict[str, object]] = {}
            self.current_step = 0
            self.current_pair = self.initial_pair
            self.current_metrics = self._evaluate(self.current_pair)
            self._archive_pair(self.current_pair, self.current_metrics)

        def _evaluate(self, pair: PolytopePair) -> Dict[str, float]:
            key = pair_cache_key(pair)
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            metrics = evaluate_pair_on_panels(
                self.searcher,
                pair,
                self.panels,
                self.encrypt_batch,
                self.projection_bank,
                self.config.surrogate,
            )
            self.cache[key] = metrics
            return metrics

        def _archive_pair(self, pair: PolytopePair, metrics: Mapping[str, float]) -> None:
            key = pair_cache_key(pair)
            old = self.archive.get(key)
            if old is None or float(metrics["score"]) > float(old["score"]):
                record: Dict[str, object] = {**pair_to_jsonable(pair), **dict(metrics)}
                self.archive[key] = record
            if len(self.archive) > self.config.archive_size * 2:
                best = sorted(
                    self.archive.items(),
                    key=lambda kv: float(kv[1]["score"]),
                    reverse=True,
                )[: self.config.archive_size]
                self.archive = dict(best)

        def get_archive(self) -> List[Dict[str, object]]:
            return sorted(
                self.archive.values(),
                key=lambda record: float(record["score"]),
                reverse=True,
            )[: self.config.archive_size]

        def _observation(self) -> np.ndarray:
            score = float(np.clip(self.current_metrics["score"], -1.0, 1.0))
            fraction = float(self.current_step / self.config.max_steps)
            return np.concatenate(
                [self.state_bits.astype(np.float32), np.asarray([score, fraction], dtype=np.float32)]
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.state_bits = pair_to_state(self.initial_pair)
            self.current_pair = self.initial_pair
            self.current_metrics = self._evaluate(self.current_pair)
            self.current_step = 0

            # Small valid perturbations diversify episodes without changing the fixed panels.
            perturbations = int(self.rng.integers(0, 4))
            for _ in range(perturbations):
                action = int(self.rng.integers(0, self.total_state_bits))
                candidate_state = self.state_bits.copy()
                candidate_state[action] ^= 1
                try:
                    candidate_pair = state_to_pair(candidate_state, self.config.num_diffs)
                except ValueError:
                    continue
                if validate_pair_constraints(candidate_pair, self.config):
                    self.state_bits = pair_to_state(candidate_pair)
                    self.current_pair = candidate_pair
                    self.current_metrics = self._evaluate(candidate_pair)
            self._archive_pair(self.current_pair, self.current_metrics)
            return self._observation(), {}

        def step(self, action):
            action = int(np.asarray(action).item())
            old_score = float(self.current_metrics["score"])
            accepted = False
            invalid = False

            if action == self.no_op_action:
                reward = self.config.no_op_penalty
            elif not 0 <= action < self.total_state_bits:
                reward = self.config.invalid_penalty
                invalid = True
            else:
                candidate_state = self.state_bits.copy()
                candidate_state[action] ^= 1
                try:
                    candidate_pair = state_to_pair(candidate_state, self.config.num_diffs)
                except ValueError:
                    candidate_pair = None

                if candidate_pair is None or not validate_pair_constraints(candidate_pair, self.config):
                    reward = self.config.invalid_penalty
                    invalid = True
                else:
                    candidate_metrics = self._evaluate(candidate_pair)
                    new_score = float(candidate_metrics["score"])
                    reward = self.config.reward_scale * (new_score - old_score) + self.config.step_penalty
                    self.state_bits = pair_to_state(candidate_pair)
                    self.current_pair = candidate_pair
                    self.current_metrics = candidate_metrics
                    self._archive_pair(candidate_pair, candidate_metrics)
                    accepted = True

            self.current_step += 1
            terminated = self.current_step >= self.config.max_steps
            info: Dict[str, object] = {
                "score": float(self.current_metrics["score"]),
                "accepted": accepted,
                "invalid": invalid,
                "cache_size": len(self.cache),
                **pair_to_jsonable(self.current_pair),
                **self.current_metrics,
            }
            return self._observation(), float(reward), terminated, False, info


    class RLProgressCallback(BaseCallback):
        def __init__(self, total_timesteps: int, label: str, verbose: int = 0):
            super().__init__(verbose)
            self.total_timesteps = int(total_timesteps)
            self.label = label
            self.print_freq = max(1, self.total_timesteps // 4)

        def _on_step(self) -> bool:
            if self.num_timesteps > 0 and self.num_timesteps % self.print_freq == 0:
                percentage = 100.0 * self.num_timesteps / self.total_timesteps
                print(
                    f"[{self.label}] PPO 进度 {percentage:5.1f}% "
                    f"({self.num_timesteps}/{self.total_timesteps})",
                    flush=True,
                )
            return True

else:  # pragma: no cover

    class PairPolytopicEnv:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "运行 PPO 搜索需要安装 gymnasium 与 stable-baselines3"
            ) from _RL_IMPORT_ERROR

def _require_rl_dependencies() -> None:
    if not _RL_AVAILABLE:
        raise ImportError(
            "运行强化学习搜索需要安装 gymnasium 和 stable-baselines3：\n"
            "  pip install gymnasium stable-baselines3"
        ) from _RL_IMPORT_ERROR


def _record_to_pair(record: Mapping[str, object]) -> PolytopePair:
    return canonical_pair(record["polytope_label_0"], record["polytope_label_1"])


def refine_archive(
    searcher: RL_SearchBestDiff,
    encrypt_batch,
    archive: Sequence[Mapping[str, object]],
    config: SearchConfig,
    worker_seed: int,
) -> List[Dict[str, object]]:
    candidates = list(archive)[: config.refine_top_k]
    if not candidates:
        return []

    feature_dim = feature_channel_count(
        config.num_diffs,
        config.surrogate.include_anchor_xor,
        config.surrogate.include_previous_right,
    ) * 16
    projection_bank = ProjectionBank.build(
        feature_dim,
        seed=worker_seed + 90_000_017,
        num_hashes=max(config.projection_hashes, 64),
        hash_width=config.projection_width,
        num_pairs=max(config.projection_pairs, 192),
    )
    panels = [
        searcher.make_panel(
            config.refine_samples,
            worker_seed + 100_000_007 + index * 1_000_003,
        )
        for index in range(config.refine_panel_count)
    ]

    refined: List[Dict[str, object]] = []
    for record in candidates:
        pair = _record_to_pair(record)
        metrics = evaluate_pair_on_panels(
            searcher,
            pair,
            panels,
            encrypt_batch,
            projection_bank,
            config.surrogate,
        )
        refined.append(
            {
                **pair_to_jsonable(pair),
                "coarse_score": float(record["score"]),
                **{f"refine_{key}": value for key, value in metrics.items()},
            }
        )

    refined.sort(key=lambda item: float(item["refine_score"]), reverse=True)

    for rank, record in enumerate(refined[: config.proxy_top_k]):
        pair = _record_to_pair(record)
        proxy = proxy_mlp_auc(
            searcher=searcher,
            pair=pair,
            encrypt_batch=encrypt_batch,
            sample_size_per_class=config.proxy_samples_per_class,
            seed=worker_seed + 200_000_033 + rank * 10_007,
            surrogate=config.surrogate,
            epochs=config.proxy_epochs,
        )
        record.update(proxy)

    for record in refined:
        proxy_advantage = float(record.get("proxy_advantage", 0.0))
        # Statistical score remains important; proxy AUC is an independent higher-fidelity signal.
        record["final_score"] = float(
            0.60 * float(record["refine_score"]) + 0.40 * proxy_advantage
        )

    refined.sort(key=lambda item: float(item["final_score"]), reverse=True)
    return refined


def run_guided_rl_search(
    searcher: RL_SearchBestDiff,
    encrypt_batch,
    seed_pool: Sequence[Mapping[str, object]],
    anchor_index: int,
    config: SearchConfig,
) -> Dict[str, object]:
    _require_rl_dependencies()
    config.validate()
    worker_seed = int(config.base_seed + anchor_index * 1_000_003)
    random.seed(worker_seed)
    np.random.seed(worker_seed & 0xFFFFFFFF)

    initial_pair = build_initial_pair(
        seed_pool,
        anchor_index=anchor_index,
        config=config,
        rng=np.random.default_rng(worker_seed),
    )
    raw_env = PairPolytopicEnv(
        searcher=searcher,
        encrypt_batch=encrypt_batch,
        initial_pair=initial_pair,
        config=config,
        worker_seed=worker_seed,
    )
    monitored = Monitor(raw_env)
    vectorized = DummyVecEnv([lambda: monitored])

    model = PPO(
        "MlpPolicy",
        vectorized,
        verbose=0,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        clip_range=config.clip_range,
        device="cpu",
        seed=worker_seed & 0x7FFFFFFF,
        policy_kwargs={"net_arch": [256, 256]},
    )
    callback = RLProgressCallback(config.total_timesteps, f"start-{anchor_index}")
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    archive = raw_env.get_archive()
    refined = refine_archive(
        searcher,
        encrypt_batch,
        archive,
        config,
        worker_seed,
    )
    vectorized.close()

    if not refined:
        raise RuntimeError("PPO 搜索未产生有效候选")
    return {
        "anchor_index": int(anchor_index),
        "initial_pair": pair_to_jsonable(initial_pair),
        "best": refined[0],
        "refined_candidates": refined,
        "coarse_archive_size": len(archive),
        "config": asdict(config),
    }


def rl_worker_task(
    seed_pool: Sequence[Mapping[str, object]],
    anchor_index: int,
    config_path: str,
    nr: int,
    config: SearchConfig,
) -> Dict[str, object]:
    try:
        searcher = RL_SearchBestDiff(config_path=config_path, nr=nr)
        cipher_dll, encrypt_batch = searcher.load_encrypt_function()
        result = run_guided_rl_search(
            searcher,
            encrypt_batch,
            seed_pool,
            anchor_index,
            config,
        )
        del cipher_dll
        return {"ok": True, **result}
    except Exception as exc:
        traceback.print_exc()
        return {
            "ok": False,
            "anchor_index": int(anchor_index),
            "error": f"{type(exc).__name__}: {exc}",
        }


def batch_rl_search_from_json(
    json_path: str,
    config_path: str,
    nr: int,
    num_starts: int,
    num_workers: int,
    config: Optional[SearchConfig] = None,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    """Run multiple independent PPO starts and save the best PAIR of polytope tuples."""
    _require_rl_dependencies()
    config = config or SearchConfig()
    config.validate()

    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    seed_pool = payload["seeds"] if isinstance(payload, dict) and "seeds" in payload else payload
    if not isinstance(seed_pool, list) or not seed_pool:
        raise ValueError("种子 JSON 必须是非空列表，或包含非空 seeds 列表")

    starts = min(int(num_starts), len(seed_pool))
    if starts < 1:
        raise ValueError("num_starts 必须为正")
    indices = list(range(starts))

    import concurrent.futures

    results: List[Dict[str, object]] = []
    worker_count = max(1, min(int(num_workers), starts))
    print(
        f"启动 {starts} 个双多胞体 PPO 起点，进程数={worker_count}，每个多胞体含 {config.num_diffs} 个差分。",
        flush=True,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                rl_worker_task,
                seed_pool,
                index,
                config_path,
                nr,
                config,
            ): index
            for index in indices
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result.get("ok"):
                best = result["best"]
                print(
                    f"[{completed}/{starts}] start={result['anchor_index']} "
                    f"final={float(best['final_score']):.6f} "
                    f"refine={float(best['refine_score']):.6f} "
                    f"proxy_auc={float(best.get('proxy_auc', 0.5)):.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[{completed}/{starts}] start={result['anchor_index']} 失败: {result['error']}",
                    flush=True,
                )

    successful = [result for result in results if result.get("ok")]
    if not successful:
        raise RuntimeError("所有 PPO 起点均失败")
    successful.sort(key=lambda result: float(result["best"]["final_score"]), reverse=True)
    best = successful[0]["best"]

    output = {
        "nr": int(nr),
        "best": best,
        "all_successful_starts": successful,
        "failed_starts": [result for result in results if not result.get("ok")],
        "search_config": asdict(config),
    }

    if output_path is None:
        output_path = f"../data/best_polytope_pair_{nr}round.json"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(f"最优双多胞体结果已保存至: {destination.resolve()}")
    return output


if __name__ == "__main__":
    CONFIG_PATH = "../Dynamic Linking/speck32_64.json"
    NR = 8
    SEED_JSON = f"../data/top_diff_seeds_{NR}round.json"

    search_config = SearchConfig(
        num_diffs=3,
        min_hw=1,
        max_hw=6,
        total_timesteps=20_000,
        coarse_samples=4096,
        refine_samples=16_384,
        proxy_samples_per_class=4096,
    )
    available_cores = os.cpu_count() or 1
    batch_rl_search_from_json(
        json_path=SEED_JSON,
        config_path=CONFIG_PATH,
        nr=NR,
        num_starts=12,
        num_workers=min(6, available_cores),
        config=search_config,
    )
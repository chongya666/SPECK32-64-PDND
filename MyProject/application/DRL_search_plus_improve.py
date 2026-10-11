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
import re
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

'''统计差分汉明重量'''
def hamming_weight(diff: Sequence[int]) -> int:
    return int(int(diff[0]).bit_count() + int(diff[1]).bit_count())
'''同一将差分变为（int,int）格式'''
def normalize_difference(diff: Sequence[int]) -> WordDiff:
    if len(diff) != 2:
        raise ValueError(f"差分必须包含两个 16-bit 字，实际为 {diff!r}")
    left, right = int(diff[0]), int(diff[1])
    if not (0 <= left < UINT16_MODULUS and 0 <= right < UINT16_MODULUS):
        raise ValueError(f"差分超出 uint16 范围: {(left, right)}")
    return left, right
'''把一个多胞体差分元组转换成同一、可比较的标准形式'''
def canonical_polytope(polytope: Sequence[Sequence[int]]) -> PolytopeDiff:
    normalized = tuple(sorted(normalize_difference(d) for d in polytope))
    if not normalized:
        raise ValueError("多胞体至少需要一个非零输入差分")
    if any(d == (0, 0) for d in normalized):
        raise ValueError("差分元组不能包含零差分；P0 已是锚点")
    if len(set(normalized)) != len(normalized):
        raise ValueError("同一多胞体内不能出现重复差分")
    return normalized
'''把双多胞体转换成不区分类别顺序的标准形式'''
def canonical_pair(polytope_0: Sequence[Sequence[int]], polytope_1: Sequence[Sequence[int]]) -> PolytopePair:
    p0 = canonical_polytope(polytope_0)
    p1 = canonical_polytope(polytope_1)
    if len(p0) != len(p1):
        raise ValueError("两个类别必须具有相同数量的非锚点差分")
    if p0 == p1:
        raise ValueError("两个类别的多胞体差分元组不能完全相同")
    return (p0, p1) if p0 < p1 else (p1, p0)
'''类似于canonical_polytope但不排序'''
def ordered_polytope(polytope: Sequence[Sequence[int]]) -> PolytopeDiff:
    normalized = tuple(normalize_difference(d) for d in polytope)
    if not normalized:
        raise ValueError("多胞体至少需要一个非零输入差分")
    if any(d == (0, 0) for d in normalized):
        raise ValueError("差分元组不能包含零差分；P0 已是锚点")
    if len(set(normalized)) != len(normalized):
        raise ValueError("同一多胞体内不能出现重复差分")
    return normalized
'''保持两个类别和各自内部差分的顺序'''
def ordered_pair(
    polytope_0: Sequence[Sequence[int]],
    polytope_1: Sequence[Sequence[int]],
) -> PolytopePair:
    p0 = ordered_polytope(polytope_0)
    p1 = ordered_polytope(polytope_1)
    if len(p0) != len(p1):
        raise ValueError("两个类别必须具有相同数量的非锚点差分")
    if set(p0) == set(p1):
        raise ValueError("两个类别的多胞体差分元组不能完全相同")
    return p0, p1
'''把 tuple 和 NumPy 类型转换为普通 Python类型，方便JSON保存'''
def pair_to_jsonable(pair: PolytopePair) -> Dict[str, List[List[int]]]:
    canonical = canonical_pair(pair[0], pair[1])
    return {
        "polytope_label_0": [[int(a), int(b)] for a, b in canonical[0]],
        "polytope_label_1": [[int(a), int(b)] for a, b in canonical[1]],
    }
'''给一个双多胞体生成唯一缓存 ID'''
def pair_cache_key(pair: PolytopePair) -> str:
    canonical = canonical_pair(pair[0], pair[1])
    payload = json.dumps(
        pair_to_jsonable(canonical),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class CryptoPanel:
    keys: np.ndarray#随机密钥集合
    plaintext0: np.ndarray#随机锚点明文集合
    seed: int#随机种子
    @property
    def sample_size(self) -> int:
        return int(self.plaintext0.shape[0])#返回样本数量
'''
在整个程序中负责：加载密码参数、调用C++SPEKC、生成样本、加密多胞体、构造输出差分等
'''
class RL_SearchBestDiff:
    def __init__(self, config_path: str, nr: int):
        self.config_path = str(config_path)#SPECK配置JSON路径
        self.config = self.load_config(config_path)
        self.nr = int(nr)
        if not 1 <= self.nr <= 22:
            raise ValueError(f"SPECK32/64 轮数必须位于 [1,22]，当前为 {self.nr}")
        #读取JSON配置
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
        self.word_size = np.dtype(self.data_type).itemsize * 8#计算单个字长度
        self.cipher_path = str(self.config["saved_path"])#读取动态链接库位置
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
                ctypes.POINTER(self.ctypes_type),
                ctypes.POINTER(self.ctypes_type),
                ctypes.c_int,
                ctypes.POINTER(self.ctypes_type),
                ctypes.c_int,
                ctypes.POINTER(self.ctypes_type),
                ctypes.c_int,
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
    '''生成固定的明文-密钥样本集合'''
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
    '''批量执行SPEKC加密'''
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
    '''对一个多胞体对应的锚点明文和所有相关明文进行批量加密'''
    def encrypt_polytope(
        self,
        polytope: Sequence[Sequence[int]],
        panel: CryptoPanel,
        encrypt_batch=None,
    ) -> np.ndarray:
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
    '''输出密文差分'''
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
    '''调试代码过程中的早期接口'''
    def make_train_data(self, diffs, sample_size, encrypt_batch=None):
        seed = int.from_bytes(os.urandom(8), "little", signed=False)
        panel = self.make_panel(sample_size, seed)
        return self.make_xor_outputs(diffs, panel, encrypt_batch)
    '''代码调试过程中早期将整数数组转换为二进制的函数'''
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
'''16bit循环右移'''
def ror16(x: np.ndarray, r: int) -> np.ndarray:
    r %= 16
    value = x.astype(np.uint32, copy=False)
    return (((value >> r) | (value << (16 - r))) & 0xFFFF).astype(np.uint16)
'''无密钥上一轮右字特征'''
def speck_previous_right_word(ciphertext: np.ndarray) -> np.ndarray:
    return ror16(np.bitwise_xor(ciphertext[:, 1], ciphertext[:, 0]), 2)
'''计算神经网络一共多少个特征通道'''
def feature_channel_count(
    num_diffs: int,
    include_anchor_xor: bool = True,
    include_previous_right: bool = True,
) -> int:
    channels = 2 * (num_diffs + 1)
    if include_anchor_xor:
        channels += 2 * num_diffs
    if include_previous_right:
        channels += num_diffs + 1
        channels += num_diffs
    return channels
'''特征构造'''
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
'''JSD与三个代理统计指标'''
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
'''比较标签 0 和标签 1 的联合分布'''
def projected_js_score(
    class_0: np.ndarray,
    class_1: np.ndarray,
    projection_bank: ProjectionBank,
    smoothing: float = 0.5,
) -> float:
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
'''两个类别的一阶单bit概率差异'''
def marginal_effect_score(class_0: np.ndarray, class_1: np.ndarray) -> float:
    return float(np.mean(np.abs(class_0.mean(axis=0) - class_1.mean(axis=0))))

def correlation_gap_score(
    class_0: np.ndarray,
    class_1: np.ndarray,
    projection_bank: ProjectionBank,
) -> float:
    idx_a = projection_bank.pair_indices[:, 0]
    idx_b = projection_bank.pair_indices[:, 1]
    x0a = class_0[:, idx_a].astype(np.float32) * 2.0 - 1.0
    x0b = class_0[:, idx_b].astype(np.float32) * 2.0 - 1.0
    x1a = class_1[:, idx_a].astype(np.float32) * 2.0 - 1.0
    x1b = class_1[:, idx_b].astype(np.float32) * 2.0 - 1.0
    corr_0 = np.mean(x0a * x0b, axis=0)
    corr_1 = np.mean(x1a * x1b, axis=0)
    return float(np.mean(np.abs(corr_0 - corr_1)) / 2.0)
'''输入结构相似度'''
def _diff_jaccard_similarity(a: WordDiff, b: WordDiff) -> float:
    x = (a[0] << 16) | a[1]
    y = (b[0] << 16) | b[1]
    union = (x | y).bit_count()
    if union == 0:
        return 1.0
    intersection = (x & y).bit_count()
    return float(intersection / union)
'''统计两个多胞体之间完全相同的差分数量'''
def input_overlap_ratio(pair: PolytopePair) -> float:
    shared = len(set(pair[0]).intersection(pair[1]))
    return float(shared / len(pair[0]))
'''计算每个多胞体内部三差分之间的平均 Jaccard 相似度'''
def within_group_similarity(pair: PolytopePair) -> float:
    values: List[float] = []
    for polytope in pair:
        for i in range(len(polytope)):
            for j in range(i + 1, len(polytope)):
                values.append(_diff_jaccard_similarity(polytope[i], polytope[j]))
    return float(np.mean(values)) if values else 0.0
'''计算两组多胞体之间的近似重复程度'''
def cross_group_similarity(pair: PolytopePair) -> float:
    directional = []
    for source, target in ((pair[0], pair[1]), (pair[1], pair[0])):
        directional.extend(
            max(_diff_jaccard_similarity(diff, other) for other in target)
            for diff in source
        )
    return float(np.mean(directional)) if directional else 0.0
'''PPO中统计代理评分的配置'''
@dataclass(frozen=True)
class SurrogateConfig:
    include_anchor_xor: bool = True#输入特征中使用锚点XOR
    include_previous_right: bool = True#使用无密钥上一轮右字
    js_weight: float = 0.50
    marginal_weight: float = 0.15
    correlation_weight: float = 0.35
    null_js_scale: float = 0.50
    null_gap_scale: float = 1.0 / math.sqrt(2.0)
    observed_floor_weight: float = 0.10
    stability_weight: float = 0.25
    exact_overlap_penalty: float = 0.20
    cross_similarity_penalty: float = 0.05
    within_similarity_penalty: float = 0.03
def _corrected_signal(
    observed: float,
    null_value: float,
    null_scale: float,
    observed_floor_weight: float,
) -> float:
    return float(
        max(0.0, observed - null_scale * null_value)
        + observed_floor_weight * observed
    )
'''在不存在类别差异的情况下，估计有限样本自身会产生多大的假差异'''
def _within_class_null_metrics(
    features: np.ndarray,
    projection_bank: ProjectionBank,
) -> Tuple[float, float, float]:
    first = features[0::2]
    second = features[1::2]
    size = min(first.shape[0], second.shape[0])
    if size < 128:
        return 0.0, 0.0, 0.0
    first = first[:size]
    second = second[:size]
    return (
        projected_js_score(first, second, projection_bank),
        marginal_effect_score(first, second),
        correlation_gap_score(first, second, projection_bank),
    )

'''PPO统计代理评分核心函数'''
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
        observed_js = projected_js_score(flat_0, flat_1, projection_bank)
        observed_marginal = marginal_effect_score(flat_0, flat_1)
        observed_correlation = correlation_gap_score(
            flat_0, flat_1, projection_bank
        )
        null_0 = _within_class_null_metrics(flat_0, projection_bank)
        null_1 = _within_class_null_metrics(flat_1, projection_bank)
        null_js = 0.5 * (null_0[0] + null_1[0])
        null_marginal = 0.5 * (null_0[1] + null_1[1])
        null_correlation = 0.5 * (null_0[2] + null_1[2])
        corrected_js = _corrected_signal(
            observed_js,
            null_js,
            surrogate.null_js_scale,
            surrogate.observed_floor_weight,
        )
        corrected_marginal = _corrected_signal(
            observed_marginal,
            null_marginal,
            surrogate.null_gap_scale,
            surrogate.observed_floor_weight,
        )
        corrected_correlation = _corrected_signal(
            observed_correlation,
            null_correlation,
            surrogate.null_gap_scale,
            surrogate.observed_floor_weight,
        )
        raw = (
            surrogate.js_weight * corrected_js
            + surrogate.marginal_weight * corrected_marginal
            + surrogate.correlation_weight * corrected_correlation
        )
        panel_records.append(
            {
                "projected_js": observed_js,
                "marginal_gap": observed_marginal,
                "correlation_gap": observed_correlation,
                "null_projected_js": null_js,
                "null_marginal_gap": null_marginal,
                "null_correlation_gap": null_correlation,
                "corrected_projected_js": corrected_js,
                "corrected_marginal_gap": corrected_marginal,
                "corrected_correlation_gap": corrected_correlation,
                "raw_score": raw,
            }
        )
    raw_scores = np.asarray(
        [record["raw_score"] for record in panel_records],
        dtype=np.float64,
    )
    mean_raw = float(np.mean(raw_scores))
    relative_std = float(
        np.std(raw_scores) / (mean_raw + 1e-12)
    )
    stability = float(1.0 / (1.0 + relative_std))
    stability_factor = (
        1.0 - surrogate.stability_weight
        + surrogate.stability_weight * stability
    )
    overlap = input_overlap_ratio(pair)
    cross_similarity = cross_group_similarity(pair)
    within_similarity = within_group_similarity(pair)
    structural_factor = max(
        0.50,
        1.0
        - surrogate.exact_overlap_penalty * overlap
        - surrogate.cross_similarity_penalty * cross_similarity
        - surrogate.within_similarity_penalty * within_similarity,
    )
    score = mean_raw * stability_factor * structural_factor
    def mean_metric(name: str) -> float:
        return float(np.mean([record[name] for record in panel_records]))
    hw_summary = pair_hamming_summary(pair)
    return {
        "score": float(score),
        **hw_summary,
        "projected_js": mean_metric("projected_js"),
        "marginal_gap": mean_metric("marginal_gap"),
        "correlation_gap": mean_metric("correlation_gap"),
        "null_projected_js": mean_metric("null_projected_js"),
        "null_marginal_gap": mean_metric("null_marginal_gap"),
        "null_correlation_gap": mean_metric("null_correlation_gap"),
        "corrected_projected_js": mean_metric("corrected_projected_js"),
        "corrected_marginal_gap": mean_metric("corrected_marginal_gap"),
        "corrected_correlation_gap": mean_metric("corrected_correlation_gap"),
        "stability": stability,
        "input_overlap": overlap,
        "cross_input_similarity": cross_similarity,
        "within_input_similarity": within_similarity,
        "structural_factor": structural_factor,
    }

'''手工实现二分类AUC'''
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
'''轻量卷积残差代理PDND'''
def proxy_mlp_auc(
    searcher: RL_SearchBestDiff,
    pair: PolytopePair,
    encrypt_batch,
    sample_size_per_class: int,
    seed: int,
    surrogate: SurrogateConfig,
    epochs: int = 5,
    batch_size: int = 2048,
    device_name: str = "cpu",
) -> Dict[str, float]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("proxy_mlp_auc 需要 PyTorch") from exc
    if sample_size_per_class < 512:
        raise ValueError("proxy 样本量过小")
    pair = canonical_pair(pair[0], pair[1])
    def make_split(panel_seed: int):
        panel_0 = searcher.make_panel(sample_size_per_class, panel_seed)
        panel_1 = searcher.make_panel(sample_size_per_class, panel_seed + 1)
        features_0 = vertices_to_bit_features(
            searcher.encrypt_polytope(pair[0], panel_0, encrypt_batch),
            surrogate.include_anchor_xor,
            surrogate.include_previous_right,
        )
        features_1 = vertices_to_bit_features(
            searcher.encrypt_polytope(pair[1], panel_1, encrypt_batch),
            surrogate.include_anchor_xor,
            surrogate.include_previous_right,
        )
        x = np.concatenate([features_0, features_1], axis=0).astype(np.float32)
        x = x * 2.0 - 1.0
        y = np.concatenate(
            [
                np.zeros(sample_size_per_class, dtype=np.float32),
                np.ones(sample_size_per_class, dtype=np.float32),
            ]
        )
        rng = np.random.default_rng(panel_seed + 2)
        permutation = rng.permutation(y.size)
        return x[permutation], y[permutation]
    x_train, y_train = make_split(seed)
    x_val, y_val = make_split(seed + 10_000_019)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    torch.manual_seed(seed & 0xFFFFFFFF)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed & 0xFFFFFFFF)
    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int):
            super().__init__()
            self.conv1 = nn.Conv1d(
                width,
                width,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
            self.bn1 = nn.BatchNorm1d(width)
            self.conv2 = nn.Conv1d(
                width,
                width,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
            self.bn2 = nn.BatchNorm1d(width)
            self.act = nn.GELU()
        def forward(self, x):
            residual = x
            x = self.act(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return self.act(x + residual)
    class ProxyPDND(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()
            width = 64
            self.stem = nn.Sequential(
                nn.Conv1d(input_channels, width, kernel_size=1, bias=False),
                nn.BatchNorm1d(width),
                nn.GELU(),
            )
            self.blocks = nn.Sequential(
                ResidualBlock(width, 1),
                ResidualBlock(width, 2),
                ResidualBlock(width, 4),
            )
            self.head = nn.Sequential(
                nn.Linear(2 * width, 96),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(96, 1),
            )
        def forward(self, x):
            x = self.blocks(self.stem(x))
            pooled = torch.cat(
                [x.mean(dim=-1), x.amax(dim=-1)],
                dim=1,
            )
            return self.head(pooled).squeeze(1)
    model = ProxyPDND(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=1e-4,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    model.eval()
    logits_parts = []
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val),
            torch.from_numpy(y_val),
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    with torch.no_grad():
        for xb, _ in validation_loader:
            xb = xb.to(device, non_blocking=True)
            logits_parts.append(model(xb).float().cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)

    auc = _binary_auc(y_val.astype(np.int8), logits)
    accuracy = float(np.mean((logits >= 0.0) == (y_val >= 0.5)))
    return {
        "proxy_auc": float(auc),
        "proxy_accuracy": accuracy,
        "proxy_advantage": float(max(0.0, 2.0 * auc - 1.0)),
    }


'''PPO全部参数'''
@dataclass
class SearchConfig:
    num_diffs: int = 3
    min_hw: int = 1
    max_hw: int = 5
    search_group_hw_max: int = 7
    max_steps: int = 96
    coarse_samples: int = 8192
    coarse_panel_count: int = 3
    projection_hashes: int = 96
    projection_width: int = 6
    projection_pairs: int = 256
    total_timesteps: int = 40_000
    learning_rate: float = 2e-4
    n_steps: int = 512
    batch_size: int = 128
    gamma: float = 0.97
    gae_lambda: float = 0.92
    ent_coef: float = 0.005
    clip_range: float = 0.20
    reward_scale: float = 100.0
    reward_clip: float = 2.0
    invalid_penalty: float = -0.10
    no_op_penalty: float = -0.002
    step_penalty: float = -0.001
    archive_size: int = 128
    elite_restart_probability: float = 0.60
    min_reset_perturbations: int = 1
    max_reset_perturbations: int = 4
    initial_quality_weight: float = 0.50
    initial_within_diversity_weight: float = 0.20
    initial_cross_diversity_weight: float = 0.10
    initial_low_hw_weight: float = 0.20
    initial_max_hw: int = 4
    initial_group_hw_max: int = 6
    anchor_hw_quota: Tuple[int, ...] = (0, 5, 4, 2, 1)
    require_exact_seed_round: bool = True
    seed_round_decay: float = 0.08
    refine_top_k: int = 24
    refine_samples: int = 65_536
    refine_panel_count: int = 4
    refine_projection_width: int = 8
    proxy_top_k: int = 4
    proxy_samples_per_class: int = 8192
    proxy_epochs: int = 5
    proxy_batch_size: int = 2048
    proxy_device: str = "cpu"
    global_rerank_top_k: int = 12
    global_proxy_samples_per_class: int = 32_768
    global_proxy_epochs: int = 8
    global_proxy_repeats: int = 3
    global_proxy_batch_size: int = 2048
    global_proxy_device: str = "auto"
    global_proxy_rank_weight: float = 0.70
    proxy_lcb_std_weight: float = 0.50
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
        if self.refine_top_k < 1 or self.global_rerank_top_k < 1:
            raise ValueError("refine_top_k/global_rerank_top_k 必须为正")
        if not 0.0 <= self.elite_restart_probability <= 1.0:
            raise ValueError("elite_restart_probability 必须位于 [0,1]")
        if self.min_reset_perturbations < 0:
            raise ValueError("min_reset_perturbations 不能为负")
        if self.max_reset_perturbations < self.min_reset_perturbations:
            raise ValueError("reset扰动范围非法")
        if not 0.0 <= self.global_proxy_rank_weight <= 1.0:
            raise ValueError("global_proxy_rank_weight 必须位于 [0,1]")
        initial_weight_sum = (
            self.initial_quality_weight
            + self.initial_within_diversity_weight
            + self.initial_cross_diversity_weight
            + self.initial_low_hw_weight
        )
        if not np.isclose(initial_weight_sum, 1.0):
            raise ValueError("初始pair四个权重之和必须为1")
        if not self.min_hw <= self.initial_max_hw <= self.max_hw:
            raise ValueError("initial_max_hw 必须位于 [min_hw,max_hw]")
        minimum_group_hw = self.num_diffs * self.min_hw
        if not minimum_group_hw <= self.initial_group_hw_max <= self.num_diffs * self.initial_max_hw:
            raise ValueError("initial_group_hw_max 与初始单差分 HW 约束不一致")
        if not self.initial_group_hw_max <= self.search_group_hw_max <= self.num_diffs * self.max_hw:
            raise ValueError("search_group_hw_max 必须不小于初始组预算且不超过理论上限")
        if any(int(value) < 0 for value in self.anchor_hw_quota):
            raise ValueError("anchor_hw_quota 不能包含负数")
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
    return ordered_pair(p0, p1)
def polytope_hamming_weight(polytope: Sequence[WordDiff]) -> int:
    return int(sum(hamming_weight(diff) for diff in polytope))
def pair_hamming_summary(pair: PolytopePair) -> Dict[str, int]:
    group_0 = polytope_hamming_weight(pair[0])
    group_1 = polytope_hamming_weight(pair[1])
    all_hws = [hamming_weight(diff) for polytope in pair for diff in polytope]
    return {
        "group_0_hw": int(group_0),
        "group_1_hw": int(group_1),
        "total_hw": int(group_0 + group_1),
        "max_diff_hw": int(max(all_hws)),
    }
def validate_pair_constraints(pair: PolytopePair, config: SearchConfig) -> bool:
    try:
        pair = ordered_pair(pair[0], pair[1])
    except ValueError:
        return False
    if len(pair[0]) != config.num_diffs:
        return False
    for polytope in pair:
        if polytope_hamming_weight(polytope) > config.search_group_hw_max:
            return False
        for diff in polytope:
            hw = hamming_weight(diff)
            if hw < config.min_hw or hw > config.max_hw:
                return False
    return True
def validate_initial_pair_constraints(pair: PolytopePair, config: SearchConfig) -> bool:
    try:
        pair = ordered_pair(pair[0], pair[1])
    except ValueError:
        return False
    if len(pair[0]) != config.num_diffs:
        return False
    for polytope in pair:
        if polytope_hamming_weight(polytope) > config.initial_group_hw_max:
            return False
        for diff in polytope:
            hw = hamming_weight(diff)
            if hw < config.min_hw or hw > config.initial_max_hw:
                return False
    return True
def _input_jaccard_distance(a: WordDiff, b: WordDiff) -> float:
    return float(1.0 - _diff_jaccard_similarity(a, b))
def _effective_seed_quality(
    record: Mapping[str, object],
    target_round: int,
    config: SearchConfig,
) -> float:
    quality = float(
        record.get(
            "quality",
            record.get("mmr_score", record.get("score", 0.0)),
        )
    )
    source_round = int(record.get("source_nr", target_round))
    round_weight = math.exp(
        -config.seed_round_decay * abs(target_round - source_round)
    )
    return float(quality * round_weight)
def _low_hw_score(hw: int, minimum_hw: int, maximum_hw: int) -> float:
    if maximum_hw <= minimum_hw:
        return 1.0
    return float(np.clip((maximum_hw - hw) / (maximum_hw - minimum_hw), 0.0, 1.0))
'''从64个种子中构造两组三差分PPO初始状态'''
def build_initial_pair(
    seed_pool: Sequence[Mapping[str, object]],
    anchor_index: int,
    config: SearchConfig,
    rng: np.random.Generator,
    target_round: int,
) -> PolytopePair:
    candidates: List[WordDiff] = []
    qualities: Dict[WordDiff, float] = {}
    for record in seed_pool:
        try:
            diff = normalize_difference(record["diff"])
        except Exception:
            continue
        hw = hamming_weight(diff)
        # 初始 pair 使用比 PPO 搜索更严格的单差分上限。
        if config.min_hw <= hw <= config.initial_max_hw and diff != (0, 0):
            effective = _effective_seed_quality(record, target_round, config)
            if diff not in qualities:
                candidates.append(diff)
                qualities[diff] = effective
            elif effective > qualities[diff]:
                qualities[diff] = effective
    if len(candidates) < 2 * config.num_diffs:
        raise ValueError(
            f"有效低 HW 种子数量不足：至少需要 {2 * config.num_diffs} 个，"
            f"实际为 {len(candidates)}"
        )
    anchor_record = seed_pool[anchor_index % len(seed_pool)]
    try:
        anchor = normalize_difference(anchor_record["diff"])
    except Exception:
        anchor = candidates[anchor_index % len(candidates)]
    if anchor not in qualities or hamming_weight(anchor) > config.initial_max_hw:
        anchor = min(
            candidates,
            key=lambda diff: (hamming_weight(diff), -qualities[diff]),
        )
    random_tie = {diff: float(rng.random()) for diff in candidates}
    ordered = sorted(
        candidates,
        key=lambda diff: (
            qualities[diff],
            _low_hw_score(hamming_weight(diff), config.min_hw, config.initial_max_hw),
            random_tie[diff],
        ),
        reverse=True,
    )
    def choose_diverse(
        selected: List[WordDiff],
        forbidden: set,
        count: int,
        cross_reference: Sequence[WordDiff] = (),
    ) -> List[WordDiff]:
        result = list(selected)
        while len(result) < count:
            current_group_hw = polytope_hamming_weight(result)
            available = [
                diff
                for diff in ordered
                if diff not in forbidden
                and diff not in result
                and hamming_weight(diff) <= config.initial_max_hw
                and current_group_hw + hamming_weight(diff) <= config.initial_group_hw_max
            ]
            if not available:
                raise RuntimeError(
                    "无法在 initial_group_hw_max 预算内构造初始多胞体；"
                    "请检查低 HW 种子数量或适当放宽初始组预算"
                )
            def candidate_score(diff: WordDiff) -> Tuple[float, float, float]:
                within_distance = (
                    min(_input_jaccard_distance(diff, chosen) for chosen in result)
                    if result
                    else 1.0
                )
                cross_distance = (
                    min(_input_jaccard_distance(diff, chosen) for chosen in cross_reference)
                    if cross_reference
                    else 1.0
                )
                low_hw = _low_hw_score(
                    hamming_weight(diff),
                    config.min_hw,
                    config.initial_max_hw,
                )
                score = (
                    config.initial_quality_weight * qualities[diff]
                    + config.initial_within_diversity_weight * within_distance
                    + config.initial_cross_diversity_weight * cross_distance
                    + config.initial_low_hw_weight * low_hw
                )
                return score, qualities[diff], low_hw

            choice = max(available, key=candidate_score)
            result.append(choice)
        return result
    poly_0 = choose_diverse([anchor], set(), config.num_diffs)
    poly_1 = choose_diverse(
        [],
        set(poly_0),
        config.num_diffs,
        cross_reference=poly_0,
    )
    pair = ordered_pair(poly_0, poly_1)
    if not validate_initial_pair_constraints(pair, config):
        raise RuntimeError("内部错误：生成的初始 pair 不满足低 HW 初始约束")
    if not validate_pair_constraints(pair, config):
        raise RuntimeError("内部错误：生成的初始 pair 不满足 PPO 搜索约束")
    return pair
'''强化学习环境'''
if _RL_AVAILABLE:
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
            self.initial_pair = ordered_pair(initial_pair[0], initial_pair[1])
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
            pair = canonical_pair(pair[0], pair[1])
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
            base_pair = self.initial_pair
            if (
                self.archive
                and self.rng.random() < self.config.elite_restart_probability
            ):
                best_record = max(
                    self.archive.values(),
                    key=lambda record: float(record["score"]),
                )
                base_pair = ordered_pair(
                    best_record["polytope_label_0"],
                    best_record["polytope_label_1"],
                )
            self.state_bits = pair_to_state(base_pair)
            self.current_pair = base_pair
            self.current_metrics = self._evaluate(self.current_pair)
            self.current_step = 0
            perturbations = int(
                self.rng.integers(
                    self.config.min_reset_perturbations,
                    self.config.max_reset_perturbations + 1,
                )
            )
            for _ in range(perturbations):
                action = int(self.rng.integers(0, self.total_state_bits))
                candidate_state = self.state_bits.copy()
                candidate_state[action] ^= 1
                try:
                    candidate_pair = state_to_pair(
                        candidate_state,
                        self.config.num_diffs,
                    )
                except ValueError:
                    continue
                if validate_pair_constraints(candidate_pair, self.config):
                    self.state_bits = candidate_state
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
                    reward = (
                        self.config.reward_scale * (new_score - old_score)
                        + self.config.step_penalty
                    )
                    reward = float(
                        np.clip(
                            reward,
                            -self.config.reward_clip,
                            self.config.reward_clip,
                        )
                    )
                    self.state_bits = candidate_state
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
else:
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
def _rank01_from_values(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size <= 1:
        return np.ones(values_array.size, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(values_array.size, dtype=np.float64)
    ranks[order] = np.arange(values_array.size, dtype=np.float64)
    return ranks / (values_array.size - 1)
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
        num_hashes=max(config.projection_hashes, 128),
        hash_width=config.refine_projection_width,
        num_pairs=max(config.projection_pairs, 384),
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
                **{
                    f"refine_{key}": value
                    for key, value in metrics.items()
                },
            }
        )
    refined.sort(
        key=lambda item: float(item["refine_score"]),
        reverse=True,
    )
    proxy_candidates = refined[: min(config.proxy_top_k, len(refined))]
    for rank, record in enumerate(proxy_candidates):
        pair = _record_to_pair(record)
        proxy = proxy_mlp_auc(
            searcher=searcher,
            pair=pair,
            encrypt_batch=encrypt_batch,
            sample_size_per_class=config.proxy_samples_per_class,
            seed=worker_seed + 200_000_033 + rank * 10_007,
            surrogate=config.surrogate,
            epochs=config.proxy_epochs,
            batch_size=config.proxy_batch_size,
            device_name=config.proxy_device,
        )
        record.update(proxy)

    refine_ranks = _rank01_from_values(
        [float(record["refine_score"]) for record in refined]
    )
    evaluated_auc = [
        float(record.get("proxy_auc", 0.5))
        for record in proxy_candidates
    ]
    proxy_ranks = _rank01_from_values(evaluated_auc)

    for index, record in enumerate(refined):
        record["refine_rank_score"] = float(refine_ranks[index])
        record["proxy_rank_score"] = 0.0
        record["final_score"] = float(0.35 * refine_ranks[index])

    for index, record in enumerate(proxy_candidates):
        record["proxy_rank_score"] = float(proxy_ranks[index])
        record["final_score"] = float(
            0.35 * record["refine_rank_score"]
            + 0.65 * record["proxy_rank_score"]
        )
    refined.sort(
        key=lambda item: float(item["final_score"]),
        reverse=True,
    )
    return refined
def _deduplicate_global_candidates(
    successful: Sequence[Mapping[str, object]],
    limit: int,
) -> List[Dict[str, object]]:
    best_by_key: Dict[str, Dict[str, object]] = {}
    for result in successful:
        for record in result.get("refined_candidates", []):
            pair = _record_to_pair(record)
            key = pair_cache_key(pair)
            copied = dict(record)
            old = best_by_key.get(key)
            if old is None or float(copied["refine_score"]) > float(old["refine_score"]):
                best_by_key[key] = copied
    candidates = sorted(
        best_by_key.values(),
        key=lambda record: float(record["refine_score"]),
        reverse=True,
    )
    return candidates[:limit]
'''最终代理重排'''
def global_rerank_candidates(
    config_path: str,
    nr: int,
    successful: Sequence[Mapping[str, object]],
    config: SearchConfig,
) -> List[Dict[str, object]]:
    candidates = _deduplicate_global_candidates(
        successful,
        config.global_rerank_top_k,
    )
    if not candidates:
        return []
    searcher = RL_SearchBestDiff(config_path=config_path, nr=nr)
    cipher_dll, encrypt_batch = searcher.load_encrypt_function()
    for candidate_index, record in enumerate(candidates):
        pair = _record_to_pair(record)
        auc_values = []
        accuracy_values = []
        for repeat in range(config.global_proxy_repeats):
            proxy = proxy_mlp_auc(
                searcher=searcher,
                pair=pair,
                encrypt_batch=encrypt_batch,
                sample_size_per_class=config.global_proxy_samples_per_class,
                seed=(
                    config.base_seed
                    + 500_000_047
                    + candidate_index * 100_003
                    + repeat * 10_007
                ),
                surrogate=config.surrogate,
                epochs=config.global_proxy_epochs,
                batch_size=config.global_proxy_batch_size,
                device_name=config.global_proxy_device,
            )
            auc_values.append(float(proxy["proxy_auc"]))
            accuracy_values.append(float(proxy["proxy_accuracy"]))
        auc_mean = float(np.mean(auc_values))
        auc_std = float(np.std(auc_values))
        auc_lcb = float(
            auc_mean - config.proxy_lcb_std_weight * auc_std
        )
        record["global_proxy_auc_values"] = auc_values
        record["global_proxy_auc_mean"] = auc_mean
        record["global_proxy_auc_std"] = auc_std
        record["global_proxy_auc_lcb"] = auc_lcb
        record["global_proxy_accuracy_mean"] = float(
            np.mean(accuracy_values)
        )
    refine_ranks = _rank01_from_values(
        [float(record["refine_score"]) for record in candidates]
    )
    proxy_ranks = _rank01_from_values(
        [float(record["global_proxy_auc_lcb"]) for record in candidates]
    )
    proxy_weight = config.global_proxy_rank_weight
    for index, record in enumerate(candidates):
        record["global_refine_rank"] = float(refine_ranks[index])
        record["global_proxy_rank"] = float(proxy_ranks[index])
        record["final_score"] = float(
            (1.0 - proxy_weight) * refine_ranks[index]
            + proxy_weight * proxy_ranks[index]
        )
    candidates.sort(
        key=lambda record: (
            float(record["final_score"]),
            float(record["global_proxy_auc_lcb"]),
            float(record["refine_score"]),
        ),
        reverse=True,
    )
    del cipher_dll
    return candidates
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
        target_round=searcher.nr,
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
        "initial_pair_hw": pair_hamming_summary(initial_pair),
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
'''种子文件读取'''
def _infer_source_round(path: str, payload: object, target_round: int) -> int:
    if isinstance(payload, dict) and "nr" in payload:
        try:
            return int(payload["nr"])
        except Exception:
            pass
    match = re.search(r"(\d+)round", Path(path).name)
    return int(match.group(1)) if match else int(target_round)
def _load_seed_pool(
    json_paths: str | Sequence[str],
    target_round: int,
    config: SearchConfig,
) -> Tuple[List[Dict[str, object]], List[str]]:
    paths = [str(json_paths)] if isinstance(json_paths, (str, os.PathLike)) else [str(p) for p in json_paths]
    if not paths:
        raise ValueError("至少需要一个种子JSON文件")
    best_by_diff: Dict[WordDiff, Dict[str, object]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        source_round = _infer_source_round(path, payload, target_round)
        if config.require_exact_seed_round and source_round != int(target_round):
            raise ValueError(
                f"种子文件轮数与目标轮数不一致：目标 NR={target_round}，"
                f"文件 {path} 对应 NR={source_round}。"
            )
        records = (
            payload["seeds"]
            if isinstance(payload, dict) and "seeds" in payload
            else payload
        )
        if not isinstance(records, list):
            raise ValueError(f"种子文件格式错误: {path}")
        for raw_record in records:
            if not isinstance(raw_record, Mapping) or "diff" not in raw_record:
                continue
            try:
                diff = normalize_difference(raw_record["diff"])
            except Exception:
                continue
            record = dict(raw_record)
            record["source_nr"] = int(source_round)
            record["source_path"] = path
            effective = _effective_seed_quality(record, target_round, config)
            old = best_by_diff.get(diff)
            if old is None or effective > _effective_seed_quality(
                old, target_round, config
            ):
                best_by_diff[diff] = record
    seed_pool = list(best_by_diff.values())
    seed_pool.sort(
        key=lambda record: _effective_seed_quality(
            record,
            target_round,
            config,
        ),
        reverse=True,
    )
    if not seed_pool:
        raise ValueError("没有读取到有效种子")
    return seed_pool, paths
'''PPO起点选择'''
def _low_hw_anchor_order(
    seed_pool: Sequence[Mapping[str, object]],
    num_starts: int,
    quota_by_hw: Sequence[int],
) -> List[int]:
    groups: Dict[int, List[int]] = {}
    for index, record in enumerate(seed_pool):
        try:
            hw = hamming_weight(normalize_difference(record["diff"]))
        except Exception:
            continue
        groups.setdefault(hw, []).append(index)
    target = min(int(num_starts), len(seed_pool))
    remaining_quota = {
        hw: max(0, int(quota_by_hw[hw])) if hw < len(quota_by_hw) else 0
        for hw in sorted(groups)
    }
    cursors = {hw: 0 for hw in groups}
    order: List[int] = []
    used = set()
    while len(order) < target and any(value > 0 for value in remaining_quota.values()):
        progressed = False
        for hw in sorted(groups):
            if remaining_quota.get(hw, 0) <= 0:
                continue
            indices = groups[hw]
            cursor = cursors[hw]
            while cursor < len(indices) and indices[cursor] in used:
                cursor += 1
            cursors[hw] = cursor
            if cursor >= len(indices):
                remaining_quota[hw] = 0
                continue
            index = indices[cursor]
            order.append(index)
            used.add(index)
            cursors[hw] += 1
            remaining_quota[hw] -= 1
            progressed = True
            if len(order) >= target:
                break
        if not progressed:
            break
    fallback = []
    for hw in sorted(groups):
        fallback.extend(index for index in groups[hw] if index not in used)
    for index in fallback:
        if len(order) >= target:
            break
        order.append(index)
        used.add(index)
    return order
'''文件路径函数'''
def seed_json_path_for_round(
    nr: int,
    data_dir: str = "../data",
) -> str:
    nr = int(nr)
    if not 1 <= nr <= 22:
        raise ValueError(f"SPECK32/64 轮数必须位于 [1,22]，当前 NR={nr}")
    return str(Path(data_dir) / f"top_diff_seeds_{nr}round.json")
def output_json_path_for_round(
    nr: int,
    data_dir: str = "../data",
) -> str:
    nr = int(nr)
    if not 1 <= nr <= 22:
        raise ValueError(f"SPECK32/64 轮数必须位于 [1,22]，当前 NR={nr}")
    return str(Path(data_dir) / f"best_polytope_pair_{nr}round.json")
def validate_seed_json_for_round(seed_json_path: str, nr: int) -> None:
    path = Path(seed_json_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {nr} 轮种子差分文件：{path}\\n"
            f"请先运行种子筛选程序生成 top_diff_seeds_{nr}round.json。"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload_nr = payload.get("nr")
        if payload_nr is not None and int(payload_nr) != int(nr):
            raise ValueError(
                f"种子文件内容中的 nr={payload_nr}，"
                f"但当前目标 NR={nr}：{path}"
            )
        records = payload.get("seeds")
    else:
        records = payload

    if not isinstance(records, list) or not records:
        raise ValueError(
            f"种子文件必须是非空列表，或包含非空 seeds 列表：{path}"
        )
'''
仅根据 NR 自动读取 top_diff_seeds_{NR}round.json，
然后执行该轮的 PPO 双多胞体搜索和代理 PDND 重排。
'''
def run_rl_search_for_round(
    nr: int,
    config_path: str,
    data_dir: str = "../data",
    num_starts: int = 16,
    num_workers: int = 1,
    config: Optional[SearchConfig] = None,
) -> Dict[str, object]:
    nr = int(nr)
    seed_json = seed_json_path_for_round(nr, data_dir)
    output_json = output_json_path_for_round(nr, data_dir)
    validate_seed_json_for_round(seed_json, nr)
    print(
        f"[R{nr}] 读取本轮种子文件：{Path(seed_json).resolve()}",
        flush=True,
    )
    print(
        f"[R{nr}] 搜索结果将保存至：{Path(output_json).resolve()}",
        flush=True,
    )
    return batch_rl_search_from_json(
        json_path=seed_json,
        config_path=config_path,
        nr=nr,
        num_starts=num_starts,
        num_workers=num_workers,
        config=config,
        output_path=output_json,
    )
'''整个搜索流程总控函数'''
def batch_rl_search_from_json(
    json_path: str | Sequence[str],
    config_path: str,
    nr: int,
    num_starts: int,
    num_workers: int,
    config: Optional[SearchConfig] = None,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    _require_rl_dependencies()
    config = config or SearchConfig()
    config.validate()
    seed_pool, seed_paths = _load_seed_pool(
        json_path,
        target_round=nr,
        config=config,
    )
    starts = min(int(num_starts), len(seed_pool))
    if starts < 1:
        raise ValueError("num_starts 必须为正")
    indices = _low_hw_anchor_order(
        seed_pool,
        starts,
        quota_by_hw=config.anchor_hw_quota,
    )
    anchor_hw_counts: Dict[int, int] = {}
    for index in indices:
        hw = hamming_weight(normalize_difference(seed_pool[index]["diff"]))
        anchor_hw_counts[hw] = anchor_hw_counts.get(hw, 0) + 1
    import concurrent.futures
    results: List[Dict[str, object]] = []
    worker_count = max(1, min(int(num_workers), len(indices)))
    print(
        f"启动 {len(indices)} 个双多胞体 PPO 起点，进程数={worker_count}，"
        f"种子文件数={len(seed_paths)}，每个多胞体含 {config.num_diffs} 个差分，"
        f"锚点HW分布={dict(sorted(anchor_hw_counts.items()))}。",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:
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
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            result = future.result()
            results.append(result)
            if result.get("ok"):
                best = result["best"]
                print(
                    f"[{completed}/{len(indices)}] start={result['anchor_index']} "
                    f"local_final={float(best['final_score']):.6f} "
                    f"refine={float(best['refine_score']):.6f} "
                    f"proxy_auc={float(best.get('proxy_auc', 0.5)):.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[{completed}/{len(indices)}] "
                    f"start={result['anchor_index']} 失败: {result['error']}",
                    flush=True,
                )

    successful = [result for result in results if result.get("ok")]
    if not successful:
        raise RuntimeError("所有 PPO 起点均失败")

    print(
        f"开始全局高保真重排：候选数最多 "
        f"{config.global_rerank_top_k}，每个候选重复 "
        f"{config.global_proxy_repeats} 次代理PDND训练。",
        flush=True,
    )
    global_candidates = global_rerank_candidates(
        config_path=config_path,
        nr=nr,
        successful=successful,
        config=config,
    )
    if not global_candidates:
        raise RuntimeError("全局重排没有产生有效候选")
    best = global_candidates[0]

    output = {
        "schema": "speck32_polytope_pair_search_low_hw_budget_v6",
        "nr": int(nr),
        "seed_json_paths": seed_paths,
        "best": best,
        "global_reranked_candidates": global_candidates,
        "all_successful_starts": successful,
        "failed_starts": [
            result for result in results if not result.get("ok")
        ],
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
    NR = 7
    CONFIG_PATH = "../Dynamic Linking/speck32_64.json"
    DATA_DIR = "../data"
    search_config = SearchConfig(
        num_diffs=3,
        min_hw=1,
        max_hw=5,
        search_group_hw_max=7,
        initial_max_hw=4,
        initial_group_hw_max=6,
        initial_quality_weight=0.50,
        initial_within_diversity_weight=0.20,
        initial_cross_diversity_weight=0.10,
        initial_low_hw_weight=0.20,
        anchor_hw_quota=(0, 5, 4, 2, 1),
        max_steps=96,
        total_timesteps=40_000,
        require_exact_seed_round=True,
        coarse_samples=8192,
        coarse_panel_count=3,
        projection_hashes=96,
        projection_width=6,
        projection_pairs=256,
        refine_top_k=24,
        refine_samples=65_536,
        refine_panel_count=4,
        refine_projection_width=8,
        proxy_top_k=4,
        proxy_samples_per_class=8192,
        proxy_epochs=5,
        proxy_device="cpu",
        global_rerank_top_k=12,
        global_proxy_samples_per_class=32_768,
        global_proxy_epochs=8,
        global_proxy_repeats=3,
        global_proxy_device="auto",
        global_proxy_rank_weight=0.70,
    )
    available_cores = os.cpu_count() or 1
    run_rl_search_for_round(
        nr=NR,
        config_path=CONFIG_PATH,
        data_dir=DATA_DIR,
        num_starts=12,
        num_workers=min(6, available_cores),
        config=search_config,
    )

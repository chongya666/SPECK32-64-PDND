from __future__ import annotations
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

MASK16 = 0xFFFF
BIT_SHIFTS = np.arange(15, -1, -1, dtype=np.uint16)
WordDiff = Tuple[int, int]
PolytopeDiff = Tuple[WordDiff, ...]
PolytopePair = Tuple[PolytopeDiff, PolytopeDiff]

def normalize_difference(diff: Sequence[int]) -> WordDiff:
    if len(diff) != 2:
        raise ValueError(f"差分必须包含两个16-bit字，实际为 {diff!r}")
    a, b = int(diff[0]), int(diff[1])
    if not (0 <= a <= MASK16 and 0 <= b <= MASK16):
        raise ValueError(f"差分超出uint16范围: {(a, b)}")
    if a == 0 and b == 0:
        raise ValueError("非锚点输入差分不能为(0,0)")
    return a, b

def canonical_polytope(poly: Sequence[Sequence[int]]) -> PolytopeDiff:
    result = tuple(normalize_difference(d) for d in poly)

    if len(result) != 3:
        raise ValueError(
            f"当前密钥恢复脚本要求三差分多胞体，实际差分数={len(result)}"
        )
    if len(set(result)) != len(result):
        raise ValueError("同一多胞体内部差分不能重复")
    return result

def canonical_pair(
    p0: Sequence[Sequence[int]],
    p1: Sequence[Sequence[int]],
) -> PolytopePair:
    a = canonical_polytope(p0)
    b = canonical_polytope(p1)

    if set(a) == set(b):
        raise ValueError("两类多胞体不能由完全相同的差分集合构成")
    return a, b

def _extract_pair_record(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        if "polytope_label_0" in payload and "polytope_label_1" in payload:
            return payload
        best = payload.get("best")
        if isinstance(best, Mapping):
            if "polytope_label_0" in best and "polytope_label_1" in best:
                return best
            nested = best.get("best")
            if (
                isinstance(nested, Mapping)
                and "polytope_label_0" in nested
                and "polytope_label_1" in nested
            ):
                return nested
        for name in (
            "global_reranked_candidates",
            "refined_candidates",
            "candidates",
        ):
            records = payload.get(name)
            if isinstance(records, list) and records:
                first = records[0]

                if (
                    isinstance(first, Mapping)
                    and "polytope_label_0" in first
                    and "polytope_label_1" in first
                ):
                    return first
    raise ValueError(
        "搜索结果JSON中未找到polytope_label_0/polytope_label_1"
    )


def load_polytope_pair(path: str | Path) -> PolytopePair:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    record = _extract_pair_record(payload)
    return canonical_pair(
        record["polytope_label_0"],
        record["polytope_label_1"],
    )

def rol16_np(x: np.ndarray, r: int) -> np.ndarray:
    r %= 16

    x32 = np.asarray(
        x,
        dtype=np.uint16,
    ).astype(
        np.uint32,
        copy=False,
    )

    return (
        ((x32 << r) | (x32 >> (16 - r))) & MASK16
    ).astype(np.uint16)

def ror16_np(x: np.ndarray, r: int) -> np.ndarray:
    r %= 16

    x32 = np.asarray(
        x,
        dtype=np.uint16,
    ).astype(
        np.uint32,
        copy=False,
    )

    return (
        ((x32 >> r) | (x32 << (16 - r))) & MASK16
    ).astype(np.uint16)

def rol16_scalar(x: int, r: int) -> int:
    r %= 16
    x &= MASK16

    return (
        (x << r) | (x >> (16 - r))
    ) & MASK16

def ror16_scalar(x: int, r: int) -> int:
    r %= 16
    x &= MASK16

    return (
        (x >> r) | (x << (16 - r))
    ) & MASK16

def speck_round_scalar(
    x: int,
    y: int,
    k: int,
) -> Tuple[int, int]:
    x = (
        ror16_scalar(x, 7) + y
    ) & MASK16

    x ^= int(k) & MASK16

    y = rol16_scalar(y, 2) ^ x
    return x & MASK16, y & MASK16


def expand_round_keys(
    master_key: Sequence[int],
    rounds: int,
) -> List[int]:

    if len(master_key) != 4:
        raise ValueError(
            "SPECK32/64主密钥必须由4个uint16字组成"
        )

    if not 1 <= rounds <= 22:
        raise ValueError(
            "轮数必须位于[1,22]"
        )
    b = int(master_key[0]) & MASK16
    a0 = int(master_key[1]) & MASK16
    a1 = int(master_key[2]) & MASK16
    a2 = int(master_key[3]) & MASK16
    round_keys: List[int] = []

    for i in range(rounds):
        round_keys.append(b)
        if i + 1 < rounds:
            a, b = speck_round_scalar(
                a0,
                b,
                i,
            )
            a0, a1, a2 = (
                a1,
                a2,
                a,
            )
    return round_keys


def encrypt_fixed_key_vertices(
    plaintext_vertices: np.ndarray,
    round_keys: Sequence[int],
) -> np.ndarray:
    p = np.asarray(
        plaintext_vertices,
        dtype=np.uint16,
    )
    if (
        p.ndim != 3
        or p.shape[0] != 4
        or p.shape[2] != 2
    ):
        raise ValueError(
            f"明文顶点应为(4,N,2)，实际为{p.shape}"
        )
    x = p[..., 0].copy()
    y = p[..., 1].copy()
    for rk in round_keys:

        x = (
            (
                ror16_np(x, 7).astype(np.uint32)
                + y.astype(np.uint32)
            )
            & MASK16
        ).astype(np.uint16)

        x ^= np.uint16(rk)

        y = rol16_np(y, 2) ^ x
    return np.stack(
        [x, y],
        axis=-1,
    ).astype(
        np.uint16,
        copy=False,
    )

def inverse_one_round_vertices(
    ciphertext_vertices: np.ndarray,
    guessed_subkey: int,
) -> np.ndarray:
    c = np.asarray(
        ciphertext_vertices,
        dtype=np.uint16,
    )
    if (
        c.ndim != 3
        or c.shape[0] != 4
        or c.shape[2] != 2
    ):
        raise ValueError(
            f"密文顶点应为(4,N,2)，实际为{c.shape}"
        )
    xp = c[..., 0]
    yp = c[..., 1]

    y = ror16_np(
        yp ^ xp,
        2,
    )
    tmp = (
        (
            xp ^ np.uint16(
                int(guessed_subkey) & MASK16
            )
        ).astype(np.uint32)
        - y.astype(np.uint32)
    ) & MASK16

    x = rol16_np(
        tmp.astype(np.uint16),
        7,
    )
    return np.stack(
        [x, y],
        axis=-1,
    ).astype(
        np.uint16,
        copy=False,
    )

def speck32_self_test() -> bool:
    key = [
        0x0100,
        0x0908,
        0x1110,
        0x1918,
    ]

    p = np.array(
        [
            [
                [
                    0x6574,
                    0x694C,
                ]
            ]
        ],
        dtype=np.uint16,
    )

    p = np.repeat(
        p,
        4,
        axis=0,
    )
    c = encrypt_fixed_key_vertices(
        p,
        expand_round_keys(
            key,
            22,
        ),
    )
    return (
        int(c[0, 0, 0]) == 0xA868
        and
        int(c[0, 0, 1]) == 0x42F2
    )

def words_to_bits(
    words: np.ndarray,
) -> np.ndarray:

    words = np.asarray(
        words,
        dtype=np.uint16,
    )

    if (
        words.ndim != 2
        or words.shape[1] != 2
    ):
        raise ValueError(
            f"words必须为(N,2)，实际为{words.shape}"
        )

    return (
        (
            words[..., None]
            >> BIT_SHIFTS
        )
        & 1
    ).astype(
        np.uint8,
        copy=False,
    )

def one_word_to_bits(
    words: np.ndarray,
) -> np.ndarray:

    words = np.asarray(
        words,
        dtype=np.uint16,
    )

    if words.ndim != 1:
        raise ValueError(
            f"words必须为(N,)，实际为{words.shape}"
        )

    return (
        (
            words[:, None]
            >> BIT_SHIFTS
        )
        & 1
    ).astype(
        np.uint8,
        copy=False,
    )

def speck_previous_right_word(
    state: np.ndarray,
) -> np.ndarray:
    return ror16_np(
        np.bitwise_xor(
            state[:, 1],
            state[:, 0],
        ),
        2,
    )

def vertices_to_bit_features(
    vertices: np.ndarray,
    include_anchor_xor: bool = True,
    include_previous_right: bool = True,
) -> np.ndarray:

    vertices = np.asarray(
        vertices,
        dtype=np.uint16,
    )

    if (
        vertices.ndim != 3
        or vertices.shape[0] != 4
        or vertices.shape[2] != 2
    ):
        raise ValueError(
            f"vertices必须为(4,N,2)，实际为{vertices.shape}"
        )

    channels: List[np.ndarray] = []

    for vertex in vertices:

        bits = words_to_bits(
            vertex
        )

        channels.extend(
            [
                bits[:, 0, :],
                bits[:, 1, :],
            ]
        )

    if include_anchor_xor:

        anchor = vertices[0]

        for vertex in vertices[1:]:

            bits = words_to_bits(
                anchor ^ vertex
            )

            channels.extend(
                [
                    bits[:, 0, :],
                    bits[:, 1, :],
                ]
            )

    if include_previous_right:

        previous = [
            speck_previous_right_word(vertex)
            for vertex in vertices
        ]

        channels.extend(
            one_word_to_bits(w)
            for w in previous
        )
        anchor_previous = previous[0]
        channels.extend(
            one_word_to_bits(
                anchor_previous ^ w
            )
            for w in previous[1:]
        )
    return np.stack(
        channels,
        axis=1,
    ).astype(
        np.uint8,
        copy=False,
    )

class SqueezeExcite1d(nn.Module):

    def __init__(
        self,
        channels: int,
        reduction: int,
    ):
        super().__init__()

        hidden = max(
            8,
            channels // max(
                1,
                reduction,
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(
                channels,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                channels,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        scale = self.fc(
            x.mean(dim=2)
        ).unsqueeze(2)

        return x * scale


class MultiScaleCircularResidualBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float,
        se_reduction: int,
    ):
        super().__init__()

        branch_channels = (
            channels // 2
        )

        self.branch3 = nn.Conv1d(
            channels,
            branch_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            padding_mode="circular",
            bias=False,
        )

        self.branch5 = nn.Conv1d(
            channels,
            branch_channels,
            kernel_size=5,
            dilation=dilation,
            padding=2 * dilation,
            padding_mode="circular",
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(
            channels
        )

        self.fuse = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.norm2 = nn.BatchNorm1d(
            channels
        )

        self.se = SqueezeExcite1d(
            channels,
            se_reduction,
        )

        self.dropout = nn.Dropout1d(
            dropout
        )

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        merged = torch.cat(
            [
                self.branch3(x),
                self.branch5(x),
            ],
            dim=1,
        )

        merged = self.activation(
            self.norm1(
                merged
            )
        )

        merged = self.dropout(
            merged
        )

        merged = self.norm2(
            self.fuse(
                merged
            )
        )

        merged = self.se(
            merged
        )

        return self.activation(
            residual + merged
        )


class BitAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            width
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(
            width
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                width,
                2 * width,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                2 * width,
                width,
            ),
            nn.Dropout(
                dropout
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        tokens = x.transpose(
            1,
            2,
        )

        normalized = self.norm1(
            tokens
        )

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        tokens = (
            tokens + attended
        )

        tokens = (
            tokens
            + self.ffn(
                self.norm2(
                    tokens
                )
            )
        )

        return tokens.transpose(
            1,
            2,
        )


class PolytopicPairND(nn.Module):

    def __init__(
        self,
        input_channels: int = 21,
        width: int = 96,
        dilations: Sequence[int] = (
            1,
            2,
            4,
            7,
            1,
            2,
            4,
            7,
        ),
        dense_units: int = 384,
        dropout: float = 0.10,
        attention_heads: int = 4,
        attention_layers: int = 1,
        se_reduction: int = 8,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                width,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(
                width
            ),
            nn.GELU(),
        )

        self.position_bias = nn.Parameter(
            torch.zeros(
                1,
                width,
                16,
            )
        )

        nn.init.normal_(
            self.position_bias,
            mean=0.0,
            std=0.02,
        )

        self.blocks = nn.ModuleList(
            [
                MultiScaleCircularResidualBlock(
                    width,
                    dilation,
                    dropout,
                    se_reduction,
                )
                for dilation in dilations
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                BitAttentionBlock(
                    width,
                    attention_heads,
                    dropout,
                )
                for _ in range(
                    attention_layers
                )
            ]
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                width * 16,
                dense_units,
            ),
            nn.BatchNorm1d(
                dense_units
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                dense_units,
                dense_units // 2,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                dense_units // 2,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = (
            x.float()
            .mul_(2.0)
            .sub_(1.0)
        )

        x = (
            self.stem(x)
            + self.position_bias
        )

        for block in self.blocks:
            x = block(x)

        for attention in self.attention_blocks:
            x = attention(x)

        return self.head(
            x
        ).squeeze(1)

@dataclass
class LoadedModel:
    model: PolytopicPairND
    threshold: float
    source_type: str
    source_path: str

def _remove_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    data = dict(
        state_dict
    )

    if (
        data
        and
        all(
            str(k).startswith("module.")
            for k in data
        )
    ):
        return {
            str(k)[len("module."):]: v
            for k, v in data.items()
        }

    return data


def load_enhanced_pdnd(
    path: str | Path,
    device: torch.device,
    threshold_override: Optional[float] = None,
) -> LoadedModel:
    path = Path(path)

    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        payload = torch.load(
            path,
            map_location=device,
        )

    if (
        isinstance(payload, Mapping)
        and
        "model_state_dict" in payload
    ):
        state_dict = _remove_module_prefix(
            payload["model_state_dict"]
        )

        model_cfg = dict(
            payload.get(
                "model_config",
                {},
            )
        )

        input_channels = int(
            payload.get(
                "input_channels",
                21,
            )
        )

        validation = payload.get(
            "validation",
            {},
        )

        stored_threshold = (
            float(
                validation.get(
                    "decision_threshold",
                    0.0,
                )
            )
            if isinstance(
                validation,
                Mapping,
            )
            else 0.0
        )

        source_type = "checkpoint"

    elif isinstance(
        payload,
        Mapping,
    ):
        state_dict = _remove_module_prefix(
            payload
        )

        model_cfg = {}
        input_channels = 21
        stored_threshold = 0.0
        source_type = "state_dict"

    else:
        raise TypeError(
            f"无法识别权重文件格式: {path}"
        )

    model = PolytopicPairND(
        input_channels=input_channels,
        width=int(
            model_cfg.get(
                "width",
                96,
            )
        ),
        dilations=tuple(
            model_cfg.get(
                "dilations",
                (
                    1,
                    2,
                    4,
                    7,
                    1,
                    2,
                    4,
                    7,
                ),
            )
        ),
        dense_units=int(
            model_cfg.get(
                "dense_units",
                384,
            )
        ),
        dropout=float(
            model_cfg.get(
                "dropout",
                0.10,
            )
        ),
        attention_heads=int(
            model_cfg.get(
                "attention_heads",
                4,
            )
        ),
        attention_layers=int(
            model_cfg.get(
                "attention_layers",
                1,
            )
        ),
        se_reduction=int(
            model_cfg.get(
                "se_reduction",
                8,
            )
        ),
    ).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()
    threshold = (
        stored_threshold
        if threshold_override is None
        else float(
            threshold_override
        )
    )

    return LoadedModel(
        model=model,
        threshold=threshold,
        source_type=source_type,
        source_path=str(
            path.resolve()
        ),
    )

def make_balanced_labels(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:

    if n < 2:
        raise ValueError(
            "样本数至少为2"
        )

    n0 = n // 2
    n1 = n - n0

    y = np.concatenate(
        [
            np.zeros(
                n0,
                dtype=np.uint8,
            ),
            np.ones(
                n1,
                dtype=np.uint8,
            ),
        ]
    )
    rng.shuffle(y)

    return y


def generate_plaintext_vertices_for_labels(
    labels: np.ndarray,
    pair: PolytopePair,
    rng: np.random.Generator,
) -> np.ndarray:

    labels = np.asarray(
        labels,
        dtype=np.uint8,
    )

    n = labels.size

    anchor = rng.integers(
        0,
        1 << 16,
        size=(
            n,
            2,
        ),
        dtype=np.uint16,
    )

    vertices = np.empty(
        (
            4,
            n,
            2,
        ),
        dtype=np.uint16,
    )

    vertices[0] = anchor

    p0 = np.asarray(
        pair[0],
        dtype=np.uint16,
    )

    p1 = np.asarray(
        pair[1],
        dtype=np.uint16,
    )

    for j in range(3):

        selected = np.where(
            labels[:, None] == 0,
            p0[j][None, :],
            p1[j][None, :],
        ).astype(
            np.uint16,
            copy=False,
        )

        vertices[j + 1] = (
            anchor ^ selected
        )
    return vertices


def random_master_key(
    rng: np.random.Generator,
) -> np.ndarray:

    return rng.integers(
        0,
        1 << 16,
        size=4,
        dtype=np.uint16,
    )

@torch.inference_mode()
def infer_logits(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    use_amp: bool = True,
) -> np.ndarray:

    outputs: List[np.ndarray] = []

    for start in range(
        0,
        features.shape[0],
        batch_size,
    ):

        end = min(
            start + batch_size,
            features.shape[0],
        )

        x = torch.from_numpy(
            np.ascontiguousarray(
                features[start:end]
            )
        ).to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(
                use_amp
                and
                device.type == "cuda"
            ),
        ):
            logits = model(x)

        outputs.append(
            logits
            .float()
            .cpu()
            .numpy()
        )

    return np.concatenate(
        outputs
    )


def accuracy_from_logits(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float,
) -> float:

    prediction = (
        np.asarray(logits)
        >= float(threshold)
    ).astype(np.uint8)

    return float(
        np.mean(
            prediction
            ==
            np.asarray(
                labels,
                dtype=np.uint8,
            )
        )
    )


def mean_log_likelihood(
    labels: np.ndarray,
    logits: np.ndarray,
) -> float:

    y = np.asarray(
        labels,
        dtype=np.uint8,
    )

    z = np.asarray(
        logits,
        dtype=np.float64,
    )

    ll = np.where(
        y == 1,
        -np.logaddexp(
            0.0,
            -z,
        ),
        -np.logaddexp(
            0.0,
            z,
        ),
    )

    return float(
        np.mean(ll)
    )

@dataclass
class ResponseTestResult:
    model_rounds: int
    attack_rounds: int
    repetitions: int
    samples_per_repetition: int

    correct_mean: float
    correct_std: float

    wrong_mean: float
    wrong_std: float

    gap_mean: float

    correct_scores: List[float]
    wrong_scores: List[float]

def evaluate_correct_and_wrong_same_data(
    loaded: LoadedModel,
    pair: PolytopePair,
    attack_rounds: int,
    master_key: np.ndarray,
    true_subkey: int,
    wrong_subkey: int,
    total_samples: int,
    rng: np.random.Generator,
    device: torch.device,
    sample_chunk: int = 65536,
    inference_batch_size: int = 4096,
    use_amp: bool = True,
) -> Tuple[float, float]:

    round_keys = expand_round_keys(
        master_key,
        attack_rounds,
    )

    if int(
        round_keys[-1]
    ) != int(
        true_subkey
    ):
        raise AssertionError(
            "true_subkey与密钥扩展结果不一致"
        )

    correct_hits = 0
    wrong_hits = 0
    seen = 0

    while seen < total_samples:

        n = min(
            sample_chunk,
            total_samples - seen,
        )

        labels = make_balanced_labels(
            n,
            rng,
        )

        plaintext_vertices = (
            generate_plaintext_vertices_for_labels(
                labels,
                pair,
                rng,
            )
        )

        ciphertext_vertices = (
            encrypt_fixed_key_vertices(
                plaintext_vertices,
                round_keys,
            )
        )

        recovered_true = (
            inverse_one_round_vertices(
                ciphertext_vertices,
                true_subkey,
            )
        )

        features_true = (
            vertices_to_bit_features(
                recovered_true,
                True,
                True,
            )
        )

        logits_true = infer_logits(
            loaded.model,
            features_true,
            device,
            inference_batch_size,
            use_amp,
        )

        correct_hits += int(
            np.sum(
                (
                    logits_true
                    >= loaded.threshold
                ).astype(
                    np.uint8
                )
                ==
                labels
            )
        )

        recovered_wrong = (
            inverse_one_round_vertices(
                ciphertext_vertices,
                wrong_subkey,
            )
        )

        features_wrong = (
            vertices_to_bit_features(
                recovered_wrong,
                True,
                True,
            )
        )

        logits_wrong = infer_logits(
            loaded.model,
            features_wrong,
            device,
            inference_batch_size,
            use_amp,
        )

        wrong_hits += int(
            np.sum(
                (
                    logits_wrong
                    >= loaded.threshold
                ).astype(
                    np.uint8
                )
                ==
                labels
            )
        )

        seen += n

        del plaintext_vertices
        del ciphertext_vertices
        del recovered_true
        del recovered_wrong
        del features_true
        del features_wrong
        del logits_true
        del logits_wrong

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        correct_hits / total_samples,
        wrong_hits / total_samples,
    )


def run_mirzaali_response_test(
    *,
    model_rounds: int,
    pair_path: str,
    weights_path: str,
    total_samples: int = 1_000_000,
    repetitions: int = 100,
    seed: int = 20260712,
    threshold_override: Optional[float] = None,
    sample_chunk: int = 65536,
    inference_batch_size: int = 4096,
    use_amp: bool = True,
    output_dir: str = "./key_recovery_results",
) -> ResponseTestResult:

    attack_rounds = (
        model_rounds + 1
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pair = load_polytope_pair(
        pair_path
    )

    loaded = load_enhanced_pdnd(
        weights_path,
        device,
        threshold_override,
    )

    rng = np.random.default_rng(
        seed
        +
        model_rounds
        *
        10_000_019
    )

    correct_scores: List[float] = []
    wrong_scores: List[float] = []

    print("=" * 96)

    print(
        f"Mirzaali-style: "
        f"R{model_rounds} PDND -> "
        f"R{attack_rounds} SPECK32/64"
    )

    print(
        f"device={device}, "
        f"threshold(logit)="
        f"{loaded.threshold:+.6f}"
    )

    print(
        f"samples/trial="
        f"{total_samples:,}, "
        f"trials={repetitions}"
    )

    print("=" * 96)

    for rep in range(
        repetitions
    ):

        master_key = random_master_key(
            rng
        )

        round_keys = expand_round_keys(
            master_key,
            attack_rounds,
        )

        true_subkey = int(
            round_keys[-1]
        )

        wrong_subkey = int(
            rng.integers(
                0,
                1 << 16,
            )
        )

        while wrong_subkey == true_subkey:
            wrong_subkey = int(
                rng.integers(
                    0,
                    1 << 16,
                )
            )

        t0 = time.time()

        correct_acc, wrong_acc = (
            evaluate_correct_and_wrong_same_data(
                loaded=loaded,
                pair=pair,
                attack_rounds=attack_rounds,
                master_key=master_key,
                true_subkey=true_subkey,
                wrong_subkey=wrong_subkey,
                total_samples=total_samples,
                rng=rng,
                device=device,
                sample_chunk=sample_chunk,
                inference_batch_size=inference_batch_size,
                use_amp=use_amp,
            )
        )

        elapsed = (
            time.time() - t0
        )

        correct_scores.append(
            correct_acc
        )

        wrong_scores.append(
            wrong_acc
        )

        print(
            f"[{rep+1:03d}/{repetitions}] "
            f"true=0x{true_subkey:04x} "
            f"correct={correct_acc:.4%} | "
            f"wrong=0x{wrong_subkey:04x} "
            f"wrong_acc={wrong_acc:.4%} | "
            f"gap={(correct_acc-wrong_acc):+.4%} "
            f"| {elapsed:.1f}s"
        )

    c = np.asarray(
        correct_scores,
        dtype=np.float64,
    )

    w = np.asarray(
        wrong_scores,
        dtype=np.float64,
    )

    result = ResponseTestResult(
        model_rounds=model_rounds,
        attack_rounds=attack_rounds,
        repetitions=repetitions,
        samples_per_repetition=total_samples,

        correct_mean=float(
            c.mean()
        ),

        correct_std=float(
            c.std(ddof=1)
            if repetitions > 1
            else 0.0
        ),

        wrong_mean=float(
            w.mean()
        ),

        wrong_std=float(
            w.std(ddof=1)
            if repetitions > 1
            else 0.0
        ),

        gap_mean=float(
            (c - w).mean()
        ),

        correct_scores=[
            float(v)
            for v in c
        ],

        wrong_scores=[
            float(v)
            for v in w
        ],
    )

    out_dir = Path(
        output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out_dir
        /
        f"mirzaali_style_R"
        f"{model_rounds}_to_R"
        f"{attack_rounds}.json"
    )

    path.write_text(
        json.dumps(
            asdict(result),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"结果已保存: "
        f"{path.resolve()}"
    )

    return result

@dataclass
class ExhaustiveOneRun:
    model_rounds: int
    attack_rounds: int
    samples: int
    true_subkey: int
    true_rank: int
    true_score: float
    top1_key: int
    top1_score: float
    top10_keys: List[int]
    elapsed_seconds: float

@dataclass
class ExhaustiveSummary:
    model_rounds: int
    attack_rounds: int
    samples: int
    repetitions: int
    mean_rank: float
    median_rank: float
    sr_at_1: float
    sr_at_10: float
    sr_at_100: float
    runs: List[Dict[str, Any]]

def make_attack_dataset(
    pair: PolytopePair,
    attack_rounds: int,
    master_key: np.ndarray,
    total_samples: int,
    rng: np.random.Generator,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    labels = make_balanced_labels(
        total_samples,
        rng,
    )

    plaintext_vertices = (
        generate_plaintext_vertices_for_labels(
            labels,
            pair,
            rng,
        )
    )

    round_keys = expand_round_keys(
        master_key,
        attack_rounds,
    )

    ciphertext_vertices = (
        encrypt_fixed_key_vertices(
            plaintext_vertices,
            round_keys,
        )
    )

    true_subkey = int(
        round_keys[-1]
    )

    return (
        ciphertext_vertices,
        labels,
        true_subkey,
    )


def score_candidate_key_block(
    loaded: LoadedModel,
    ciphertext_vertices: np.ndarray,
    labels: np.ndarray,
    keys: np.ndarray,
    device: torch.device,
    inference_batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    n = labels.size
    features_list: List[np.ndarray] = []
    for key in keys:

        recovered = (
            inverse_one_round_vertices(
                ciphertext_vertices,
                int(key),
            )
        )
        features_list.append(
            vertices_to_bit_features(
                recovered,
                True,
                True,
            )
        )

    all_features = np.concatenate(
        features_list,
        axis=0,
    )

    logits = infer_logits(
        loaded.model,
        all_features,
        device,
        batch_size=inference_batch_size,
        use_amp=use_amp,
    ).reshape(
        len(keys),
        n,
    )

    y = labels.astype(
        np.uint8
    )[None, :]

    ll = np.where(
        y == 1,

        -np.logaddexp(
            0.0,
            -logits.astype(
                np.float64
            ),
        ),

        -np.logaddexp(
            0.0,
            logits.astype(
                np.float64
            ),
        ),
    )

    return ll.mean(
        axis=1
    )

def exhaustive_last_subkey_search(
    *,
    model_rounds: int,
    pair_path: str,
    weights_path: str,
    total_samples: int,
    seed: int,
    threshold_override: Optional[float] = None,
    key_block_size: int = 16,
    inference_batch_size: int = 4096,
    use_amp: bool = True,
    progress_every_keys: int = 1024,
) -> ExhaustiveOneRun:

    attack_rounds = (
        model_rounds + 1
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pair = load_polytope_pair(
        pair_path
    )

    loaded = load_enhanced_pdnd(
        weights_path,
        device,
        threshold_override,
    )

    rng = np.random.default_rng(
        seed
    )

    master_key = random_master_key(
        rng
    )

    (
        ciphertext_vertices,
        labels,
        true_subkey,
    ) = make_attack_dataset(
        pair,
        attack_rounds,
        master_key,
        total_samples,
        rng,
    )

    scores = np.empty(
        1 << 16,
        dtype=np.float64,
    )

    keys_all = np.arange(
        1 << 16,
        dtype=np.uint32,
    )

    t0 = time.time()
    for start in range(
        0,
        1 << 16,
        key_block_size,
    ):

        end = min(
            start + key_block_size,
            1 << 16,
        )

        keys = keys_all[
            start:end
        ]

        scores[start:end] = (
            score_candidate_key_block(
                loaded=loaded,
                ciphertext_vertices=ciphertext_vertices,
                labels=labels,
                keys=keys,
                device=device,
                inference_batch_size=inference_batch_size,
                use_amp=use_amp,
            )
        )

        if (
            end % progress_every_keys == 0
            or
            end == (1 << 16)
        ):

            elapsed = (
                time.time() - t0
            )

            current_order = np.argsort(
                scores[:end],
                kind="mergesort",
            )[::-1]

            current_best = int(
                current_order[0]
            )
            print(
                f"[R{attack_rounds}] "
                f"keys={end:5d}/65536 "
                f"elapsed={elapsed:.1f}s "
                f"current_best="
                f"0x{current_best:04x}"
            )

    elapsed = (
        time.time() - t0
    )

    order = np.argsort(
        scores,
        kind="mergesort",
    )[::-1]

    true_rank = int(
        np.where(
            order == true_subkey
        )[0][0]
    ) + 1

    result = ExhaustiveOneRun(
        model_rounds=model_rounds,
        attack_rounds=attack_rounds,

        samples=total_samples,

        true_subkey=true_subkey,
        true_rank=true_rank,

        true_score=float(
            scores[true_subkey]
        ),

        top1_key=int(
            order[0]
        ),

        top1_score=float(
            scores[
                order[0]
            ]
        ),
        top10_keys=[
            int(k)
            for k in order[:10]
        ],

        elapsed_seconds=float(
            elapsed
        ),
    )

    print("=" * 96)
    print(
        f"R{model_rounds} PDND "
        f"-> R{attack_rounds}: "
        f"N={total_samples:,}, "
        f"true=0x{true_subkey:04x}, "
        f"rank={true_rank}/65536, "
        f"top1=0x{int(order[0]):04x}, "
        f"time={elapsed:.1f}s"
    )
    print(
        "Top-10:",
        " ".join(
            f"0x{int(k):04x}"
            for k in order[:10]
        ),
    )

    print("=" * 96)

    return result

def run_exhaustive_repetitions(
    *,
    model_rounds: int,
    pair_path: str,
    weights_path: str,
    sample_sizes: Sequence[int] = (1024,),
    repetitions: int = 5,
    base_seed: int = 20260712,
    threshold_override: Optional[float] = None,
    key_block_size: int = 16,
    inference_batch_size: int = 4096,
    use_amp: bool = True,
    output_dir: str = "./key_recovery_results",
) -> List[ExhaustiveSummary]:
    summaries: List[
        ExhaustiveSummary
    ] = []

    out_dir = Path(
        output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for n in sample_sizes:

        runs: List[
            ExhaustiveOneRun
        ] = []

        for rep in range(
            repetitions
        ):

            seed = (
                base_seed
                +
                int(n) * 1009
                +
                rep * 10_000_019
            )

            run = exhaustive_last_subkey_search(
                model_rounds=model_rounds,
                pair_path=pair_path,
                weights_path=weights_path,
                total_samples=int(n),
                seed=seed,
                threshold_override=threshold_override,
                key_block_size=key_block_size,
                inference_batch_size=inference_batch_size,
                use_amp=use_amp,
            )

            runs.append(
                run
            )

        ranks = np.asarray(
            [
                r.true_rank
                for r in runs
            ],
            dtype=np.int64,
        )

        summary = ExhaustiveSummary(
            model_rounds=model_rounds,
            attack_rounds=model_rounds + 1,

            samples=int(n),
            repetitions=repetitions,

            mean_rank=float(
                ranks.mean()
            ),

            median_rank=float(
                np.median(
                    ranks
                )
            ),

            sr_at_1=float(
                np.mean(
                    ranks <= 1
                )
            ),

            sr_at_10=float(
                np.mean(
                    ranks <= 10
                )
            ),

            sr_at_100=float(
                np.mean(
                    ranks <= 100
                )
            ),

            runs=[
                asdict(r)
                for r in runs
            ],
        )

        summaries.append(
            summary
        )

        path = (
            out_dir
            /
            f"exhaustive_R"
            f"{model_rounds}_to_R"
            f"{model_rounds + 1}_N"
            f"{int(n)}.json"
        )

        path.write_text(
            json.dumps(
                asdict(summary),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"结果已保存: "
            f"{path.resolve()}"
        )

    return summaries

def main() -> None:

    print("=" * 100)
    print("SPECK32/64 enhanced-PDND key recovery")
    print("=" * 100)

    if not speck32_self_test():
        raise RuntimeError(
            "SPECK32/64 self-test failed: "
            "请检查左右字顺序、轮函数和密钥扩展顺序"
        )

    print("SPECK32/64 self-test passed.")
    print()

    # 1. 选择实验模式
    # "response"
    # "exhaustive"
    RUN_MODE = "response"

    RESPONSE_MODEL_ROUNDS = [
        5,
        6,
        7,
        8,
    ]
    RESPONSE_SAMPLES = 1_000_000
    RESPONSE_REPETITIONS = 100
    RESPONSE_SAMPLE_CHUNK = 10_000
    # Exhaustive完整密钥搜索参数
    EXHAUSTIVE_MODEL_ROUND = 7
    EXHAUSTIVE_SAMPLE_SIZES = [
        256,
        512,
        1024,
        2048,
    ]
    EXHAUSTIVE_REPETITIONS = 20
    KEY_BLOCK_SIZE = 16
    RANDOM_SEED = 20260712
    INFERENCE_BATCH_SIZE = 4096
    USE_AMP = True
    THRESHOLD_OVERRIDE = None
    SCRIPT_DIR = Path(
        __file__
    ).resolve().parent

    PROJECT_DIR = (
        SCRIPT_DIR.parent
    )
    PAIR_PATHS = {
        5: PROJECT_DIR
        / "data"
        / "best_polytope_pair_5round.json",

        6: PROJECT_DIR
        / "data"
        / "best_polytope_pair_6round.json",

        7: PROJECT_DIR
        / "data"
        / "best_polytope_pair_7round.json",

        8: PROJECT_DIR
        / "data"
        / "best_polytope_pair_8round.json",
    }

    MODEL_TO_PAIR_ROUND = {
        5: 5,
        6: 6,
        7: 7,
        8: 7,
    }

    WEIGHT_PATHS = {
        5: SCRIPT_DIR
        / "pdnd_accuracy_improved_checkpoints"
        / "speck32_5r_improved_best_weights.pth",

        6: SCRIPT_DIR
        / "pdnd_accuracy_improved_checkpoints"
        / "speck32_6r_improved_best_weights.pth",

        7: SCRIPT_DIR
        / "pdnd_accuracy_improved_checkpoints"
        / "speck32_7r_improved_best_weights.pth",

        8: SCRIPT_DIR
        / "pdnd_accuracy_improved_checkpoints"
        / "speck32_8r_improved_best_weights.pth",
    }

    OUTPUT_DIR = (
        PROJECT_DIR
        /
        "key_recovery_results"
    )

    RUN_MODE = (
        RUN_MODE
        .lower()
        .strip()
    )
    if RUN_MODE not in {
        "response",
        "exhaustive",
    }:
        raise ValueError(
            f"RUN_MODE={RUN_MODE!r} 无效，"
            "只能设置为 'response' 或 'exhaustive'"
        )

    print()
    print("-" * 100)
    print("实验文件路径检查")
    print("-" * 100)

    print(
        f"脚本目录："
        f"{SCRIPT_DIR}"
    )

    print(
        f"项目目录："
        f"{PROJECT_DIR}"
    )

    print(
        f"结果目录："
        f"{OUTPUT_DIR}"
    )

    print()

    if RUN_MODE == "response":
        rounds_to_check = (
            RESPONSE_MODEL_ROUNDS
        )
    else:
        rounds_to_check = [
            EXHAUSTIVE_MODEL_ROUND
        ]

    for r in rounds_to_check:

        pair_round = (
            MODEL_TO_PAIR_ROUND[r]
        )

        pair_file = Path(
            PAIR_PATHS[
                pair_round
            ]
        )

        weight_file = Path(
            WEIGHT_PATHS[
                r
            ]
        )

        print(
            f"R{r}差分文件："
            f"{pair_file}"
        )

        print(
            f"  exists = "
            f"{pair_file.exists()}"
        )

        print(
            f"R{r}模型权重："
            f"{weight_file}"
        )

        print(
            f"  exists = "
            f"{weight_file.exists()}"
        )

        if not pair_file.exists():
            raise FileNotFoundError(
                f"\n找不到R{r}差分文件：\n"
                f"{pair_file}"
            )

        if not weight_file.exists():
            raise FileNotFoundError(
                f"\n找不到R{r}模型权重：\n"
                f"{weight_file}"
            )

    Path(
        OUTPUT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "实验文件路径检查通过。"
    )
    print("-" * 100)
    print()

    if RUN_MODE == "response":

        print("#" * 100)

        print(
            "运行模式："
            "Mirzaali-style correct-key / wrong-key response test"
        )

        print("#" * 100)

        all_results = []

        for model_rounds in (
            RESPONSE_MODEL_ROUNDS
        ):

            pair_round = (
                MODEL_TO_PAIR_ROUND[
                    model_rounds
                ]
            )

            pair_path = (
                PAIR_PATHS[
                    pair_round
                ]
            )

            weights_path = (
                WEIGHT_PATHS[
                    model_rounds
                ]
            )

            print()
            print("=" * 100)

            print(
                f"开始实验："
                f"R{model_rounds} PDND "
                f"-> R{model_rounds + 1} SPECK32/64"
            )

            print(
                f"差分文件："
                f"{pair_path}"
            )

            print(
                f"模型权重："
                f"{weights_path}"
            )

            print(
                f"样本数量："
                f"{RESPONSE_SAMPLES:,}"
            )

            print(
                f"重复次数："
                f"{RESPONSE_REPETITIONS}"
            )

            print("=" * 100)

            result = (
                run_mirzaali_response_test(
                    model_rounds=model_rounds,
                    pair_path=pair_path,
                    weights_path=weights_path,
                    total_samples=RESPONSE_SAMPLES,
                    repetitions=RESPONSE_REPETITIONS,
                    seed=RANDOM_SEED,
                    threshold_override=THRESHOLD_OVERRIDE,
                    sample_chunk=RESPONSE_SAMPLE_CHUNK,
                    inference_batch_size=INFERENCE_BATCH_SIZE,
                    use_amp=USE_AMP,
                    output_dir=OUTPUT_DIR,
                )
            )

            all_results.append(
                result
            )

        summary_path = (
            Path(
                OUTPUT_DIR
            )
            /
            "mirzaali_style_all_rounds_summary.json"
        )

        summary_path.write_text(
            json.dumps(
                [
                    asdict(result)
                    for result in all_results
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 100)

        print(
            "Response实验全部完成"
        )

        print(
            f"汇总结果："
            f"{summary_path.resolve()}"
        )

        print("=" * 100)

        return

    if RUN_MODE == "exhaustive":

        model_rounds = (
            EXHAUSTIVE_MODEL_ROUND
        )

        pair_round = (
            MODEL_TO_PAIR_ROUND[
                model_rounds
            ]
        )

        pair_path = (
            PAIR_PATHS[
                pair_round
            ]
        )

        weights_path = (
            WEIGHT_PATHS[
                model_rounds
            ]
        )

        print("#" * 100)

        print(
            "运行模式："
            "完整2^16末轮子密钥枚举"
        )

        print(
            f"R{model_rounds} PDND "
            f"-> R{model_rounds + 1} SPECK32/64"
        )

        print(
            f"样本量："
            f"{EXHAUSTIVE_SAMPLE_SIZES}"
        )

        print(
            f"重复次数："
            f"{EXHAUSTIVE_REPETITIONS}"
        )

        print("#" * 100)

        summaries = (
            run_exhaustive_repetitions(
                model_rounds=model_rounds,
                pair_path=pair_path,
                weights_path=weights_path,
                sample_sizes=EXHAUSTIVE_SAMPLE_SIZES,
                repetitions=EXHAUSTIVE_REPETITIONS,
                base_seed=RANDOM_SEED,
                threshold_override=THRESHOLD_OVERRIDE,
                key_block_size=KEY_BLOCK_SIZE,
                inference_batch_size=INFERENCE_BATCH_SIZE,
                use_amp=USE_AMP,
                output_dir=OUTPUT_DIR,
            )
        )

        print()
        print("=" * 100)
        print("完整密钥搜索结果")
        print("=" * 100)

        for summary in summaries:

            print(
                f"N="
                f"{summary.samples:,}"
            )

            print(
                f"平均Rank="
                f"{summary.mean_rank:.2f}"
            )

            print(
                f"Rank中位数="
                f"{summary.median_rank:.2f}"
            )

            print(
                f"SR@1="
                f"{summary.sr_at_1:.4%}"
            )

            print(
                f"SR@10="
                f"{summary.sr_at_10:.4%}"
            )

            print(
                f"SR@100="
                f"{summary.sr_at_100:.4%}"
            )

            print("-" * 80)

        return

if __name__ == "__main__":
    main()
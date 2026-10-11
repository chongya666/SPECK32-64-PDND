from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import concurrent.futures
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import norm, rankdata

from DRL_search_polytopic_pair import (
    CryptoPanel,
    ProjectionBank,
    RL_SearchBestDiff,
    hamming_weight,
    normalize_difference,
)


WordDiff = Tuple[int, int]
_WORKER_SEARCHER: RL_SearchBestDiff | None = None
_WORKER_DLL = None
_WORKER_ENCRYPT_BATCH = None
_WORKER_PANELS: List[CryptoPanel] = []
_WORKER_PROJECTIONS: ProjectionBank | None = None


@dataclass(frozen=True)
class ScreeningConfig:
    max_hw: int = 4

    # Stage 1: cheap but globally comparable evaluation.
    coarse_sample_size: int = 4096
    coarse_panel_count: int = 2
    coarse_keep: int = 800
    coarse_chunk_size: int = 64

    # Stage 2: independent, larger panels.
    refine_sample_size: int = 32_768
    refine_panel_count: int = 3
    refine_chunk_size: int = 16

    num_projection_hashes: int = 24
    projection_width: int = 8
    top_k_bits: int = 6
    familywise_alpha: float = 0.01

    final_n: int = 64
    diversity_lambda: float = 0.25
    base_seed: int = 20260712
    num_workers: int = min(8, os.cpu_count() or 1)

    def validate(self) -> None:
        if not 1 <= self.max_hw <= 32:
            raise ValueError("max_hw 必须位于 [1,32]")
        if self.coarse_sample_size < 128 or self.refine_sample_size < 128:
            raise ValueError("样本量过小")
        if self.coarse_panel_count < 1 or self.refine_panel_count < 1:
            raise ValueError("panel_count 必须为正")
        if self.coarse_keep < self.final_n:
            raise ValueError("coarse_keep 必须不小于 final_n")
        if not 0.0 <= self.diversity_lambda <= 1.0:
            raise ValueError("diversity_lambda 必须位于 [0,1]")


def generate_low_hw_candidates(max_hw: int = 4) -> List[WordDiff]:
    candidates: List[WordDiff] = []
    for hw in range(1, max_hw + 1):
        for positions in combinations(range(32), hw):
            value = 0
            for position in positions:
                value |= 1 << (31 - position)
            candidates.append(((value >> 16) & 0xFFFF, value & 0xFFFF))
    return candidates


def _init_worker(
    config_path: str,
    nr: int,
    sample_size: int,
    panel_count: int,
    panel_seed: int,
    num_projection_hashes: int,
    projection_width: int,
) -> None:
    global _WORKER_SEARCHER, _WORKER_DLL, _WORKER_ENCRYPT_BATCH
    global _WORKER_PANELS, _WORKER_PROJECTIONS

    _WORKER_SEARCHER = RL_SearchBestDiff(config_path=config_path, nr=nr)
    _WORKER_DLL, _WORKER_ENCRYPT_BATCH = _WORKER_SEARCHER.load_encrypt_function()
    _WORKER_PANELS = [
        _WORKER_SEARCHER.make_panel(sample_size, panel_seed + index * 1_000_003)
        for index in range(panel_count)
    ]
    _WORKER_PROJECTIONS = ProjectionBank.build(
        feature_dim=32,
        seed=panel_seed + 97,
        num_hashes=num_projection_hashes,
        hash_width=projection_width,
        num_pairs=1,  # pair projections are not needed in seed screening.
    )


def _xor_words_to_bits(xor_words: np.ndarray) -> np.ndarray:
    xor_words = np.asarray(xor_words, dtype=np.uint16)
    if xor_words.ndim != 2 or xor_words.shape[1] != 2:
        raise ValueError(f"期望 XOR 输出形状为 (N,2)，实际为 {xor_words.shape}")
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return ((xor_words[..., None] >> shifts) & 1).astype(np.uint8).reshape(xor_words.shape[0], 32)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _projection_js_uniform(bits: np.ndarray, projection_bank: ProjectionBank) -> Tuple[float, np.ndarray]:
    width = projection_bank.hash_indices.shape[1]
    weights = 1 << np.arange(width, dtype=np.uint32)
    bins = 1 << width
    uniform = np.full(bins, 1.0 / bins, dtype=np.float64)
    js_values = []
    entropy_deficits = []

    for indices in projection_bank.hash_indices:
        codes = np.sum(bits[:, indices].astype(np.uint32) * weights, axis=1)
        counts = np.bincount(codes.astype(np.int64), minlength=bins).astype(np.float64)
        counts += 0.5  # Jeffreys smoothing stabilizes sparse bins.
        probability = counts / counts.sum()
        js_values.append(_js_divergence(probability, uniform))
        entropy = -float(np.sum(probability * np.log2(probability)))
        entropy_deficits.append(1.0 - entropy / width)
    return float(np.mean(js_values)), np.asarray(entropy_deficits, dtype=np.float64)


def _evaluate_one_panel(
    xor_words: np.ndarray,
    projection_bank: ProjectionBank,
    top_k_bits: int,
    z_critical: float,
) -> Dict[str, object]:
    bits = _xor_words_to_bits(xor_words)
    signed_bias = bits.mean(axis=0) - 0.5
    absolute_bias = np.abs(signed_bias)
    k = min(max(1, top_k_bits), 32)
    top_k = np.partition(absolute_bias, 32 - k)[-k:]
    projected_js, entropy_deficits = _projection_js_uniform(bits, projection_bank)
    z_scores = 2.0 * math.sqrt(bits.shape[0]) * absolute_bias

    return {
        "signed_bias": signed_bias,
        "marginal_rms": float(np.sqrt(np.mean(absolute_bias**2))),
        "topk_bias": float(np.mean(top_k)),
        "max_bias": float(np.max(absolute_bias)),
        "projected_js_uniform": projected_js,
        "entropy_deficits": entropy_deficits,
        "significant_bits": np.flatnonzero(z_scores >= z_critical).astype(int).tolist(),
    }


def _aggregate_runs(diff: WordDiff, runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    bias_stack = np.stack([run["signed_bias"] for run in runs], axis=0)
    deficit_stack = np.stack([run["entropy_deficits"] for run in runs], axis=0)
    strengths = np.linalg.norm(bias_stack, axis=1)

    cosine_values = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            denominator = strengths[i] * strengths[j]
            cosine_values.append(
                0.0 if denominator < 1e-15 else float(np.dot(bias_stack[i], bias_stack[j]) / denominator)
            )
    direction_consistency = (
        float(np.mean([(value + 1.0) / 2.0 for value in cosine_values]))
        if cosine_values
        else 1.0
    )
    rms_values = np.asarray([run["marginal_rms"] for run in runs], dtype=np.float64)
    js_values = np.asarray([run["projected_js_uniform"] for run in runs], dtype=np.float64)
    relative_cv = 0.5 * (
        np.std(rms_values) / (np.mean(rms_values) + 1e-12)
        + np.std(js_values) / (np.mean(js_values) + 1e-12)
    )
    magnitude_stability = float(1.0 / (1.0 + relative_cv))
    stability = 0.5 * direction_consistency + 0.5 * magnitude_stability

    mean_bias = bias_stack.mean(axis=0)
    mean_deficit = deficit_stack.mean(axis=0)
    # Compact signature used only for diversity selection, not as a cryptographic claim.
    signature = np.concatenate([mean_bias, mean_deficit])
    norm = np.linalg.norm(signature)
    if norm > 0:
        signature = signature / norm

    significant_union = sorted(
        set(bit for run in runs for bit in run["significant_bits"])
    )
    input_value = (diff[0] << 16) | diff[1]
    input_active_bits = [31 - bit for bit in range(32) if (input_value >> bit) & 1]

    return {
        "diff": [int(diff[0]), int(diff[1])],
        "input_hw": hamming_weight(diff),
        "input_active_bits": input_active_bits,
        "marginal_rms": float(np.mean(rms_values)),
        "topk_bias": float(np.mean([run["topk_bias"] for run in runs])),
        "max_bias": float(np.mean([run["max_bias"] for run in runs])),
        "projected_js_uniform": float(np.mean(js_values)),
        "stability": float(stability),
        "significant_output_bits": significant_union,
        "signed_biases": mean_bias.astype(float).tolist(),
        "signature": signature.astype(float).tolist(),
    }


def evaluate_chunk_worker(
    chunk: Sequence[WordDiff],
    top_k_bits: int,
    familywise_alpha: float,
) -> List[Dict[str, object]]:
    if (
        _WORKER_SEARCHER is None
        or _WORKER_ENCRYPT_BATCH is None
        or _WORKER_PROJECTIONS is None
        or not _WORKER_PANELS
    ):
        raise RuntimeError("worker 尚未初始化")

    normalized_chunk = tuple(sorted(normalize_difference(diff) for diff in chunk))
    z_critical = float(norm.isf(familywise_alpha / (2.0 * 32)))
    per_diff_runs: List[List[Dict[str, object]]] = [[] for _ in normalized_chunk]

    for panel in _WORKER_PANELS:
        xor_outputs = _WORKER_SEARCHER.make_xor_outputs(
            normalized_chunk,
            panel,
            _WORKER_ENCRYPT_BATCH,
        )
        if len(xor_outputs) != len(normalized_chunk):
            raise RuntimeError("加密输出数量与候选数量不一致")
        for index, xor_words in enumerate(xor_outputs):
            per_diff_runs[index].append(
                _evaluate_one_panel(
                    xor_words,
                    _WORKER_PROJECTIONS,
                    top_k_bits,
                    z_critical,
                )
            )

    return [
        _aggregate_runs(diff, runs)
        for diff, runs in zip(normalized_chunk, per_diff_runs)
    ]


def _run_stage(
    candidates: Sequence[WordDiff],
    config_path: str,
    nr: int,
    sample_size: int,
    panel_count: int,
    chunk_size: int,
    stage_seed: int,
    config: ScreeningConfig,
) -> List[Dict[str, object]]:
    chunks = [
        candidates[index : index + chunk_size]
        for index in range(0, len(candidates), chunk_size)
    ]
    results: List[Dict[str, object]] = []
    workers = max(1, min(config.num_workers, len(chunks)))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            config_path,
            nr,
            sample_size,
            panel_count,
            stage_seed,
            config.num_projection_hashes,
            config.projection_width,
        ),
    ) as executor:
        futures = [
            executor.submit(
                evaluate_chunk_worker,
                chunk,
                config.top_k_bits,
                config.familywise_alpha,
            )
            for chunk in chunks
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            # Fail fast: a partial ranking must never be silently saved as a complete result.
            chunk_result = future.result()
            results.extend(chunk_result)
            print(
                f"  完成 chunk {completed}/{len(chunks)}，累计 {len(results)}/{len(candidates)} 个候选",
                flush=True,
            )

    if len(results) != len(candidates):
        raise RuntimeError(f"阶段结果缺失: {len(results)} != {len(candidates)}")
    return results


def _percentile_quality(records: List[Dict[str, object]]) -> None:
    metrics = {
        "marginal_rms": 0.20,
        "topk_bias": 0.15,
        "projected_js_uniform": 0.40,
        "stability": 0.25,
    }
    count = len(records)
    quality = np.zeros(count, dtype=np.float64)
    for metric, weight in metrics.items():
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        percentile = (rankdata(values, method="average") - 1.0) / max(1, count - 1)
        quality += weight * percentile
    for index, record in enumerate(records):
        record["quality"] = float(quality[index])


def _signature_similarity(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    signature_a = np.asarray(a["signature"], dtype=np.float64)
    signature_b = np.asarray(b["signature"], dtype=np.float64)
    cosine = float(np.dot(signature_a, signature_b))

    diff_a = normalize_difference(a["diff"])
    diff_b = normalize_difference(b["diff"])
    value_a = (diff_a[0] << 16) | diff_a[1]
    value_b = (diff_b[0] << 16) | diff_b[1]
    union = (value_a | value_b).bit_count()
    input_similarity = 1.0 if union == 0 else 1.0 - (value_a ^ value_b).bit_count() / union
    return float(0.75 * max(0.0, cosine) + 0.25 * input_similarity)


def select_diverse_seeds(
    records: Sequence[Dict[str, object]],
    final_n: int,
    diversity_lambda: float,
) -> List[Dict[str, object]]:
    remaining = sorted(records, key=lambda record: float(record["quality"]), reverse=True)
    selected: List[Dict[str, object]] = []

    while remaining and len(selected) < final_n:
        if not selected:
            choice = remaining[0]
            mmr_score = float(choice["quality"])
        else:
            choice = None
            mmr_score = -np.inf
            for candidate in remaining:
                redundancy = max(_signature_similarity(candidate, chosen) for chosen in selected)
                score = float(candidate["quality"]) - diversity_lambda * redundancy
                if score > mmr_score:
                    choice = candidate
                    mmr_score = score
            assert choice is not None
        copied = dict(choice)
        copied["mmr_score"] = float(mmr_score)
        selected.append(copied)
        remaining.remove(choice)
    return selected


def screen_starting_seeds(
    config_path: str,
    nr: int,
    config: ScreeningConfig,
    output_path: str | None = None,
) -> Dict[str, object]:
    config.validate()
    candidates = generate_low_hw_candidates(config.max_hw)
    print(
        f"阶段一：{len(candidates)} 个 HW<= {config.max_hw} 候选，"
        f"{config.coarse_panel_count}×{config.coarse_sample_size} 固定样本面板。"
    )
    coarse = _run_stage(
        candidates,
        config_path,
        nr,
        config.coarse_sample_size,
        config.coarse_panel_count,
        config.coarse_chunk_size,
        config.base_seed,
        config,
    )
    _percentile_quality(coarse)
    coarse.sort(key=lambda record: float(record["quality"]), reverse=True)
    coarse_keep = coarse[: min(config.coarse_keep, len(coarse))]

    refine_candidates = [normalize_difference(record["diff"]) for record in coarse_keep]
    print(
        f"阶段二：复筛前 {len(refine_candidates)} 个候选，"
        f"{config.refine_panel_count}×{config.refine_sample_size} 独立固定面板。"
    )
    refined = _run_stage(
        refine_candidates,
        config_path,
        nr,
        config.refine_sample_size,
        config.refine_panel_count,
        config.refine_chunk_size,
        config.base_seed + 50_000_021,
        config,
    )
    _percentile_quality(refined)
    refined.sort(key=lambda record: float(record["quality"]), reverse=True)
    seeds = select_diverse_seeds(refined, config.final_n, config.diversity_lambda)

    payload = {
        "schema": "speck32_seed_pool_v2",
        "nr": int(nr),
        "purpose": "diverse building blocks for joint pair-of-polytopes optimization",
        "config": asdict(config),
        "seeds": seeds,
    }
    if output_path is None:
        output_path = f"../data/top_diff_seeds_{nr}round.json"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"已保存 {len(seeds)} 个多样化种子至: {destination.resolve()}")
    return payload


if __name__ == "__main__":
    CONFIG_PATH = "../Dynamic Linking/speck32_64.json"
    NR = 5
    cfg = ScreeningConfig(
        max_hw=4,
        coarse_sample_size=4096,
        coarse_panel_count=2,
        coarse_keep=800,
        refine_sample_size=32_768,
        refine_panel_count=3,
        final_n=64,
        num_workers=min(8, os.cpu_count() or 1),
    )
    result = screen_starting_seeds(CONFIG_PATH, NR, cfg)
    print("\n最终种子：")
    for rank, seed in enumerate(result["seeds"], start=1):
        left, right = seed["diff"]
        print(
            f"{rank:02d}. ({left:#06x}, {right:#06x}) "
            f"HW={seed['input_hw']} quality={seed['quality']:.6f} "
            f"MMR={seed['mmr_score']:.6f}"
        )
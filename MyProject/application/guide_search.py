from __future__ import annotations
import os
#将底层数学库自身的线程数限制为1
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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy.stats import norm, rankdata

from DRL_search_plus_improve import (
    CryptoPanel,#固定实验数据，候选差分A和候选差分B使用相同的随机明文和密钥
    ProjectionBank,#保存随机选择的比特组合
    RL_SearchBestDiff,#借用该类的SPECK加密、样本数据生成以及输出差分计算
    hamming_weight,#计算一个输入差分的汉明重量
    normalize_difference,##把输入差分转换成程序规定的表示形式
)
#每个子进程启动后，只加载一次Speck动态链接库、加密函数、固定明文密钥、随机投影、零假设噪声指标
WordDiff = Tuple[int, int]
_WORKER_SEARCHER: RL_SearchBestDiff | None = None
_WORKER_DLL = None
_WORKER_ENCRYPT_BATCH = None
_WORKER_PANELS: List[CryptoPanel] = []
_WORKER_PROJECTIONS: ProjectionBank | None = None
_WORKER_NULL_METRICS: List[Dict[str, float]] = []

@dataclass(frozen=True)
class ScreeningConfig:
    #配置文件保存在JSON中，来确保可重复性
    profile_name: str = "custom"
    max_hw: int = 4
    # Stage 1: 粗筛参数
    coarse_sample_size: int = 8192#每个明文-密钥集合8192个样本
    coarse_panel_count: int = 2#每个候选用两个独立的明文-密钥集合评价
    coarse_keep: int = 800#保存800粗筛结果个进入复筛
    coarse_chunk_size: int = 64#每64个候选差分做为一个任务块
    # Stage 2: 复筛参数
    refine_sample_size: int = 65536#每个明文-密钥集合65536个样本
    refine_panel_count: int = 4#每个候选用四个独立的明文-密钥集合评价
    refine_chunk_size: int = 16#每16个差分做为一个任务块
    num_projection_hashes: int = 48#用于计算多比特联合分布，随机生成48组比特组合
    projection_width: int = 6#粗筛中每组随机组合选择6个输出差分bit
    refine_projection_width: int = 8#复筛中每组随机组合选择8个输出差分bit
    top_k_bits: int = 6#32个bit中选取绝对偏差最大的6个bit
    familywise_alpha: float = 0.01#整体显著性水平设置
    final_n: int = 64#最终输出的种子数量
    diversity_lambda: float = 0.25#MMR多样性惩罚系数
    '''
    选出64个候选种子后，需要将这些种子交给PPO联合搜索
    候选选择逻辑为：候选得=质量分-0.25×与已选择的候选的最大相似度
    得分高但是和已经选择的种子太相似，会受到惩罚
    '''
    null_repeats: int = 4#有限样本噪声校正参数（进行4次取平均）
    '''在有限样本情况下，即使输出的bit是均匀随机的，有限样本会产生非零数值'''
    null_floor_weight: float = 0.10#噪声校正中保留原始观测值的百分之10
    '''校正后的指标=max(0,观测值-随机基线)+0.1×观测值'''
    within_hw_rank_weight: float = 0.60#z综合所有HW候选排名和同一HW层内部排名
    '''同HW层排名占百分之60，全局排名占百分之40'''
    hw_prior_weight: float = 0.20#综合低HW偏好与统计质量得到最终质量分数
    '''低HW偏好占百分之20，统计质量占百分之80'''
    coarse_hw_quota: Tuple[int, ...] = (0, 32, 288, 280, 200)#粗筛保留800个候选，各HW所占数量
    final_hw_quota: Tuple[int, ...] = (0, 24, 20, 12, 8)#复筛后保留64个候选，各HW所占数量
    base_seed: int = 20260712#固定随机种子
    num_workers: int = min(8, os.cpu_count() or 1)#进程数
    #检查配置是否合法
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
        if not 0.0 <= self.within_hw_rank_weight <= 1.0:
            raise ValueError("within_hw_rank_weight 必须位于 [0,1]")
        if not 0.0 <= self.hw_prior_weight <= 1.0:
            raise ValueError("hw_prior_weight 必须位于 [0,1]")
        if self.null_repeats < 1:
            raise ValueError("null_repeats 必须为正")
        if not 0.0 <= self.null_floor_weight <= 1.0:
            raise ValueError("null_floor_weight 必须位于 [0,1]")
        if not 1 <= self.projection_width <= 16:
            raise ValueError("projection_width 必须位于 [1,16]")
        if not 1 <= self.refine_projection_width <= 16:
            raise ValueError("refine_projection_width 必须位于 [1,16]")
MAX_SPECK32_ROUNDS = 22
'''自动计算每个HW层应该保留多少候选'''
def _allocate_hw_quota(
    total_n: int,
    max_hw: int,
    weights: Sequence[float],
) -> Tuple[int, ...]:
    if total_n < 1:
        raise ValueError("total_n 必须为正")
    if not 1 <= max_hw <= 32:
        raise ValueError("max_hw 必须位于 [1,32]")
    capacities = [0] + [math.comb(32, hw) for hw in range(1, max_hw + 1)]
    '''capacities=[0,32,496,4960,35960]'''
    target = min(int(total_n), sum(capacities))
    usable_weights = [
        max(0.0, float(weights[hw - 1])) if hw - 1 < len(weights) else 1.0
        for hw in range(1, max_hw + 1)
    ]
    if sum(usable_weights) <= 0.0:
        usable_weights = [1.0] * max_hw
    weight_sum = float(sum(usable_weights))
    raw = [target * weight / weight_sum for weight in usable_weights]#按比例计算每层名额
    quota = [0] + [
        min(capacities[hw], int(math.floor(raw[hw - 1])))
        for hw in range(1, max_hw + 1)
    ]
    remaining = target - sum(quota)
    #确定剩余名额应该分给哪些HW层
    order = sorted(
        range(1, max_hw + 1),
        key=lambda hw: (
            raw[hw - 1] - math.floor(raw[hw - 1]),
            usable_weights[hw - 1],
            -hw,
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for hw in order:
            if quota[hw] < capacities[hw]:
                quota[hw] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return tuple(quota)

'''指定HW各分层权重'''
def _round_adaptive_weights(nr: int, stage: str) -> Tuple[float, ...]:
    if stage not in {"coarse", "final"}:
        raise ValueError("stage 必须为 coarse 或 final")
    if stage == "coarse":
        return (0.04, 0.36, 0.35, 0.25)
    return (0.375, 0.3125, 0.1875, 0.125)

'''根据轮数在自动配置筛选参数'''
def make_screening_config(
    nr: int,
    profile: str = "auto",
    *,
    max_hw: int = 4,
    coarse_keep: int = 800,
    final_n: int = 64,
    hw_prior_weight: float = 0.20,
    base_seed: int = 20260712,
    num_workers: Optional[int] = None,
) -> ScreeningConfig:
    """
    Profiles:
      auto          : 根据轮数自动增加统计样本量。
      fast          : 统一使用较小样本，用于快速检查程序。
      balanced      : 使用统一的中等规模参数。
      high_fidelity : 使用更大样本、更多明文-密钥集合（面板）、更多随机投影和更多零假设重复。
    """
    nr = int(nr)
    if not 1 <= nr <= MAX_SPECK32_ROUNDS:
        raise ValueError(
            f"SPECK32/64 轮数必须位于 [1,{MAX_SPECK32_ROUNDS}]，当前为 {nr}"
        )
    if profile not in {"auto", "fast", "balanced", "high_fidelity"}:
        raise ValueError(
            "profile 必须为 auto、fast、balanced 或 high_fidelity"
        )
    workers = (
        min(8, os.cpu_count() or 1)
        if num_workers is None
        else max(1, int(num_workers))
    )
    #轮数自适应:
    #1～4轮：信号较强，样本量较小；
    #5～6轮：增加粗筛和复筛样本；
    #7～8轮：增加面板数和零假设重复；
    #9轮以上：使用更大面板减少高轮弱信号噪声。
    if nr <= 4:
        params = dict(
            coarse_sample_size=4_096,
            coarse_panel_count=2,
            refine_sample_size=32_768,
            refine_panel_count=3,
            num_projection_hashes=32,
            projection_width=8,
            refine_projection_width=8,
            null_repeats=3,
            within_hw_rank_weight=0.40,
        )
    elif nr <= 6:
        params = dict(
            coarse_sample_size=8_192,
            coarse_panel_count=2,
            refine_sample_size=65_536,
            refine_panel_count=3,
            num_projection_hashes=48,
            projection_width=6,
            refine_projection_width=8,
            null_repeats=4,
            within_hw_rank_weight=0.55,
        )
    elif nr <= 8:
        params = dict(
            coarse_sample_size=8_192 if nr == 7 else 16_384,
            coarse_panel_count=3,
            refine_sample_size=65_536 if nr == 7 else 131_072,
            refine_panel_count=4,
            num_projection_hashes=64,
            projection_width=6,
            refine_projection_width=8,
            null_repeats=6,
            within_hw_rank_weight=0.60,
        )
    else:
       #在极高轮次时，剩余信号可能极其微弱;更大的独立面板可以减少，但无法消除由有限样本噪声引起的错误排名。
        params = dict(
            coarse_sample_size=16_384,
            coarse_panel_count=4,
            refine_sample_size=131_072,
            refine_panel_count=5,
            num_projection_hashes=96,
            projection_width=6,
            refine_projection_width=8,
            null_repeats=8,
            within_hw_rank_weight=0.60,
        )

    if profile == "fast":
        params.update(
            coarse_sample_size=4_096,
            coarse_panel_count=2,
            refine_sample_size=32_768,
            refine_panel_count=3,
            num_projection_hashes=32,
            projection_width=6,
            refine_projection_width=8,
            null_repeats=2,
        )
    elif profile == "balanced":
        params.update(
            coarse_sample_size=8_192,
            coarse_panel_count=2,
            refine_sample_size=65_536,
            refine_panel_count=4,
            num_projection_hashes=48,
            projection_width=6,
            refine_projection_width=8,
            null_repeats=4,
        )
    elif profile == "high_fidelity":
        params.update(
            coarse_sample_size=max(16_384, int(params["coarse_sample_size"]) * 2),
            coarse_panel_count=max(4, int(params["coarse_panel_count"]) + 1),
            refine_sample_size=max(131_072, int(params["refine_sample_size"]) * 2),
            refine_panel_count=max(5, int(params["refine_panel_count"]) + 1),
            num_projection_hashes=max(96, int(params["num_projection_hashes"]) * 2),
            null_repeats=max(8, int(params["null_repeats"]) * 2),
        )
    #计算粗筛阶段各HW层名额
    coarse_quota = _allocate_hw_quota(
        coarse_keep,
        max_hw,
        _round_adaptive_weights(nr, "coarse"),
    )
    #计算复筛阶段各HW层名额
    final_quota = _allocate_hw_quota(
        final_n,
        max_hw,
        _round_adaptive_weights(nr, "final"),
    )
    #构建配置对象
    config = ScreeningConfig(
        profile_name=profile,
        max_hw=max_hw,
        coarse_keep=coarse_keep,
        final_n=final_n,
        coarse_hw_quota=coarse_quota,
        final_hw_quota=final_quota,
        hw_prior_weight=float(hw_prior_weight),
        # 不同轮次不会使用完全相同的数据面板。
        base_seed=int(base_seed) + nr * 1_000_003,
        num_workers=workers,
        **params,
    )
    config.validate()
    return config

'''枚举所有满足低于最高汉明权重的非零输入差分'''
def generate_low_hw_candidates(max_hw: int = 4) -> List[WordDiff]:
    candidates: List[WordDiff] = []
    for hw in range(1, max_hw + 1):
        for positions in combinations(range(32), hw):
            value = 0
            for position in positions:
                value |= 1 << (31 - position)
            candidates.append(((value >> 16) & 0xFFFF, value & 0xFFFF))
    return candidates

'''初始化每一个子进程'''
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
    global _WORKER_PANELS, _WORKER_PROJECTIONS, _WORKER_NULL_METRICS
    _WORKER_SEARCHER = RL_SearchBestDiff(config_path=config_path, nr=nr)
    _WORKER_DLL, _WORKER_ENCRYPT_BATCH = _WORKER_SEARCHER.load_encrypt_function()
    _WORKER_PANELS = [
        _WORKER_SEARCHER.make_panel(sample_size, panel_seed + index * 1000003)
        for index in range(panel_count)
    ]
    _WORKER_NULL_METRICS = []
    _WORKER_PROJECTIONS = ProjectionBank.build(
        feature_dim=32,
        seed=panel_seed + 97,
        num_hashes=num_projection_hashes,
        hash_width=projection_width,
        num_pairs=1,
    )

'''将两个16-bit输出差分字（N,2）转换为（N,32）的比特矩阵'''
def _xor_words_to_bits(xor_words: np.ndarray) -> np.ndarray:
    xor_words = np.asarray(xor_words, dtype=np.uint16)
    if xor_words.ndim != 2 or xor_words.shape[1] != 2:
        raise ValueError(f"期望 XOR 输出形状为 (N,2)，实际为 {xor_words.shape}")
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    return ((xor_words[..., None] >> shifts) & 1).astype(np.uint8).reshape(xor_words.shape[0], 32)
'''
计算两个离散概率分布之间的詹森-香农散度(JSD)
即计算候选差分经过若干轮SPECK加密后得到的多比特联合输出分布与理想均匀分布之间的差异
'''
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:#计算KL散度
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)#计算JSD

'''检测多bit间的联合偏差'''
def _projection_js_uniform(bits: np.ndarray, projection_bank: ProjectionBank) -> Tuple[float, np.ndarray]:
    width = projection_bank.hash_indices.shape[1]#读取每个随机组合包含多少bit
    weights = 1 << np.arange(width, dtype=np.uint32)
    bins = 1 << width#相当于2^width
    uniform = np.full(bins, 1.0 / bins, dtype=np.float64)
    js_values = []#记录每组随机比特组合的JSD
    entropy_deficits = []#记录每组组合的熵缺失
    for indices in projection_bank.hash_indices:#逐组处理随机选择的比特位置
        codes = np.sum(bits[:, indices].astype(np.uint32) * weights, axis=1)#把每个样本的多个bit状态转换成一个整数编号
        counts = np.bincount(codes.astype(np.int64), minlength=bins).astype(np.float64)#计算每一种联合状态出现多少次
        counts += 0.5  # Jeffreys平滑，防止某些状态从未出现时出现数值不稳定
        probability = counts / counts.sum()#把频数转换为概率分布
        js_values.append(_js_divergence(probability, uniform))#计算当前联合分布与均匀分布间的JSD
        entropy = -float(np.sum(probability * np.log2(probability)))#计算香农熵
        entropy_deficits.append(1.0 - entropy / width)#width个二进制变量最大熵为width bit，计算熵缺失，值越接近0越随机，值越大存在明显的非均匀结构
    return float(np.mean(js_values)), np.asarray(entropy_deficits, dtype=np.float64)

'''评价一个候选差分'''
def _evaluate_bits(
    bits: np.ndarray,
    projection_bank: ProjectionBank,
    top_k_bits: int,
    z_critical: float,
) -> Dict[str, object]:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 2 or bits.shape[1] != 32:
        raise ValueError(f"期望 bit 矩阵形状为 (N,32)，实际为 {bits.shape}")
    signed_bias = bits.mean(axis=0) - 0.5#有符号边缘偏差
    absolute_bias = np.abs(signed_bias)#绝对偏差
    k = min(max(1, top_k_bits), 32)
    top_k = np.partition(absolute_bias, 32 - k)[-k:]#取绝对偏差最大的k个输出bit
    projected_js, entropy_deficits = _projection_js_uniform(bits, projection_bank)#检测多bit联合分布对均匀分布的偏离
    z_scores = 2.0 * math.sqrt(bits.shape[0]) * absolute_bias#计算每个bit分布偏离多少倍标准差
    return {
        "signed_bias": signed_bias,#有符号偏差
        "marginal_rms": float(np.sqrt(np.mean(absolute_bias**2))),#全部输出bit的整体一阶偏差强度
        "topk_bias": float(np.mean(top_k)),#取绝对偏差最大的k个输出bit并求偏差均值
        "max_bias": float(np.max(absolute_bias)),#最强单bit偏差
        "projected_js_uniform": projected_js,#多bit联合分布对均匀分布的偏离
        "entropy_deficits": entropy_deficits,#熵缺失
        "significant_bits": np.flatnonzero(z_scores >= z_critical).astype(int).tolist(),#具有显著偏差的bit
    }

'''包装函数'''
def _evaluate_one_panel(
    xor_words: np.ndarray,
    projection_bank: ProjectionBank,
    top_k_bits: int,
    z_critical: float,
) -> Dict[str, object]:
    return _evaluate_bits(
        _xor_words_to_bits(xor_words),
        projection_bank,
        top_k_bits,
        z_critical,
    )

'''估计纯随机数据在有限样本下自然会产生多大的“假偏差”，防止把有限样本噪声误判为密码学偏差。'''
def _make_null_metrics(
    sample_size: int,
    projection_bank: ProjectionBank,
    top_k_bits: int,
    z_critical: float,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))#创建固定随机数生成器
    runs = [
        _evaluate_bits(
            rng.integers(0, 2, size=(sample_size, 32), dtype=np.uint8),
            projection_bank,
            top_k_bits,
            z_critical,
        )
        for _ in range(repeats)
    ]
    return {
        "marginal_rms": float(np.mean([run["marginal_rms"] for run in runs])),
        "topk_bias": float(np.mean([run["topk_bias"] for run in runs])),
        "projected_js_uniform": float(
            np.mean([run["projected_js_uniform"] for run in runs])
        ),
    }

'''进行随机噪声校正'''
def _null_correct_metric(
    observed: float,
    null_value: float,
    floor_weight: float,
) -> float:
    return float(max(0.0, observed - null_value) + floor_weight * observed)

'''判断某个候选差分是否在不同随机明文-密钥数据上表现稳定良好'''
def _aggregate_runs(diff: WordDiff, runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    bias_stack = np.stack([run["signed_bias"] for run in runs], axis=0)
    deficit_stack = np.stack([run["entropy_deficits"] for run in runs], axis=0)
    strengths = np.linalg.norm(bias_stack, axis=1)#计算每个面板偏差向量的L2范数
    cosine_values = []#保存不同面板偏差方向之间的余弦相似度
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            denominator = strengths[i] * strengths[j]
            cosine_values.append(
                0.0
                if denominator < 1e-15
                else float(np.dot(bias_stack[i], bias_stack[j]) / denominator)
            )
    direction_consistency = (
        float(np.mean([(value + 1.0) / 2.0 for value in cosine_values]))
        if cosine_values
        else 1.0
    )#direction_consistency越接近1，不同面板上的偏差方向越一致
    rms_raw = np.asarray([run["marginal_rms"] for run in runs], dtype=np.float64)
    topk_raw = np.asarray([run["topk_bias"] for run in runs], dtype=np.float64)
    js_raw = np.asarray(
        [run["projected_js_uniform"] for run in runs], dtype=np.float64
    )
    rms_corrected = np.asarray(
        [run["marginal_rms_corrected"] for run in runs], dtype=np.float64
    )
    topk_corrected = np.asarray(
        [run["topk_bias_corrected"] for run in runs], dtype=np.float64
    )
    js_corrected = np.asarray(
        [run["projected_js_corrected"] for run in runs], dtype=np.float64
    )

    relative_cv = 0.5 * (
        np.std(rms_corrected) / (np.mean(rms_corrected) + 1e-12)
        + np.std(js_corrected) / (np.mean(js_corrected) + 1e-12)
    )#计算两个核心指标的相对变异程度
    magnitude_stability = float(1.0 / (1.0 + relative_cv))#不同面板上的值越接近，magnitude_stability越接近1
    stability = 0.5 * direction_consistency + 0.5 * magnitude_stability#确定最终的稳定性
    mean_bias = bias_stack.mean(axis=0)#计算多个面板上平均后的有符号偏差
    mean_deficit = deficit_stack.mean(axis=0)#计算平均熵缺失
    signature = np.concatenate([mean_bias, mean_deficit])#将两类信息拼接为一个特征向量
    '''后续用于衡量不同候选差分是否“统计行为类似”'''
    signature_norm = np.linalg.norm(signature)
    if signature_norm > 0:
        signature = signature / signature_norm
    significant_union = sorted(
        set(bit for run in runs for bit in run["significant_bits"])#把所有面板中层显著偏离随即对bit位置合并起来去重
    )
    input_value = (diff[0] << 16) | diff[1]
    input_active_bits = [
        31 - bit for bit in range(32) if (input_value >> bit) & 1#找出这个差分具体激活了哪些bit
    ]

    return {
        "diff": [int(diff[0]), int(diff[1])],
        "input_hw": hamming_weight(diff),
        "input_active_bits": input_active_bits,
        #原始统计指标
        "marginal_rms_raw": float(np.mean(rms_raw)),
        "topk_bias_raw": float(np.mean(topk_raw)),
        "projected_js_uniform_raw": float(np.mean(js_raw)),
        #噪声校正后的指标
        "marginal_rms": float(np.mean(rms_corrected)),
        "topk_bias": float(np.mean(topk_corrected)),
        "projected_js_uniform": float(np.mean(js_corrected)),
        "max_bias": float(np.mean([run["max_bias"] for run in runs])),
        "stability": float(stability),
        "significant_output_bits": significant_union,
        "signed_biases": mean_bias.astype(float).tolist(),
        "signature": signature.astype(float).tolist(),
    }


'''每个子进程真正评价一批候选'''
def evaluate_chunk_worker(
    chunk: Sequence[WordDiff],
    top_k_bits: int,
    familywise_alpha: float,
    null_repeats: int,
    null_floor_weight: float,
) -> List[Dict[str, object]]:
    global _WORKER_NULL_METRICS
    if (
        _WORKER_SEARCHER is None
        or _WORKER_ENCRYPT_BATCH is None
        or _WORKER_PROJECTIONS is None
        or not _WORKER_PANELS
    ):
        raise RuntimeError("worker 尚未初始化")
    normalized_chunk = tuple(sorted(normalize_difference(diff) for diff in chunk))
    z_critical = float(norm.isf(familywise_alpha / (2.0 * 32)))#计算比特显著性检验阈值
    '''多重比较校正，防止32个bit中偶然出现一个大偏差被误判'''
    per_diff_runs: List[List[Dict[str, object]]] = [[] for _ in normalized_chunk]
    if not _WORKER_NULL_METRICS:
        _WORKER_NULL_METRICS = [
            _make_null_metrics(
                sample_size=panel.sample_size,
                projection_bank=_WORKER_PROJECTIONS,
                top_k_bits=top_k_bits,
                z_critical=z_critical,
                repeats=null_repeats,
                seed=panel.seed + 700_000_019,
            )
            for panel in _WORKER_PANELS
        ]
    for panel_index, panel in enumerate(_WORKER_PANELS):
        xor_outputs = _WORKER_SEARCHER.make_xor_outputs(
            normalized_chunk,
            panel,
            _WORKER_ENCRYPT_BATCH,
        )
        if len(xor_outputs) != len(normalized_chunk):
            raise RuntimeError("加密输出数量与候选数量不一致")
        null_metrics = _WORKER_NULL_METRICS[panel_index]
        for index, xor_words in enumerate(xor_outputs):
            record = _evaluate_one_panel(
                xor_words,
                _WORKER_PROJECTIONS,
                top_k_bits,
                z_critical,
            )
            record["marginal_rms_corrected"] = _null_correct_metric(
                float(record["marginal_rms"]),
                null_metrics["marginal_rms"],
                null_floor_weight,
            )
            record["topk_bias_corrected"] = _null_correct_metric(
                float(record["topk_bias"]),
                null_metrics["topk_bias"],
                null_floor_weight,
            )
            record["projected_js_corrected"] = _null_correct_metric(
                float(record["projected_js_uniform"]),
                null_metrics["projected_js_uniform"],
                null_floor_weight,
            )
            per_diff_runs[index].append(record)
    return [
        _aggregate_runs(diff, runs)
        for diff, runs in zip(normalized_chunk, per_diff_runs)
    ]

'''执行完整的粗筛或复筛'''
def _run_stage(
    candidates: Sequence[WordDiff],
    config_path: str,
    nr: int,
    sample_size: int,
    panel_count: int,
    chunk_size: int,
    stage_seed: int,
    projection_width: int,
    config: ScreeningConfig,
) -> List[Dict[str, object]]:
    chunks = [
        candidates[index : index + chunk_size]
        for index in range(0, len(candidates), chunk_size)
    ]
    results: List[Dict[str, object]] = []
    workers = max(1, min(config.num_workers, len(chunks)))
    #创建多进程池
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
            projection_width,
        ),
    ) as executor:
        #为每一个chunk提交一个异步任务
        futures = [
            executor.submit(
                evaluate_chunk_worker,
                chunk,
                config.top_k_bits,
                config.familywise_alpha,
                config.null_repeats,
                config.null_floor_weight,
            )
            for chunk in chunks
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            chunk_result = future.result()
            results.extend(chunk_result)
            print(
                f"  完成 chunk {completed}/{len(chunks)}，累计 {len(results)}/{len(candidates)} 个候选",
                flush=True,
            )

    if len(results) != len(candidates):
        raise RuntimeError(f"阶段结果缺失: {len(results)} != {len(candidates)}")
    return results

'''把各种指标转换为0~1排名，0为排名最低，1为排名最高'''
def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.ones(values.size, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1)#通过将排名映射到0~1，不同指标间可以加权组合

'''计算每个候选差分的综合质量分数'''
def _percentile_quality(
    records: List[Dict[str, object]],
    within_hw_weight: float,
    hw_prior_weight: float,
    max_hw: int,
) -> None:
    #四类指标权重
    metrics = {
        "marginal_rms": 0.20,
        "topk_bias": 0.15,
        "projected_js_uniform": 0.40,
        "stability": 0.25,
    }
    count = len(records)
    global_quality = np.zeros(count, dtype=np.float64)
    within_quality = np.zeros(count, dtype=np.float64)
    for metric, weight in metrics.items():
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        global_quality += weight * _rank01(values)
    hw_values = sorted({int(record["input_hw"]) for record in records})
    for hw in hw_values:
        indices = [
            index for index, record in enumerate(records)
            if int(record["input_hw"]) == hw
        ]
        for metric, weight in metrics.items():
            values = np.asarray(
                [records[index][metric] for index in indices],
                dtype=np.float64,
            )
            ranks = _rank01(values)
            for local_index, record_index in enumerate(indices):
                within_quality[record_index] += weight * ranks[local_index]
    for index, record in enumerate(records):
        hw = int(record["input_hw"])
        if max_hw <= 1:
            low_hw_score = 1.0
        else:
            low_hw_score = float(np.clip((max_hw - hw) / (max_hw - 1), 0.0, 1.0))
        #统计质量=百分之四十全局排名+百分之六十同HW层内排名
        statistical_quality = float(
            (1.0 - within_hw_weight) * global_quality[index]
            + within_hw_weight * within_quality[index]
        )
        #最终质量=百分之八十统计质量+百分之二十低HW偏好
        final_quality = float(
            (1.0 - hw_prior_weight) * statistical_quality
            + hw_prior_weight * low_hw_score
        )
        record["quality_global"] = float(global_quality[index])
        record["quality_within_hw"] = float(within_quality[index])
        record["statistical_quality"] = statistical_quality
        record["low_hw_score"] = low_hw_score
        record["quality"] = final_quality


'''读取某个HW层的配额'''
def _quota_for_hw(quota: Sequence[int], hw: int) -> int:
    return int(quota[hw]) if hw < len(quota) else 0

'''根据HW分层配额保留粗筛后的候选差分'''
def stratified_top_k(
    records: Sequence[Dict[str, object]],
    total_n: int,
    quota_by_hw: Sequence[int],
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []#保存候选
    selected_ids = set()#保存已选差分ID
    for hw in sorted({int(record["input_hw"]) for record in records}):
        quota = _quota_for_hw(quota_by_hw, hw)
        if quota <= 0:
            continue
        group = sorted(
            (record for record in records if int(record["input_hw"]) == hw),
            key=lambda record: float(record["quality"]),
            reverse=True,
        )
        for record in group[:quota]:
            key = tuple(record["diff"])
            if key not in selected_ids:
                selected.append(record)
                selected_ids.add(key)

    if len(selected) < total_n:#若某个HW层候选数量不足，则启动补位，所有未入选候选重新按质量排序
        remaining = sorted(
            records,
            key=lambda record: float(record["quality"]),
            reverse=True,
        )
        for record in remaining:
            key = tuple(record["diff"])
            if key in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(key)
            if len(selected) >= total_n:
                break
    return selected[:total_n]

'''计算两个候选种子之间的综合相似度,这个函数用于MMR'''
def _signature_similarity(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    signature_a = np.asarray(a["signature"], dtype=np.float64)
    signature_b = np.asarray(b["signature"], dtype=np.float64)
    cosine = float(np.dot(signature_a, signature_b))#两个向量已经归一化，所以点积就是余弦相似度
    '''两个差分经过 SPECK后表现出来的统计模式有多相似。'''
    diff_a = normalize_difference(a["diff"])
    diff_b = normalize_difference(b["diff"])
    value_a = (diff_a[0] << 16) | diff_a[1]
    value_b = (diff_b[0] << 16) | diff_b[1]
    union = (value_a | value_b).bit_count()#两个输入差分所有活跃bit位的并集大小
    input_similarity = 1.0 if union == 0 else 1.0 - (value_a ^ value_b).bit_count() / union#利用两个差分的不同活跃比特位除两个差分的活跃比特位并集，计算活跃bit集合Jaccard相似度
    return float(0.75 * max(0.0, cosine) + 0.25 * input_similarity)#最终相似度=百分之七十五输出统计特征相似度+百分之二十五输入活跃bit结构相似度


'''使用MMR选择质量高且不冗余的候选'''
def select_diverse_seeds(
    records: Sequence[Dict[str, object]],
    final_n: int,
    diversity_lambda: float,
) -> List[Dict[str, object]]:
    remaining = sorted(
        records,
        key=lambda record: float(record["quality"]),
        reverse=True,
    )
    selected: List[Dict[str, object]] = []
    while remaining and len(selected) < final_n:
        if not selected:#第一个候选挑选质量分最高的候选
            choice = remaining[0]
            mmr_score = float(choice["quality"])
        else:#之后的候选，找到当前候选与所有已选种子的最大相似度
            choice = None
            mmr_score = -np.inf
            for candidate in remaining:
                redundancy = max(
                    _signature_similarity(candidate, chosen)
                    for chosen in selected
                )
                score = float(candidate["quality"]) - diversity_lambda * redundancy
                '''score=质量得分-0.25×最大相似度'''
                if score > mmr_score:
                    choice = candidate
                    mmr_score = score
            assert choice is not None
        copied = dict(choice)
        copied["mmr_score"] = float(mmr_score)
        selected.append(copied)
        remaining.remove(choice)
    return selected#最终返回质量高且相互间冗余度低的种子

'''最终64个种子的实际选择函数'''
def select_stratified_diverse_seeds(
    records: Sequence[Dict[str, object]],
    final_n: int,
    quota_by_hw: Sequence[int],
    diversity_lambda: float,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    selected_ids = set()
    for hw in sorted({int(record["input_hw"]) for record in records}):
        quota = _quota_for_hw(quota_by_hw, hw)
        if quota <= 0:
            continue
        group = [
            record for record in records
            if int(record["input_hw"]) == hw
        ]
        for record in select_diverse_seeds(#在当前HW层内使用MMR挑选层配额质量高且相互冗余度低的候选
            group,
            final_n=min(quota, len(group)),
            diversity_lambda=diversity_lambda,
        ):
            key = tuple(record["diff"])
            if key not in selected_ids:
                selected.append(record)
                selected_ids.add(key)

    remaining = [
        record for record in records
        if tuple(record["diff"]) not in selected_ids
    ]
    '''当某个层因容量不足没凑够候选，则从所有剩余候选中全局补位，但这次计算MMR是与所有已经选择的HW层候选比较'''
    while remaining and len(selected) < final_n:
        choice = None
        best_score = -np.inf
        for candidate in remaining:
            redundancy = (
                max(
                    _signature_similarity(candidate, chosen)
                    for chosen in selected
                )
                if selected
                else 0.0
            )
            score = float(candidate["quality"]) - diversity_lambda * redundancy
            if score > best_score:
                choice = candidate
                best_score = score
        assert choice is not None
        copied = dict(choice)
        copied["mmr_score"] = float(best_score)
        selected.append(copied)
        selected_ids.add(tuple(choice["diff"]))
        remaining.remove(choice)
    selected.sort(
        key=lambda record: float(record["mmr_score"]),
        reverse=True,
    )
    return selected[:final_n]

'''主函数'''
def screen_starting_seeds(
    config_path: str,
    nr: int,
    config: ScreeningConfig,
    output_path: str | None = None,
) -> Dict[str, object]:
    config.validate()
    candidates = generate_low_hw_candidates(config.max_hw)
    print(
        f"\n========== SPECK32/64 R{nr} 种子筛选 =========="\
        f"\n阶段一：{len(candidates)} 个 HW<= {config.max_hw} 候选，"
        f"{config.coarse_panel_count}×{config.coarse_sample_size} 固定样本面板。"
    )
    #粗筛
    coarse = _run_stage(
        candidates,
        config_path,
        nr,
        config.coarse_sample_size,
        config.coarse_panel_count,
        config.coarse_chunk_size,
        config.base_seed,
        config.projection_width,
        config,
    )
    _percentile_quality(
        coarse,
        config.within_hw_rank_weight,
        config.hw_prior_weight,
        config.max_hw,
    )
    coarse.sort(key=lambda record: float(record["quality"]), reverse=True)
    coarse_keep = stratified_top_k(
        coarse,
        total_n=min(config.coarse_keep, len(coarse)),
        quota_by_hw=config.coarse_hw_quota,
    )
    #复筛
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
        config.refine_projection_width,
        config,
    )
    _percentile_quality(
        refined,
        config.within_hw_rank_weight,
        config.hw_prior_weight,
        config.max_hw,
    )
    refined.sort(key=lambda record: float(record["quality"]), reverse=True)
    seeds = select_stratified_diverse_seeds(
        refined,
        final_n=config.final_n,
        quota_by_hw=config.final_hw_quota,
        diversity_lambda=config.diversity_lambda,
    )
    #构造输出的JSON文件
    payload = {
        "schema": "speck32_seed_pool_low_hw_prior_v5",
        "nr": int(nr),
        "round_profile": config.profile_name,
        "purpose": "low-HW-prior diverse building blocks for joint pair-of-polytopes optimization",
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

'''打印出最终的64个种子'''
def print_seed_summary(result: Mapping[str, object]) -> None:
    seeds = result.get("seeds", [])
    print("\n最终种子：")
    for rank, seed in enumerate(seeds, start=1):
        left, right = seed["diff"]
        print(
            f"{rank:02d}. ({left:#06x}, {right:#06x}) "
            f"HW={int(seed['input_hw'])} "
            f"quality={float(seed['quality']):.6f} "
            f"stat={float(seed['statistical_quality']):.6f} "
            f"lowHW={float(seed['low_hw_score']):.3f} "
            f"global={float(seed['quality_global']):.6f} "
            f"within_hw={float(seed['quality_within_hw']):.6f} "
            f"MMR={float(seed['mmr_score']):.6f}"
        )


if __name__ == "__main__":
    nr = 7
    CONFIG_PATH = "../Dynamic Linking/speck32_64.json"
    # auto 会根据 nr 自动调整统计样本量。
    # 可选："auto"、"fast"、"balanced"、"high_fidelity"。
    PROFILE = "auto"
    OUTPUT_PATH = f"../data/top_diff_seeds_{nr}round.json"
    cfg = make_screening_config(
        nr=nr,
        profile=PROFILE,
        max_hw=4,
        coarse_keep=800,
        final_n=64,
        hw_prior_weight=0.20,
        base_seed=20260712,
        num_workers=min(8, os.cpu_count() or 1),
    )
    print(
        f"开始搜索 SPECK32/64 的 {nr} 轮种子差分："
        f"profile={cfg.profile_name}, "
        f"coarse={cfg.coarse_panel_count}×{cfg.coarse_sample_size}, "
        f"refine={cfg.refine_panel_count}×{cfg.refine_sample_size}, "
        f"hw_prior_weight={cfg.hw_prior_weight}, "
        f"coarse_hw_quota={cfg.coarse_hw_quota}, "
        f"final_hw_quota={cfg.final_hw_quota}",
        flush=True,
    )
    result = screen_starting_seeds(
        config_path=CONFIG_PATH,
        nr=nr,
        config=cfg,
        output_path=OUTPUT_PATH,
    )
    print_seed_summary(result)
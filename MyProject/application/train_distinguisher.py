from __future__ import annotations
import copy
import gc
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from DRL_search_plus_improve import (
    PolytopeDiff,
    RL_SearchBestDiff,
    canonical_pair,
    feature_channel_count,
    pair_to_jsonable,
    vertices_to_bit_features,
)
'''单个训练阶段配置'''
@dataclass(frozen=True)
class StageConfig:
    nr: int
    train_samples: int
    val_samples: int
    epochs: int
'''完整实验配置'''
@dataclass
class ExperimentConfig:
    cipher_config_path: str = "../Dynamic Linking/speck32_64.json"
    search_result_path: Optional[str] = None#自动差分搜索结果路径
    # search_result_path 不存在时使用。
    fallback_polytope_0: PolytopeDiff = (
        (0x0040, 0x0000),
        (0x0000, 0x8000),
        (0x0060, 0x0000),
    )
    fallback_polytope_1: PolytopeDiff = (
        (0x0020, 0x0000),
        (0x0040, 0x8000),
        (0x0010, 0x2000),
    )
    stages: Tuple[StageConfig, ...] = field(default_factory=tuple)#训练阶段列表
    test_samples: int = 200_000#独立测试集的总样本数
    repetitions: int = 3#独立重复训练三次
    include_anchor_xor: bool = True#控制是否启用非锚点密文相对于锚点密文的XOR特征
    include_previous_right: bool = True#控制是否加入无密钥恢复上一轮右字特征
    batch_size: int = 4096#每个batch包含4096个样本
    loader_workers: int = 0
    # 增强网络：多尺度循环卷积 + 通道注意力 + 比特位置自注意力。
    width: int = 96#主干网络的隐藏通道数量
    dilations: Tuple[int, ...] = (1, 2, 4, 7, 1, 2, 4, 7)#每个数对应一个多尺度循环残差块的扩张率
    #扩张率不是“卷积核变大”，而是控制卷积观察相邻位置的间隔
    dense_units: int = 384#分类头第一层全连接神经元数量
    dropout: float = 0.10#训练时随机关闭百分之十的神经元
    attention_heads: int = 4#比特位置自注意力使用 4 个注意力头
    attention_layers: int = 1#1个bit位置自注意力层
    se_reduction: int = 8#SE 通道注意力中的压缩比例
    # 优化配置。
    base_lr: float = 1.5e-4#循环学习率最低值
    max_lr: float = 8e-4#循环学习率最高值
    lr_cycle_epochs: int = 10#控制一个完整学习率周期大约多少个epoch
    weight_decay: float = 1e-5#AdamW的权重衰减系数
    grad_clip_norm: float = 1.0#限制梯度整体L2范数不超过1
    auc_rank_weight: float = 0.20#最终训练损失=BCE+0.2×排序损失
    label_smoothing: float = 0.01#将标签0->0.01,1->0.99防止模型对训练数据过度自信
    ema_decay: float = 0.999#模型参数指数滑动平均;新EMA=0.999×旧EMA+0.001×当前模型参数
    ''' 
    隔若干 epoch 生成一批全新的训练样本；
    这会增加数据生成时间，但可显著扩大有效训练数据量。
    '''
    refresh_train_every: int = 5#每训练5个epoch重新生成一套新训练数据
    patience: int = 18#如果验证AUC连续18个epoch没有显著提升，则提前停止训练
    min_auc_delta: float = 1e-4#AUC在epoch中显著提升规定参数
    seed: int = 20260712
    use_amp: bool = True#控制CUDA下是否启用自动混合精度
    calibrate_threshold: bool = True#是否在验证集上寻找最佳分类阈值
    # 只保存改进网络的最终最佳纯权重文件和一份结果 JSON。
    output_dir: str = "./pdnd_accuracy_improved_checkpoints"
    keep_repetition_weights: bool = False#是否保留每一次重复实验的权重
    '''
    是否启用预训练。
    True：允许加载上一轮已有权重，或按照 stages 先训练低轮阶段。
    False：完全取消预训练，不加载上一轮权重，也不训练低轮阶段， 模型从随机初始化开始，仅训练最终目标轮。
    '''
    use_pretraining: bool = False
    # 下面参数仅当 use_pretraining=True 时生效。
    auto_load_previous_round: bool = True#自动搜索文件中保存的上一轮权重文件
    previous_weights_dir: Optional[str] = None#此参数用来指定上一轮权重所在目录
    strict_previous_weights: bool = True#控制PyTorch加载权重时是否严格匹配
    #检查配置是否合法
    def validate(self) -> None:
        if not self.stages:
            raise ValueError("至少需要一个训练阶段")
        rounds = [stage.nr for stage in self.stages]
        if rounds != sorted(rounds):
            raise ValueError("stages 应按轮数从低到高排列")
        for stage in self.stages:
            if not 1 <= stage.nr <= 22:
                raise ValueError(f"非法轮数 {stage.nr}")
            if stage.train_samples <= 0 or stage.train_samples % 2:
                raise ValueError("train_samples 必须是正偶数")
            if stage.val_samples <= 0 or stage.val_samples % 2:
                raise ValueError("val_samples 必须是正偶数")
            if stage.epochs <= 0:
                raise ValueError("epochs 必须为正")
        if self.test_samples <= 0 or self.test_samples % 2:
            raise ValueError("test_samples 必须是正偶数")
        if self.repetitions < 1:
            raise ValueError("repetitions 必须为正")
        if self.batch_size < 2:
            raise ValueError("batch_size 必须至少为 2")
        if self.width < 16 or self.width % 2:
            raise ValueError("width 必须是至少 16 的偶数")
        if self.attention_layers < 0:
            raise ValueError("attention_layers 不能为负")
        if self.attention_layers and self.width % self.attention_heads:
            raise ValueError("width 必须能被 attention_heads 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing 必须位于 [0,0.5)")
        if not 0.0 <= self.auc_rank_weight:
            raise ValueError("auc_rank_weight 不能为负")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay 必须位于 (0,1)")
        if self.refresh_train_every < 0:
            raise ValueError("refresh_train_every 不能为负")
        if self.previous_weights_dir is not None and not str(self.previous_weights_dir).strip():
            raise ValueError("previous_weights_dir 不能是空字符串")
'''尽可能统一Python、Numpy和Pytorch的随机状态'''
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

'''读取正式训练使用的两组三差分元组'''
def load_polytope_pair(config: ExperimentConfig):
    if config.search_result_path:
        path = Path(config.search_result_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            best = payload.get("best", payload)
            if "polytope_label_0" in best and "polytope_label_1" in best:
                return canonical_pair(best["polytope_label_0"], best["polytope_label_1"])
            raise ValueError(f"搜索结果文件缺少 polytope_label_0/1: {path}")
        print(f"警告：未找到搜索结果 {path}，使用 fallback 元组。", flush=True)
    return canonical_pair(config.fallback_polytope_0, config.fallback_polytope_1)

'''为一个类别生成指定数量的PDND输入特征'''
def generate_class_features(
    searcher: RL_SearchBestDiff,
    polytope: PolytopeDiff,
    sample_size: int,
    seed: int,
    encrypt_batch,
    include_anchor_xor: bool,
    include_previous_right: bool,
) -> np.ndarray:
    panel = searcher.make_panel(sample_size, seed)#生成随机面板
    vertices = searcher.encrypt_polytope(polytope, panel, encrypt_batch)#生成多胞体密文
    return vertices_to_bit_features(
        vertices,
        include_anchor_xor=include_anchor_xor,
        include_previous_right=include_previous_right,
    )
'''生成标签 0 和标签 1 数量完全相同的数据集'''
def generate_balanced_dataset(
    searcher: RL_SearchBestDiff,
    pair,
    total_samples: int,
    seed: int,
    encrypt_batch,
    include_anchor_xor: bool,
    include_previous_right: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if total_samples <= 0 or total_samples % 2:
        raise ValueError("total_samples 必须是正偶数")
    half = total_samples // 2
    # 两类使用独立随机明文和独立随机密钥。避免同索引样本共享明文/密钥造成伪相关。
    features_0 = generate_class_features(
        searcher,
        pair[0],
        half,
        seed,
        encrypt_batch,
        include_anchor_xor,
        include_previous_right,
    )
    features_1 = generate_class_features(
        searcher,
        pair[1],
        half,
        seed + 1,
        encrypt_batch,
        include_anchor_xor,
        include_previous_right,
    )
    x = np.concatenate([features_0, features_1], axis=0).astype(np.uint8, copy=False)
    y = np.concatenate(
        [np.zeros(half, dtype=np.float32), np.ones(half, dtype=np.float32)]
    )
    rng = np.random.default_rng(seed + 2)
    permutation = rng.permutation(total_samples)
    return (
        torch.from_numpy(np.ascontiguousarray(x[permutation])),
        torch.from_numpy(np.ascontiguousarray(y[permutation])),
    )
'''把Tensor数据封装成DataLoader'''
def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

'''SE 通道注意力，让网络自动学习 96 个特征通道中哪些更重要'''
class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = max(8, channels // max(1, reduction))
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),#压缩通道维度Squeeze压缩
            nn.GELU(),
            nn.Linear(hidden, channels),#把维度还原为原始通道数量Excitation激发
            nn.Sigmoid(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x.mean(dim=2)).unsqueeze(2)
        return x * scale
'''建立一个多尺度循环残差卷积块'''
class MultiScaleCircularResidualBlock(nn.Module):
    """同时使用 3 与 5 的循环卷积核，捕捉不同尺度的 ARX 比特依赖。"""
    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float,
        se_reduction: int,
    ):
        super().__init__()
        branch_channels = channels // 2
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
        self.norm1 = nn.BatchNorm1d(channels)
        self.fuse = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcite1d(channels, se_reduction)#每个残差块内部都有一个SE
        self.dropout = nn.Dropout1d(dropout)
        self.activation = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x#保留输入用于残差连接
        merged = torch.cat([self.branch3(x), self.branch5(x)], dim=1)
        merged = self.activation(self.norm1(merged))
        merged = self.dropout(merged)
        merged = self.norm2(self.fuse(merged))
        merged = self.se(merged)
        return self.activation(residual + merged)
'''在全部 16 个 bit 位置之间建立全局依赖'''
class BitAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2)
'''构建完整正式的PDND网络'''
class PolytopicPairND(nn.Module):
    def __init__(
        self,
        input_channels: int,
        width: int,
        dilations: Sequence[int],
        dense_units: int,
        dropout: float,
        attention_heads: int,
        attention_layers: int,
        se_reduction: int,
    ):
        super().__init__()
        #将输入的21个密码特征通道映射到96个隐藏通道
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        # 告诉网络不同 bit 位置并不是完全等价的。
        self.position_bias = nn.Parameter(torch.zeros(1, width, 16))
        nn.init.normal_(self.position_bias, mean=0.0, std=0.02)
        #构造 8 个多尺度循环残差块。
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
        #创建自注意力层
        self.attention_blocks = nn.ModuleList(
            [BitAttentionBlock(width, attention_heads, dropout) for _ in range(attention_layers)]
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 16, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units // 2, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().mul_(2.0).sub_(1.0)
        x = self.stem(x) + self.position_bias
        for block in self.blocks:
            x = block(x)
        for attention in self.attention_blocks:
            x = attention(x)
        return self.head(x).squeeze(1)
'''保存一份模型参数的指数滑动平均版本'''
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
    #每个训练batch后更新EMA参数
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            current = model_state[name].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(current)

#不依赖 sklearn，手工计算AUC。
def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    n_pos = int(positive.sum())
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
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
#在验证集上寻找使balanced accuracy最大的logit阈值
def best_balanced_threshold(labels: np.ndarray, logits: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    logits = np.asarray(logits, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(-logits, kind="mergesort")
    sorted_scores = logits[order]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    )
    tp = tp[group_ends]
    fp = fp[group_ends]
    tpr = tp / n_pos
    tnr = (n_neg - fp) / n_neg
    balanced = 0.5 * (tpr + tnr)
    best_index = int(np.argmax(balanced))
    return float(sorted_scores[group_ends[best_index]])

'''#根据标签、logit和阈值计算完整评价指标'''
def metrics_from_logits(
    labels: np.ndarray,
    logits: np.ndarray,
    mean_loss: float,
    threshold: float = 0.0,
) -> Dict[str, float]:
    labels = labels.astype(np.int8)
    predictions = (logits >= threshold).astype(np.int8)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, labels.size)
    auc = binary_auc(labels, logits)
    standard_error = math.sqrt(0.25 / max(1, labels.size))
    z_over_random = (accuracy - 0.5) / standard_error
    return {
        "loss": float(mean_loss),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "auc": float(auc),
        "advantage": float(2.0 * accuracy - 1.0),
        "z_over_random": float(z_over_random),
        "decision_threshold": float(threshold),
    }
'''在整个验证集或测试集上运行网络，收集所有 logit、标签以及平均损失'''
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    labels_all: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * batch_y.numel()
            labels_all.append(batch_y.cpu().numpy())
            logits_all.append(logits.float().cpu().numpy())
    labels = np.concatenate(labels_all)
    logits = np.concatenate(logits_all)
    return total_loss / labels.size, labels, logits

'''封装完整验证或测试过程。'''
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold: Optional[float] = 0.0,
) -> Dict[str, float]:
    mean_loss, labels, logits = collect_logits(model, loader, criterion, device, use_amp)
    chosen_threshold = (
        best_balanced_threshold(labels, logits) if threshold is None else float(threshold)
    )
    calibrated = metrics_from_logits(labels, logits, mean_loss, chosen_threshold)
    zero_metrics = metrics_from_logits(labels, logits, mean_loss, 0.0)
    calibrated["accuracy_at_zero"] = zero_metrics["accuracy"]
    calibrated["balanced_accuracy_at_zero"] = zero_metrics["balanced_accuracy"]
    return calibrated
def pairwise_auc_ranking_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = logits[labels >= 0.5]
    negative = logits[labels < 0.5]
    count = min(positive.numel(), negative.numel())
    if count == 0:
        return logits.new_zeros(())
    positive = positive[torch.randperm(positive.numel(), device=logits.device)[:count]]
    negative = negative[torch.randperm(negative.numel(), device=logits.device)[:count]]
    return F.softplus(-(positive - negative)).mean()
def smooth_binary_targets(labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon <= 0.0:
        return labels
    return labels * (1.0 - 2.0 * epsilon) + epsilon

'''这是整个训练程序的核心函数，负责训练一个轮数阶段'''
def train_stage(
    model: nn.Module,
    stage: StageConfig,
    searcher: RL_SearchBestDiff,
    encrypt_batch,
    pair,
    config: ExperimentConfig,
    device: torch.device,
    repetition_seed: int,
) -> Dict[str, object]:
    print(
        f"生成 {stage.nr} 轮验证集 {stage.val_samples:,}；"
        f"训练集每次 {stage.train_samples:,} ...",
        flush=True,
    )
    x_val, y_val = generate_balanced_dataset(
        searcher,
        pair,
        stage.val_samples,
        repetition_seed + stage.nr * 100_003 + 50_000_021,#同一阶段训练过程中验证集固定不变，保证不同epoch可比较
        encrypt_batch,
        config.include_anchor_xor,
        config.include_previous_right,
    )
    val_loader = make_loader(
        x_val,
        y_val,
        config.batch_size,
        False,
        device,
        config.loader_workers,
    )
    # 先生成第一批训练数据，以确定 steps_per_epoch。
    train_refresh_index = 0
    x_train, y_train = generate_balanced_dataset(
        searcher,
        pair,
        stage.train_samples,
        repetition_seed + stage.nr * 100_003,
        encrypt_batch,
        config.include_anchor_xor,
        config.include_previous_right,
    )
    train_loader = make_loader(
        x_train,
        y_train,
        config.batch_size,
        True,
        device,
        config.loader_workers,
    )
    #定义优化器和调度器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay,
        amsgrad=True,
    )
    steps_per_epoch = max(1, len(train_loader))
    half_cycle = max(1, steps_per_epoch * config.lr_cycle_epochs // 2)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=config.base_lr,
        max_lr=config.max_lr,
        step_size_up=half_cycle,
        step_size_down=half_cycle,
        cycle_momentum=False,
        mode="triangular2",
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.use_amp and device.type == "cuda"
    )
    ema = ModelEMA(model, config.ema_decay)
    best_auc = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_validation: Optional[Dict[str, float]] = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []
    for epoch in range(1, stage.epochs + 1):
        if (
            config.refresh_train_every > 0
            and epoch > 1
            and (epoch - 1) % config.refresh_train_every == 0
        ):
            del x_train, y_train, train_loader
            gc.collect()
            train_refresh_index += 1
            refresh_seed = (
                repetition_seed
                + stage.nr * 100_003
                + train_refresh_index * 70_000_027
            )
            print(
                f"R{stage.nr}: 刷新训练集 #{train_refresh_index}，seed={refresh_seed}",
                flush=True,
            )
            x_train, y_train = generate_balanced_dataset(
                searcher,
                pair,
                stage.train_samples,
                refresh_seed,
                encrypt_batch,
                config.include_anchor_xor,
                config.include_previous_right,
            )
            train_loader = make_loader(
                x_train,
                y_train,
                config.batch_size,
                True,
                device,
                config.loader_workers,
            )

        model.train()
        train_loss = 0.0
        train_labels: List[np.ndarray] = []
        train_logits: List[np.ndarray] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.use_amp and device.type == "cuda",
            ):
                logits = model(batch_x)
                targets = smooth_binary_targets(batch_y, config.label_smoothing)
                bce_loss = criterion(logits, targets)
                rank_loss = pairwise_auc_ranking_loss(logits, batch_y)
                loss = bce_loss + config.auc_rank_weight * rank_loss

            scaler.scale(loss).backward()#反向传播
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            train_loss += float(loss.item()) * batch_y.numel()
            train_labels.append(batch_y.detach().cpu().numpy())
            train_logits.append(logits.detach().float().cpu().numpy())
        train_y = np.concatenate(train_labels)
        train_z = np.concatenate(train_logits)
        train_metrics = metrics_from_logits(
            train_y,
            train_z,
            train_loss / train_y.size,
            threshold=0.0,
        )
        val_metrics = evaluate_model(
            ema.module,
            val_loader,
            criterion,
            device,
            config.use_amp,
            threshold=None if config.calibrate_threshold else 0.0,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "validation": val_metrics,
            }
        )
        print(
            f"R{stage.nr} E{epoch:03d}/{stage.epochs} "
            f"train loss={train_metrics['loss']:.5f} "
            f"acc={train_metrics['accuracy']:.4%} auc={train_metrics['auc']:.6f} | "
            f"val loss={val_metrics['loss']:.5f} "
            f"acc={val_metrics['accuracy']:.4%} "
            f"acc@0={val_metrics['accuracy_at_zero']:.4%} "
            f"auc={val_metrics['auc']:.6f} "
            f"thr={val_metrics['decision_threshold']:+.5f}",
            flush=True,
        )
        if val_metrics["auc"] > best_auc + config.min_auc_delta:
            best_auc = val_metrics["auc"]
            epochs_without_improvement = 0
            best_state = copy.deepcopy(ema.module.state_dict())
            best_validation = dict(val_metrics)
            best_epoch = int(epoch)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(
                f"R{stage.nr}: 验证 AUC 连续 {config.patience} 轮无显著提高，提前停止。",
                flush=True,
            )
            break
    if best_state is None or best_validation is None:
        raise RuntimeError(f"R{stage.nr}: 未产生可用的最佳模型权重")
    # 把最佳EMA权重加载回正式模型
    model.load_state_dict(best_state)
    del x_train, y_train, x_val, y_val, train_loader, val_loader, ema
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "history": history,
    }
'''集中传递网络结构参数，避免在多个位置重复构造模型'''
def build_model(
    config: ExperimentConfig,
    input_channels: int,
    device: torch.device,
) -> PolytopicPairND:
    return PolytopicPairND(
        input_channels=input_channels,
        width=config.width,
        dilations=config.dilations,
        dense_units=config.dense_units,
        dropout=config.dropout,
        attention_heads=config.attention_heads,
        attention_layers=config.attention_layers,
        se_reduction=config.se_reduction,
    ).to(device)
'''自动查找上一轮网络权重文件'''
def find_previous_round_weights(
    config: ExperimentConfig,
    output_dir: Path,
    previous_round: int,
    repetition: int,
) -> Optional[Path]:
    search_dir = Path(config.previous_weights_dir) if config.previous_weights_dir else output_dir
    if not search_dir.exists():
        print(
            f"上一轮权重目录不存在：{search_dir.resolve()}，将训练上一轮网络。",
            flush=True,
        )
        return None
    exact_names = (
        f"rep{repetition:02d}_speck32_{previous_round}r_best_weights.pth",
        f"speck32_{previous_round}r_improved_best_weights.pth",
        f"speck32_{previous_round}r_best_weights.pth",
        f".rep{repetition:02d}_speck32_{previous_round}r_best_weights.pth",
    )
    for name in exact_names:
        candidate = search_dir / name
        if candidate.is_file():
            return candidate
    candidates = []
    for pattern in (
        f"rep*_speck32_{previous_round}r_best_weights.pth",
        f"*speck32_{previous_round}r*weights*.pth",
    ):
        candidates.extend(path for path in search_dir.glob(pattern) if path.is_file())
    unique_candidates = list({path.resolve(): path for path in candidates}.values())
    if not unique_candidates:
        return None
    selected = max(unique_candidates, key=lambda path: path.stat().st_mtime)
    print(
        f"未找到当前 repetition 的精确上一轮权重，自动选择最近修改文件："
        f"{selected.resolve()}",
        flush=True,
    )
    return selected

'''将上一轮权重加载到当前模型'''
def load_previous_round_weights(
    model: nn.Module,
    weights_path: Path,
    device: torch.device,
    strict: bool,
) -> Dict[str, object]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在：{weights_path.resolve()}")

    try:
        payload = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(weights_path, map_location=device)
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        source_type = "checkpoint"
    elif isinstance(payload, Mapping):
        state_dict = payload
        source_type = "state_dict"
    else:
        raise TypeError(
            f"无法识别权重文件格式：{weights_path.resolve()}；"
            "应为 state_dict 或包含 model_state_dict 的字典。"
        )
    state_dict = dict(state_dict)
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key)[len("module."):]: value
            for key, value in state_dict.items()
        }
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    print(f"已加载上一轮权重：{weights_path.resolve()}", flush=True)
    print(
        f"加载类型={source_type}，strict={strict}，"
        f"missing_keys={len(missing_keys)}，"
        f"unexpected_keys={len(unexpected_keys)}",
        flush=True,
    )

    return {
        "path": str(weights_path.resolve()),
        "source_type": source_type,
        "strict": bool(strict),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }

'''完成一次完整独立实验'''
def train_one_repetition(
    config: ExperimentConfig,
    pair,
    repetition: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    repetition_seed = config.seed + repetition * 10_000_019
    seed_everything(repetition_seed)
    input_channels = feature_channel_count(
        len(pair[0]),
        config.include_anchor_xor,
        config.include_previous_right,
    )
    model = build_model(config, input_channels, device)
    target_stage = config.stages[-1]
    previous_round = target_stage.nr - 1
    stages_to_run = list(config.stages)
    pretraining_info: Dict[str, object] = {
        "target_round": int(target_stage.nr),
        "previous_round": int(previous_round),
        "mode": "train_configured_stages",
        "loaded_weights": None,
        "executed_stages": [],
    }
    if not config.use_pretraining:
        # 完全取消预训练：不加载上一轮权重，也不执行任何低轮阶段。
        stages_to_run = [target_stage]
        pretraining_info["mode"] = "disabled_random_initialization"
        print(
            f"预训练已关闭：模型从随机权重开始，仅训练目标轮 R{target_stage.nr}。",
            flush=True,
        )
    #启用预训练且找到上一轮权重加载权重，并跳过低轮阶段，直接训练目标轮
    elif config.auto_load_previous_round and previous_round >= 1:
        previous_weights_path = find_previous_round_weights(
            config=config,
            output_dir=output_dir,
            previous_round=previous_round,
            repetition=repetition,
        )
        if previous_weights_path is not None:
            loaded_info = load_previous_round_weights(
                model=model,
                weights_path=previous_weights_path,
                device=device,
                strict=config.strict_previous_weights,
            )
            stages_to_run = [target_stage]
            pretraining_info["mode"] = "loaded_existing_previous_round"
            pretraining_info["loaded_weights"] = loaded_info
            print(
                f"检测到 {previous_round} 轮权重，跳过低轮训练，"
                f"直接训练 {target_stage.nr} 轮。",
                flush=True,
            )
        else:
            # 启用预训练但找不到权重按照 stages 中配置的低轮阶段先训练，再训练目标轮
            lower_round_stages = [
                stage for stage in config.stages
                if stage.nr < target_stage.nr
            ]
            if not lower_round_stages:
                raise FileNotFoundError(
                    f"未找到 {previous_round} 轮权重，且 stages 中没有低轮预训练阶段。"
                )
            pretraining_info["mode"] = "trained_lower_round_stages"
            print(
                f"未找到 {previous_round} 轮权重，将按照 stages 先训练低轮阶段 "
                f"{[stage.nr for stage in lower_round_stages]}，"
                f"再训练目标轮 {target_stage.nr}。",
                flush=True,
            )
    else:
        lower_round_stages = [
            stage for stage in config.stages
            if stage.nr < target_stage.nr
        ]
        if lower_round_stages:
            pretraining_info["mode"] = "trained_lower_round_stages"
            print(
                f"已关闭上一轮权重自动加载，将按照 stages 执行预训练："
                f"{[stage.nr for stage in lower_round_stages]} → {target_stage.nr}。",
                flush=True,
            )
        else:
            pretraining_info["mode"] = "no_lower_round_stage"
            print(
                f"stages 中没有低轮阶段，将从随机权重直接训练目标轮 "
                f"R{target_stage.nr}。",
                flush=True,
            )
    stage_results: List[Dict[str, object]] = []
    for stage in stages_to_run:
        searcher = RL_SearchBestDiff(config.cipher_config_path, stage.nr)
        cipher_dll, encrypt_batch = searcher.load_encrypt_function()
        result = train_stage(
            model,
            stage,
            searcher,
            encrypt_batch,
            pair,
            config,
            device,
            repetition_seed,
        )
        stage_results.append({"stage": asdict(stage), **result})
        pretraining_info["executed_stages"].append(int(stage.nr))
        del cipher_dll, searcher
        gc.collect()
    # 最终测试只针对最后一个目标轮数，并使用目标阶段验证集校准得到的阈值。
    target_threshold = float(
        stage_results[-1]["best_validation"].get("decision_threshold", 0.0)
    )
    searcher = RL_SearchBestDiff(config.cipher_config_path, target_stage.nr)
    cipher_dll, encrypt_batch = searcher.load_encrypt_function()
    x_test, y_test = generate_balanced_dataset(
        searcher,
        pair,
        config.test_samples,
        repetition_seed + 900_000_049,
        encrypt_batch,
        config.include_anchor_xor,
        config.include_previous_right,
    )
    test_loader = make_loader(
        x_test,
        y_test,
        config.batch_size,
        False,
        device,
        config.loader_workers,
    )
    test_metrics = evaluate_model(
        model,
        test_loader,
        nn.BCEWithLogitsLoss(),
        device,
        config.use_amp,
        threshold=target_threshold,
    )
    repetition_weights_path = (
        output_dir
        / f".rep{repetition:02d}_speck32_{target_stage.nr}r_best_weights.pth"
    )
    cpu_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(cpu_state, repetition_weights_path)

    del cipher_dll, searcher, x_test, y_test, test_loader, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "repetition": repetition,
        "seed": repetition_seed,
        "pretraining": pretraining_info,
        "stage_results": stage_results,
        "test": test_metrics,
        "temporary_weights_path": str(repetition_weights_path),
    }
'''汇总多次独立测试结果'''
def aggregate_repetitions(results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    metric_names = [
        "loss",
        "accuracy",
        "accuracy_at_zero",
        "balanced_accuracy",
        "balanced_accuracy_at_zero",
        "tpr",
        "tnr",
        "auc",
        "advantage",
        "z_over_random",
        "decision_threshold",
    ]
    aggregate: Dict[str, object] = {}
    for metric in metric_names:
        values = np.asarray(
            [result["test"][metric] for result in results],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate

'''正式训练程序的顶层控制函数'''
def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    config.validate()
    # 开启预训练时，低轮预训练阶段和目标轮训练阶段统一使用上一轮的
    # 最优输入差分元组。例如目标轮 nr=8 时，R7 预训练和 R8 正式训练
    # 均读取 best_polytope_pair_7round.json，而不再读取 8 轮差分元组。
    target_round = config.stages[-1].nr
    pair_source_round = target_round
    if config.use_pretraining and target_round > 1:
        pair_source_round = target_round - 1
        if config.search_result_path:
            configured_path = Path(config.search_result_path)
            config.search_result_path = str(
                configured_path.with_name(
                    f"best_polytope_pair_{pair_source_round}round.json"
                )
            )
        print(
            f"预训练已开启：R{pair_source_round} 预训练和 R{target_round} 目标轮训练"
            f"统一使用 R{pair_source_round} 最优输入差分元组。",
            flush=True,
        )
    pair = load_polytope_pair(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print("SPECK32/64 增强双结构化多胞体神经区分器")
    print(f"设备: {device}")
    print(f"类别 0: {[[hex(a), hex(b)] for a, b in pair[0]]}")
    print(f"类别 1: {[[hex(a), hex(b)] for a, b in pair[1]]}")
    print(
        "输入通道: "
        f"{feature_channel_count(len(pair[0]), config.include_anchor_xor, config.include_previous_right)}"
    )
    print(
        f"网络: width={config.width}, attention={config.attention_layers}层/"
        f"{config.attention_heads}头, EMA={config.ema_decay}, "
        f"rank_weight={config.auc_rank_weight}"
    )
    print("=" * 96)
    repetitions: List[Dict[str, object]] = []
    for repetition in range(config.repetitions):
        print(f"\n--- 独立重复 {repetition + 1}/{config.repetitions} ---", flush=True)
        repetitions.append(
            train_one_repetition(config, pair, repetition, device, output_dir)
        )
    best_repetition = max(
        repetitions,
        key=lambda item: float(
            item["stage_results"][-1]["best_validation"]["auc"]
        ),
    )
    best_source_path = Path(best_repetition["temporary_weights_path"])
    best_weights_path = (
        output_dir / f"speck32_{target_round}r_improved_best_weights.pth"
    )
    shutil.copy2(best_source_path, best_weights_path)
    if not config.keep_repetition_weights:
        for item in repetitions:
            temporary_path = Path(item["temporary_weights_path"])
            if temporary_path.exists():
                temporary_path.unlink()
            item["temporary_weights_path"] = None

    output = {
        "schema": "speck32_pdnd_pair_training_optional_pretraining_v4",
        "pair": pair_to_jsonable(pair),
        "pair_source_round": int(pair_source_round),
        "config": asdict(config),
        "device": str(device),
        "best_model": {
            "selected_repetition": int(best_repetition["repetition"]),
            "selection_metric": "validation_auc",
            "validation_auc": float(
                best_repetition["stage_results"][-1]["best_validation"]["auc"]
            ),
            "weights_path": str(best_weights_path),
        },
        "repetitions": repetitions,
        "aggregate_test": aggregate_repetitions(repetitions),
    }
    result_path = output_dir / f"speck32_pdnd_{target_round}r_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(f"最终最佳纯权重已保存至: {best_weights_path.resolve()}")
    print(f"实验结果已保存至: {result_path.resolve()}")
    print(json.dumps(output["aggregate_test"], indent=2, ensure_ascii=False))
    return output
'''预设轮数训练计划'''
def make_target_stages(target_round: int) -> Tuple[StageConfig, ...]:
    plans = {
       5: (
            StageConfig(4, 1_000_000, 100_000, 30),
            StageConfig(5, 10_000_000, 1_000_000, 60),
        ),
        6: (
            StageConfig(5, 2_000_000, 200_000, 40),
            StageConfig(6, 10_000_000, 1_000_000, 80),
        ),
        7: (
            StageConfig(6, 5_000_000, 500_000, 50),
            StageConfig(7, 10_000_000, 1_000_000, 100),
        ),
        8: (
            StageConfig(7, 5_000_000, 500_000, 60),
            StageConfig(8, 10_000_000, 1_000_000, 120),
        ),
        9: (
            StageConfig(8, 6_000_000, 600_000, 70),
            StageConfig(9, 10_000_000, 1_000_000, 140),
        ),
    }
    if target_round not in plans:
        raise ValueError("默认计划仅提供 5、6、7、8、9 轮")
    return plans[target_round]
if __name__ == "__main__":
    TARGET_ROUND = 8
    # True：启用预训练。
    # False：完全取消预训练，只从随机权重训练目标轮。
    USE_PRETRAINING = True
    experiment = ExperimentConfig(
        search_result_path=f"../data/best_polytope_pair_{TARGET_ROUND}round.json",
        stages=make_target_stages(TARGET_ROUND),
        use_pretraining=USE_PRETRAINING,
        # 仅在 USE_PRETRAINING=True 时生效。
        auto_load_previous_round=True,
        test_samples=1_000_000,
        repetitions=3,
    )
    run_experiment(experiment)
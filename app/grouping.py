"""地区对照报告的共用年龄分组（纯函数，不依赖 Web/数据库）。

输入：
- 作业快照中各年龄段的固定顺序 ``age_order``；
- 必须保留的分界（在哪些年龄段之后必须切开）；
- 最低有效样本量 ``min_ess``；
- 每个 (地区, 年龄段) 格内的个人校正权重列表。

规则：
1. 只允许合并**相邻**年龄段，且保留分界两侧永远不得落入同一组；
2. 每个地区在每个合并组上都必须达标，有效样本量
   ``ESS = (Σw)² / Σ(w²)``，没有任何样本的空格计 0；
3. 输出组数最多的完整分法（组数相同时取更靠前段更短的确定性分法）；
4. 若不存在覆盖全部年龄段的完整分法，则按保留分界把年龄段切成若干区间，
   逐区间报告“可分组 / 无法分组”，**不产出任何部分方案**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# 与权重求解器对外承诺的残差容差同量级：
# 理论上恰好在阈值上的格按达标处理，避免浮点尾差翻转结论。
ESS_TOL = 1.0e-9


def effective_sample_size(weights: Iterable[float]) -> float:
    """Kish 有效样本量 (Σw)²/Σ(w²)；空权重列表（空格）计 0。"""
    sum_w = 0.0
    sum_w2 = 0.0
    for w in weights:
        sum_w += w
        sum_w2 += w * w
    if sum_w2 <= 0.0:
        return 0.0
    return (sum_w * sum_w) / sum_w2


@dataclass
class CellESS:
    """单个 (地区, 年龄段) 原子格的有效样本量。"""

    region: str
    age_group: str
    ess: float
    sample_count: int          # 该格内样本人数（0 即空格）
    meets_threshold: bool      # ess >= min_ess（含容差）


@dataclass
class GroupESS:
    """一个合并组在某地区上的有效样本量。"""

    region: str
    ess: float
    sample_count: int
    meets_threshold: bool


@dataclass
class AgeGroupBlock:
    """一个合并后的年龄组（由若干相邻原子年龄段组成）。"""

    group_index: int
    age_groups: list[str]
    cut_after_last: bool                  # 该组之后是否为保留分界
    cells: list[GroupESS]                 # 各地区在该组上的 ESS


@dataclass
class UngroupableSegment:
    """保留分界切开后、无法在段内形成达标的完整分法的区间。"""

    segment_index: int
    age_groups: list[str]
    # 段内每个原子年龄段 × 每个地区的格内 ESS（空格计 0）
    cell_ess: list[CellESS]
    # 整个段不做任何合并时，各地区的 ESS（仍可能为 0）
    whole_segment_ess: list[GroupESS]


@dataclass
class GroupingResult:
    feasible: bool
    age_group_order: list[str]
    regions: list[str]
    keep_boundaries: list[str]
    min_ess: float
    # feasible=True 时给出完整分法
    groups: list[AgeGroupBlock] = field(default_factory=list)
    # feasible=False 时逐区间给出诊断（不产出部分方案）
    ungroupable_segments: list[UngroupableSegment] = field(default_factory=list)
    # 全部原子格 ESS，两种情形都返回，便于核对合并前后数值
    atomic_cell_ess: list[CellESS] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 输入整理
# ---------------------------------------------------------------------------

def _meets(ess: float, min_ess: float) -> bool:
    return ess + ESS_TOL >= min_ess


def build_cells(
    age_order: list[str],
    regions: list[str],
    cell_weights: dict[tuple[str, str], list[float]],
    min_ess: float,
) -> list[CellESS]:
    cells: list[CellESS] = []
    for age in age_order:
        for region in regions:
            weights = cell_weights.get((region, age), [])
            ess = effective_sample_size(weights)
            cells.append(CellESS(
                region=region,
                age_group=age,
                ess=ess,
                sample_count=len(weights),
                meets_threshold=_meets(ess, min_ess),
            ))
    return cells


def _cell_index(age_idx: int, region_idx: int, n_regions: int) -> int:
    return age_idx * n_regions + region_idx


def _segment_ess_with_threshold(
    start: int,
    end: int,
    cells: list[CellESS],
    regions: list[str],
    cell_weights: dict[tuple[str, str], list[float]],
    min_ess: float,
) -> list[GroupESS]:
    out: list[GroupESS] = []
    for r_idx, region in enumerate(regions):
        merged: list[float] = []
        count = 0
        for a_idx in range(start, end):
            cell = cells[_cell_index(a_idx, r_idx, len(regions))]
            count += cell.sample_count
            merged.extend(cell_weights.get((region, cell.age_group), []))
        ess = effective_sample_size(merged)
        out.append(GroupESS(
            region=region,
            ess=ess,
            sample_count=count,
            meets_threshold=_meets(ess, min_ess),
        ))
    return out


def _block_valid(seg_ess: list[GroupESS]) -> bool:
    return all(g.meets_threshold for g in seg_ess)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def plan_grouping(
    age_order: list[str],
    regions: list[str],
    cell_weights: dict[tuple[str, str], list[float]],
    keep_boundaries: list[str],
    min_ess: float,
) -> GroupingResult:
    """求组数最多的完整相邻合并分法；不可行时按保留分界逐段诊断。

    ``keep_boundaries`` 为年龄段代码列表：在每个代码对应的年龄段之后强制
    切开（最后一个年龄段之后不构成界，调用方应在 API 层拒绝）。
    """
    n = len(age_order)
    n_regions = len(regions)
    boundary_set = set(keep_boundaries)
    # 强制切开位置：cut[k] 为 True 表示第 k 个年龄段（0-based）之后必须切开
    forced_cut = [age in boundary_set for age in age_order]

    cells = build_cells(age_order, regions, cell_weights, min_ess)

    # 预计算每个候选连续段在各地区上的合并 ESS 与合法性
    # seg_valid[i][j]: 年龄段 [i, j) 是否对所有地区达标
    seg_ess_cache: dict[tuple[int, int], list[GroupESS]] = {}
    seg_valid: dict[tuple[int, int], bool] = {}
    for i in range(n):
        for j in range(i + 1, n + 1):
            ges = _segment_ess_with_threshold(
                i, j, cells, regions, cell_weights, min_ess
            )
            seg_ess_cache[(i, j)] = ges
            seg_valid[(i, j)] = _block_valid(ges)

    # DP：dp[e] = 覆盖 [0, e) 的最优分法
    # 用 (组数, 组起点序列) 表示。择优规则：
    #   1) 组数最多；
    #   2) 组数相同时，组起点序列字典序更小 —— 切点尽量靠前、前段尽量短，
    #      保证任何输入下结果唯一且确定。
    # 约束：段 [s, e) 对所有地区达标，且内部不含强制分界
    # （forced_cut[s:e-1] 全 False）。
    dp: dict[int, tuple[int, tuple[int, ...]] | None] = {0: (0, ())}
    for e in range(1, n + 1):
        best: tuple[int, tuple[int, ...]] | None = None
        for s in range(e):
            if s not in dp or dp[s] is None or dp[s][1] is None:
                continue
            # 段 (s, e) 内部不能跨越强制分界
            if any(forced_cut[k] for k in range(s, e - 1)):
                continue
            if not seg_valid[(s, e)]:
                continue
            cand_count = dp[s][0] + 1
            cand_cuts = dp[s][1] + (s,)  # 新组从 s 开始
            cand = (cand_count, cand_cuts)
            if best is None or cand_count > best[0] or \
                    (cand_count == best[0] and cand_cuts < best[1]):
                best = cand
        dp[e] = best

    atomic_cells = cells

    if dp[n] is not None:
        starts = list(dp[n][1]) + [n]
        groups: list[AgeGroupBlock] = []
        for gi in range(len(starts) - 1):
            s, e = starts[gi], starts[gi + 1]
            ges = seg_ess_cache[(s, e)]
            groups.append(AgeGroupBlock(
                group_index=gi,
                age_groups=age_order[s:e],
                cut_after_last=(e < n and forced_cut[e - 1]),
                cells=ges,
            ))
        return GroupingResult(
            feasible=True,
            age_group_order=list(age_order),
            regions=list(regions),
            keep_boundaries=list(keep_boundaries),
            min_ess=min_ess,
            groups=groups,
            atomic_cell_ess=atomic_cells,
        )

    # 不可行：按强制分界切段，逐段独立判定能否在段内完整分组
    segments: list[UngroupableSegment] = []
    seg_start = 0
    seg_index = 0
    for k in range(n):
        if forced_cut[k]:
            seg_end = k + 1
            _append_segment(
                segments, seg_index, seg_start, seg_end, age_order, regions,
                cells, cell_weights, min_ess, n_regions, seg_valid,
            )
            seg_index += 1
            seg_start = seg_end
    if seg_start < n:
        _append_segment(
            segments, seg_index, seg_start, n, age_order, regions,
            cells, cell_weights, min_ess, n_regions, seg_valid,
        )

    return GroupingResult(
        feasible=False,
        age_group_order=list(age_order),
        regions=list(regions),
        keep_boundaries=list(keep_boundaries),
        min_ess=min_ess,
        ungroupable_segments=segments,
        atomic_cell_ess=atomic_cells,
    )


def _append_segment(
    segments: list[UngroupableSegment],
    seg_index: int,
    start: int,
    end: int,
    age_order: list[str],
    regions: list[str],
    cells: list[CellESS],
    cell_weights: dict[tuple[str, str], list[float]],
    min_ess: float,
    n_regions: int,
    seg_valid: dict[tuple[int, int], bool],
) -> None:
    """段内再跑一次小规模可达性 DP：能完整分组则跳过，否则记录诊断。"""
    m = end - start
    reachable = [False] * (m + 1)
    reachable[0] = True
    for e in range(1, m + 1):
        for s in range(e):
            if reachable[s] and seg_valid[(start + s, start + e)]:
                reachable[e] = True
                break
    if reachable[m]:
        return  # 该段本身可分组，不列入“无法分组”清单

    seg_cells = [
        cells[_cell_index(a, r, n_regions)]
        for a in range(start, end) for r in range(n_regions)
    ]
    whole = _segment_ess_with_threshold(
        start, end, cells, regions, cell_weights, min_ess
    )
    segments.append(UngroupableSegment(
        segment_index=seg_index,
        age_groups=age_order[start:end],
        cell_ess=seg_cells,
        whole_segment_ess=whole,
    ))

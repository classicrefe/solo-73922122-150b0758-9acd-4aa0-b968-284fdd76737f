"""年龄分组求解器单元测试。

覆盖：ESS 公式（含空格计 0）、只合相邻、保留分界、组数最多的最优性、
并列裁决确定性、不可行时按分界切段诊断、不产出部分方案。
"""

from __future__ import annotations

import pytest

from app import grouping


def uniform_cells(regions, ages, n_per_cell=1, w=1.0):
    """每个 (地区, 年龄段) 格放 n_per_cell 个等权 w 的样本。"""
    return {
        (r, a): [w] * n_per_cell
        for r in regions for a in ages
    }


def test_ess_formula_equal_weights_equals_count():
    # 等权 w：ESS = (n w)² / (n w²) = n
    for n in (1, 2, 5, 10):
        assert grouping.effective_sample_size([3.0] * n) == pytest.approx(float(n))


def test_ess_formula_unequal_weights():
    # 手算：w=[1,2,3] -> (6)²/(1+4+9)=36/14
    assert grouping.effective_sample_size([1.0, 2.0, 3.0]) == \
        pytest.approx(36.0 / 14.0)


def test_empty_cell_counts_zero():
    assert grouping.effective_sample_size([]) == 0.0
    res = grouping.plan_grouping(
        ["a1"], ["R1"], {("R1", "a1"): []}, [], 1.0
    )
    assert res.feasible is False
    cell = res.atomic_cell_ess[0]
    assert cell.ess == 0.0 and cell.sample_count == 0
    assert cell.meets_threshold is False
    assert len(res.ungroupable_segments) == 1
    assert res.groups == []  # 不产出部分方案


def test_single_age_group_passes_when_all_regions_meet():
    cw = {("R1", "a1"): [1.0, 1.0], ("R2", "a1"): [1.0, 1.0]}
    res = grouping.plan_grouping(["a1"], ["R1", "R2"], cw, [], 2.0)
    assert res.feasible is True
    assert [g.age_groups for g in res.groups] == [["a1"]]
    for g in res.groups:
        assert {c.region: c.meets_threshold for c in g.cells} == \
            {"R1": True, "R2": True}


def test_all_singletons_when_threshold_met():
    ages = ["a1", "a2", "a3", "a4"]
    cw = uniform_cells(["R1", "R2"], ages, n_per_cell=2, w=1.0)
    res = grouping.plan_grouping(ages, ["R1", "R2"], cw, [], 2.0)
    assert res.feasible is True
    assert [g.age_groups for g in res.groups] == [[a] for a in ages]
    assert len(res.groups) == 4  # 组数最多：全部不合并


def test_merges_adjacent_only_and_maximizes_group_count():
    # 每格 1 人等权：单格 ESS=1，两格合并 ESS=2，三格 ESS=3
    ages = ["a1", "a2", "a3", "a4"]
    cw = uniform_cells(["R1"], ages, n_per_cell=1)
    res = grouping.plan_grouping(ages, ["R1"], cw, [], 2.0)
    assert res.feasible is True
    blocks = [g.age_groups for g in res.groups]
    # 4 个原子段、每段至少 2 个原子 => 最多 2 组
    assert len(blocks) == 2
    assert blocks == [["a1", "a2"], ["a3", "a4"]]
    # 每组都是相邻年龄段，且 ESS 达标
    for g in res.groups:
        assert g.cells[0].ess == pytest.approx(2.0)
        assert g.cells[0].meets_threshold is True


def test_threshold_three_forces_triples():
    ages = ["a1", "a2", "a3", "a4", "a5", "a6"]
    cw = uniform_cells(["R1"], ages, n_per_cell=1)
    res = grouping.plan_grouping(ages, ["R1"], cw, [], 3.0)
    blocks = [g.age_groups for g in res.groups]
    assert blocks == [["a1", "a2", "a3"], ["a4", "a5", "a6"]]


def test_keep_boundary_never_crossed():
    # 4 段等权单样本，阈值 2；强制在 a2 后切开
    ages = ["a1", "a2", "a3", "a4"]
    cw = uniform_cells(["R1"], ages, n_per_cell=1)
    res = grouping.plan_grouping(ages, ["R1"], cw, ["a2"], 2.0)
    assert res.feasible is True
    blocks = [g.age_groups for g in res.groups]
    # 分界两侧各自成对，绝不出现跨 a2|a3 的组合
    assert blocks == [["a1", "a2"], ["a3", "a4"]]
    assert res.groups[0].cut_after_last is True
    assert res.groups[1].cut_after_last is False


def test_keep_boundary_makes_feasible_problem_infeasible():
    # 无分界时三段全合并可行（ESS=3）；强制在 a2 后切开后，
    # 右段 [a3] 单独 ESS=1 < 2，无法分组
    ages = ["a1", "a2", "a3"]
    cw = uniform_cells(["R1"], ages, n_per_cell=1)
    ok = grouping.plan_grouping(ages, ["R1"], cw, [], 2.0)
    assert ok.feasible is True
    bad = grouping.plan_grouping(ages, ["R1"], cw, ["a2"], 2.0)
    assert bad.feasible is False
    assert bad.groups == []  # 无部分方案
    seg_ages = [s.age_groups for s in bad.ungroupable_segments]
    assert seg_ages == [["a3"]]
    seg = bad.ungroupable_segments[0]
    # 诊断里给出短板格与整段 ESS
    assert seg.cell_ess[0].ess == pytest.approx(1.0)
    assert seg.whole_segment_ess[0].ess == pytest.approx(1.0)
    assert seg.whole_segment_ess[0].meets_threshold is False


def test_infeasible_reports_each_bad_segment_separately():
    # 6 段，强制分界在 a2、a4 后 => 三个区间，每区 2 段；
    # 中间区间每格只有 1 个样本（区间合计 2 人），其余区间 4 人。
    # 阈值 2 时中间区间整段 ESS=2 恰好达标 => 完整分法存在；
    # 阈值 3 时中间区间 ESS=2 不达标，仅该区间进入无法分组清单。
    ages = ["a1", "a2", "a3", "a4", "a5", "a6"]
    cw = uniform_cells(["R1"], ages, n_per_cell=2)
    for a in ("a3", "a4"):
        cw[("R1", a)] = [1.0]  # 中间区间降为单样本格
    ok = grouping.plan_grouping(ages, ["R1"], cw, ["a2", "a4"], 2.0)
    assert ok.feasible is True
    # 阈值 2 下：首尾区间每格 2 人（ESS=2）可单独成组，中间区间整段 ESS=2
    assert [g.age_groups for g in ok.groups] == [
        ["a1"], ["a2"], ["a3", "a4"], ["a5"], ["a6"]]

    strict = grouping.plan_grouping(ages, ["R1"], cw, ["a2", "a4"], 3.0)
    assert strict.feasible is False
    assert strict.groups == []
    strict_segs = [s.age_groups for s in strict.ungroupable_segments]
    assert strict_segs == [["a3", "a4"]]
    seg = strict.ungroupable_segments[0]
    assert seg.whole_segment_ess[0].ess == pytest.approx(2.0)
    assert seg.whole_segment_ess[0].meets_threshold is False


def test_every_region_must_meet_in_every_group():
    # R1 每格 2 人、R2 每格 1 人；阈值 2 => R2 必须靠合并达标
    ages = ["a1", "a2", "a3"]
    cw = {}
    for a in ages:
        cw[("R1", a)] = [1.0, 1.0]
        cw[("R2", a)] = [1.0]
    res = grouping.plan_grouping(ages, ["R1", "R2"], cw, [], 2.0)
    assert res.feasible is True
    # R2 单格 ESS=1 不达标，任何单原子组都不允许 => 至少每组合 2 段
    for g in res.groups:
        by_region = {c.region: c for c in g.cells}
        assert by_region["R1"].meets_threshold
        assert by_region["R2"].meets_threshold
        assert by_region["R2"].ess >= 2.0 - 1e-9
    assert len(res.groups) == 1  # 3 段每段最少 2 原子 => 只能 1 组


def test_weak_region_drives_infeasibility_diagnosis():
    ages = ["a1", "a2"]
    cw = {
        ("R1", "a1"): [1.0, 1.0], ("R1", "a2"): [1.0, 1.0],
        ("R2", "a1"): [1.0],       ("R2", "a2"): [],       # R2 有空格
    }
    res = grouping.plan_grouping(ages, ["R1", "R2"], cw, [], 2.0)
    # R2 全段只有 1 人，ESS=1 < 2 => 整段无法分组
    assert res.feasible is False
    assert [s.age_groups for s in res.ungroupable_segments] == [["a1", "a2"]]
    seg = res.ungroupable_segments[0]
    r2_cells = [c for c in seg.cell_ess if c.region == "R2"]
    assert {c.age_group: c.ess for c in r2_cells} == {"a1": 1.0, "a2": 0.0}
    whole_r2 = [c for c in seg.whole_segment_ess if c.region == "R2"][0]
    assert whole_r2.ess == pytest.approx(1.0)
    assert whole_r2.sample_count == 1


def test_deterministic_tie_break_prefers_earlier_cuts():
    # 4 段等权单样本、阈值 2：两种 2 组分法 [(a1a2)(a3a4)] 与
    # 不存在其它 2 组分法；构造一个真正并列的场景：
    # 阈值 1 时所有原子段都达标，最优是 4 个单段组（唯一）。
    # 用自定义权重制造“两种切法组数相同”：
    # 每格等权 1 人、阈值 1 => 单段全达标，4 组唯一最优。
    ages = ["a1", "a2", "a3", "a4"]
    cw = uniform_cells(["R1"], ages, n_per_cell=1)
    res = grouping.plan_grouping(ages, ["R1"], cw, [], 1.0)
    first = [g.age_groups for g in res.groups]
    # 重复调用结果稳定
    res2 = grouping.plan_grouping(ages, ["R1"], cw, [], 1.0)
    assert [g.age_groups for g in res2.groups] == first
    assert first == [["a1"], ["a2"], ["a3"], ["a4"]]

    # 真正的并列场景：a2 为空格（ESS=0），a1、a3 各有 2 个等权样本，
    # 阈值 1。最大组数为 2，存在两种同分切法：
    #   [a1][a2,a3] 与 [a1,a2][a3]
    # 裁决规则取切点更靠前的前者（组起点元组字典序更小）。
    tie_cw = {
        ("R1", "a1"): [1.0, 1.0],
        ("R1", "a2"): [],
        ("R1", "a3"): [1.0, 1.0],
    }
    res3 = grouping.plan_grouping(["a1", "a2", "a3"], ["R1"], tie_cw, [], 1.0)
    assert res3.feasible is True
    assert [g.age_groups for g in res3.groups] == [["a1"], ["a2", "a3"]]
    # 另一种切法 [a1,a2][a3] 同样合法且同为 2 组（并列由裁决规则打破）
    ess_a1a2 = grouping.effective_sample_size(
        tie_cw[("R1", "a1")] + tie_cw[("R1", "a2")])
    ess_a3 = grouping.effective_sample_size(tie_cw[("R1", "a3")])
    assert min(ess_a1a2, ess_a3) >= 1.0


def test_boundary_cut_flags_and_group_indices():
    ages = ["a1", "a2", "a3", "a4"]
    cw = uniform_cells(["R1"], ages, n_per_cell=1)
    res = grouping.plan_grouping(ages, ["R1"], cw, ["a2"], 2.0)
    assert [g.group_index for g in res.groups] == [0, 1]
    assert res.groups[0].cut_after_last is True
    assert res.groups[1].cut_after_last is False


def test_atomic_cell_ess_always_returned():
    ages = ["a1", "a2"]
    cw = {
        ("R1", "a1"): [1.0], ("R1", "a2"): [2.0, 2.0],
        ("R2", "a1"): [1.0, 1.0], ("R2", "a2"): [1.0],
    }
    res = grouping.plan_grouping(ages, ["R1", "R2"], cw, [], 1.0)
    assert len(res.atomic_cell_ess) == 4
    idx = {(c.region, c.age_group): c for c in res.atomic_cell_ess}
    assert idx[("R1", "a2")].ess == pytest.approx(2.0)  # 等权 2 人
    assert idx[("R1", "a2")].sample_count == 2
    # 不可行情形同样返回原子格
    bad = grouping.plan_grouping(ages, ["R1", "R2"], cw, [], 5.0)
    assert bad.feasible is False
    assert len(bad.atomic_cell_ess) == 4


def test_multiple_boundaries_segment_diagnostics():
    # 5 段，分界 a1、a3 后 => 区间 [a1] [a2,a3] [a4,a5]；
    # 阈值 3，单样本等权：区间长度 1 与 2 都不达标，仅长度 2 的也不达标
    ages = ["a1", "a2", "a3", "a4", "a5"]
    cw = uniform_cells(["R1"], ages, n_per_cell=1)
    res = grouping.plan_grouping(ages, ["R1"], cw, ["a1", "a3"], 3.0)
    assert res.feasible is False
    segs = [s.age_groups for s in res.ungroupable_segments]
    assert segs == [["a1"], ["a2", "a3"], ["a4", "a5"]]
    assert [s.segment_index for s in res.ungroupable_segments] == [0, 1, 2]


def test_missing_cell_treated_as_empty():
    # 完全没有出现的格按空格计 0
    ages = ["a1", "a2"]
    cw = {("R1", "a1"): [1.0]}  # 其余格缺失；整段 ESS 最多为 1
    res = grouping.plan_grouping(ages, ["R1"], cw, [], 2.0)
    assert res.feasible is False
    idx = {(c.region, c.age_group): c for c in res.atomic_cell_ess}
    assert idx[("R1", "a2")].ess == 0.0
    assert idx[("R1", "a2")].sample_count == 0
    assert [s.age_groups for s in res.ungroupable_segments] == [["a1", "a2"]]


def test_merged_ess_uses_weights_not_headcount():
    # 不等权合并：w=[1,3] -> ESS=16/10=1.6；阈值 1.5 达标、1.8 不达标
    ages = ["a1", "a2"]
    cw = {("R1", "a1"): [1.0], ("R1", "a2"): [3.0]}
    ok = grouping.plan_grouping(ages, ["R1"], cw, [], 1.5)
    assert ok.feasible is True
    assert ok.groups[0].cells[0].ess == pytest.approx(1.6)
    bad = grouping.plan_grouping(ages, ["R1"], cw, [], 1.8)
    assert bad.feasible is False

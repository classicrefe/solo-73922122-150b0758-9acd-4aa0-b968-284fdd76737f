"""/age-grouping 端到端测试。

覆盖：成功分组（组数最多、ESS 数值核对）、不可行分段诊断（无部分方案）、
409 语义（作业不存在/非成功/失效）、422 严格校验、OpenAPI 暴露。
"""

import time

import pytest


# 3 地区 × 4 年龄段，每格 1 人，d 全为 1；目标取 d 的合计 => 最优解 w=d=1，
# 因此每个非空格 ESS=1，合并 k 个相邻格后 ESS=k。
AGES = ["youth", "mid", "senior", "elder"]
REGIONS = ["East", "South", "West"]
SAMPLES = [
    {"person_id": f"{r[0]}{i}", "region": r, "age_group": a, "d": 1}
    for r in REGIONS
    for i, a in enumerate(AGES)
]
TARGETS = [
    {"dimension": "region", "code": r, "population": len(AGES)}
    for r in REGIONS
] + [
    {"dimension": "age_group", "code": a, "population": len(REGIONS)}
    for a in AGES
]


def poll(c, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = c.get(f"/jobs/{job_id}").json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.05)
    raise TimeoutError


@pytest.fixture()
def succeeded_job(client):
    c, _ = client
    assert c.put("/samples", json={"samples": SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": TARGETS}).status_code == 200
    r = c.post("/calculate")
    job = poll(c, r.json()["job_id"])
    assert job["status"] == "succeeded"
    return c, job["job_id"]


def test_grouping_feasible_returns_max_groups(succeeded_job):
    c, job_id = succeeded_job
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES,
        "keep_boundaries": [],
        "min_effective_sample_size": 2,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True
    assert body["ungroupable_segments"] == []
    # 4 个原子段、每段至少 2 个 => 最多 2 组
    assert [g["age_groups"] for g in body["groups"]] == [
        ["youth", "mid"], ["senior", "elder"]]
    assert len(body["groups"]) == 2
    # 每地区在每组都达标
    for g in body["groups"]:
        assert len(g["cells"]) == len(REGIONS)
        for cell in g["cells"]:
            assert cell["meets_threshold"] is True
            assert cell["effective_sample_size"] >= 2.0 - 1e-9
    # ESS 数值：2 个等权单人格合并 => ESS=2
    first = body["groups"][0]["cells"][0]
    assert first["sample_count"] == 2
    assert first["effective_sample_size"] == pytest.approx(2.0)
    # 无分界时 cut_after_last 全为 False
    assert all(g["cut_after_last"] is False for g in body["groups"])
    # 原子格 ESS 全量返回：12 个格，每格 1 人 ESS=1
    assert len(body["atomic_cell_ess"]) == len(REGIONS) * len(AGES)
    assert all(x["effective_sample_size"] == pytest.approx(1.0)
               for x in body["atomic_cell_ess"])


def test_grouping_threshold_one_keeps_all_atomic_groups(succeeded_job):
    c, job_id = succeeded_job
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES,
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    body = r.json()
    assert body["feasible"] is True
    assert [g["age_groups"] for g in body["groups"]] == [[a] for a in AGES]


def test_grouping_threshold_three_forces_one_big_group(succeeded_job):
    c, job_id = succeeded_job
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES,
        "keep_boundaries": [],
        "min_effective_sample_size": 3,
    })
    body = r.json()
    assert body["feasible"] is True
    # 4 个原子段、每组至少 3 个 => 无法切成两组，唯一完整分法是 1 大组
    assert [g["age_groups"] for g in body["groups"]] == [AGES]
    for cell in body["groups"][0]["cells"]:
        assert cell["effective_sample_size"] == pytest.approx(4.0)
        assert cell["sample_count"] == 4


def test_grouping_with_keep_boundary_feasible(succeeded_job):
    c, job_id = succeeded_job
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES,
        "keep_boundaries": ["mid"],
        "min_effective_sample_size": 2,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    # 分界 mid|senior 两侧各自成对，没有跨分界的组
    assert [g["age_groups"] for g in body["groups"]] == [
        ["youth", "mid"], ["senior", "elder"]]
    assert body["groups"][0]["cut_after_last"] is True
    assert body["groups"][1]["cut_after_last"] is False


def test_grouping_infeasible_reports_segments_no_partial_plan(succeeded_job):
    c, job_id = succeeded_job
    # 分界在 youth 后：左区 [youth] 单格 ESS=1 < 2，永远无法达标；
    # 右区 [mid,senior,elder] 整段 ESS=3 可以达标
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES,
        "keep_boundaries": ["youth"],
        "min_effective_sample_size": 2,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["groups"] == []  # 绝不返回部分方案
    segs = body["ungroupable_segments"]
    assert [s["age_groups"] for s in segs] == [["youth"]]
    seg = segs[0]
    # 每个地区在该区间的格 ESS=1，不达标
    assert len(seg["cell_ess"]) == len(REGIONS)
    assert all(x["effective_sample_size"] == pytest.approx(1.0)
               for x in seg["cell_ess"])
    assert all(x["meets_threshold"] is False for x in seg["cell_ess"])
    assert all(x["meets_threshold"] is False
               for x in seg["whole_segment_ess"])
    # 原子格信息仍然返回
    assert len(body["atomic_cell_ess"]) == len(REGIONS) * len(AGES)


def test_grouping_unknown_job_is_409(client):
    c, _ = client
    r = c.post("/age-grouping", json={
        "job_id": "deadbeef",
        "age_group_order": ["a"],
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    assert r.status_code == 409


def test_grouping_on_invalidated_job_is_409(succeeded_job):
    c, job_id = succeeded_job
    # 任何输入修改都会删除作业
    r = c.put("/samples", json={"samples": SAMPLES})
    assert r.status_code == 200
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES,
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    assert r.status_code == 409


def test_grouping_on_non_succeeded_job_is_409(client):
    c, _ = client
    # 冲突输入 => 作业 infeasible
    conflict_samples = [
        {"person_id": "X", "region": "R1", "age_group": "a1", "d": 1},
        {"person_id": "Y", "region": "R1", "age_group": "a2", "d": 1},
        {"person_id": "Z", "region": "R2", "age_group": "a1", "d": 1},
    ]
    conflict_targets = [
        {"dimension": "region", "code": "R1", "population": 4},
        {"dimension": "age_group", "code": "a1", "population": 1},
    ]
    c.put("/samples", json={"samples": conflict_samples})
    c.put("/targets", json={"targets": conflict_targets})
    ack = c.post("/calculate").json()
    job = poll(c, ack["job_id"])
    assert job["status"] == "infeasible"
    r = c.post("/age-grouping", json={
        "job_id": ack["job_id"],
        "age_group_order": ["a1", "a2"],
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    assert r.status_code == 409


def test_grouping_on_running_job_is_409(client, monkeypatch):
    c, _ = client
    c.put("/samples", json={"samples": SAMPLES})
    c.put("/targets", json={"targets": TARGETS})
    # 让求解阻塞，作业停留在 running 时发起分组请求
    import app.solver as solver_mod
    import threading

    started = threading.Event()

    def slow(*a, **k):
        started.set()
        time.sleep(1.0)
        return solver_mod.QPResult(status="failed", reason="hold")

    monkeypatch.setattr(solver_mod, "solve_qp", slow)
    ack = c.post("/calculate").json()
    assert started.wait(5)
    r = c.post("/age-grouping", json={
        "job_id": ack["job_id"],
        "age_group_order": AGES,
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    assert r.status_code == 409
    poll(c, ack["job_id"])  # 等作业收尾，避免线程残留


def test_grouping_age_order_mismatch_is_422(succeeded_job):
    c, job_id = succeeded_job
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES[:3],  # 缺 elder
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    assert r.status_code == 422
    r = c.post("/age-grouping", json={
        "job_id": job_id,
        "age_group_order": AGES + ["nonexistent"],
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    assert r.status_code == 422


def test_grouping_request_validation_422(client):
    c, _ = client
    base = {
        "job_id": "j1",
        "age_group_order": ["a1", "a2"],
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    }
    # 空年龄段顺序
    r = c.post("/age-grouping", json={**base, "age_group_order": []})
    assert r.status_code == 422
    # 重复年龄段
    r = c.post("/age-grouping", json={**base, "age_group_order": ["a1", "a1"]})
    assert r.status_code == 422
    # 布尔/非正阈值
    r = c.post("/age-grouping", json={**base, "min_effective_sample_size": True})
    assert r.status_code == 422
    r = c.post("/age-grouping", json={**base, "min_effective_sample_size": 0})
    assert r.status_code == 422
    # 分界引用不存在的年龄段
    r = c.post("/age-grouping", json={**base, "keep_boundaries": ["zzz"]})
    assert r.status_code == 422
    # 最后一个年龄段不能作为分界
    r = c.post("/age-grouping", json={**base, "keep_boundaries": ["a2"]})
    assert r.status_code == 422
    # 重复分界
    r = c.post("/age-grouping", json={
        **base, "age_group_order": ["a1", "a2", "a3"],
        "keep_boundaries": ["a1", "a1"]})
    assert r.status_code == 422
    # 额外字段禁止
    r = c.post("/age-grouping", json={**base, "extra": 1})
    assert r.status_code == 422
    # 空 job_id
    r = c.post("/age-grouping", json={**base, "job_id": "   "})
    assert r.status_code == 422


def test_grouping_openapi_documented(client):
    c, _ = client
    spec = c.get("/openapi.json").json()
    assert "/age-grouping" in spec["paths"]
    schema = spec["components"]["schemas"]
    assert "AgeGroupingRequest" in schema
    assert "AgeGroupingResponse" in schema
    assert c.get("/docs").status_code == 200

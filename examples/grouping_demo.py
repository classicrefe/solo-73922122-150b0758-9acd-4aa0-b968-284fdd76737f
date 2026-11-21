#!/usr/bin/env python3
"""地区对照年龄分组端到端演示（仅用标准库 urllib）。

用法:
    python3 examples/grouping_demo.py [BASE_URL]

依次演示：
  1. 提交样本/目标并等待一个 succeeded 校准作业（West 地区没有 elder 样本，
     该空格 ESS 计 0）；
  2. 可分组情形：阈值 2，返回组数最多的完整相邻合并分法与各格 ESS；
  3. 不可分组情形：保留 youth|mid 分界后 [youth] 区间永远无法达标，
     接口返回无法分组的区间诊断且不给出任何部分方案；
  4. 失效/不存在作业返回 409。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent

REGIONS = ["East", "South", "West"]
AGES = ["youth", "mid", "senior", "elder"]


def call(method: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def wait_job(job_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body = call("GET", f"/jobs/{job_id}")
        if body["status"] not in ("queued", "running"):
            return status, body
        time.sleep(0.2)
    raise TimeoutError(job_id)


def build_dataset():
    """每格 1 人、d=1；West  deliberately 没有 elder（空格）。"""
    samples = []
    pid = 0
    for r in REGIONS:
        for i, a in enumerate(AGES):
            if r == "West" and a == "elder":
                continue  # 空格：West × elder 无样本
            pid += 1
            samples.append({"person_id": f"P{pid}", "region": r,
                            "age_group": a, "d": 1})
    targets = [
        # 地区人口 = 该地区样本人数（w=d=1 时恰好可行）
        {"dimension": "region", "code": "East", "population": 4},
        {"dimension": "region", "code": "South", "population": 4},
        {"dimension": "region", "code": "West", "population": 3},
    ] + [
        {"dimension": "age_group", "code": a,
         # elder 只有 East/South 两人
         "population": 2 if a == "elder" else 3}
        for a in AGES
    ]
    return {"samples": samples}, {"targets": targets}


def main() -> None:
    samples_payload, targets_payload = build_dataset()
    print("=" * 72)
    print("准备：提交样本/目标并取得 succeeded 作业")
    print("=" * 72)
    print("PUT /samples ->", call("PUT", "/samples", samples_payload)[0])
    print("PUT /targets ->", call("PUT", "/targets", targets_payload)[0])
    _, ack = call("POST", "/calculate")
    _, job = wait_job(ack["job_id"])
    assert job["status"] == "succeeded", job
    job_id = job["job_id"]
    print("作业成功:", job_id)

    print()
    print("=" * 72)
    print("情形 1：可分组（min_ess=2，无保留分界）")
    print("=" * 72)
    req1 = json.loads((HERE / "grouping_feasible_request.json").read_text())
    req1["job_id"] = job_id
    status, body = call("POST", "/age-grouping", req1)
    print("HTTP", status, "| feasible =", body["feasible"])
    assert status == 200 and body["feasible"] is True
    assert body["ungroupable_segments"] == []
    for g in body["groups"]:
        print(f"  组 {g['group_index']}: {g['age_groups']}"
              f"{'  <保留分界>' if g['cut_after_last'] else ''}")
        for cell in g["cells"]:
            print(f"    {cell['region']:6s} ESS={cell['effective_sample_size']:.4f}"
                  f"  人数={cell['sample_count']}  达标={cell['meets_threshold']}")
            assert cell["meets_threshold"]
    # West 缺 elder 格：任何切开都会让含 elder 的组在 West 上 ESS<=1，
    # 因此唯一完整分法是四段全合并（West 整段 3 人 ESS=3 达标）
    assert [g["age_groups"] for g in body["groups"]] == [AGES]
    # 空格 ESS=0 仍在原子格信息中可见
    west_elder = [c for c in body["atomic_cell_ess"]
                  if c["region"] == "West" and c["age_group"] == "elder"][0]
    assert west_elder["effective_sample_size"] == 0.0
    assert west_elder["sample_count"] == 0
    print("  原子格 West/elder 为空格，ESS =", west_elder["effective_sample_size"])

    print()
    print("=" * 72)
    print("情形 1b：补齐空格后，组数最多的 2 组分法")
    print("=" * 72)
    full_samples = {"samples": samples_payload["samples"] + [
        {"person_id": "P99", "region": "West", "age_group": "elder", "d": 1}]}
    # 自洽目标：每个地区 4 人、每个年龄段 3 人（w=d=1）
    full_targets = {"targets": [
        {"dimension": "region", "code": r, "population": 4} for r in REGIONS
    ] + [
        {"dimension": "age_group", "code": a, "population": 3} for a in AGES
    ]}
    call("PUT", "/samples", full_samples)
    call("PUT", "/targets", full_targets)
    _, ack = call("POST", "/calculate")
    _, job = wait_job(ack["job_id"])
    assert job["status"] == "succeeded"
    job_id = job["job_id"]
    req1b = dict(req1, job_id=job_id)
    status, body = call("POST", "/age-grouping", req1b)
    print("HTTP", status, "| feasible =", body["feasible"])
    assert status == 200 and body["feasible"] is True
    for g in body["groups"]:
        print(f"  组 {g['group_index']}: {g['age_groups']}")
        for cell in g["cells"]:
            print(f"    {cell['region']:6s} ESS={cell['effective_sample_size']:.4f}"
                  f"  人数={cell['sample_count']}  达标={cell['meets_threshold']}")
    assert [g["age_groups"] for g in body["groups"]] == [
        ["youth", "mid"], ["senior", "elder"]]
    assert len(body["groups"]) == 2  # 组数最多

    print()
    print("=" * 72)
    print("情形 2：不可分组（保留 youth|mid 分界，阈值 2）")
    print("=" * 72)
    req2 = json.loads((HERE / "grouping_infeasible_request.json").read_text())
    req2["job_id"] = job_id
    status, body = call("POST", "/age-grouping", req2)
    print("HTTP", status, "| feasible =", body["feasible"])
    assert status == 200 and body["feasible"] is False
    assert body["groups"] == []  # 不开放部分方案
    for seg in body["ungroupable_segments"]:
        print(f"  无法分组区间 {seg['segment_index']}: {seg['age_groups']}")
        for c in seg["cell_ess"]:
            print(f"    原子格 {c['region']}/{c['age_group']} "
                  f"ESS={c['effective_sample_size']:.4f} 达标={c['meets_threshold']}")
        for c in seg["whole_segment_ess"]:
            print(f"    整段   {c['region']:6s} "
                  f"ESS={c['effective_sample_size']:.4f} 达标={c['meets_threshold']}")
    assert [s["age_groups"] for s in body["ungroupable_segments"]] == [["youth"]]

    print()
    print("=" * 72)
    print("情形 3：失效/不存在作业返回 409")
    print("=" * 72)
    status, body = call("POST", "/age-grouping", {
        "job_id": "nonexistent-job",
        "age_group_order": AGES,
        "keep_boundaries": [],
        "min_effective_sample_size": 1,
    })
    print("不存在作业 -> HTTP", status, "|", body.get("detail"))
    assert status == 409
    # 输入修改后旧作业永久失效
    call("PUT", "/samples", samples_payload)
    status, body = call("POST", "/age-grouping", {**req1, "job_id": job_id})
    print("输入修改后的旧作业 -> HTTP", status, "|", body.get("detail"))
    assert status == 409

    print("\n全部分组演示断言通过 ✔")


if __name__ == "__main__":
    main()

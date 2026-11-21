"""FastAPI 应用：样本/目标更正、后台权重计算、结果取用。"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .db import Database
from .jobs import JobManager
from .grouping import plan_grouping
from .schemas import (
    AgeGroupingRequest,
    AgeGroupingResponse,
    AgeGroupBlockOut,
    CellESSOut,
    GroupCellESSOut,
    JobResult,
    SamplesPayload,
    TargetsPayload,
    TargetTotalsItem,
    UngroupableSegmentOut,
    WeightItem,
    WholeSegmentESSOut,
)

DB_PATH = os.environ.get("WEIGHTS_DB", "/data/weights.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    app.state.db = Database(DB_PATH)
    app.state.jobs = JobManager(app.state.db)
    yield
    app.state.jobs.shutdown()
    app.state.db.close()


app = FastAPI(
    title="回访问卷样本权重校正 API",
    version="1.0.0",
    description=(
        "根据地区/年龄段人口总量，对至多 200 人的基础权重 d 做二次规划校正，"
        "满足 0.5d ≤ w ≤ 2d 且目标人群权重合计等于人口总量，"
        "最小化 Σ(w−d)²/d。任何样本/目标修改都会令旧结果与在途计算永久失效。"
    ),
    lifespan=lifespan,
)


def db() -> Database:
    return app.state.db


def jobs() -> JobManager:
    return app.state.jobs


# ---------------------------------------------------------------- 输入更正
@app.put("/samples", status_code=200, tags=["输入数据"],
         summary="整体替换样本（至多 200 人；使所有旧结果与在途计算失效）")
def put_samples(payload: SamplesPayload):
    samples = [s.model_dump() for s in payload.samples]
    epoch = db().replace_samples(samples)
    return {"epoch": epoch, "count": len(samples),
            "message": "样本已更新，旧结果与在途计算已永久失效，须重新计算"}


@app.get("/samples", tags=["输入数据"], summary="查看当前样本")
def get_samples():
    snap = db().snapshot()
    return {"epoch": snap["epoch"], "samples": snap["samples"]}


@app.put("/targets", status_code=200, tags=["输入数据"],
         summary="整体替换人口目标（至多 12 项；使所有旧结果与在途计算失效）")
def put_targets(payload: TargetsPayload):
    targets = [t.model_dump() for t in payload.targets]
    epoch = db().replace_targets(targets)
    return {"epoch": epoch, "count": len(targets),
            "message": "目标已更新，旧结果与在途计算已永久失效，须重新计算"}


@app.get("/targets", tags=["输入数据"], summary="查看当前人口目标")
def get_targets():
    snap = db().snapshot()
    return {"epoch": snap["epoch"], "targets": snap["targets"]}


# ---------------------------------------------------------------- 计算/结果
@app.post("/calculate", status_code=202, tags=["计算"],
          summary="提交后台权重计算（不阻塞，返回 job_id）")
def calculate():
    snap = db().snapshot()
    if not snap["samples"]:
        raise HTTPException(422, "当前没有样本，无法计算：请先 PUT /samples")
    if not snap["targets"]:
        raise HTTPException(422, "当前没有目标，无法计算：请先 PUT /targets")
    job = jobs().submit()
    return JSONResponse(
        status_code=202,
        content={"job_id": job["job_id"], "epoch": job["epoch"],
                 "status": job["status"],
                 "poll": f"/jobs/{job['job_id']}"},
    )


def _serialize_job(row: dict) -> dict:
    result = json.loads(row["result"]) if row["result"] else None
    weights = totals = None
    objective = l1 = lmax = None
    iis = None
    if row["status"] == "succeeded" and result:
        weights = [WeightItem(**x).model_dump() for x in result["weights"]]
        totals = [TargetTotalsItem(**x).model_dump() for x in result["target_totals"]]
        objective = result["objective"]
        l1 = result["residual_abs_sum"]
        lmax = result["residual_max_abs"]
    elif row["status"] == "infeasible" and result:
        iis = result["infeasible_targets"]

    out = JobResult(
        job_id=row["job_id"],
        epoch=row["epoch"],
        status=row["status"],
        submitted_at=row["submitted_at"],
        finished_at=row["finished_at"],
        input_snapshot=json.loads(row["input_snapshot"]),
        error=row["error"],
        weights=weights,
        objective=objective,
        residual_abs_sum=l1,
        residual_max_abs=lmax,
        target_totals=totals,
        infeasible_targets=iis,
    )
    return out.model_dump(exclude_none=False)


@app.get("/jobs", tags=["计算"], summary="列出最近作业")
def list_jobs():
    return {"epoch": db().get_epoch(), "jobs": db().list_jobs()}


@app.get("/jobs/{job_id}", tags=["计算"],
         summary="查询作业状态与完整结果（仅成功时返回权重，失败/中断无半份结果）")
def get_job(job_id: str):
    row = db().get_job(job_id)
    if row is None:
        raise HTTPException(404, "作业不存在或已因输入变更而永久失效")
    return _serialize_job(row)


@app.get("/result", tags=["计算"],
         summary="取用与当前输入一致的完整权重；无可用完整结果时返回 409")
def get_result():
    row = db().latest_succeeded_job()
    if row is None:
        raise HTTPException(
            409,
            "当前没有与现有输入一致的完整权重结果（尚未计算、计算中、冲突、"
            "数值失败或进程重启均不开放半份结果）",
        )
    data = _serialize_job(dict(row))
    # 一致性核对：作业快照必须与当前输入逐字段一致
    current = db().snapshot()
    snap = data["input_snapshot"]
    if snap.get("samples") != current["samples"] or \
            snap.get("targets") != current["targets"] or \
            snap.get("epoch") != current["epoch"]:
        # 理论上不会发生（epoch 变更即清除作业），双保险
        raise HTTPException(409, "结果与当前输入不一致，已拒绝取用，请重新计算")
    return data


# ---------------------------------------------------------------- 年龄分组
@app.post(
    "/age-grouping",
    response_model=AgeGroupingResponse,
    tags=["地区对照分组"],
    summary=(
        "为地区对照报告求共用年龄分组：只合并相邻年龄段、保留指定分界，"
        "每个地区在每个合并组均达到最低有效样本量"
    ),
)
def age_grouping(payload: AgeGroupingRequest):
    row = db().get_job(payload.job_id)
    if row is None:
        # 作业不存在，或已因输入变更被永久删除（失效）
        raise HTTPException(409, "作业不存在或已因输入变更而永久失效")
    if row["status"] != "succeeded":
        raise HTTPException(
            409,
            f"作业状态为 {row['status']}，仅成功（succeeded）作业可用于年龄分组",
        )

    snapshot = json.loads(row["input_snapshot"])
    samples = snapshot["samples"]
    result = json.loads(row["result"]) if row["result"] else None
    if not result or "weights" not in result:
        # 双保险：成功作业必须带完整权重，否则视为无完整结果
        raise HTTPException(409, "作业缺少完整权重结果，无法用于年龄分组")

    # 年龄段顺序必须与作业快照中的年龄段集合完全一致（顺序由请求给定）
    snapshot_ages = {s["age_group"] for s in samples}
    if set(payload.age_group_order) != snapshot_ages or \
            len(payload.age_group_order) != len(snapshot_ages):
        raise HTTPException(
            422,
            "age_group_order 必须与作业快照中的年龄段集合完全一致；"
            f"快照年龄段={sorted(snapshot_ages)}",
        )

    weights_by_person = {item["person_id"]: float(item["w"])
                         for item in result["weights"]}
    # 地区按其在快照中首次出现的顺序排列，保证输出确定性
    regions: list[str] = []
    for s in samples:
        if s["region"] not in regions:
            regions.append(s["region"])

    # 汇总每个 (地区, 年龄段) 格内的个人权重
    cell_weights: dict[tuple[str, str], list[float]] = {}
    for s in samples:
        w = weights_by_person.get(s["person_id"])
        if w is None:
            raise HTTPException(409, "作业结果缺少样本权重，无法用于年龄分组")
        cell_weights.setdefault((s["region"], s["age_group"]), []).append(w)

    plan = plan_grouping(
        age_order=payload.age_group_order,
        regions=regions,
        cell_weights=cell_weights,
        keep_boundaries=payload.keep_boundaries,
        min_ess=payload.min_effective_sample_size,
    )

    atomic = [
        CellESSOut(
            region=c.region,
            age_group=c.age_group,
            effective_sample_size=c.ess,
            sample_count=c.sample_count,
            meets_threshold=c.meets_threshold,
        )
        for c in plan.atomic_cell_ess
    ]
    groups_out = [
        AgeGroupBlockOut(
            group_index=g.group_index,
            age_groups=g.age_groups,
            cut_after_last=g.cut_after_last,
            cells=[
                GroupCellESSOut(
                    region=x.region,
                    effective_sample_size=x.ess,
                    sample_count=x.sample_count,
                    meets_threshold=x.meets_threshold,
                )
                for x in g.cells
            ],
        )
        for g in plan.groups
    ]
    segments_out = [
        UngroupableSegmentOut(
            segment_index=seg.segment_index,
            age_groups=seg.age_groups,
            cell_ess=[
                CellESSOut(
                    region=c.region,
                    age_group=c.age_group,
                    effective_sample_size=c.ess,
                    sample_count=c.sample_count,
                    meets_threshold=c.meets_threshold,
                )
                for c in seg.cell_ess
            ],
            whole_segment_ess=[
                WholeSegmentESSOut(
                    region=x.region,
                    effective_sample_size=x.ess,
                    sample_count=x.sample_count,
                    meets_threshold=x.meets_threshold,
                )
                for x in seg.whole_segment_ess
            ],
        )
        for seg in plan.ungroupable_segments
    ]

    return AgeGroupingResponse(
        job_id=payload.job_id,
        feasible=plan.feasible,
        age_group_order=plan.age_group_order,
        regions=plan.regions,
        keep_boundaries=plan.keep_boundaries,
        min_effective_sample_size=plan.min_ess,
        groups=groups_out,
        ungroupable_segments=segments_out,
        atomic_cell_ess=atomic,
    )


@app.get("/health", tags=["系统"], summary="健康检查")
def health():
    return {"status": "ok", "epoch": db().get_epoch()}

"""请求/响应模型与严格输入校验。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_SAMPLES = 200
MAX_TARGETS = 12
MAX_D = 100
MAX_CODE_LEN = 64


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _strict_int(name: str, minimum: int, maximum: int | None = None):
    # mode="before"：在 Pydantic 类型强转之前拦截，
    # 否则 True 会先被强转为 1，布尔冒充整数将无法识别。
    def _v(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是整数")
        if value < minimum:
            raise ValueError(f"{name} 必须 >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} 必须 <= {maximum}")
        return value
    return field_validator(name, mode="before")(_v)


def _strict_code(name: str):
    def _v(value):
        if isinstance(value, bool) or not isinstance(value, str):
            raise ValueError(f"{name} 必须是非空字符串")
        value = value.strip()
        if not value:
            raise ValueError(f"{name} 不能为空字符串")
        if len(value) > MAX_CODE_LEN:
            raise ValueError(f"{name} 长度须 <= {MAX_CODE_LEN}")
        return value
    return field_validator(name, mode="before")(_v)


class SampleIn(StrictModel):
    person_id: str = Field(description="样本人员编号，非空，单次提交内唯一")
    region: str = Field(description="地区代码")
    age_group: str = Field(description="年龄段代码")
    d: int = Field(description="正整数基础权重，1 <= d <= 100")

    _v_id = _strict_code("person_id")
    _v_region = _strict_code("region")
    _v_age = _strict_code("age_group")
    _v_d = _strict_int("d", 1, MAX_D)


class SamplesPayload(StrictModel):
    samples: list[SampleIn] = Field(description=f"至多 {MAX_SAMPLES} 人")

    @field_validator("samples")
    @classmethod
    def _check(cls, v: list[SampleIn]):
        if len(v) > MAX_SAMPLES:
            raise ValueError(f"样本数至多 {MAX_SAMPLES} 人")
        ids = [s.person_id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("person_id 在本次提交中存在重复")
        return v


class TargetIn(StrictModel):
    dimension: Literal["region", "age_group"] = Field(
        description="目标维度：region 或 age_group"
    )
    code: str = Field(description="维度取值代码（对应样本的 region/age_group）")
    population: int = Field(description="非负整数人口总量")

    _v_code = _strict_code("code")
    _v_pop = _strict_int("population", 0)


class TargetsPayload(StrictModel):
    targets: list[TargetIn] = Field(description=f"至多 {MAX_TARGETS} 项")

    @field_validator("targets")
    @classmethod
    def _check(cls, v: list[TargetIn]):
        if len(v) > MAX_TARGETS:
            raise ValueError(f"目标至多 {MAX_TARGETS} 项")
        keys = [(t.dimension, t.code) for t in v]
        if len(set(keys)) != len(keys):
            raise ValueError("(dimension, code) 在本次提交中存在重复")
        return v


class WeightItem(BaseModel):
    person_id: str
    d: int
    w: float
    lower: float
    upper: float


class TargetTotalsItem(BaseModel):
    dimension: str
    code: str
    population: int
    weighted_total: float
    abs_error: float


class JobResult(BaseModel):
    job_id: str
    epoch: int
    status: str
    submitted_at: str
    finished_at: str | None = None
    input_snapshot: dict
    error: str | None = None
    # status=succeeded 时完整给出，其余状态为 None，绝不开放半份结果
    weights: list[WeightItem] | None = None
    objective: float | None = None
    residual_abs_sum: float | None = None
    residual_max_abs: float | None = None
    target_totals: list[TargetTotalsItem] | None = None
    # status=infeasible 时给出最小冲突目标子集
    infeasible_targets: list[dict] | None = None


# ============================================================
# 地区对照报告：共用年龄分组
# ====================================================

class AgeGroupingRequest(StrictModel):
    job_id: str = Field(description="有效的成功校准作业号（status=succeeded）")
    age_group_order: list[str] = Field(
        description="作业快照中各年龄段的顺序；必须与快照中的年龄段集合完全一致"
    )
    keep_boundaries: list[str] = Field(
        default_factory=list,
        description=(
            "必须保留的分界：列出的每个年龄段之后强制切开，"
            "其两侧年龄段不得并入同一组；不允许使用最后一个年龄段"
        ),
    )
    min_effective_sample_size: float = Field(
        description="最低有效样本量，正数；每个地区在每个合并组都须达到该值"
    )

    _v_job = _strict_code("job_id")

    @field_validator("age_group_order")
    @classmethod
    def _v_order(cls, v: list[str]):
        if not v:
            raise ValueError("age_group_order 至少包含一个年龄段")
        cleaned: list[str] = []
        for code in v:
            if isinstance(code, bool) or not isinstance(code, str):
                raise ValueError("年龄段代码必须是字符串")
            code = code.strip()
            if not code:
                raise ValueError("年龄段代码不能为空字符串")
            if len(code) > MAX_CODE_LEN:
                raise ValueError(f"年龄段代码长度须 <= {MAX_CODE_LEN}")
            cleaned.append(code)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("age_group_order 中存在重复年龄段")
        return cleaned

    @field_validator("keep_boundaries")
    @classmethod
    def _v_boundaries(cls, v: list[str]):
        cleaned: list[str] = []
        for code in v:
            if isinstance(code, bool) or not isinstance(code, str):
                raise ValueError("保留分界必须是年龄段代码字符串")
            code = code.strip()
            if not code:
                raise ValueError("保留分界不能为空字符串")
            if len(code) > MAX_CODE_LEN:
                raise ValueError(f"保留分界代码长度须 <= {MAX_CODE_LEN}")
            cleaned.append(code)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("keep_boundaries 中存在重复分界")
        return cleaned

    @field_validator("min_effective_sample_size", mode="before")
    @classmethod
    def _v_min_ess(cls, v):
        # 在 Pydantic 强转之前拦截：布尔是 int 子类，不拦截会被当成 1.0
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("min_effective_sample_size 必须是正数")
        if v <= 0:
            raise ValueError("min_effective_sample_size 必须 > 0")
        return float(v)

    @model_validator(mode="after")
    def _cross_check(self):
        unknown = [b for b in self.keep_boundaries
                   if b not in self.age_group_order]
        if unknown:
            raise ValueError(
                "keep_boundaries 含 age_group_order 中不存在的年龄段: "
                + ", ".join(unknown)
            )
        if self.age_group_order[-1] in self.keep_boundaries:
            raise ValueError(
                "最后一个年龄段之后没有分界，不能把它列为 keep_boundaries"
            )
        return self


class CellESSOut(BaseModel):
    region: str
    age_group: str
    effective_sample_size: float
    sample_count: int
    meets_threshold: bool


class GroupCellESSOut(BaseModel):
    region: str
    effective_sample_size: float
    sample_count: int
    meets_threshold: bool


class AgeGroupBlockOut(BaseModel):
    group_index: int
    age_groups: list[str]
    cut_after_last: bool
    cells: list[GroupCellESSOut]


class WholeSegmentESSOut(BaseModel):
    region: str
    effective_sample_size: float
    sample_count: int
    meets_threshold: bool


class UngroupableSegmentOut(BaseModel):
    segment_index: int
    age_groups: list[str]
    # 段内每个原子年龄段 × 地区的格内 ESS（空格计 0），用于定位短板
    cell_ess: list[CellESSOut]
    whole_segment_ess: list[WholeSegmentESSOut]


class AgeGroupingResponse(BaseModel):
    job_id: str
    feasible: bool
    age_group_order: list[str]
    regions: list[str]
    keep_boundaries: list[str]
    min_effective_sample_size: float
    # feasible=True：组数最多的完整分法；feasible=False 为空列表，
    # 不返回任何部分方案
    groups: list[AgeGroupBlockOut]
    # feasible=False：按保留分界切开后无法分组的区间；feasible=True 为空
    ungroupable_segments: list[UngroupableSegmentOut]
    # 全部原子 (地区, 年龄段) 格的 ESS，两种情形都返回
    atomic_cell_ess: list[CellESSOut]

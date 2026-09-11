"""Frozen vision-scan contracts. No OpenAI, Streamlit, or matplotlib imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

REASON_MAX_CHARS = 160
MISSING_EVIDENCE_MAX_ITEMS = 16
MISSING_EVIDENCE_ITEM_MAX_CHARS = 200
VIOLATED_RULE_MAX_ITEMS = 16
LOOKBACK_BARS_MIN = 2
LOOKBACK_BARS_UI_MAX = 400
DEFAULT_LOOKBACK_BARS = 100
DEFAULT_MOVING_AVERAGES: tuple[int, ...] = (10, 20, 50)
RESPONSE_SCHEMA_NAME = "vision_batch_classification"
RESPONSE_SCHEMA_PATH = Path(__file__).with_name("response_schema.json")

Verdict = Literal["match", "no_match", "uncertain"]
CandidateStatus = Literal["pending", "completed", "error", "skipped"]
RunStatus = Literal["running", "completed", "partial", "failed", "cancelled", "dry_run"]
ReviewJudgment = Literal["agree", "disagree", "unsure"]
ReviewState = Literal["unreviewed", "agree", "disagree", "unsure"]
ScanMode = Literal["live", "dry_run", "demo"]
RuleKind = Literal["required", "preferred", "disqualifier"]
ExamplePolarity = Literal["positive", "negative"]
ExampleType = Literal["market_window", "image"]
ImageDetail = Literal["auto", "low", "high"]
BlockKind = Literal["text", "image"]
BlockPurpose = Literal["instructions", "setup", "example", "candidate"]
ArtifactKind = Literal["candidate", "example"]
SkipKind = Literal["short_window", "stale", "unavailable_input"]
SortKey = Literal["ticker", "asof_date", "match_strength", "candidate_id", "arrival"]
ErrorKind = Literal[
    "auth",
    "config",
    "rate_limit",
    "transport",
    "provider",
    "refusal",
    "timeout",
    "incomplete",
    "oversize",
    "invalid_output",
    "unavailable_input",
    "stale_snapshot",
    "short_window",
    "profile_conflict",
    "market_cap_unavailable",
    "cancelled",
    "changed_input",
    "run_locked",
]

# Refusal, timeout, invalid_output, and unavailable_input never become no_match.
NON_MATCH_ERROR_KINDS: frozenset[str] = frozenset(
    {"refusal", "timeout", "invalid_output", "unavailable_input"}
)


class VisionError(Exception):
    """Base error for the vision pipeline."""


class ChartProfileConflictError(VisionError):
    def __init__(self, conflicts: tuple["ProfileFieldConflict", ...]) -> None:
        self.conflicts = tuple(conflicts)
        parts = [
            f"{c.field}: " + ", ".join(f"{sid}={value!r}" for sid, value in c.setup_values)
            for c in self.conflicts
        ]
        super().__init__("Incompatible chart profile across enabled setups: " + "; ".join(parts))


class OversizeRequestError(VisionError):
    def __init__(self, message: str, *, estimates: "RequestBudgetEstimate | None" = None) -> None:
        super().__init__(message)
        self.estimates = estimates


class SemanticValidationError(VisionError):
    def __init__(self, message: str, *, issues: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.issues = tuple(issues)


class FrozenInputConflictError(VisionError):
    pass


class RunLockedError(VisionError):
    pass


class StaleSnapshotError(VisionError):
    pass


class InsufficientLastNError(VisionError):
    """Globally insufficient last-N capacity. Do not rebuild implicitly."""

    def __init__(self, needed_bars: int) -> None:
        self.needed_bars = int(needed_bars)
        super().__init__(
            f"Derived last-N bars do not cover lookback_bars={self.needed_bars}. "
            f"Run `ns rebuild-last100 --window-size {self.needed_bars}` "
            "and retry. The pipeline will not rebuild this dataset implicitly."
        )


class ClassificationError(VisionError):
    def __init__(self, error: "AttemptError") -> None:
        super().__init__(error.message)
        self.error = error


@dataclass(frozen=True)
class ProfileFieldConflict:
    field: str
    setup_values: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class ChartProfile:
    timeframe: str
    lookback_bars: int
    volume: bool
    moving_averages: tuple[int, ...]

    def __post_init__(self) -> None:
        if str(self.timeframe).strip().lower() != "1d":
            raise VisionError(f"Unsupported timeframe {self.timeframe!r}. MVP accepts only '1d'.")
        if int(self.lookback_bars) < LOOKBACK_BARS_MIN:
            raise VisionError(f"lookback_bars must be >= {LOOKBACK_BARS_MIN}")
        object.__setattr__(self, "lookback_bars", int(self.lookback_bars))
        object.__setattr__(self, "volume", bool(self.volume))
        mas = tuple(int(x) for x in self.moving_averages)
        if any(n <= 0 for n in mas):
            raise VisionError("moving_averages must be positive integers")
        object.__setattr__(self, "moving_averages", mas)
        object.__setattr__(self, "timeframe", "1d")


@dataclass(frozen=True)
class MaAvailability:
    period: int
    available: bool
    first_valid_index: int | None
    note: str


@dataclass(frozen=True)
class ScanConfig:
    model: str
    timeout_seconds: float = 60.0
    max_output_tokens: int = 4096
    batch_size: int = 10
    max_concurrency: int = 2
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    retry_backoff_max_seconds: float = 20.0
    jitter_ratio: float = 0.25
    max_request_bytes: int = 2_500_000
    max_images_per_request: int = 24
    max_in_flight_batches: int = 2
    image_detail: ImageDetail = "auto"
    mode: ScanMode = "live"
    request_label: str | None = None

    def __post_init__(self) -> None:
        if not str(self.model).strip():
            raise VisionError("ScanConfig.model must be an explicit model id")
        if int(self.batch_size) < 1:
            raise VisionError("batch_size must be >= 1")
        if int(self.max_concurrency) < 1:
            raise VisionError("max_concurrency must be >= 1")
        if int(self.max_in_flight_batches) < 1:
            raise VisionError("max_in_flight_batches must be >= 1")
        if self.image_detail not in {"auto", "low", "high"}:
            raise VisionError("image_detail must be auto, low, or high")
        if self.mode not in {"live", "dry_run", "demo"}:
            raise VisionError("mode must be live, dry_run, or demo")
        if float(self.jitter_ratio) < 0:
            raise VisionError("jitter_ratio must be >= 0")


@dataclass(frozen=True)
class Bar:
    date: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


@dataclass(frozen=True)
class BarWindow:
    ticker: str
    timeframe: str
    bars: tuple[Bar, ...]
    provenance: str
    source_digest: str

    @property
    def asof_date(self) -> date | None:
        if not self.bars:
            return None
        return self.bars[-1].date

    @property
    def bar_count(self) -> int:
        return len(self.bars)


@dataclass(frozen=True)
class FeatureValue:
    """Latest-session features. Units are part of the contract, not display sugar."""

    close: float | None  # USD / share
    dollar_vol_avg_20: float | None  # USD / day
    adr_pct_20: float | None  # fraction; 0.04 = 4%

    def units(self) -> dict[str, str]:
        return {
            "close": "USD/share",
            "dollar_vol_avg_20": "USD/day",
            "adr_pct_20": "fraction",
        }


@dataclass(frozen=True)
class SnapshotRule:
    rule_id: str
    kind: RuleKind
    index: int
    text: str


@dataclass(frozen=True)
class FilterSnapshot:
    min_price: float | None = None
    min_dollar_vol_20d: float | None = None
    min_adr_pct_20: float | None = None
    min_market_cap: float | None = None
    max_market_cap: float | None = None

    def requires_market_cap(self) -> bool:
        return self.min_market_cap is not None or self.max_market_cap is not None


@dataclass(frozen=True)
class SetupSnapshot:
    setup_id: str
    name: str
    enabled: bool
    timeframe: str
    lookback_bars: int
    description: str
    llm_notes: str
    rules: tuple[SnapshotRule, ...]
    filters: FilterSnapshot
    chart_volume: bool
    chart_moving_averages: tuple[int, ...]
    yaml_bytes: bytes
    yaml_digest: str
    content_digest: str
    example_ids: tuple[str, ...]

    def rules_of(self, kind: RuleKind) -> tuple[SnapshotRule, ...]:
        return tuple(r for r in self.rules if r.kind == kind)

    def required_rule_ids(self) -> frozenset[str]:
        return frozenset(r.rule_id for r in self.rules if r.kind == "required")


@dataclass(frozen=True)
class ExampleInput:
    setup_id: str
    example_id: str
    polarity: ExamplePolarity
    quality: str | None
    note: str
    type: ExampleType
    ticker: str | None
    asof_date: date | None
    timeframe: str
    image_bytes: bytes | None
    window: BarWindow | None
    raw_digest: str

    @property
    def scoped_id(self) -> str:
        return f"{self.setup_id}/{self.example_id}"


@dataclass(frozen=True)
class CandidateInput:
    candidate_id: str
    ticker: str
    window: BarWindow
    features: FeatureValue
    eligible_setup_ids: tuple[str, ...]
    source_digest: str
    asof_date: date
    profile: ChartProfile


@dataclass(frozen=True)
class InputSkip:
    ticker: str
    kind: SkipKind
    message: str
    asof_date: date | None = None
    bar_count: int | None = None
    features: FeatureValue | None = None


@dataclass(frozen=True)
class ScanDiagnostics:
    """Counts that were independently recorded. Missing values stay None."""

    universe_count: int | None = None
    input_count: int | None = None
    eligible_union_count: int | None = None
    per_setup_eligible_counts: tuple[tuple[str, int], ...] | None = None
    not_eligible_count: int | None = None
    skipped_count: int | None = None
    unavailable: tuple[str, ...] = ()


COMPILER_ID = "snapshot_request_compiler_v1"


@dataclass(frozen=True)
class CompilerSnapshot:
    """Frozen prompt/schema used to compile and resume a run."""

    compiler_id: str
    instructions: str
    schema_name: str
    json_schema: Mapping[str, Any]
    fingerprint: str


@dataclass(frozen=True)
class PreparedScan:
    profile: ChartProfile
    setups: tuple[SetupSnapshot, ...]
    examples: tuple[ExampleInput, ...]
    candidates: tuple[CandidateInput, ...]
    skips: tuple[InputSkip, ...]
    config: ScanConfig
    diagnostics: ScanDiagnostics
    prepared_at: datetime
    semantic_digest: str
    raw_digest: str
    global_filters_yaml_bytes: bytes
    global_filters_raw_digest: str
    compiler: CompilerSnapshot | None = None


@dataclass(frozen=True)
class RenderedImage:
    png_bytes: bytes
    width: int
    height: int
    sha256: str
    media_type: str = "image/png"


@dataclass(frozen=True)
class ChartArtifact:
    artifact_id: str
    kind: ArtifactKind
    image: RenderedImage
    title: str
    profile: ChartProfile | None
    ma_availability: tuple[MaAvailability, ...]
    final_session_date: date | None
    ticker: str | None = None
    candidate_id: str | None = None
    setup_id: str | None = None
    example_id: str | None = None
    source: Literal["rendered", "upload"] = "rendered"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    relative_path: str
    sha256: str
    width: int
    height: int
    byte_length: int


@dataclass(frozen=True)
class RequestBlock:
    kind: BlockKind
    purpose: BlockPurpose
    text: str | None = None
    artifact_id: str | None = None
    image_digest: str | None = None


@dataclass(frozen=True)
class RequestBudgetEstimate:
    reference_image_count: int
    candidate_image_count: int
    total_image_count: int
    text_chars: int
    transport_bytes_estimate: int
    anticipated_output_tokens_estimate: int
    notes: str = (
        "All byte, image, and token figures are conservative estimates, not billed usage."
    )


@dataclass(frozen=True)
class CompiledRequest:
    batch_id: str
    candidate_ids: tuple[str, ...]
    setup_ids: tuple[str, ...]
    blocks: tuple[RequestBlock, ...]
    json_schema: Mapping[str, Any]
    schema_name: str
    fingerprint: str
    estimates: RequestBudgetEstimate
    model: str
    max_output_tokens: int = 4096
    image_detail: ImageDetail = "auto"
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class Assessment:
    setup_id: str
    verdict: Verdict
    match_strength: int | None
    reason: str
    violated_required_rule_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...]


@dataclass(frozen=True)
class CandidateClassification:
    candidate_id: str
    assessments: tuple[Assessment, ...]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class AttemptError:
    kind: ErrorKind
    message: str
    retryable: bool
    http_status: int | None = None
    provider_code: str | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class ClassificationAttempt:
    attempt_id: str
    batch_id: str
    batch_fingerprint: str
    candidate_ids: tuple[str, ...]
    setup_ids: tuple[str, ...]
    started_at: datetime
    ended_at: datetime | None
    provider: str
    model: str | None
    response_id: str | None
    usage: TokenUsage | None
    retry_after_seconds: float | None
    latency_ms: int | None
    error: AttemptError | None
    sanitized_output: Mapping[str, Any] | None
    results: tuple[CandidateClassification, ...] | None
    accepted: bool


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    status: CandidateStatus
    ticker: str
    asof_date: date
    features: FeatureValue
    eligible_setup_ids: tuple[str, ...]
    assessments: tuple[Assessment, ...]
    artifact_id: str | None
    error: AttemptError | None
    attempt_id: str | None
    source_digest: str


@dataclass(frozen=True)
class RunSummary:
    """Terminal accounting for a scan.

    ``candidates_completed`` means successfully classified only, never skipped
    or failed. ``candidates_total`` is scan candidates plus input skips, so
    ``completed + error + skipped + pending == total``.
    """

    candidates_total: int
    candidates_completed: int
    candidates_error: int
    candidates_skipped: int
    candidates_pending: int
    setup_matches: int
    attempts: int
    usage: TokenUsage | None
    status: RunStatus
    synthetic: bool = False


@dataclass(frozen=True)
class ScanOutcome:
    run_id: str
    status: RunStatus
    summary: RunSummary
    diagnostics: ScanDiagnostics
    semantic_digest: str
    raw_digest: str


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    semantic_digest: str
    raw_digest: str
    config: ScanConfig
    profile: ChartProfile
    lock_holder: str | None = None
    synthetic: bool = False


@dataclass(frozen=True)
class MatchedSetupBadge:
    setup_id: str
    setup_name: str
    match_strength: int


@dataclass(frozen=True)
class AssessmentSummary:
    setup_id: str
    setup_name: str
    verdict: Verdict
    match_strength: int | None
    reason: str


@dataclass(frozen=True)
class ReviewRecord:
    review_id: str
    run_id: str
    candidate_id: str
    setup_id: str | None
    judgment: ReviewJudgment
    note: str
    created_at: datetime
    supersedes_review_id: str | None


@dataclass(frozen=True)
class PageRequest:
    page: int = 1
    page_size: int = 50

    def __post_init__(self) -> None:
        if int(self.page) < 1:
            raise VisionError("page is 1-based and must be >= 1")
        if int(self.page_size) < 1:
            raise VisionError("page_size must be >= 1")
        object.__setattr__(self, "page", int(self.page))
        object.__setattr__(self, "page_size", int(self.page_size))


@dataclass(frozen=True)
class SortSpec:
    key: SortKey = "match_strength"
    descending: bool = True


@dataclass(frozen=True)
class ResultQuery:
    run_id: str
    setup_ids: tuple[str, ...] = ()
    verdicts: tuple[Verdict, ...] = ()
    min_match_strength: int | None = None
    max_match_strength: int | None = None
    review_state: ReviewState | Literal["any"] = "any"
    sort: SortSpec = field(default_factory=SortSpec)
    page: PageRequest = field(default_factory=PageRequest)


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    ticker: str
    asof_date: date
    status: CandidateStatus
    features: FeatureValue
    matched_setups: tuple[MatchedSetupBadge, ...]
    assessments: tuple[AssessmentSummary, ...]
    review_state: ReviewState
    current_review: ReviewRecord | None
    chart_ref: ArtifactRef | None
    error: AttemptError | None


@dataclass(frozen=True)
class ResultPage:
    items: tuple[CandidateRow, ...]
    page: int
    page_size: int
    total_candidates: int
    total_setup_matches: int
    has_next: bool
    has_prev: bool
    next_page: int | None
    prev_page: int | None


@dataclass(frozen=True)
class ResultCounts:
    candidates: int
    completed: int
    error: int
    skipped: int
    pending: int
    setup_matches: int
    unreviewed: int
    reviewed: int


@dataclass(frozen=True)
class CandidateDetail:
    row: CandidateRow
    assessments: tuple[Assessment, ...]
    eligible_setup_ids: tuple[str, ...]
    window: BarWindow | None
    artifact: ArtifactRef | None
    reviews: tuple[ReviewRecord, ...]
    attempts: tuple[ClassificationAttempt, ...]


def load_response_schema() -> dict[str, Any]:
    return __import__("json").loads(RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))

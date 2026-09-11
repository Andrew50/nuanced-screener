"""ResultReader: one row per candidate, setup-aware filters, paging, and exports."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

from .reviews import FilesystemReviewStore
from .store import Clock, FilesystemRunStore, format_dt
from .types import (
    ArtifactRef,
    Assessment,
    AssessmentSummary,
    CandidateDetail,
    CandidateInput,
    CandidateResult,
    CandidateRow,
    CandidateStatus,
    FeatureValue,
    MatchedSetupBadge,
    PageRequest,
    PreparedScan,
    ResultCounts,
    ResultPage,
    ResultQuery,
    ReviewRecord,
    ReviewState,
    SetupSnapshot,
    SortSpec,
    StoredRun,
    VisionError,
)

LIST_JOIN = ";"
ADR_DISPLAY_MULTIPLIER = 100.0


@dataclass(frozen=True)
class QueryCounts:
    """Distinct counts the UI needs. Unknown eligibility diagnostics stay off this object."""

    candidates: int
    completed: int
    pending: int
    error: int
    skipped: int
    any_match: int
    setup_matches: int
    uncertain: int
    no_match_only: int
    unreviewed: int
    reviewed: int
    reviewed_pairs: int
    per_setup_matches: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class QueryExport:
    """All rows matching the active query, not just the visible page."""

    candidates: tuple[dict[str, Any], ...]
    assessments: tuple[dict[str, Any], ...]
    flattening: str


@dataclass(frozen=True)
class NeighborPosition:
    candidate_id: str
    prev_id: str | None
    next_id: str | None
    position: int
    total: int
    page: int


def format_adr_display_pct(fraction: float | None) -> float | None:
    if fraction is None:
        return None
    return float(fraction) * ADR_DISPLAY_MULTIPLIER


def flatten_list(values: Sequence[str] | None) -> str | None:
    if not values:
        return None
    return LIST_JOIN.join(str(v) for v in values)


class FilesystemResultReader:
    """Reads committed store records. Does not call the classifier, renderer, or data vendors."""

    def __init__(
        self,
        scan_root: Path,
        *,
        store: FilesystemRunStore | None = None,
        reviews: FilesystemReviewStore | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.scan_root = Path(scan_root)
        self.store = store or FilesystemRunStore(self.scan_root, clock=clock)
        self.reviews = reviews or FilesystemReviewStore(self.scan_root, clock=clock, store=self.store)

    def list_runs(self) -> tuple[StoredRun, ...]:
        return self.store.list_runs()

    def load_run(self, run_id: str) -> StoredRun:
        return self.store.load_run(run_id)

    def load_frozen_inputs(self, run_id: str) -> PreparedScan:
        return self.store.load_frozen_inputs(run_id)

    def query(
        self,
        q: ResultQuery,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
    ) -> ResultPage:
        rows = self._filtered_rows(q, ticker_query=ticker_query, statuses=statuses)
        page = q.page.page
        size = q.page.page_size
        total = len(rows)
        start = (page - 1) * size
        chunk = rows[start : start + size]
        has_next = start + size < total
        has_prev = page > 1 and total > 0
        setup_matches = sum(len(r.matched_setups) for r in rows)
        return ResultPage(
            items=tuple(chunk),
            page=page,
            page_size=size,
            total_candidates=total,
            total_setup_matches=setup_matches,
            has_next=has_next,
            has_prev=has_prev,
            next_page=(page + 1) if has_next else None,
            prev_page=(page - 1) if has_prev else None,
        )

    def ordered_ids(
        self,
        q: ResultQuery,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
    ) -> tuple[str, ...]:
        return tuple(
            r.candidate_id for r in self._filtered_rows(q, ticker_query=ticker_query, statuses=statuses)
        )

    def neighbor(
        self,
        q: ResultQuery,
        candidate_id: str,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
    ) -> NeighborPosition | None:
        ids = self.ordered_ids(q, ticker_query=ticker_query, statuses=statuses)
        if candidate_id not in ids:
            return None
        idx = ids.index(candidate_id)
        total = len(ids)
        page = (idx // q.page.page_size) + 1
        return NeighborPosition(
            candidate_id=candidate_id,
            prev_id=ids[idx - 1] if idx > 0 else None,
            next_id=ids[idx + 1] if idx + 1 < total else None,
            position=idx + 1,
            total=total,
            page=page,
        )

    def list_pages(
        self,
        q: ResultQuery,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
    ) -> tuple[int, ...]:
        total = len(self._filtered_rows(q, ticker_query=ticker_query, statuses=statuses))
        if total == 0:
            return ()
        n = (total + q.page.page_size - 1) // q.page.page_size
        return tuple(range(1, n + 1))

    def get_detail(self, run_id: str, candidate_id: str) -> CandidateDetail:
        frozen = self.store.load_frozen_inputs(run_id)
        recorded = {r.candidate_id: r for r in self.store.list_candidate_results(run_id)}
        by_input = {c.candidate_id: c for c in frozen.candidates}
        if candidate_id not in recorded and candidate_id not in by_input:
            raise VisionError(f"unknown candidate {candidate_id} in run {run_id}")
        result = recorded.get(candidate_id) or _pending_result(by_input[candidate_id])
        names = {s.setup_id: s.name for s in frozen.setups}
        reviews = self.reviews.list_reviews(run_id, candidate_id)
        row = self._candidate_row(run_id, result, names, frozen)
        window = by_input[candidate_id].window if candidate_id in by_input else None
        attempts = tuple(
            a for a in self.store.list_attempts(run_id) if candidate_id in a.candidate_ids
        )
        artifact = row.chart_ref
        return CandidateDetail(
            row=row,
            assessments=result.assessments,
            eligible_setup_ids=result.eligible_setup_ids,
            window=window,
            artifact=artifact,
            reviews=reviews,
            attempts=attempts,
        )

    def counts(self, run_id: str) -> ResultCounts:
        ext = self.extended_counts(run_id)
        return ResultCounts(
            candidates=ext.candidates,
            completed=ext.completed,
            error=ext.error,
            skipped=ext.skipped,
            pending=ext.pending,
            setup_matches=ext.setup_matches,
            unreviewed=ext.unreviewed,
            reviewed=ext.reviewed,
        )

    def extended_counts(self, run_id: str) -> QueryCounts:
        frozen = self.store.load_frozen_inputs(run_id)
        names = {s.setup_id: s.name for s in frozen.setups}
        rows = self._all_rows(run_id, frozen, names)
        per_setup: dict[str, int] = {}
        any_match = 0
        uncertain = 0
        no_match_only = 0
        setup_matches = 0
        reviewed = 0
        reviewed_pairs = 0
        for row, result in rows:
            match_n = sum(1 for a in result.assessments if a.verdict == "match")
            setup_matches += match_n
            if match_n:
                any_match += 1
            if any(a.verdict == "uncertain" for a in result.assessments):
                uncertain += 1
            if result.status == "completed" and result.assessments and match_n == 0 and not any(
                a.verdict == "uncertain" for a in result.assessments
            ):
                no_match_only += 1
            for badge in row.matched_setups:
                per_setup[badge.setup_id] = per_setup.get(badge.setup_id, 0) + 1
            currents = self.reviews.current_reviews_for_candidate(run_id, row.candidate_id)
            reviewed_pairs += len(currents)
            if currents:
                reviewed += 1
        return QueryCounts(
            candidates=len(rows),
            completed=sum(1 for _, r in rows if r.status == "completed"),
            pending=sum(1 for _, r in rows if r.status == "pending"),
            error=sum(1 for _, r in rows if r.status == "error"),
            skipped=sum(1 for _, r in rows if r.status == "skipped"),
            any_match=any_match,
            setup_matches=setup_matches,
            uncertain=uncertain,
            no_match_only=no_match_only,
            unreviewed=len(rows) - reviewed,
            reviewed=reviewed,
            reviewed_pairs=reviewed_pairs,
            per_setup_matches=tuple(sorted(per_setup.items())),
        )

    def get_artifact_bytes(self, run_id: str, artifact_id: str) -> bytes:
        return self.store.get_artifact_bytes(run_id, artifact_id)

    def get_artifact_ref(self, run_id: str, artifact_id: str) -> ArtifactRef:
        return self.store.get_artifact_ref(run_id, artifact_id)

    def setup_snapshots(self, run_id: str) -> tuple[SetupSnapshot, ...]:
        return self.store.load_frozen_inputs(run_id).setups

    def export_query(
        self,
        q: ResultQuery,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
    ) -> QueryExport:
        frozen = self.store.load_frozen_inputs(q.run_id)
        names = {s.setup_id: s.name for s in frozen.setups}
        units = FeatureValue(None, None, None).units()
        rows = self._filtered_rows(q, ticker_query=ticker_query, statuses=statuses)
        recorded = {r.candidate_id: r for r in self.store.list_candidate_results(q.run_id)}
        by_input = {c.candidate_id: c for c in frozen.candidates}
        cand_out: list[dict[str, Any]] = []
        assess_out: list[dict[str, Any]] = []
        for row in rows:
            result = recorded.get(row.candidate_id)
            if result is None and row.candidate_id in by_input:
                result = _pending_result(by_input[row.candidate_id])
            eligible = result.eligible_setup_ids if result else ()
            cand_out.append(
                {
                    "run_id": q.run_id,
                    "candidate_id": row.candidate_id,
                    "ticker": row.ticker,
                    "asof_date": row.asof_date.isoformat(),
                    "status": row.status,
                    "close": row.features.close,
                    "close_unit": units["close"],
                    "dollar_vol_avg_20": row.features.dollar_vol_avg_20,
                    "dollar_vol_avg_20_unit": units["dollar_vol_avg_20"],
                    "adr_pct_20": row.features.adr_pct_20,
                    "adr_pct_20_unit": units["adr_pct_20"],
                    "adr_pct_20_display_pct": format_adr_display_pct(row.features.adr_pct_20),
                    "matched_setup_ids": flatten_list([b.setup_id for b in row.matched_setups]),
                    "matched_setup_names": flatten_list([b.setup_name for b in row.matched_setups]),
                    "matched_strengths": flatten_list(
                        [str(b.match_strength) for b in row.matched_setups]
                    ),
                    "eligible_setup_ids": flatten_list(eligible),
                    "review_state": row.review_state,
                    "current_review_id": row.current_review.review_id if row.current_review else None,
                    "current_review_judgment": row.current_review.judgment if row.current_review else None,
                    "current_review_note": row.current_review.note if row.current_review else None,
                    "error_kind": row.error.kind if row.error else None,
                    "error_message": row.error.message if row.error else None,
                    "chart_artifact_id": row.chart_ref.artifact_id if row.chart_ref else None,
                    "chart_relative_path": row.chart_ref.relative_path if row.chart_ref else None,
                    "chart_sha256": row.chart_ref.sha256 if row.chart_ref else None,
                }
            )
            assessments = result.assessments if result else ()
            current_by_setup = {
                rec.setup_id: rec
                for rec in self.reviews.current_reviews_for_candidate(q.run_id, row.candidate_id)
            }
            covered = {a.setup_id for a in assessments}
            for a in assessments:
                rev = current_by_setup.get(a.setup_id)
                assess_out.append(_assessment_export_row(q.run_id, row, names, a, rev))
            for sid, rev in current_by_setup.items():
                if sid is None or sid in covered:
                    continue
                assess_out.append(
                    {
                        "run_id": q.run_id,
                        "candidate_id": row.candidate_id,
                        "ticker": row.ticker,
                        "setup_id": sid,
                        "setup_name": names.get(sid, sid),
                        "verdict": None,
                        "match_strength": None,
                        "reason": None,
                        "violated_required_rule_ids": None,
                        "missing_evidence": None,
                        "review_id": rev.review_id,
                        "review_judgment": rev.judgment,
                        "review_note": rev.note,
                        "review_created_at": format_dt(rev.created_at),
                    }
                )
        flattening = (
            f"One candidate row per stock. Lists joined with {LIST_JOIN!r}. "
            "adr_pct_20 is the stored fraction (0.04 = 4%); adr_pct_20_display_pct is percent. "
            "Units are close=USD/share, dollar_vol_avg_20=USD/day, adr_pct_20=fraction. "
            "Nulls stay empty. Assessment export is normalized (one row per setup assessment) "
            "and includes the current review for that pair when present."
        )
        return QueryExport(
            candidates=tuple(cand_out),
            assessments=tuple(assess_out),
            flattening=flattening,
        )

    def export_parquet_bytes(
        self,
        q: ResultQuery,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
        table: Literal["candidates", "assessments"] = "candidates",
    ) -> bytes:
        import io

        import pandas as pd

        payload = self.export_query(q, ticker_query=ticker_query, statuses=statuses)
        records = payload.candidates if table == "candidates" else payload.assessments
        buf = io.BytesIO()
        pd.DataFrame(list(records)).to_parquet(buf, index=False)
        return buf.getvalue()

    def export_csv(
        self,
        q: ResultQuery,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
        table: Literal["candidates", "assessments"] = "candidates",
    ) -> str:
        import pandas as pd

        payload = self.export_query(q, ticker_query=ticker_query, statuses=statuses)
        records = payload.candidates if table == "candidates" else payload.assessments
        return pd.DataFrame(list(records)).to_csv(index=False)

    def _filtered_rows(
        self,
        q: ResultQuery,
        *,
        ticker_query: str = "",
        statuses: tuple[CandidateStatus, ...] = (),
    ) -> list[CandidateRow]:
        frozen = self.store.load_frozen_inputs(q.run_id)
        names = {s.setup_id: s.name for s in frozen.setups}
        selected = set(q.setup_ids)
        ticker_q = ticker_query.strip().upper()
        status_set = set(statuses)
        scored: list[tuple[Any, CandidateRow]] = []
        for row, result in self._all_rows(q.run_id, frozen, names):
            if ticker_q and ticker_q not in row.ticker.upper() and ticker_q not in row.candidate_id.upper():
                continue
            if status_set and row.status not in status_set:
                continue
            if selected:
                relevant = [a for a in result.assessments if a.setup_id in selected]
                if result.status == "completed" and result.assessments and not relevant:
                    continue
                if not result.assessments and selected:
                    if not any(sid in selected for sid in result.eligible_setup_ids):
                        continue
            else:
                relevant = list(result.assessments)
            if q.verdicts:
                if result.status != "completed":
                    continue
                if not any(a.verdict in q.verdicts for a in relevant):
                    continue
            strengths = [
                a.match_strength
                for a in relevant
                if a.verdict == "match" and a.match_strength is not None
            ]
            if q.min_match_strength is not None or q.max_match_strength is not None:
                if not strengths:
                    continue
                bound_ok = [
                    s
                    for s in strengths
                    if (q.min_match_strength is None or s >= q.min_match_strength)
                    and (q.max_match_strength is None or s <= q.max_match_strength)
                ]
                if not bound_ok:
                    continue
                sort_strength = max(bound_ok)
            else:
                sort_strength = max(strengths) if strengths else None
            if q.review_state not in {"any", None} and row.review_state != q.review_state:
                continue
            scored.append((sort_strength, row))
        ranks = self._arrival_ranks(q.run_id) if q.sort.key == "arrival" else {}
        return _sort_rows(scored, q.sort, arrival_ranks=ranks)

    def _arrival_ranks(self, run_id: str) -> dict[str, int]:
        """Stable order of first appearance in accepted/journaled attempts (earliest first)."""

        ranks: dict[str, int] = {}
        seq = 0
        attempts = list(self.store.list_attempts(run_id))
        attempts.sort(
            key=lambda a: (
                (a.ended_at or a.started_at).isoformat() if (a.ended_at or a.started_at) else "",
                a.attempt_id,
            )
        )
        for attempt in attempts:
            for cid in attempt.candidate_ids:
                if cid not in ranks:
                    ranks[cid] = seq
                    seq += 1
        return ranks

    def _all_rows(
        self,
        run_id: str,
        frozen: PreparedScan,
        names: dict[str, str],
    ) -> list[tuple[CandidateRow, CandidateResult]]:
        recorded = {r.candidate_id: r for r in self.store.list_candidate_results(run_id)}
        out: list[tuple[CandidateRow, CandidateResult]] = []
        seen: set[str] = set()
        for cand in frozen.candidates:
            result = recorded.get(cand.candidate_id) or _pending_result(cand)
            row = self._candidate_row(run_id, result, names, frozen)
            out.append((row, result))
            seen.add(cand.candidate_id)
        extras = [r for cid, r in recorded.items() if cid not in seen]
        extras.sort(key=lambda r: r.candidate_id)
        for result in extras:
            out.append((self._candidate_row(run_id, result, names, frozen), result))
        return out

    def _candidate_row(
        self,
        run_id: str,
        result: CandidateResult,
        names: dict[str, str],
        frozen: PreparedScan,
    ) -> CandidateRow:
        badges = tuple(
            MatchedSetupBadge(
                setup_id=a.setup_id,
                setup_name=names.get(a.setup_id, a.setup_id),
                match_strength=int(a.match_strength or 0),
            )
            for a in result.assessments
            if a.verdict == "match"
        )
        summaries = tuple(
            AssessmentSummary(
                setup_id=a.setup_id,
                setup_name=names.get(a.setup_id, a.setup_id),
                verdict=a.verdict,
                match_strength=a.match_strength,
                reason=a.reason,
            )
            for a in result.assessments
        )
        whole = self.reviews.current_review(run_id, result.candidate_id, setup_id=None)
        currents = self.reviews.current_reviews_for_candidate(run_id, result.candidate_id)
        current = whole or _latest_among(currents)
        state: ReviewState = current.judgment if current is not None else "unreviewed"
        chart = None
        if result.artifact_id:
            try:
                chart = self.store.get_artifact_ref(run_id, result.artifact_id)
            except VisionError:
                chart = None
        if chart is None:
            chart = self.store.find_artifact_for_candidate(run_id, result.candidate_id)
        return CandidateRow(
            candidate_id=result.candidate_id,
            ticker=result.ticker,
            asof_date=result.asof_date,
            status=result.status,
            features=result.features,
            matched_setups=badges,
            assessments=summaries,
            review_state=state,
            current_review=current,
            chart_ref=chart,
            error=result.error,
        )


def _pending_result(cand: CandidateInput) -> CandidateResult:
    return CandidateResult(
        candidate_id=cand.candidate_id,
        status="pending",
        ticker=cand.ticker,
        asof_date=cand.asof_date,
        features=cand.features,
        eligible_setup_ids=cand.eligible_setup_ids,
        assessments=(),
        artifact_id=None,
        error=None,
        attempt_id=None,
        source_digest=cand.source_digest,
    )


def _latest_among(recs: Sequence[ReviewRecord]) -> ReviewRecord | None:
    if not recs:
        return None
    return sorted(recs, key=lambda r: (r.created_at, r.review_id))[-1]


def _sort_rows(
    scored: list[tuple[int | None, CandidateRow]],
    spec: SortSpec,
    *,
    arrival_ranks: dict[str, int] | None = None,
) -> list[CandidateRow]:
    items = list(scored)
    items.sort(key=lambda item: item[1].candidate_id)
    if spec.key == "match_strength":
        if spec.descending:
            items.sort(key=lambda item: (item[0] is None, -(item[0] or 0)))
        else:
            items.sort(key=lambda item: (item[0] is None, item[0] or 0))
    elif spec.key == "ticker":
        items.sort(key=lambda item: item[1].ticker, reverse=spec.descending)
    elif spec.key == "asof_date":
        items.sort(key=lambda item: item[1].asof_date.isoformat(), reverse=spec.descending)
    elif spec.key == "candidate_id" and spec.descending:
        items.sort(key=lambda item: item[1].candidate_id, reverse=True)
    elif spec.key == "arrival":
        ranks = arrival_ranks or {}
        missing = 10**12
        items.sort(
            key=lambda item: (ranks.get(item[1].candidate_id, missing), item[1].candidate_id),
            reverse=spec.descending,
        )
    return [row for _, row in items]


def _assessment_export_row(
    run_id: str,
    row: CandidateRow,
    names: dict[str, str],
    assessment: Assessment,
    review: ReviewRecord | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "candidate_id": row.candidate_id,
        "ticker": row.ticker,
        "setup_id": assessment.setup_id,
        "setup_name": names.get(assessment.setup_id, assessment.setup_id),
        "verdict": assessment.verdict,
        "match_strength": assessment.match_strength,
        "reason": assessment.reason,
        "violated_required_rule_ids": flatten_list(assessment.violated_required_rule_ids),
        "missing_evidence": flatten_list(assessment.missing_evidence),
        "review_id": review.review_id if review else None,
        "review_judgment": review.judgment if review else None,
        "review_note": review.note if review else None,
        "review_created_at": format_dt(review.created_at) if review else None,
    }


def page_for_index(index: int, page_size: int) -> int:
    if index < 0:
        return 1
    return (index // page_size) + 1


def query_with_page(q: ResultQuery, page: int) -> ResultQuery:
    return replace(q, page=PageRequest(page=page, page_size=q.page.page_size))

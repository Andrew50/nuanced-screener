"""One Responses API attempt. Retries are owned by the runner, not the SDK."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4
import base64
import json
import logging
import time

from .types import (
    AttemptError,
    ClassificationAttempt,
    CompiledRequest,
    TokenUsage,
)
from .validation import validate_classification_payload

log = logging.getLogger("screener_loader.vision.client")

GetImage = Callable[[str], bytes]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = None
    if hasattr(headers, "get"):
        raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _usage_from(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _collect_refusals(response: Any) -> tuple[str, ...]:
    refusals: list[str] = []
    for item in getattr(response, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "refusal":
            refusals.append(str(getattr(item, "refusal", None) or item))
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                refusals.append(str(getattr(part, "refusal", None) or part))
    return tuple(refusals)


def _text_from_output_blocks(response: Any) -> str | None:
    texts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            ptype = getattr(part, "type", None)
            if ptype in {"output_text", "text"}:
                texts.append(str(getattr(part, "text", "") or ""))
    joined = "".join(texts).strip()
    return joined or None


def _collect_output_text(response: Any) -> tuple[str | None, tuple[str, ...]]:
    """Read provider text exactly once.

    The official SDK ``Response.output_text`` already concatenates ``output_text``
    blocks. Concatenating that property *and* the underlying blocks duplicates
    valid JSON and fails validation. Prefer ``output_text`` when it is non-empty;
    otherwise fall back to walking output blocks (for mocks and older shapes).
    """

    refusals = _collect_refusals(response)
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip(), refusals
    return _text_from_output_blocks(response), refusals


def _sanitize_json(text: str) -> Mapping[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, Mapping):
        return parsed
    return None


def _error(
    kind: str,
    message: str,
    *,
    retryable: bool,
    exc: BaseException | None = None,
    http_status: int | None = None,
) -> AttemptError:
    status = http_status
    if status is None and exc is not None:
        status = getattr(exc, "status_code", None)
    return AttemptError(
        kind=kind,  # type: ignore[arg-type]
        message=message,
        retryable=retryable,
        http_status=int(status) if status is not None else None,
        provider_code=type(exc).__name__ if exc is not None else None,
        retry_after_seconds=_retry_after_seconds(exc) if exc is not None else None,
    )


class OpenAIClassifier:
    """Official OpenAI SDK, Responses API, one attempt, max_retries=0."""

    provider = "openai_responses"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        get_image: GetImage | None = None,
        images: Mapping[str, bytes] | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._get_image = get_image
        self._images = dict(images or {})

    def _image_bytes(self, artifact_id: str) -> bytes:
        if artifact_id in self._images:
            return self._images[artifact_id]
        if self._get_image is None:
            raise KeyError(artifact_id)
        return self._get_image(artifact_id)

    def _make_client(self, request: CompiledRequest) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The vision classifier requires the openai package. Agent 3 should add it to extras."
            ) from e
        return OpenAI(
            api_key=self._api_key,
            max_retries=0,
            timeout=float(request.timeout_seconds),
        )

    def classify(self, request: CompiledRequest) -> ClassificationAttempt:
        started = _utc_now()
        t0 = time.monotonic()
        attempt_id = str(uuid4())

        def finish(**kwargs: Any) -> ClassificationAttempt:
            ended = _utc_now()
            return ClassificationAttempt(
                attempt_id=attempt_id,
                batch_id=request.batch_id,
                batch_fingerprint=request.fingerprint,
                candidate_ids=request.candidate_ids,
                setup_ids=request.setup_ids,
                started_at=started,
                ended_at=ended,
                provider=self.provider,
                model=request.model,
                latency_ms=int((time.monotonic() - t0) * 1000),
                **kwargs,
            )

        if not str(request.model).strip():
            err = _error("config", "CompiledRequest.model is empty", retryable=False)
            return finish(response_id=None, usage=None, retry_after_seconds=None, error=err, sanitized_output=None, results=None, accepted=False)

        if self._client is None and not (self._api_key or __import__("os").environ.get("OPENAI_API_KEY")):
            err = _error("auth", "OPENAI_API_KEY is missing; refusing to invent results", retryable=False)
            return finish(response_id=None, usage=None, retry_after_seconds=None, error=err, sanitized_output=None, results=None, accepted=False)

        try:
            content = self._provider_content(request)
        except KeyError as e:
            err = _error("unavailable_input", f"Missing image artifact {e}", retryable=False)
            return finish(response_id=None, usage=None, retry_after_seconds=None, error=err, sanitized_output=None, results=None, accepted=False)

        log.info(
            "vision classify attempt batch_id=%s fingerprint=%s model=%s images=%s",
            request.batch_id,
            request.fingerprint,
            request.model,
            sum(1 for b in request.blocks if b.kind == "image"),
        )

        try:
            client = self._make_client(request)
            response = client.responses.create(
                model=request.model,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": request.schema_name,
                        "strict": True,
                        "schema": dict(request.json_schema),
                    }
                },
                max_output_tokens=int(request.max_output_tokens),
                timeout=float(request.timeout_seconds),
            )
        except Exception as exc:
            return finish(**self._from_exception(exc))

        usage = _usage_from(response)
        response_id = getattr(response, "id", None)
        status = getattr(response, "status", None)
        text, refusals = _collect_output_text(response)
        if refusals:
            err = _error("refusal", refusals[0][:500], retryable=False)
            return finish(
                response_id=response_id,
                usage=usage,
                retry_after_seconds=None,
                error=err,
                sanitized_output={"refusal": refusals[0][:500]},
                results=None,
                accepted=False,
            )
        if status == "incomplete":
            detail = getattr(response, "incomplete_details", None)
            reason = getattr(detail, "reason", None) if detail is not None else None
            err = _error("incomplete", f"Incomplete response ({reason or 'unknown'})", retryable=True)
            return finish(
                response_id=response_id,
                usage=usage,
                retry_after_seconds=None,
                error=err,
                sanitized_output={"status": "incomplete", "reason": reason},
                results=None,
                accepted=False,
            )
        if not text:
            err = _error("invalid_output", "Provider returned no text output", retryable=True)
            return finish(
                response_id=response_id,
                usage=usage,
                retry_after_seconds=None,
                error=err,
                sanitized_output=None,
                results=None,
                accepted=False,
            )
        parsed = _sanitize_json(text)
        if parsed is None:
            err = _error("invalid_output", "Provider output was not JSON", retryable=True)
            return finish(
                response_id=response_id,
                usage=usage,
                retry_after_seconds=None,
                error=err,
                sanitized_output={"text_preview": text[:500]},
                results=None,
                accepted=False,
            )
        return finish(
            response_id=response_id,
            usage=usage,
            retry_after_seconds=None,
            error=None,
            sanitized_output=parsed,
            results=None,
            accepted=False,  # runner validates against eligible coverage
        )

    def _provider_content(self, request: CompiledRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for block in request.blocks:
            if block.kind == "text":
                content.append({"type": "input_text", "text": block.text or ""})
                continue
            raw = self._image_bytes(str(block.artifact_id))
            b64 = base64.standard_b64encode(raw).decode("ascii")
            media = "image/png"
            if raw[:3] == b"\xff\xd8\xff":
                media = "image/jpeg"
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{media};base64,{b64}",
                    "detail": request.image_detail,
                }
            )
        return content

    def _from_exception(self, exc: BaseException) -> dict[str, Any]:
        name = type(exc).__name__
        retry_after = _retry_after_seconds(exc)
        message = str(exc)
        # Keep credentials and data URIs out of stored errors.
        if "base64" in message.lower() or "sk-" in message:
            message = name
        kind = "provider"
        retryable = True
        if name in {"AuthenticationError", "PermissionDeniedError"}:
            kind, retryable = "auth", False
        elif name in {"APITimeoutError"}:
            kind, retryable = "timeout", True
        elif name in {"APIConnectionError"}:
            kind, retryable = "transport", True
        elif name in {"RateLimitError"}:
            kind, retryable = "rate_limit", True
        elif name in {"BadRequestError"}:
            kind, retryable = "config", False
            lowered = str(exc).lower()
            if "too large" in lowered or "maximum" in lowered and "image" in lowered:
                kind = "oversize"
        elif name in {"LengthFinishReasonError"}:
            kind, retryable = "incomplete", True
        elif name in {"InternalServerError", "APIStatusError", "APIError"}:
            kind, retryable = "provider", True
        err = _error(kind, message[:500], retryable=retryable, exc=exc)
        if retry_after is not None:
            err = AttemptError(
                kind=err.kind,
                message=err.message,
                retryable=err.retryable,
                http_status=err.http_status,
                provider_code=err.provider_code,
                retry_after_seconds=retry_after,
            )
        return {
            "response_id": None,
            "usage": None,
            "retry_after_seconds": err.retry_after_seconds,
            "error": err,
            "sanitized_output": None,
            "results": None,
            "accepted": False,
        }


def attach_validated_results(
    attempt: ClassificationAttempt,
    payload: Mapping[str, Any],
    *,
    candidates,
    setups,
) -> ClassificationAttempt:
    """Apply semantic validation. Invalid batches are not partial successes."""
    from dataclasses import replace

    results = validate_classification_payload(payload, candidates=candidates, setups=setups)
    return replace(attempt, results=results, accepted=True, error=None, sanitized_output=dict(payload))

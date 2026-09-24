"""Unified OpenAI-compatible client for all configured LLM backends.

The project intentionally exposes one :class:`LLMClient` regardless of the
configured model. Provider/model/base URL/API key values are loaded from the
``llm_env_files`` mapping in ``configs/experiment.yaml`` and the referenced
``.env`` file. Prompt text is shared across models and is handled separately by
``src.explainers``.

LLMs are planning components only. This module never calls the frozen predictor
and never accepts predictor probabilities as authoritative model output.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

try:
    from .utils import ConfigError, require_config_keys, resolve_project_path
except ImportError:  # pragma: no cover
    from utils import ConfigError, require_config_keys, resolve_project_path


class LLMError(RuntimeError):
    """Raised when LLM configuration, invocation, or parsing fails."""


def _is_budget_exceeded(exc: Exception) -> bool:
    """Return True when the provider explicitly reports exhausted budget."""
    body = getattr(exc, "body", None)

    if isinstance(body, Mapping):
        error = body.get("error", body)
        if isinstance(error, Mapping):
            code = error.get("code")
            if isinstance(code, str) and code.lower() == "budget_exceeded":
                return True

    return "budget_exceeded" in str(exc).lower()


@dataclass(frozen=True)
class LLMSettings:
    """Resolved settings for one configured LLM backend."""

    llm_id: str
    provider: str
    model: str
    api_key: str
    base_url: str
    request_timeout_seconds: float
    max_output_tokens: int
    max_retries: int
    retry_backoff_seconds: float


def load_llm_settings(
    experiment_config: Mapping[str, Any],
    llm_id: str,
    project_root: str | Path,
) -> LLMSettings:
    """Resolve one model's settings from ``experiment.yaml`` and its env file.

    Args:
        experiment_config: Parsed ``configs/experiment.yaml``.
        llm_id: One of the IDs listed under ``llm_models``.
        project_root: Project root used to resolve the configured env-file path.

    Returns:
        Validated immutable LLM settings.

    Raises:
        ConfigError: If the model ID or env mapping is invalid.
        FileNotFoundError: If the configured env file is missing.
    """
    configured_ids = [str(value) for value in experiment_config.get("llm_models", [])]
    if llm_id not in configured_ids:
        raise ConfigError(
            f"Unknown llm_id {llm_id!r}; configured values: {configured_ids}"
        )

    env_files = experiment_config.get("llm_env_files")
    if not isinstance(env_files, Mapping) or llm_id not in env_files:
        raise ConfigError(f"Missing llm_env_files mapping for {llm_id!r}")

    env_path = resolve_project_path(project_root, str(env_files[llm_id]))
    if not env_path.is_file():
        raise FileNotFoundError(f"Configured LLM env file not found: {env_path}")

    raw = {
        key: value
        for key, value in dotenv_values(env_path).items()
        if value is not None
    }
    required = [
        "LLM_ID",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_REQUEST_TIMEOUT_SECONDS",
        "LLM_MAX_OUTPUT_TOKENS",
        "LLM_MAX_RETRIES",
        "LLM_RETRY_BACKOFF_SECONDS",
    ]
    require_config_keys(raw, required, str(env_path))

    if str(raw["LLM_ID"]) != llm_id:
        raise ConfigError(
            f"LLM_ID mismatch in {env_path}: expected {llm_id!r}, got {raw['LLM_ID']!r}"
        )
    provider = str(raw["LLM_PROVIDER"]).lower()
    supported_providers = {"openai", "ollama_native"}
    if provider not in supported_providers:
        raise ConfigError(
            f"Unsupported LLM_PROVIDER {raw['LLM_PROVIDER']!r}; "
            f"supported values are {sorted(supported_providers)}."
        )

    base_url = str(raw["LLM_BASE_URL"]).rstrip("/")

    # Runtime-only endpoint override for local Ollama jobs.
    # Scientific model/decoding configuration remains in the env file;
    # Slurm may assign a task-specific local TCP port.
    if provider == "ollama_native":
        runtime_base_url = os.environ.get(
            "LLM_BASE_URL_OVERRIDE"
        )
        if runtime_base_url:
            base_url = runtime_base_url.rstrip("/")

    return LLMSettings(
        llm_id=llm_id,
        provider=provider,
        model=str(raw["LLM_MODEL"]),
        api_key=str(raw["LLM_API_KEY"]),
        base_url=base_url,
        request_timeout_seconds=float(raw["LLM_REQUEST_TIMEOUT_SECONDS"]),
        max_output_tokens=int(raw["LLM_MAX_OUTPUT_TOKENS"]),
        max_retries=int(raw["LLM_MAX_RETRIES"]),
        retry_backoff_seconds=float(raw["LLM_RETRY_BACKOFF_SECONDS"]),
    )


class LLMClient:
    """Uniform chat-completions client for the four configured LLM models.

    The class is deliberately thin. It sends a prompt and returns text or a
    parsed JSON object. No model-specific prompt variants or branches are used,
    preserving information parity across LLMs.
    """

    def __init__(
        self,
        llm_id: str,
        experiment_config: Mapping[str, Any],
        project_root: str | Path,
        *,
        client: Any | None = None,
    ) -> None:
        """Create a client from the model ID and configuration.

        ``client`` is injectable for unit/offline tests. Production use creates
        an :class:`openai.OpenAI` client against the configured Otago-compatible
        endpoint.
        """
        self.settings = load_llm_settings(experiment_config, llm_id, project_root)

        if client is not None:
            self._client = client
        elif self.settings.provider == "openai":
            try:
                from openai import OpenAI
            except (ImportError, AttributeError) as exc:
                raise LLMError(
                    "A compatible openai package is required. "
                    "Install the version declared in requirements.txt."
                ) from exc

            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.request_timeout_seconds,
            )
        elif self.settings.provider == "ollama_native":
            # Native Ollama transport uses urllib directly.
            self._client = None
        else:  # defensive; load_llm_settings already validates provider
            raise LLMError(
                f"Unsupported provider {self.settings.provider!r}"
            )

    @property
    def llm_id(self) -> str:
        """Return the experiment-level model ID."""
        return self.settings.llm_id

    @property
    def model(self) -> str:
        """Return the provider-facing model name from the env file."""
        return self.settings.model

    def _request_once(
        self,
        prompt: str,
        *,
        temperature: float | None,
        top_p: float | None,
        system_prompt: str | None = None,
    ) -> tuple[str, str | None]:
        """Send exactly one provider request and return text plus finish reason."""

        if self.settings.provider == "ollama_native":
            return self._request_once_ollama_native(
                prompt,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
            )

        if self.settings.provider == "openai":
            return self._request_once_openai(
                prompt,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
            )

        raise LLMError(
            f"Unsupported provider {self.settings.provider!r}"
        )

    def _request_once_openai(
        self,
        prompt: str,
        *,
        temperature: float | None,
        top_p: float | None,
        system_prompt: str | None = None,
    ) -> tuple[str, str | None]:
        """Use the existing OpenAI-compatible transport."""
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt}
            )

        messages.append({"role": "user", "content": prompt})

        request_kwargs = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_output_tokens,
        }

        # Claude Opus 4.6 exposed through the Otago gateway does not
        # accept temperature and top_p in the same request.
        if self.llm_id == "claude_opus46":
            if top_p is not None and float(top_p) < 1.0:
                request_kwargs["top_p"] = float(top_p)
            elif temperature is not None:
                request_kwargs["temperature"] = float(temperature)
        else:
            if temperature is not None:
                request_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                request_kwargs["top_p"] = float(top_p)

        if self._client is None:
            raise LLMError("OpenAI client is not initialized")

        response = self._client.chat.completions.create(
            **request_kwargs
        )

        if not response.choices:
            raise LLMError("LLM returned no choices")

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        content = choice.message.content

        if content is None or not str(content).strip():
            raise LLMError(
                "LLM returned an empty response "
                f"(finish_reason={finish_reason!r})"
            )

        return str(content).strip(), finish_reason

    def _request_once_ollama_native(
        self,
        prompt: str,
        *,
        temperature: float | None,
        top_p: float | None,
        system_prompt: str | None = None,
    ) -> tuple[str, str | None]:
        """Use Ollama's native /api/chat streaming transport.

        Thinking is disabled explicitly so that the local models return the
        structured planner response rather than spending the completion budget
        on hidden/auxiliary reasoning.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt}
            )

        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "num_predict": self.settings.max_output_tokens,
        }

        if temperature is not None:
            options["temperature"] = float(temperature)

        # Preserve the existing experiment policy:
        # top_p=None means it is not explicitly configured.
        if top_p is not None:
            options["top_p"] = float(top_p)

        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": options,
        }

        base_url = self.settings.base_url.rstrip("/")

        # Accept either root URL or an accidentally retained /v1 suffix.
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        url = f"{base_url}/api/chat"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        content_parts: list[str] = []
        thinking_chars = 0
        finish_reason: str | None = None

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.request_timeout_seconds,
            ) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue

                    event = json.loads(
                        raw_line.decode("utf-8")
                    )

                    message = event.get("message") or {}

                    content = message.get("content") or ""
                    thinking = (
                        message.get("thinking")
                        or message.get("reasoning")
                        or ""
                    )

                    if content:
                        content_parts.append(str(content))

                    if thinking:
                        thinking_chars += len(str(thinking))

                    if event.get("done"):
                        finish_reason = (
                            event.get("done_reason")
                            or "stop"
                        )
                        break

        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                detail = str(exc)

            raise LLMError(
                f"Ollama native HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise LLMError(
                f"Ollama native request failed: {exc}"
            ) from exc

        except TimeoutError as exc:
            raise LLMError(
                "Ollama native request timed out after "
                f"{self.settings.request_timeout_seconds}s"
            ) from exc

        text = "".join(content_parts).strip()

        if thinking_chars:
            raise LLMError(
                "Ollama native think=false request unexpectedly "
                f"returned reasoning content ({thinking_chars} chars)"
            )

        if not text:
            raise LLMError(
                "Ollama native request returned empty content "
                f"(finish_reason={finish_reason!r})"
            )

        return text, finish_reason

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None,
        top_p: float | None,
        system_prompt: str | None = None,
    ) -> str:
        """Send a completion request using the configured retry budget."""
        attempts = self.settings.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                text, _ = self._request_once(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt,
                )
                return text
            except Exception as exc:
                if _is_budget_exceeded(exc):
                    raise LLMError(
                        f"LLM budget exhausted for {self.settings.llm_id!r}: {exc}"
                    ) from exc

                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(self.settings.retry_backoff_seconds * (2**attempt))

        raise LLMError(
            f"LLM request failed for {self.settings.llm_id!r} "
            f"after {attempts} attempt(s): {last_error}"
        ) from last_error

    def complete_json(
        self,
        prompt: str,
        *,
        temperature: float | None,
        top_p: float | None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Request and validate one JSON object under one shared retry budget.

        Also records non-invasive audit metadata in ``last_json_audit``.
        The public return type remains unchanged.
        """
        attempts = self.settings.max_retries + 1
        last_error: Exception | None = None

        self.last_json_audit = {
            "attempt_count": 0,
            "first_attempt_direct_json_object": False,
            "first_attempt_json_valid": False,
            "final_json_valid": False,
            "recovered_after_retry": False,
        }

        for attempt in range(attempts):
            try:
                text, finish_reason = self._request_once(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt,
                )

                if attempt == 0:
                    try:
                        direct_payload = json.loads(_strip_json_fence(text))
                        self.last_json_audit[
                            "first_attempt_direct_json_object"
                        ] = isinstance(direct_payload, dict)
                    except Exception:
                        self.last_json_audit[
                            "first_attempt_direct_json_object"
                        ] = False

                try:
                    payload = _extract_json_object(text)
                except LLMError as exc:
                    preview = text[:300].replace("\n", "\\n")
                    raise LLMError(
                        "LLM response could not be parsed as one unambiguous JSON object "
                        f"(finish_reason={finish_reason!r}, "
                        f"response_chars={len(text)}, "
                        f"response_preview={preview!r}): {exc}"
                    ) from exc

                self.last_json_audit.update(
                    {
                        "attempt_count": attempt + 1,
                        "first_attempt_json_valid": attempt == 0,
                        "final_json_valid": True,
                        "recovered_after_retry": attempt > 0,
                    }
                )

                return payload

            except Exception as exc:
                if attempt == 0:
                    self.last_json_audit["first_attempt_json_valid"] = False

                self.last_json_audit["attempt_count"] = attempt + 1

                if _is_budget_exceeded(exc):
                    raise LLMError(
                        f"LLM budget exhausted for {self.settings.llm_id!r}: {exc}"
                    ) from exc

                last_error = exc
                if attempt >= attempts - 1:
                    break

                time.sleep(
                    self.settings.retry_backoff_seconds * (2**attempt)
                )

        self.last_json_audit["final_json_valid"] = False
        self.last_json_audit["recovered_after_retry"] = False

        raise LLMError(
            f"LLM JSON request failed for {self.settings.llm_id!r} "
            f"after {attempts} attempt(s): {last_error}"
        ) from last_error


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract exactly one outermost valid JSON object from model output.

    A direct JSON object is accepted immediately. If surrounding prose is
    present, all decodable JSON objects are located together with their text
    spans. Objects fully contained inside a larger decodable object are treated
    as nested structures rather than independent candidates. Exactly one
    outermost JSON object must remain.
    """
    cleaned = _strip_json_fence(text)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        return payload
    if payload is not None:
        raise LLMError("LLM JSON response must be an object")

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []

    for start_index, char in enumerate(cleaned):
        if char != "{":
            continue

        try:
            candidate, consumed = decoder.raw_decode(cleaned[start_index:])
        except json.JSONDecodeError:
            continue

        if isinstance(candidate, dict):
            end_index = start_index + consumed
            candidates.append((start_index, end_index, candidate))

    if not candidates:
        raise LLMError("No valid JSON object found in LLM response")

    outermost: list[tuple[int, int, dict[str, Any]]] = []

    for start_index, end_index, candidate in candidates:
        contained = False

        for other_start, other_end, _ in candidates:
            if (
                other_start <= start_index
                and end_index <= other_end
                and (other_start < start_index or end_index < other_end)
            ):
                contained = True
                break

        if not contained:
            outermost.append((start_index, end_index, candidate))

    unique_outermost: list[dict[str, Any]] = []
    seen_serialized: set[str] = set()

    for _, _, candidate in outermost:
        canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        if canonical not in seen_serialized:
            seen_serialized.add(canonical)
            unique_outermost.append(candidate)

    if len(unique_outermost) != 1:
        raise LLMError(
            f"Ambiguous LLM response: found {len(unique_outermost)} "
            "distinct outermost JSON objects"
        )

    return unique_outermost[0]


def _strip_json_fence(text: str) -> str:
    """Remove a single surrounding Markdown JSON/code fence if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()

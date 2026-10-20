"""OpenAI-compatible request/response models (the subset this server supports)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StreamOptions(BaseModel):
    include_usage: bool = False


class _Base(BaseModel):
    # Clients send many optional fields (user, logit_bias, ...): ignore unknown ones.
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    # vLLM-style extension, handy for benchmarks: keep generating past EOS.
    ignore_eos: bool = False

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v: str | list[str] | None) -> str | list[str] | None:
        stops = [v] if isinstance(v, str) else (v or [])
        if len(stops) > 4:
            raise ValueError("at most 4 stop sequences are supported")
        if any(not s for s in stops):
            raise ValueError("stop sequences must be non-empty")
        return v

    @property
    def stop_list(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        return (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)


class CompletionRequest(_Base):
    prompt: str | list[str]
    echo: bool = False
    logprobs: int | None = None

    @field_validator("prompt")
    @classmethod
    def _check_prompt(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, list) and len(v) != 1:
            raise ValueError("exactly one prompt per request is supported")
        text = v if isinstance(v, str) else v[0]
        if not text:
            raise ValueError("prompt must be non-empty")
        return v

    @property
    def prompt_text(self) -> str:
        return self.prompt if isinstance(self.prompt, str) else self.prompt[0]


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart]

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content)


class ChatCompletionRequest(_Base):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    logprobs: bool = False

    @property
    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


def error_body(
    message: str, type_: str, *, param: str | None = None, code: str | int | None = None
) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}

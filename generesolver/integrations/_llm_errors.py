from __future__ import annotations


class LLMResponseError(RuntimeError):
    def __init__(self, message: str, *, raw_payload: str = "") -> None:
        super().__init__(message)
        self.raw_payload = raw_payload

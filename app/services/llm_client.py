from __future__ import annotations

from functools import lru_cache
from typing import Any

import requests
from dotenv import load_dotenv

from app.config import Settings, get_settings

load_dotenv()


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.llm_api_key.strip()
        self.base_url = self.settings.llm_base_url.strip()
        self.model = self.settings.llm_model.strip()
        self.timeout = max(int(self.settings.llm_timeout), 1)
        self.default_temperature = float(self.settings.llm_temperature)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._request(messages=messages, temperature=temperature)

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._request(
            messages=messages,
            temperature=self.default_temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )

    def _request(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None = None,
    ) -> str:
        self._validate_configuration()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise RuntimeError(f"LLM request timed out after {self.timeout} seconds.") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        if response.status_code != 200:
            detail = self._extract_error_detail(response)
            raise RuntimeError(
                f"LLM API returned non-200 status {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("LLM API returned invalid JSON.") from exc

        return self._extract_text(data)

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM response JSON is invalid: missing choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("LLM response JSON is invalid: choice item is not an object.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("LLM response JSON is invalid: missing message object.")

        content = message.get("content")
        if isinstance(content, str):
            result = content.strip()
            if result:
                return result

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            result = "\n".join(parts).strip()
            if result:
                return result

        raise ValueError("LLM response JSON is invalid: content field is missing or empty.")

    def _extract_error_detail(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or f"HTTP {response.status_code}"

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail
            return str(payload)
        return str(payload)

    def _validate_configuration(self) -> None:
        missing_items = []
        if not self.api_key:
            missing_items.append("LLM_API_KEY")
        if not self.base_url:
            missing_items.append("LLM_BASE_URL")
        if not self.model:
            missing_items.append("LLM_MODEL")

        if missing_items:
            missing_text = ", ".join(missing_items)
            raise RuntimeError(f"LLM client is not configured. Missing: {missing_text}")


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> str:
    return get_llm_client().chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )

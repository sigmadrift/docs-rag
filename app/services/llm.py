"""OpenAI 호환 chat completions 스트리밍 클라이언트.

Ollama, vLLM, 사내 LLM 서버 모두 같은 API를 제공하므로 base_url/model 설정만 바꾸면 교체된다.
"""
import json
from collections.abc import AsyncIterator

import httpx

from app.core.config import get_settings


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    """토큰 델타를 순서대로 yield한다."""
    s = get_settings()
    async with (
        httpx.AsyncClient(base_url=s.llm_base_url, timeout=httpx.Timeout(120, connect=5)) as client,
        client.stream(
            "POST",
            "/chat/completions",
            headers={"Authorization": f"Bearer {s.llm_api_key}"},
            json={"model": s.llm_model, "messages": messages, "stream": True},
        ) as resp,
    ):
        if resp.status_code != 200:
            body = (await resp.aread()).decode(errors="replace")[:500]
            raise RuntimeError(f"LLM 서버 오류 {resp.status_code}: {body}")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0]["delta"].get("content")
            if delta:
                yield delta

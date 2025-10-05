"""Utility for running Browser Use tasks locally."""

from __future__ import annotations

import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent, Browser, ChatOpenAI


def _build_llm(model: str) -> ChatOpenAI:
    """Instantiate ChatOpenAI aligned with our OpenAI SDK configuration."""

    llm_kwargs: dict[str, object] = {"model": model}

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CEREBRAS_API_KEY")
    if api_key:
        llm_kwargs["api_key"] = api_key

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("CEREBRAS_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url

    organization = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")
    if organization:
        llm_kwargs["organization"] = organization

    project = os.getenv("OPENAI_PROJECT")
    if project:
        llm_kwargs["project"] = project

    return ChatOpenAI(**llm_kwargs)


def run_browser_task(
    task: str,
    *,
    model: str = "gpt-4.1-mini",
    headless: bool = False,
    max_steps: Optional[int] = None,
) -> None:
    """Spin up a local browser and execute the provided automation instructions.

    Args:
        task: Natural-language instructions for the browser agent.
        model: LLM model identifier understood by `browser_use.ChatOpenAI`.
        headless: When False (default) the browser window is shown locally.
        max_steps: Optional step limit fed to the agent. `None` uses the library default.

    Raises:
        ValueError: If `task` is empty.
        RuntimeError: If called while an asyncio event loop is already running.
    """

    if not task or not task.strip():
        raise ValueError("task must be a non-empty string")

    async def _execute() -> None:
        browser = Browser(headless=headless)
        agent = Agent(task=task, browser=browser, llm=_build_llm(model))

        try:
            if max_steps is None:
                await agent.run()
            else:
                await agent.run(max_steps=max_steps)
        finally:
            await browser.stop()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_execute())
    else:
        raise RuntimeError(
            "run_browser_task() must be called from a synchronous context. "
            "Use 'await agent.run(...)' directly if already inside an event loop."
        )


if __name__ == "__main__":
    run_browser_task("make an account for lichess, use as email preston@")

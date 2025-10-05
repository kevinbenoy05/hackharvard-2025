"""Utility for running Browser Use tasks locally."""

from __future__ import annotations

import asyncio
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent, Browser, ChatOpenAI


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
        agent = Agent(task=task, browser=browser, llm=ChatOpenAI(model=model))

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

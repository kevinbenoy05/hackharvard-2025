# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Parallel Browser SDK built on top of [Browser Use](https://github.com/browser-use/browser-use), designed to execute browser automation tasks in parallel using AI agents. The project enables:

1. **Parallel task execution** with separate browser profiles for independent tasks
2. **Branched task execution** where multiple tasks share a common browser storage state (cookies, local storage, etc.)
3. **LLM-powered browser automation** using Google's Gemini models (default: gemini-2.5-flash)

## Core Architecture

### Main Components

- **`browser_tasks.py`**: Core SDK implementation containing all parallel execution logic
  - `ParallelBrowserSDK`: Main SDK class for orchestrating parallel browser tasks
  - `ParallelTask`: Task definition dataclass
  - `BrowserConfig`: Configuration for browser instances and profile management
  - `run_parallel_tasks()`: Convenience function for simple parallel execution
  - `run_branched_tasks()`: Convenience function for tasks branching from shared storage state
  - `_export_storage_state()`: Internal utility to extract browser storage state from agents

- **`main.py`**: Interactive CLI entry point with three test modes:
  1. SDK test with simple parallel tasks
  2. Convenience function test
  3. Branched tasks test (shares cookies/storage across tasks)

### Browser Profile Management

The SDK handles two distinct execution patterns:

1. **Separate Profiles** (`separate_profiles=True`): Each task gets an isolated browser profile in `./temp_agent_{agent_id}/` directories. Use for independent tasks.

2. **Shared Storage State** (`storage_state` provided): All tasks start from the same browser state (cookies, localStorage). Use for tasks that need to branch from an authenticated session or shared context. Creates temporary profiles under system temp directories.

### Storage State Workflow

For branched tasks:
1. Run an initial task to establish browser state (e.g., login)
2. Export storage state using `_export_storage_state(agent)`
3. Pass storage state to `run_branched_tasks()` to spawn parallel tasks that inherit the state

## Development Commands

### Setup
```bash
# Install dependencies (requires Python 3.13+)
uv sync

# Alternative without uv
pip install -r requirements.txt
```

### Running
```bash
# Main interactive CLI
python main.py

# Select option 1-3 for different test modes
```

### Environment Configuration

Required in `.env`:
```
GOOGLE_API_KEY="your_gemini_api_key"
```

The SDK defaults to Google's Gemini (via `ChatGoogle` from browser-use), configured in `ParallelBrowserSDK.__init__()` with `llm_model="gemini-2.5-flash"`.

## Key Implementation Details

- **Error handling**: `execute_parallel()` uses `asyncio.gather(..., return_exceptions=True)` to ensure all tasks complete even if some fail
- **Cleanup**: Temporary user data directories are automatically removed in finally blocks
- **LLM instance sharing**: A single LLM instance is shared across all parallel agents in a batch to optimize resource usage
- **Agent execution**: Each agent runs asynchronously via `agent.run(max_steps=...)` with configurable step limits

## Extending the SDK

To add support for different LLM providers, modify `ParallelBrowserSDK._create_llm()` to return different chat model instances based on configuration or environment variables.

# Browserbase Starter Template

This backend project demonstrates how to launch a [Browser Use](https://github.com/browser-use/browser-use)
agent on a remote browser provided by [Browserbase](https://browserbase.com/).
It handles the Browserbase session lifecycle for you, connects Browser Use to the
session via CDP, and lets you configure your preferred LLM provider with
environment variables.

## Prerequisites

- Python 3.13 (matches the `pyproject.toml` requirement)
- A Browserbase account with an API key (and optionally a project ID)
- An API key for your preferred LLM provider (OpenAI, Google AI Studio, Anthropic, ...)

## Installation

```bash
uv sync
```

> If you are not using `uv`, install dependencies with `pip install -r requirements.txt`
> or `pip install browser-use httpx python-dotenv`.

## Configuration

Create a `.env` file (or export the variables directly) with at least the
following entries:

```env
BROWSERBASE_API_KEY="your_browserbase_api_key"
BROWSERBASE_PROJECT_ID="optional_project_id"
BROWSER_USE_LLM_PROVIDER="openai"          # or google, anthropic
BROWSER_USE_LLM_MODEL="gpt-4.1-mini"       # optional override per provider
OPENAI_API_KEY="your_openai_key"           # provider specific secrets
# GOOGLE_API_KEY="your_google_key"
# ANTHROPIC_API_KEY="your_anthropic_key"

# Optional extras
# BROWSERBASE_API_URL="https://api.browserbase.com/v1"
# BROWSER_USE_TASK="Custom agent instructions"
# BROWSER_USE_MAX_STEPS="20"
```

> Only set the secrets required by the LLM provider you plan to use.
>
> `python-dotenv` is included, so the script will automatically read values from `.env`.

## Running the template

```bash
python main.py
```

The script will:

1. Authenticate with Browserbase and create a new browser session
2. Connect Browser Use to the session via the returned WebSocket endpoint
3. Run the agent with your chosen LLM and task prompt
4. Print the agent's final result and clean up both the Browser Use browser and the Browserbase session

## Customisation Tips

- Pass additional Browserbase settings by editing the call to
  `browserbase.session()` in `main.py` (for example to select a region or tweak
  browser flags).
- Replace the default task description with `BROWSER_USE_TASK` to automate your own workflow.
- Extend `build_llm()` if you would like to support more providers or custom
  client configuration.

Happy automating!

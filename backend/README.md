# Parallel Browser SDK

This backend project provides a powerful SDK for running browser automation tasks in parallel using [Browser Use](https://github.com/browser-use/browser-use).

## Features

- 🚀 **Parallel Execution**: Run multiple browser tasks simultaneously with isolated profiles
- 🔄 **Branched Tasks**: Execute tasks from a shared browser state (cookies, sessions)
- 💬 **Conversational Agent**: AI assistant that clarifies vague requests and uses real-time tools
- 🛠️ **Built-in Tools**: Weather, date/time, location, calculations for the conversational agent
- 🤖 **Auto-Parallelization**: Agent automatically deploys multiple browsers when tasks can run in parallel
- 🧠 **Smart Mode**: Enhanced intelligence with self-verification, reflection, and error correction

## Prerequisites

- Python 3.13+
- Google AI API key (for Gemini model)

## Installation

```bash
uv sync
```

> If you are not using `uv`, install dependencies with `pip install -r requirements.txt`
> or `pip install browser-use httpx python-dotenv`.

## Configuration

Create a `.env` file with your Google API key:

```env
GOOGLE_API_KEY="your_google_api_key"
```

Optional environment variables:
- `QUIET=1` - Disable conversational prompts and prints
- `NON_INTERACTIVE=1` - Skip clarification questions

## Usage

### Interactive CLI

```bash
python main.py
```

Choose from:
1. Test SDK with simple parallel tasks
2. Test convenience functions
3. Test branched tasks from freshly captured state
4. **Test conversational agent (with built-in tools)**

### Conversational Agent Example

The conversational agent asks clarifying questions, uses real-time tools, and automatically deploys multiple parallel agents when beneficial:

#### Example 1: Proactive Tool Usage
```
Enter your vague task description: get me to California

🤖 Initial request: get me to California

🔧 Using tool: get_user_location()
📊 Result: Boston, Massachusetts

🔧 Using tool: get_current_date()
📊 Result: Saturday, October 04, 2025

❓ What dates are you looking to travel, and do you have a preference for which California city (LA, SF, San Diego)?
👤 Your answer: next week, LA

✅ Refined task: Search for flights from Boston to Los Angeles (LAX) for the week of October 11-18, 2025, and display the cheapest options with times and prices.

🔍 Analyzing if task can be parallelized...
✨ Running single agent task
   Reason: Single flight search query
```

#### Example 2: Automatic Parallelization
```
Enter your vague task description: compare flight prices from Boston to NYC on Kayak, Google Flights, and Expedia

❓ What are your preferred travel dates?
👤 Your answer: next weekend

🔧 Using tool: get_current_date()
📊 Result: Saturday, October 4, 2025

✅ Refined task: Compare flight prices from Boston to NYC for October 11-13, 2025 across multiple booking sites

🔍 Analyzing if task can be parallelized...

🚀 Deploying 3 parallel agents!
📋 Reason: Task involves comparing prices across multiple independent websites
   Agent 1: Go to Kayak.com and search for flights from Boston to NYC for October 11-13, 2025, extract cheapest option
   Agent 2: Go to Google Flights and search for flights from Boston to NYC for October 11-13, 2025, extract cheapest option
   Agent 3: Go to Expedia.com and search for flights from Boston to NYC for October 11-13, 2025, extract cheapest option

[3 browsers running in parallel...]

💡 RESULTS:
--- Agent 1 ---
Kayak: Boston to NYC, Oct 11-13: $187 (JetBlue)

--- Agent 2 ---
Google Flights: Boston to NYC, Oct 11-13: $179 (Delta)

--- Agent 3 ---
Expedia: Boston to NYC, Oct 11-13: $195 (American Airlines)

✅ Success rate: 100%
⏱️  Total execution time: 18.3s
```

### Available Tools

The conversational agent has access to:
- **get_current_date()** - Current date
- **get_current_datetime()** - Current date and time
- **get_user_location()** - User's approximate location based on IP (city, region, country)
- **get_weather(location)** - Weather for any city
- **calculate(expression)** - Basic math calculations

### Programmatic Usage

```python
from browser_tasks import run_conversational_task

result = await run_conversational_task(
    initial_query="compare prices for laptops on Amazon, Best Buy, and Newegg",
    max_steps=15,
    headless=True,
    enable_parallel_agents=True  # Default: True
)

print(f"Refined task: {result['refined_task']}")
print(f"Number of agents: {len(result['task_descriptions'])}")
print(f"Success rate: {result['execution_result']['success_rate']:.0%}")

# Access individual agent results
for i, exec_result in enumerate(result['execution_result']['execution_results'], 1):
    if exec_result.success:
        print(f"Agent {i}: {exec_result.result.final_result()}")
```

### How Auto-Parallelization Works

1. **Task Analysis**: After clarification and refinement, the agent analyzes if the task can benefit from parallel execution
2. **Smart Splitting**: Tasks are split when:
   - Comparing multiple websites/options
   - Gathering info from independent sources
   - Multiple subtasks with no dependencies
3. **Parallel Execution**: Multiple browser agents run simultaneously with isolated profiles
4. **Aggregated Results**: All results are collected and presented together

**Disable auto-parallelization:**
```python
result = await run_conversational_task(
    initial_query="your task",
    enable_parallel_agents=False  # Force single agent
)
```

### 🧠 Smart Mode

All agents run in **Smart Mode** by default, which adds enhanced intelligence:

#### What Smart Mode Does

1. **Self-Verification**: Agents verify their actions succeeded
   - After applying filters, checks if results match criteria
   - Detects when UI interactions fail silently
   
2. **Reflection & Self-Correction**: Agents recognize mistakes and adapt
   - Example: If setting "arrival by 6 PM" but seeing 8:30 PM flights, tries different approach
   - Automatically switches strategies when something doesn't work

3. **Custom Verification Tool**: Built-in `verify_page_state()` action
   ```python
   # Agent can call this to verify its actions
   verify_page_state(
       expected_condition="flights arriving before 6 PM",
       current_observation="seeing flights at 8:30 PM"
   )
   # Returns: ❌ MISMATCH - try different approach
   ```

4. **Enhanced Error Handling**: 
   - 5 retry attempts instead of 3
   - Explains what went wrong and tries alternatives
   - Prefers accuracy over speed

5. **Structured Output**: Extracts data in clear, organized formats

#### Example: Smart Mode in Action

```
Task: Find flights from DC to Boston arriving by 6 PM

❌ Without Smart Mode:
- Applies filter
- Assumes it worked
- Returns flights arriving at 8:30 PM ✗

✅ With Smart Mode:
- Applies filter
- Verifies results match criteria
- Detects 8:30 PM arrival doesn't match "by 6 PM"
- Tries alternative filtering approach
- Manually filters results if needed
- Returns only flights arriving by 6 PM ✓
```

**Disable smart mode (not recommended):**
```python
result = await run_parallel_tasks(
    task_descriptions=["your task"],
    enable_smart_mode=False  # Disables reflection and verification
)
```

Happy automating!

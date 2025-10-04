import os
import json
import shutil
import asyncio
import inspect
import tempfile
import uuid
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
from browser_use import Agent, ChatGoogle, Browser

load_dotenv()

@dataclass
class ParallelTask:
    """Represents a single task to be executed by an agent"""
    task_description: str
    max_steps: int = 10
    agent_id: Optional[str] = None

    def __post_init__(self):
        if self.agent_id is None:
            self.agent_id = str(uuid.uuid4())[:8]

@dataclass
class ParallelExecutionResult:
    """Result of parallel task execution"""
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class BrowserConfig:
    """Browser configuration for agents"""
    headless: bool = True
    storage_state: Optional[str] = None
    separate_profiles: bool = True
    profile_prefix: str = "temp_agent"
    temp_user_data_dirs: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.storage_state and self.separate_profiles:
            self.separate_profiles = False


async def _resolve_maybe_awaitable(value: Any) -> Any:
    """Return awaited value when necessary"""
    if inspect.isawaitable(value):
        return await value
    return value


async def _export_storage_state(agent: Agent) -> Dict[str, Any]:
    """Extract storage state dict from an agent's browser session"""
    candidates = []

    session = getattr(agent, "browser_session", None)
    if session is not None:
        candidates.append(session)

    browser = getattr(agent, "browser", None)
    if browser is not None and browser not in candidates:
        candidates.append(browser)

    for source in candidates:
        for attr_name in ("export_storage_state", "storage_state", "get_storage_state", "_cdp_get_storage_state"):
            attr = getattr(source, attr_name, None)
            if attr is None:
                continue

            try:
                result = attr() if callable(attr) else attr
            except TypeError as exc:
                if "path" in str(exc) and callable(attr):
                    try:
                        result = attr(path=None)
                    except Exception:
                        continue
                else:
                    continue
            except Exception:
                continue

            try:
                result = await _resolve_maybe_awaitable(result)
            except TypeError:
                continue

            if isinstance(result, dict):
                return result

            if isinstance(result, str) and os.path.exists(result):
                try:
                    with open(result, "r", encoding="utf-8") as file:
                        return json.load(file)
                except (OSError, json.JSONDecodeError):
                    continue

    raise RuntimeError("Unable to export storage state from agent; provide a manual storage state export.")

class ParallelBrowserSDK:
    """SDK for executing browser tasks in parallel"""

    def __init__(self, llm_api_key: Optional[str] = None, llm_model: str = "gemini-2.5-flash"):
        self.llm_api_key = llm_api_key or os.environ.get("GOOGLE_API_KEY")
        self.llm_model = llm_model

    def _create_llm(self):
        """Create LLM instance"""
        return ChatGoogle(
            model=self.llm_model,
            api_key=self.llm_api_key
        )

    def _create_browser(self, config: BrowserConfig, agent_id: str) -> Browser:
        """Create browser instance based on configuration"""
        if config.storage_state:
            temp_profile_dir = tempfile.mkdtemp(prefix="tmp-browser-use-profile-")
            config.temp_user_data_dirs.append(temp_profile_dir)
            return Browser(
                headless=config.headless,
                storage_state=config.storage_state,
                user_data_dir=temp_profile_dir
            )

        profile_dir = f"./{config.profile_prefix}_{agent_id}" if config.separate_profiles else None

        return Browser(
            headless=config.headless,
            user_data_dir=profile_dir,
            storage_state=None
        )

    async def execute_parallel(self,
                             tasks: List[ParallelTask],
                             browser_config: Optional[BrowserConfig] = None) -> Dict[str, Any]:
        """
        Execute multiple browser tasks in parallel

        Args:
            tasks: List of ParallelTask objects
            browser_config: Browser configuration for all agents

        Returns:
            Dict with execution results, timing, and statistics
        """
        if browser_config is None:
            browser_config = BrowserConfig()

        cleanup_dirs = browser_config.temp_user_data_dirs

        try:
            llm = self._create_llm()
            start_time = asyncio.get_event_loop().time()

            # Create agents for all tasks
            agents = []
            for task in tasks:
                browser = self._create_browser(browser_config, task.agent_id)
                agent = Agent(
                    task=task.task_description,
                    browser=browser,
                    llm=llm
                )
                agents.append((agent, task))

            # Execute all agents in parallel
            agent_tasks = [
                agent.run(max_steps=task.max_steps)
                for agent, task in agents
            ]

            results = await asyncio.gather(*agent_tasks, return_exceptions=True)

            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            # Process results
            execution_results = []
            successful_count = 0
            failed_count = 0

            for i, (result, (agent, task)) in enumerate(zip(results, agents)):
                if isinstance(result, Exception):
                    execution_results.append(ParallelExecutionResult(
                        task_id=task.agent_id,
                        success=False,
                        result=None,
                        error=str(result),
                        execution_time=total_time
                    ))
                    failed_count += 1
                else:
                    execution_results.append(ParallelExecutionResult(
                        task_id=task.agent_id,
                        success=True,
                        result=result,
                        error=None,
                        execution_time=total_time
                    ))
                    successful_count += 1

            return {
                'execution_results': execution_results,
                'total_time': total_time,
                'successful_count': successful_count,
                'failed_count': failed_count,
                'total_tasks': len(tasks),
                'success_rate': successful_count / len(tasks) if tasks else 0
            }
        finally:
            for temp_dir in cleanup_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)
            browser_config.temp_user_data_dirs.clear()

# Convenience functions for common use cases
async def run_parallel_tasks(task_descriptions: List[str],
                           max_steps: int = 10,
                           headless: bool = True,
                           llm_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Simple function to run multiple tasks in parallel with separate profiles

    Args:
        task_descriptions: List of task descriptions
        max_steps: Maximum steps per agent
        headless: Run browsers in headless mode
        llm_api_key: API key for LLM

    Returns:
        Execution results
    """
    sdk = ParallelBrowserSDK(llm_api_key=llm_api_key)

    tasks = [
        ParallelTask(task_description=desc, max_steps=max_steps)
        for desc in task_descriptions
    ]

    config = BrowserConfig(headless=headless, separate_profiles=True)

    return await sdk.execute_parallel(tasks, config)

async def run_branched_tasks(storage_state: Union[str, Dict[str, Any]],
                           parallel_task_descriptions: List[str],
                           max_steps: int = 10,
                           headless: bool = True,
                           llm_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Simple function to run tasks that branch from an existing browser storage state

    Args:
        storage_state: Path to storage state file or storage state dict
        parallel_task_descriptions: List of parallel task descriptions
        max_steps: Maximum steps per agent
        headless: Run browsers in headless mode
        llm_api_key: API key for LLM

    Returns:
        Execution results from parallel execution
    """
    sdk = ParallelBrowserSDK(llm_api_key=llm_api_key)

    parallel_tasks = [
        ParallelTask(task_description=desc, max_steps=max_steps)
        for desc in parallel_task_descriptions
    ]

    # Handle storage state - could be file path or dict
    temp_file: Optional[str] = None
    if isinstance(storage_state, dict):
        # If it's a dict, save it to a temporary file
        temp_file = f"./temp_storage_{uuid.uuid4().hex[:8]}.json"
        with open(temp_file, 'w') as f:
            json.dump(storage_state, f)
        storage_file = temp_file
    else:
        # Assume it's a file path
        storage_file = storage_state

    config = BrowserConfig(
        headless=headless,
        storage_state=storage_file,
        separate_profiles=False
    )

    try:
        return await sdk.execute_parallel(parallel_tasks, config)
    finally:
        if temp_file:
            try:
                os.remove(temp_file)
            except OSError:
                pass

# Example usage and testing
async def test_sdk():
    """Test the SDK with simple examples"""
    sdk = ParallelBrowserSDK()

    print("🧪 Testing parallel execution with separate profiles...")

    # Test 1: Simple parallel tasks
    tasks = [
        ParallelTask("Go to https://httpbin.org/get and extract the IP address", max_steps=5),
        ParallelTask("Go to https://httpbin.org/user-agent and extract the user agent", max_steps=5),
    ]

    config = BrowserConfig(headless=True, separate_profiles=True)
    result = await sdk.execute_parallel(tasks, config)

    print(f"✅ Completed {result['successful_count']}/{result['total_tasks']} tasks")
    print(f"⏱️  Total time: {result['total_time']:.2f} seconds")

    return result

if __name__ == "__main__":
    print("🚀 Parallel Browser SDK")
    print("Choose test option:")
    print("1. Test SDK with simple parallel tasks")
    print("2. Test convenience functions")
    print("3. Test branched tasks from freshly captured state")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "2":
        print("Testing simple convenience functions...")

        async def test_convenience():
            result = await run_parallel_tasks([
                "Go to https://httpbin.org/ip and get my IP",
                "Go to https://httpbin.org/headers and get headers"
            ], headless=True)

            print(f"Success rate: {result['success_rate']:.0%}")
            return result

        asyncio.run(test_convenience())

    elif choice == "3":
        print("Testing branched tasks starting from a freshly captured storage state...")

        async def test_branched():
            sdk = ParallelBrowserSDK()
            llm = sdk._create_llm()

            browser = Browser(headless=True, keep_alive=True)
            agent = Agent(
                task="Go to https://httpbin.org/cookies/set/test_session/abc123 and verify the cookie is set",
                browser=browser,
                llm=llm
            )

            storage_state: Optional[Dict[str, Any]] = None

            try:
                await agent.run(max_steps=8)
                storage_state = await _export_storage_state(agent)
            finally:
                try:
                    await browser.stop()
                except Exception:
                    pass

            if not storage_state:
                print("⚠️  Unable to capture storage state from the initial task.")
                return

            print("🌟 Running branched tasks with captured storage state...")

            result = await run_branched_tasks(
                storage_state=storage_state,
                parallel_task_descriptions=[
                    "Go to https://httpbin.org/cookies and verify my test_session cookie is present",
                    "Go to https://httpbin.org/get and extract my IP address",
                    "Go to https://httpbin.org/user-agent and extract the user agent"
                ],
                headless=True,
                llm_api_key=sdk.llm_api_key
            )

            print("🎉 Branched tasks completed!")
            print(f"Success rate: {result['success_rate']:.0%}")
            print(f"Total time: {result['total_time']:.2f} seconds")

            return result

        asyncio.run(test_branched())

    else:
        asyncio.run(test_sdk())

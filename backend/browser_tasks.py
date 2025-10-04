"""Browser task utilities built on top of browser-use."""

import asyncio
import inspect
import json
import os
import shutil
import tempfile
import uuid
import contextlib
import sys
import base64
import queue
import signal
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from browser_use import Agent, Browser, ChatGoogle, Tools
from browser_use.agent.service import ActionResult
from dotenv import load_dotenv
import sounddevice as sd
import websockets
from bubus import EventBus as _PatchedEventBus

load_dotenv()

# Create enhanced tools with reflection capability
enhanced_tools = Tools()


# Raise the EventBus memory warning threshold so browser-use can retain richer history
# without tripping the 50MB safeguard. Use env var BROWSER_EVENTBUS_MEMORY_WARN_MB to
# customize (set to <=0 or inf to disable warnings entirely).
_EVENTBUS_MEMORY_WARN_RAW = os.environ.get("BROWSER_EVENTBUS_MEMORY_WARN_MB", "inf")
try:
    _EVENTBUS_MEMORY_WARN_MB = float(_EVENTBUS_MEMORY_WARN_RAW)
except ValueError:
    _EVENTBUS_MEMORY_WARN_MB = float("inf")


def _patched_eventbus_memory_check(self: _PatchedEventBus) -> None:
    """Custom memory check using the configured warning threshold."""

    import logging

    if not math.isfinite(_EVENTBUS_MEMORY_WARN_MB) or _EVENTBUS_MEMORY_WARN_MB <= 0:
        return

    total_bytes = 0
    bus_details: list[tuple[str, int, int, int]] = []

    for bus in list(_PatchedEventBus.all_instances):
        try:
            bus_bytes = 0

            for event in bus.event_history.values():
                bus_bytes += sys.getsizeof(event)
                if hasattr(event, "__dict__"):
                    for value in event.__dict__.values():
                        if isinstance(value, (str, bytes, list, dict, tuple, set)):
                            bus_bytes += sys.getsizeof(value)

            if bus.event_queue and hasattr(bus.event_queue, "_queue"):
                queue_storage = bus.event_queue._queue  # type: ignore[attr-defined]
                for queued_event in queue_storage:
                    bus_bytes += sys.getsizeof(queued_event)
                    if hasattr(queued_event, "__dict__"):
                        for value in queued_event.__dict__.values():
                            if isinstance(value, (str, bytes, list, dict, tuple, set)):
                                bus_bytes += sys.getsizeof(value)

            total_bytes += bus_bytes
            bus_details.append(
                (
                    bus.name,
                    bus_bytes,
                    len(bus.event_history),
                    bus.event_queue.qsize() if bus.event_queue else 0,
                )
            )
        except Exception:
            continue

    total_mb = total_bytes / (1024 * 1024)
    if total_mb <= _EVENTBUS_MEMORY_WARN_MB:
        return

    details: list[str] = []
    for name, bytes_used, history_size, queue_size in sorted(bus_details, key=lambda entry: entry[1], reverse=True):
        mb = bytes_used / (1024 * 1024)
        if mb > 0.1:
            details.append(f"  - {name}: {mb:.1f}MB (history={history_size}, queue={queue_size})")

    warning_msg = (
        f"\n⚠️  WARNING: Total EventBus memory usage is {total_mb:.1f}MB (> {_EVENTBUS_MEMORY_WARN_MB:.0f}MB limit)\n"
        f"Active EventBus instances: {len(_PatchedEventBus.all_instances)}\n"
    )

    if details:
        warning_msg += "Memory breakdown:\n" + "\n".join(details[:5])
        if len(details) > 5:
            warning_msg += f"\n  ... and {len(details) - 5} more"

    warning_msg += "\nConsider:\n"
    warning_msg += "  - Reducing max_history_size\n"
    warning_msg += "  - Clearing completed EventBus instances with stop(clear=True)\n"
    warning_msg += "  - Reducing event payload sizes\n"

    logging.getLogger("bubus").warning(warning_msg)


if hasattr(_PatchedEventBus, "_check_total_memory_usage"):
    _PatchedEventBus._check_total_memory_usage = _patched_eventbus_memory_check


@enhanced_tools.action(
    description='Verify if the current page state matches expected criteria. Use this to check if filters, searches, or actions worked correctly.'
)
def verify_page_state(
    expected_condition: str,
    current_observation: str,
) -> ActionResult:
    """
    Verify if the current page matches expected conditions.
    
    Args:
        expected_condition: What you expect to see (e.g., "flights arriving before 6 PM")
        current_observation: What you actually see on the page (e.g., "flights showing 8:30 PM arrival")
    
    Returns:
        ActionResult with verification status and recommendations
    """
    # Simple heuristic check - in practice, could use LLM for more sophisticated verification
    matches = expected_condition.lower() in current_observation.lower()
    
    if matches:
        return ActionResult(
            extracted_content=f"✅ VERIFIED: Page state matches expected condition.\nExpected: {expected_condition}\nObserved: {current_observation}",
            include_in_memory=True
        )
    else:
        return ActionResult(
            extracted_content=f"❌ MISMATCH DETECTED: Page state does NOT match expected condition.\nExpected: {expected_condition}\nActual: {current_observation}\n\n⚠️ RECOMMENDATION: Try a different approach or manually filter the results.",
            include_in_memory=True,
            error="Verification failed - expected condition not met"
        )


# Quiet/interactive controls via environment
_QUIET_MODE = os.environ.get("QUIET", "0") == "1"
_NON_INTERACTIVE = os.environ.get("NON_INTERACTIVE", "0") == "1"

# Voice transcription configuration
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SAMPLE_RATE = 16_000
CHANNELS = 1
FRAMES_PER_CHUNK = int(SAMPLE_RATE * 0.02)


async def get_microphone_audio(shutdown_event: asyncio.Event) -> AsyncGenerator[bytes, None]:
    """Yield raw PCM16 audio chunks from the system microphone until shutdown_event is set."""
    audio_queue: queue.Queue[bytes] = queue.Queue()

    def _callback(indata, _frames, _time, status):
        if status:
            print(f"[sounddevice] {status}", file=sys.stderr)
        audio_queue.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAMES_PER_CHUNK,
        dtype="int16",
        channels=CHANNELS,
        callback=_callback,
    )

    with stream:
        while not shutdown_event.is_set():
            try:
                chunk = await asyncio.to_thread(audio_queue.get, True, 0.1)
            except queue.Empty:
                continue
            if chunk:
                yield chunk


async def stream_microphone_audio(
    websocket: websockets.WebSocketClientProtocol,
    shutdown_event: asyncio.Event,
) -> None:
    """Stream microphone audio to the realtime endpoint until shutdown_event is set."""
    try:
        async for audio_chunk in get_microphone_audio(shutdown_event):
            audio_b64 = base64.b64encode(audio_chunk).decode("ascii")
            payload = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
            await websocket.send(json.dumps(payload))
            if shutdown_event.is_set():
                break
    finally:
        shutdown_event.set()


def build_session_update(prompt: str = "", language: str = "en", vad_threshold: float = 0.3) -> str:
    """Create the session.update payload for configuring transcription."""
    session_update = {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe",
                "prompt": prompt,
                "language": language,
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": vad_threshold,  # Lower threshold = less sensitive to silence
                "prefix_padding_ms": 500,
                "silence_duration_ms": 2000,
            },
        },
    }
    return json.dumps(session_update)


@contextlib.asynccontextmanager
async def realtime_connection(api_key: str):
    """Async context manager that yields an authenticated websocket connection."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }
    websocket = await websockets.connect(REALTIME_URL, additional_headers=headers)
    try:
        yield websocket
    finally:
        await websocket.close()


async def get_voice_input(prompt_text: str, debug: bool = False) -> str:
    """Get voice input from the user and return transcribed text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    print(f"🎤 {prompt_text}")
    print("   (Speak now, will auto-detect when you're done...)")

    shutdown_event = asyncio.Event()
    transcribed_text = ""
    speech_started = False
    speech_stopped = False

    async with realtime_connection(api_key) as websocket:
        await websocket.send(build_session_update("User responding to browser agent question.", vad_threshold=0.3))

        audio_task = asyncio.create_task(
            stream_microphone_audio(websocket, shutdown_event)
        )

        try:
            while not shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

                event = json.loads(message)
                event_type = event.get("type")
                if not event_type:
                    continue

                if debug:
                    print(f"[DEBUG] Event: {event_type}")

                # Track when speech starts and stops
                if event_type == "input_audio_buffer.speech_started":
                    speech_started = True
                    if debug:
                        print("[DEBUG] Speech started")
                
                elif event_type == "input_audio_buffer.speech_stopped":
                    speech_stopped = True
                    if debug:
                        print("[DEBUG] Speech stopped, waiting for transcription...")
                
                # Only process transcription after speech has started AND stopped
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript")
                    if debug:
                        print(f"[DEBUG] Transcription received: {transcript}")
                    
                    if transcript and speech_started:
                        # Accumulate transcription (in case there are multiple chunks)
                        if transcribed_text:
                            transcribed_text += " " + transcript
                        else:
                            transcribed_text = transcript
                        
                        # Only stop if we've had a proper speech start->stop cycle
                        if speech_stopped:
                            if debug:
                                print("[DEBUG] Complete utterance detected, stopping")
                            shutdown_event.set()

                elif event_type == "error":
                    print("Transcription error:", json.dumps(event, indent=2))
        finally:
            shutdown_event.set()
            try:
                await websocket.close()
            except websockets.exceptions.ConnectionClosed:
                pass
            try:
                await audio_task
            except websockets.exceptions.ConnectionClosed:
                pass

    return transcribed_text.strip()


async def speak_text(text: str) -> None:
    """Convert text to speech and play it."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        # Save to temporary file and play
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        try:
            # Play audio using system player
            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", tmp_path], check=True)
            elif sys.platform == "linux":
                subprocess.run(["mpg123", tmp_path], check=True)
            elif sys.platform == "win32":
                try:
                    # Use -nodisp to hide the ffplay window, -autoexit to close when done.
                    subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp_path], check=True, capture_output=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback if ffplay is not installed or fails
                    print("INFO: 'ffplay' not found. Falling back to default media player. Install FFmpeg for direct audio playback.")
                    subprocess.run(["start", tmp_path], shell=True, check=True)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")


@contextlib.contextmanager
def _suppress_native_stderr() -> Any:
    """Temporarily redirect the C-level stderr (fd=2) to os.devnull.

    This silences early absl/gRPC warnings that bypass Python's sys.stderr.
    """

    try:
        fd = sys.stderr.fileno()
    except Exception:
        # Fallback: if no real fileno, just yield
        yield
        return

    saved_fd = os.dup(fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, fd)
        yield
    finally:
        try:
            os.dup2(saved_fd, fd)
        finally:
            os.close(saved_fd)
            os.close(devnull_fd)


@dataclass
class ParallelTask:
    """Represents a single task to be executed by an agent."""

    task_description: str
    max_steps: int = 10
    agent_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.agent_id is None:
            self.agent_id = str(uuid.uuid4())[:8]


@dataclass
class ParallelExecutionResult:
    """Result of parallel task execution."""

    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class BrowserConfig:
    """Browser configuration for agents."""
    cdp_url: Optional[str] = None  # URL to connect to existing Chrome instance
    headless: bool = True
    storage_state: Optional[str] = None
    separate_profiles: bool = True
    profile_prefix: str = "temp_agent"
    temp_user_data_dirs: List[str] = field(default_factory=list)
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0

    def __post_init__(self) -> None:
        if self.storage_state and self.separate_profiles:
            self.separate_profiles = False


async def _resolve_maybe_awaitable(value: Any) -> Any:
    """Return awaited value when necessary."""

    if inspect.isawaitable(value):
        return await value
    return value


async def _export_storage_state(agent: Agent) -> Dict[str, Any]:
    """Extract storage state dict from an agent's browser session."""

    candidates = []

    session = getattr(agent, "browser_session", None)
    if session is not None:
        candidates.append(session)

    browser = getattr(agent, "browser", None)
    if browser is not None and browser not in candidates:
        candidates.append(browser)

    for source in candidates:
        for attr_name in (
            "export_storage_state",
            "storage_state",
            "get_storage_state",
            "_cdp_get_storage_state",
        ):
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

    raise RuntimeError(
        "Unable to export storage state from agent; provide a manual storage state export."
    )

class ParallelBrowserSDK:
    """SDK for executing browser tasks in parallel."""

    def __init__(self, llm_api_key: Optional[str] = None, llm_model: str = "gemini-2.5-flash") -> None:
        self.llm_api_key = llm_api_key or os.environ.get("GOOGLE_API_KEY")
        self.llm_model = llm_model

    def _create_llm(self) -> ChatGoogle:
        """Create LLM instance."""
        with _suppress_native_stderr():
            llm = ChatGoogle(model=self.llm_model, api_key=self.llm_api_key)
        return llm

    def _create_browser(self, config: BrowserConfig, agent_id: str) -> Browser:
        """Create browser instance based on configuration."""
        # NEW: Check if we should connect to existing Chrome session
        print("Checking for open browser...")
        if hasattr(config, 'cdp_url') and config.cdp_url:
            return Browser(
                cdp_url=config.cdp_url,  # Connect to existing Chrome
                headless=False,  # Must be False when using CDP
            )
        print("No open browser found, launching new guest window...")
        if config.storage_state:
            temp_profile_dir = tempfile.mkdtemp(prefix="tmp-browser-use-profile-")
            config.temp_user_data_dirs.append(temp_profile_dir)
            return Browser(
                headless=config.headless,
                storage_state=config.storage_state,
                user_data_dir=temp_profile_dir,
                disable_security=True,  # Speed up page loads
            )

        profile_dir = (
            f"./{config.profile_prefix}_{agent_id}" if config.separate_profiles else None
        )

        return Browser(
            headless=config.headless,
            user_data_dir=profile_dir,
            storage_state=None,
            disable_security=True,  # Speed up page loads
        )

    async def _execute_task_with_retry(
        self,
        agent: Agent,
        task: ParallelTask,
        browser_config: BrowserConfig,
        llm: ChatGoogle,
    ) -> Any:
        """Execute a single task with retry logic for CDP errors."""

        last_exception = None
        max_retries = browser_config.max_retries if browser_config.retry_on_failure else 1

        for attempt in range(max_retries):
            try:
                result = await agent.run(max_steps=task.max_steps)
                return result
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if this is a retryable CDP error
                is_cdp_error = any(phrase in error_msg for phrase in [
                    "no node with given id found",
                    "no node found for given backend id",
                    "cdp error",
                    "failed to click element",
                    "node is detached from document",
                ])

                # Only retry on CDP errors and if we have retries left
                if is_cdp_error and attempt < max_retries - 1:
                    if not _QUIET_MODE:
                        print(f"⚠️  CDP error on task {task.agent_id} (attempt {attempt + 1}/{max_retries}), retrying...")

                    await asyncio.sleep(browser_config.retry_delay)

                    # Recreate browser and agent for retry
                    try:
                        await agent.browser.stop()
                    except Exception:
                        pass

                    browser = self._create_browser(browser_config, task.agent_id)
                    agent = Agent(
                        task=task.task_description,
                        browser=browser,
                        llm=llm,
                        tools=enhanced_tools,
                        max_failures=3,
                        use_thinking=False,  # Disable to avoid LLM timeouts
                    )
                else:
                    # Non-retryable error or out of retries
                    raise last_exception

        raise last_exception

    async def execute_parallel(
        self,
        tasks: List[ParallelTask],
        browser_config: Optional[BrowserConfig] = None,
    ) -> Dict[str, Any]:
        """Execute multiple browser tasks in parallel."""

        if browser_config is None:
            browser_config = BrowserConfig()

        cleanup_dirs = browser_config.temp_user_data_dirs

        try:
            llm = self._create_llm()
            start_time = asyncio.get_event_loop().time()

            agents = []
            for task in tasks:
                browser = self._create_browser(browser_config, task.agent_id)
                await browser.start()
                agent = Agent(
                    task=task.task_description,
                    browser=browser,
                    llm=llm,
                    tools=enhanced_tools,  # Add reflection tools
                    max_failures=3,  # Reduce failures to save memory
                    use_thinking=False,  # Disable to avoid LLM timeouts
                )
                agents.append((agent, task))

            agent_tasks = [
                self._execute_task_with_retry(agent, task, browser_config, llm)
                for agent, task in agents
            ]

            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            for agent, task in agents:
                try:
                    await agent.browser.stop()
                except Exception:
                    pass
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            execution_results: List[ParallelExecutionResult] = []
            successful_count = 0
            failed_count = 0

            for result, (agent, task) in zip(results, agents):
                if isinstance(result, Exception):
                    execution_results.append(
                        ParallelExecutionResult(
                            task_id=task.agent_id,
                            success=False,
                            result=None,
                            error=str(result),
                            execution_time=total_time,
                        )
                    )
                    failed_count += 1
                else:
                    execution_results.append(
                        ParallelExecutionResult(
                            task_id=task.agent_id,
                            success=True,
                            result=result,
                            error=None,
                            execution_time=total_time,
                        )
                    )
                    successful_count += 1

            return {
                "execution_results": execution_results,
                "total_time": total_time,
                "successful_count": successful_count,
                "failed_count": failed_count,
                "total_tasks": len(tasks),
                "success_rate": successful_count / len(tasks) if tasks else 0,
            }
        finally:
            for temp_dir in cleanup_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)
            browser_config.temp_user_data_dirs.clear()


# Enhanced system message for smarter agent behavior
SMART_AGENT_SYSTEM_MESSAGE = """
🧠 ENHANCED INTELLIGENCE INSTRUCTIONS:

1. VERIFY YOUR ACTIONS:
   - After applying filters or settings, ALWAYS verify they were applied correctly
   - If something doesn't match your goal, try a different approach immediately
   - Don't just assume an action worked - check the actual page state

2. SELF-CORRECT WHEN NEEDED:
   - If you see results that don't match your filters/criteria, recognize this immediately
   - Example: If you set "arrival by 6 PM" but see 8:30 PM flights, the filter didn't work
   - Try alternative methods: different selectors, UI approaches, or manual filtering

3. BE PRECISE WITH INTERACTIONS:
   - Use exact selectors and specific actions
   - Wait for elements to load before interacting
   - Prefer clicking visible, stable elements over dynamic ones

4. EXTRACT STRUCTURED DATA:
   - When asked to find or compare information, extract it in a clear, structured format
   - Use tables, lists, or JSON-like output for multiple items
   - Include all relevant details (prices, times, names, etc.)

5. THINK STEP-BY-STEP:
   - Before completing a task, review if all requirements are met
   - If unsure, double-check the page state matches your task goals
   - Don't be satisfied with partial completion

6. HANDLE FAILURES GRACEFULLY:
   - If an approach fails 2-3 times, try a completely different method
   - Explain what went wrong and what you're trying instead
   - Consider if manual data extraction might be more reliable than automated filters

Remember: Being thorough and accurate is more important than being fast.
"""


async def run_parallel_tasks(
    task_descriptions: List[str],
    max_steps: int = 10,
    headless: bool = True,
    llm_api_key: Optional[str] = None,
    enable_smart_mode: bool = True,
) -> Dict[str, Any]:
    """Simple function to run multiple tasks in parallel with separate profiles.
    
    Args:
        task_descriptions: List of task descriptions to execute
        max_steps: Maximum steps per agent
        headless: Run browsers in headless mode
        llm_api_key: Optional Google API key
        enable_smart_mode: Enable enhanced intelligence with reflection and self-correction
    """

    sdk = ParallelBrowserSDK(llm_api_key=llm_api_key)

    tasks = []
    for desc in task_descriptions:
        # Add smart mode instructions to task if enabled
        if enable_smart_mode:
            enhanced_desc = f"{desc}\n\n{SMART_AGENT_SYSTEM_MESSAGE}"
            tasks.append(ParallelTask(task_description=enhanced_desc, max_steps=max_steps))
        else:
            tasks.append(ParallelTask(task_description=desc, max_steps=max_steps))

    config = BrowserConfig(headless=headless, separate_profiles=True)

    return await sdk.execute_parallel(tasks, config)


async def run_branched_tasks(
    storage_state: Union[str, Dict[str, Any]],
    parallel_task_descriptions: List[str],
    max_steps: int = 10,
    headless: bool = True,
    llm_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run tasks that branch from an existing browser storage state."""

    sdk = ParallelBrowserSDK(llm_api_key=llm_api_key)

    parallel_tasks = [
        ParallelTask(task_description=desc, max_steps=max_steps)
        for desc in parallel_task_descriptions
    ]

    temp_file: Optional[str] = None
    if isinstance(storage_state, dict):
        temp_file = f"./temp_storage_{uuid.uuid4().hex[:8]}.json"
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(storage_state, file)
        storage_file = temp_file
    else:
        storage_file = storage_state

    config = BrowserConfig(headless=headless, storage_state=storage_file, separate_profiles=False)

    try:
        return await sdk.execute_parallel(parallel_tasks, config)
    finally:
        if temp_file:
            try:
                os.remove(temp_file)
            except OSError:
                pass


async def test_sdk() -> Dict[str, Any]:
    """Test the SDK with simple examples."""

    sdk = ParallelBrowserSDK()

    print("🧪 Testing parallel execution with separate profiles...")

    tasks = [
        ParallelTask("Go to https://httpbin.org/get and extract the IP address", max_steps=5),
        ParallelTask("Go to https://httpbin.org/user-agent and extract the user agent", max_steps=5),
    ]

    config = BrowserConfig(headless=True, separate_profiles=True)
    result = await sdk.execute_parallel(tasks, config)

    print(f"✅ Completed {result['successful_count']}/{result['total_tasks']} tasks")
    print(f"⏱️  Total time: {result['total_time']:.2f} seconds")

    return result


async def test_branched_tasks() -> Optional[Dict[str, Any]]:
    """Test branched tasks starting from a freshly captured storage state."""

    sdk = ParallelBrowserSDK()
    llm = sdk._create_llm()

    browser = Browser(headless=True, keep_alive=True, disable_security=True)
    agent = Agent(
        task="Go to https://httpbin.org/cookies/set/test_session/abc123 and verify the cookie is set",
        browser=browser,
        llm=llm,
        tools=enhanced_tools,
        max_failures=3,
        use_thinking=False,
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
        return None

    print("🌟 Running branched tasks with captured storage state...")

    result = await run_branched_tasks(
        storage_state=storage_state,
        parallel_task_descriptions=[
            "Go to https://httpbin.org/cookies and verify my test_session cookie is present",
            "Go to https://httpbin.org/get and extract my IP address",
            "Go to https://httpbin.org/user-agent and extract the user agent",
        ],
        headless=True,
        llm_api_key=sdk.llm_api_key,
    )

    print("🎉 Branched tasks completed!")
    print(f"Success rate: {result['success_rate']:.0%}")
    print(f"Total time: {result['total_time']:.2f} seconds")

    return result


class ConversationalBrowserAgent:
    """Agent that clarifies user queries before executing browser tasks."""

    def __init__(self, llm_api_key: Optional[str] = None, llm_model: str = "gemini-2.5-flash") -> None:
        self.sdk = ParallelBrowserSDK(llm_api_key=llm_api_key, llm_model=llm_model)
        self.llm_api_key = llm_api_key or os.environ.get("GOOGLE_API_KEY")
        self.llm_model = llm_model
        self._tool_functions = self._register_tools()

    def _register_tools(self) -> Dict[str, callable]:
        """Register simple tool functions."""
        from datetime import datetime
        import httpx

        def get_current_datetime() -> str:
            """Get the current date and time."""
            return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        def get_current_date() -> str:
            """Get the current date."""
            return datetime.now().strftime("%A, %B %d, %Y")

        def get_user_location() -> str:
            """Get the user's approximate location based on IP address."""
            try:
                # Try ipgeolocation.io API (accurate and reliable)
                response = httpx.get("https://ipgeolocation.abstractapi.com/v1/?api_key=free", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    city = data.get("city", "")
                    region = data.get("region", "")
                    country = data.get("country", "")

                    location_parts = []
                    if city:
                        location_parts.append(city)
                    if region and region != city:
                        location_parts.append(region)
                    if country:
                        location_parts.append(country)

                    if location_parts:
                        return ", ".join(location_parts)
            except Exception:
                pass

            try:
                # Fallback to ipinfo.io (no API key needed for basic info)
                response = httpx.get("https://ipinfo.io/json", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    city = data.get("city", "")
                    region = data.get("region", "")
                    country = data.get("country", "")

                    location_parts = []
                    if city:
                        location_parts.append(city)
                    if region and region != city:
                        location_parts.append(region)
                    if country:
                        location_parts.append(country)

                    if location_parts:
                        return ", ".join(location_parts)
            except Exception:
                pass

            try:
                # Final fallback to ipapi.com (simple and free)
                response = httpx.get("https://ipapi.co/json/", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    city = data.get("city", "")
                    region = data.get("region", "")
                    country = data.get("country_name", "")

                    location_parts = []
                    if city:
                        location_parts.append(city)
                    if region and region != city:
                        location_parts.append(region)
                    if country:
                        location_parts.append(country)

                    if location_parts:
                        return ", ".join(location_parts)
            except Exception:
                pass

            return "Unable to determine location"

        def get_weather(location: str) -> str:
            """Get current weather for a location using wttr.in service."""
            try:
                response = httpx.get(f"https://wttr.in/{location}?format=j1", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    current = data["current_condition"][0]
                    temp_f = current["temp_F"]
                    desc = current["weatherDesc"][0]["value"]
                    return f"{location}: {temp_f}°F, {desc}"
                return f"Unable to fetch weather for {location}"
            except Exception:
                return f"Unable to fetch weather for {location}"

        def calculate(expression: str) -> str:
            """Safely evaluate a simple math expression."""
            try:
                # Basic safety: only allow numbers, operators, and parentheses
                allowed_chars = set("0123456789+-*/(). ")
                if not all(c in allowed_chars for c in expression):
                    return "Invalid expression - only basic math operators allowed"
                result = eval(expression, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error evaluating expression: {str(e)}"

        return {
            "get_current_datetime": get_current_datetime,
            "get_current_date": get_current_date,
            "get_user_location": get_user_location,
            "get_weather": get_weather,
            "calculate": calculate,
        }

    def _get_tool_declarations(self) -> List[Dict[str, Any]]:
        """Get Gemini-compatible tool declarations."""
        return [
            {
                "name": "get_current_datetime",
                "description": "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_current_date",
                "description": "Get the current date",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_user_location",
                "description": "Get the user's approximate location (city, region, country) based on their IP address",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location (e.g., 'Boston', 'New York')"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Evaluate a simple math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate (e.g., '2 + 2', '10 * 5')"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]

    async def _call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool function."""
        if tool_name not in self._tool_functions:
            return f"Unknown tool: {tool_name}"
        
        try:
            func = self._tool_functions[tool_name]
            result = func(**tool_args)
            return result
        except Exception as e:
            return f"Error calling {tool_name}: {str(e)}"

    async def _generate_clarification_question_with_facts(self, conversation_context: str, clarification_round: int) -> tuple[str, List[str]]:
        """Generate a clarification question using the LLM with tool support.
        
        Returns:
            tuple: (question_text, list_of_new_facts_learned)
        """
        
        facts_learned = []
        question = await self._generate_clarification_question(conversation_context, clarification_round, facts_learned)
        return question, facts_learned

    async def _generate_clarification_question(self, conversation_context: str, clarification_round: int, facts_learned: Optional[List[str]] = None) -> str:
        """Generate a clarification question using the LLM with tool support."""
        
        if facts_learned is None:
            facts_learned = []

        import google.generativeai as genai
        import logging
        import os

        # Suppress gRPC warnings
        os.environ['GRPC_VERBOSITY'] = 'ERROR'
        os.environ['GLOG_minloglevel'] = '2'
        logging.getLogger('absl').setLevel(logging.ERROR)

        with _suppress_native_stderr():
            genai.configure(api_key=self.llm_api_key)
            
            # Create tools for Gemini
            tools = []
            for tool_decl in self._get_tool_declarations():
                tools.append(genai.protos.Tool(
                    function_declarations=[
                        genai.protos.FunctionDeclaration(
                            name=tool_decl["name"],
                            description=tool_decl["description"],
                            parameters=genai.protos.Schema(
                                type=genai.protos.Type.OBJECT,
                                properties={
                                    k: genai.protos.Schema(type=genai.protos.Type.STRING, description=v.get("description", ""))
                                    for k, v in tool_decl["parameters"]["properties"].items()
                                },
                                required=tool_decl["parameters"]["required"]
                            )
                        )
                    ]
                ))
            
            model = genai.GenerativeModel(self.llm_model, tools=tools)

        prompt = f"""You are a helpful assistant that asks specific clarification questions about browser automation tasks.

{conversation_context}

This is clarification round {clarification_round} of 3. Based on the user's request and any previous answers, ask ONE specific question that hasn't been answered yet to make the task more actionable for a browser automation agent.

You have access to tools for getting current date/time, weather, location, and doing calculations. **USE THESE TOOLS PROACTIVELY** when they would help:
- If user mentions travel/flights but no departure city → Call get_user_location() FIRST, then ask about dates/preferences
- If user mentions "today"/"tomorrow" without specific date → Call get_current_date() FIRST
- If planning outdoor activities → Call get_weather() to help with suggestions

CONTEXT AWARENESS:
- "get me to [place]" or "fly to [place]" = user wants flight booking
- "ip" or "where I am" = user wants you to use get_user_location() tool
- Don't ask obvious questions - use tools to fill gaps automatically

Focus on:
- Travel dates (if booking flights/hotels)
- Specific preferences (time of day, price range, airline, etc.)
- Success criteria or expected outcomes
- ANY info that would make search results more relevant

CRITICAL RULES - MUST FOLLOW:
- NEVER ask about information already shown above (location, date, weather, etc.)
- If you see "User location: Boston" above, DO NOT ask "where are you flying from?"
- If you see "Today's date: Saturday, October 04, 2025" above, DO NOT ask "what date?"
- Build ONLY on gaps in information
- Be natural and conversational

Ask a concise, specific question (one sentence). Do not explain or add preamble.

Ask your clarification question:"""

        with _suppress_native_stderr():
            response = await model.generate_content_async(prompt)
        
        # Handle function calls - loop until we get text response
        max_tool_iterations = 5
        iteration = 0
        conversation_history = [{"role": "user", "parts": [{"text": prompt}]}]
        
        while iteration < max_tool_iterations:
            if not response.candidates or not response.candidates[0].content.parts:
                break
            
            has_function_call = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    has_function_call = True
                    fc = part.function_call
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}
                    
                    if not _QUIET_MODE:
                        print(f"🔧 Using tool: {tool_name}({tool_args})")
                    
                    tool_result = await self._call_tool(tool_name, tool_args)
                    
                    # Track facts learned from tools
                    if tool_name == "get_user_location":
                        facts_learned.append(f"User location: {tool_result}")
                    elif tool_name == "get_current_date":
                        facts_learned.append(f"Today's date: {tool_result}")
                    elif tool_name == "get_current_datetime":
                        facts_learned.append(f"Current date/time: {tool_result}")
                    elif tool_name == "get_weather":
                        facts_learned.append(f"Weather info: {tool_result}")
                    
                    if not _QUIET_MODE:
                        print(f"📊 Result: {tool_result}\n")
                    
                    # Add to conversation history
                    conversation_history.append({"role": "model", "parts": [part]})
                    conversation_history.append({
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": tool_name,
                                "response": {"result": tool_result}
                            }
                        }]
                    })
                    
                    # Get next response
                    with _suppress_native_stderr():
                        response = await model.generate_content_async(conversation_history)
                    break
            
            if not has_function_call:
                break
            
            iteration += 1
        
        # Extract text from final response
        try:
            return response.text.strip()
        except (ValueError, AttributeError):
            # If still can't get text, return a generic question
            return "Can you provide more details about what you'd like to accomplish?"

    async def _refine_task_description(self, original_query: str, qa_pairs: List[tuple]) -> str:
        """Refine the task description based on Q&A exchanges."""

        import google.generativeai as genai
        import logging
        import os

        # Suppress gRPC warnings
        os.environ['GRPC_VERBOSITY'] = 'ERROR'
        os.environ['GLOG_minloglevel'] = '2'
        logging.getLogger('absl').setLevel(logging.ERROR)

        with _suppress_native_stderr():
            genai.configure(api_key=self.llm_api_key)
            
            # Create tools for Gemini
            tools = []
            for tool_decl in self._get_tool_declarations():
                tools.append(genai.protos.Tool(
                    function_declarations=[
                        genai.protos.FunctionDeclaration(
                            name=tool_decl["name"],
                            description=tool_decl["description"],
                            parameters=genai.protos.Schema(
                                type=genai.protos.Type.OBJECT,
                                properties={
                                    k: genai.protos.Schema(type=genai.protos.Type.STRING, description=v.get("description", ""))
                                    for k, v in tool_decl["parameters"]["properties"].items()
                                },
                                required=tool_decl["parameters"]["required"]
                            )
                        )
                    ]
                ))
            
            model = genai.GenerativeModel(self.llm_model, tools=tools)

        qa_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

        prompt = f"""You are a helpful assistant that creates precise browser automation task descriptions.

Original user request: "{original_query}"

Clarifications:
{qa_context}

You have access to tools for getting current date/time, weather, location, and doing calculations. Use them to enrich the task description with relevant real-time information when needed.

CONTEXT UNDERSTANDING:
- "California" without a specific city = likely means major airports like Los Angeles (LAX), San Francisco (SFO), or San Diego (SAN)
- For flights: Always specify BOTH departure and arrival cities/airports clearly
- For dates: Use specific dates from tools if available (not "today" - use actual date)
- For locations: Use city names, not states alone

IMPORTANT: Create a specific, actionable task that includes:
1. The exact action (search, book, find, compare, etc.)
2. All specific details (cities, dates, websites, criteria)
3. What to return or extract
4. Any filters or preferences mentioned

Based on the original request and the clarifications, create a single, clear, specific task description for a browser automation agent.

Output ONLY the refined task description (2-3 sentences max), no explanation or preamble."""

        with _suppress_native_stderr():
            response = await model.generate_content_async(prompt)
        
        # Handle function calls - loop until we get text response
        max_tool_iterations = 5
        iteration = 0
        conversation_history = [{"role": "user", "parts": [{"text": prompt}]}]
        
        while iteration < max_tool_iterations:
            if not response.candidates or not response.candidates[0].content.parts:
                break
            
            has_function_call = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    has_function_call = True
                    fc = part.function_call
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}
                    
                    if not _QUIET_MODE:
                        print(f"🔧 Using tool: {tool_name}({tool_args})")
                    
                    tool_result = await self._call_tool(tool_name, tool_args)
                    
                    if not _QUIET_MODE:
                        print(f"📊 Result: {tool_result}\n")
                    
                    # Add to conversation history
                    conversation_history.append({"role": "model", "parts": [part]})
                    conversation_history.append({
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": tool_name,
                                "response": {"result": tool_result}
                            }
                        }]
                    })
                    
                    # Get next response
                    with _suppress_native_stderr():
                        response = await model.generate_content_async(conversation_history)
                    break
            
            if not has_function_call:
                break
            
            iteration += 1
        
        # Extract text from final response
        try:
            return response.text.strip()
        except (ValueError, AttributeError):
            # If still can't get text, return a fallback
            return "Unable to generate task description - please try again"

    async def _analyze_task_parallelization(self, refined_task: str) -> Dict[str, Any]:
        """Analyze if task should be split into multiple parallel agents."""

        import google.generativeai as genai
        import logging
        import os

        # Suppress gRPC warnings
        os.environ['GRPC_VERBOSITY'] = 'ERROR'
        os.environ['GLOG_minloglevel'] = '2'
        logging.getLogger('absl').setLevel(logging.ERROR)

        with _suppress_native_stderr():
            genai.configure(api_key=self.llm_api_key)
            model = genai.GenerativeModel(self.llm_model)

        prompt = f"""You are a task analysis expert for browser automation. Analyze the following task and determine if it would benefit from being split into multiple parallel browser agents.

Task: "{refined_task}"

Consider splitting into parallel agents if:
- The task involves comparing multiple options/websites (e.g., comparing prices, features)
- The task requires gathering information from multiple independent sources
- The task has multiple independent subtasks that don't depend on each other
- Parallel execution would be significantly faster than sequential

DO NOT split if:
- The task is a simple single-site operation
- Subtasks depend on results from previous subtasks
- The task requires maintaining state/session between steps
- The system might be overwhelmed by too many parallel tasks

IMPORTANT: If splitting, provide ONLY 2-3 specific subtasks (NOT 4+) to avoid system overload.

Format your response as JSON:
{{
  "should_split": true/false,
  "reason": "brief explanation",
  "parallel_tasks": ["task 1", "task 2", ...] or null
}}

Respond with ONLY the JSON, no additional text."""

        with _suppress_native_stderr():
            response = await model.generate_content_async(prompt)
        
        try:
            import json
            # Try to extract JSON from response
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            return result
        except Exception as e:
            if not _QUIET_MODE:
                print(f"⚠️  Could not parse parallelization analysis: {e}")
            return {"should_split": False, "reason": "parsing error", "parallel_tasks": None}

    async def run_conversational_task(
        self,
        initial_query: str,
        max_steps: int = 70,
        headless: bool = True,
        enable_parallel_agents: bool = True,
        use_existing_chrome: bool = False,  # NEW: Flag to use existing Chrome session
        cdp_url: str = "http://localhost:9222", # NEW: CDP URL for existing Chrome
        
    ) -> Dict[str, Any]:
        """Run a browser task with conversational clarification.
        
        Args:
            initial_query: User's initial vague task description
            max_steps: Maximum steps for each agent
            headless: Run browsers in headless mode
            enable_parallel_agents: Allow automatic deployment of multiple parallel agents
        """

        if not _QUIET_MODE:
            print(f"\n🤖 Initial request: {initial_query}")
            print("\nLet me ask a few questions to better understand what you need...\n")

        qa_pairs: List[tuple] = []
        conversation_context = initial_query
        known_facts: List[str] = []  # Track facts learned from tools

        # Clarification rounds (skip if NON_INTERACTIVE)
        total_rounds = 0 if _NON_INTERACTIVE else 2
        for round_num in range(1, total_rounds + 1):
            # Build context with previous Q&As and known facts
            context_parts = [initial_query]
            
            if known_facts:
                context_parts.append(f"\nKnown information from tools:\n" + "\n".join([f"- {fact}" for fact in known_facts]))
            
            if qa_pairs:
                prev_context = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(qa_pairs)])
                context_parts.append(f"\nPrevious clarifications:\n{prev_context}")
            
            conversation_context = "\n".join(context_parts)

            # Generate question (may call tools and add to known_facts)
            question, new_facts = await self._generate_clarification_question_with_facts(conversation_context, round_num)
            known_facts.extend(new_facts)

            if not _QUIET_MODE:
                print(f"❓ {question}")
                # Speak the question
                await speak_text(question)

            # Get voice input from user
            user_answer = await get_voice_input("Your answer", debug=True)
            print(f"👤 You said: {user_answer}\n")
            qa_pairs.append((question, user_answer))
            if not _QUIET_MODE:
                print()

        # Refine the task
        refined_task = await self._refine_task_description(initial_query, qa_pairs)
        if not _QUIET_MODE:
            print(f"✅ Refined task: {refined_task}\n")

        # Analyze if task should be parallelized
        task_descriptions = [refined_task]
        parallelization_info = None
        
        if enable_parallel_agents:
            if not _QUIET_MODE:
                print("🔍 Analyzing if task can be parallelized...\n")
            
            parallelization_info = await self._analyze_task_parallelization(refined_task)
            
            if parallelization_info.get("should_split") and parallelization_info.get("parallel_tasks"):
                task_descriptions = parallelization_info["parallel_tasks"]
                if not _QUIET_MODE:
                    print(f"🚀 Deploying {len(task_descriptions)} parallel agents!")
                    print(f"📋 Reason: {parallelization_info.get('reason', 'N/A')}")
                    for i, task in enumerate(task_descriptions, 1):
                        print(f"   Agent {i}: {task}")
                    print()
            else:
                if not _QUIET_MODE:
                    print(f"✨ Running single agent task")
                    if parallelization_info.get("reason"):
                        print(f"   Reason: {parallelization_info['reason']}")
                    print()

        # Limit max_steps to avoid memory issues
        adjusted_max_steps = min(max_steps, 15)  # Cap at 40 to prevent memory overflow
        if not _QUIET_MODE and adjusted_max_steps != max_steps:
            print(f"⚙️  Limited max steps to {adjusted_max_steps} to prevent memory issues\n")

        # Execute using existing SDK with smart mode enabled
        if use_existing_chrome:
            config = BrowserConfig(
                cdp_url=cdp_url,
                headless=False,  # Must be False when using CDP
                separate_profiles=False
            )
            result = await self.sdk.execute_parallel(
                tasks=[ParallelTask(task_description=desc, max_steps=adjusted_max_steps) for desc in task_descriptions],
                browser_config=config
            )
        else:
            result = await run_parallel_tasks(
                task_descriptions=task_descriptions,
                max_steps=adjusted_max_steps,
                headless=headless,
                llm_api_key=self.sdk.llm_api_key,
                enable_smart_mode=True,
            )

        return {
            "original_query": initial_query,
            "qa_pairs": qa_pairs,
            "refined_task": refined_task,
            "parallelization_info": parallelization_info,
            "task_descriptions": task_descriptions,
            "execution_result": result,
        }


async def run_conversational_task(
    initial_query: str,
    max_steps: int = 20,
    headless: bool = True,
    llm_api_key: Optional[str] = None,
    enable_parallel_agents: bool = True,
    use_existing_chrome: bool = False,  #  Flag to use existing Chrome session
    cdp_url: str = "http://localhost:9222",  # CDP URL for existing Chrome
) -> Dict[str, Any]:
    """Convenience function to run a conversational browser task.
    
    Args:
        initial_query: User's initial vague task description
        max_steps: Maximum steps for each agent
        headless: Run browsers in headless mode
        llm_api_key: Optional Google API key
        enable_parallel_agents: Allow automatic deployment of multiple parallel agents
    """

    import logging
    import warnings

    # Suppress all the noise
    warnings.filterwarnings("ignore")
    logging.getLogger("browser_use").setLevel(logging.ERROR)
    logging.getLogger("cdp_use").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GLOG_minloglevel'] = '2'

    agent = ConversationalBrowserAgent(llm_api_key=llm_api_key)
    return await agent.run_conversational_task(initial_query, max_steps, headless, enable_parallel_agents, use_existing_chrome, cdp_url)


__all__ = [
    "ParallelTask",
    "ParallelExecutionResult",
    "BrowserConfig",
    "ParallelBrowserSDK",
    "ConversationalBrowserAgent",
    "run_parallel_tasks",
    "run_branched_tasks",
    "run_conversational_task",
    "test_sdk",
    "test_branched_tasks",
]

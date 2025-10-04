import concurrent.futures
import itertools
from browser_use import Agent as BrowserUseAgent, ChatGoogle, Browser
from dotenv import load_dotenv

load_dotenv()

def run_browser_task(task: str, browser: Browser) -> str:
    """Runs a BrowserUseAgent task using a PRE-EXISTING browser instance."""
    print(f"Starting task: {task[:30]}...")
    agent = BrowserUseAgent(
        task=task,
        llm=ChatGoogle(model="gemini-flash-latest"),
        browser=browser,
    )
    result = agent.run_sync()
    print(f"Finished task: {task[:30]}...")
    return str(result)

# --- Main script execution ---
tasks = [
    "Go to gmail.com and tell me the subject of the first unread email",
    "What is the cheapest and best rated e-scooter on Amazon right now?",
]

# The CDP URL for the running Chrome instance
cdp_url = "http://localhost:9222"

try:
    # 1. Connect to the browser you opened with the modified shortcut
    print("Connecting to your active browser instance...")
    local_browser = Browser(cdp_url=cdp_url)

    print("Starting concurrent agent run...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_browser_task, tasks, itertools.repeat(local_browser)))

    print("\n--- Results ---")
    for i, result in enumerate(results):
        print(f"Result from Task {i+1}:\n{result}\n")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure Google Chrome is running and was started with your modified shortcut.")

finally:
    print("\nScript finished. Your browser remains open.")
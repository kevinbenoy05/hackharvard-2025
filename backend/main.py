import os
from browser_use import Agent, ChatGoogle, Browser
from dotenv import load_dotenv
import asyncio


load_dotenv()

async def main():
    # Use Gemini Flash - fast, cheap, and has larger context window
    llm = ChatGoogle(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY")
    )

    # Initialize browser with optimized settings
    browser = Browser(
        headless=False,
        window_size={'width': 1280, 'height': 800},
    )

    task = "go to csail.mit.edu and summarize the events for next week"

    # Create agent with optimized settings
    agent = Agent(
        task=task,
        browser=browser,
        llm=llm,
        flash_mode=True,
        max_actions_per_step=10,
    )

    try:
        history = await agent.run(max_steps=50)

        # Display results
        if history.is_done():
            print("\n✅ Task completed successfully!")
            result = history.final_result()
            if result:
                print(f"Result: {result}")
        else:
            print("\nTask did not complete")

    finally:
        # Always clean up browser resources
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())

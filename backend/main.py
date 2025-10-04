from browser_use import Agent, ChatOpenAI, Browser
from dotenv import load_dotenv
import asyncio


load_dotenv()

browser = Browser(
	headless=True,  # Show browser window
	window_size={'width': 1000, 'height': 700},  # Set window size
)

async def main():
    global browser
    llm = ChatOpenAI(model="gpt-4.1-nano")
    task = "go to priceline.com and find me a flight from boston to washington dc that departs from 4 to 9 pm tomorrow, give me the cheapest one"
    agent = Agent(
        task=task,
        browser=browser,
        llm=ChatOpenAI(model='gpt-4.1-nano'),
    )
    agent = Agent(task=task, llm=llm)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

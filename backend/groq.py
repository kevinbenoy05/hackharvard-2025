from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv
load_dotenv()

agent1 = Agent(
    task="Find the number of stars of the browser-use repo",
    llm=ChatGoogle(model="gemini-flash-latest"),
    # browser=Browser(use_cloud=True),  # Uses Browser-Use cloud for the browser
)
agent1.run_sync()
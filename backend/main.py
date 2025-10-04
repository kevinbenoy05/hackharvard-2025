import asyncio

from browser_tasks import run_parallel_tasks, test_branched_tasks, test_sdk


def main() -> None:
    print("🚀 Parallel Browser SDK")
    print("Choose test option:")
    print("1. Test SDK with simple parallel tasks")
    print("2. Test convenience functions")
    print("3. Test branched tasks from freshly captured state")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "2":
        print("Testing simple convenience functions...")

        async def test_convenience() -> None:
            result = await run_parallel_tasks(
                [
                    "Go to https://httpbin.org/ip and get my IP",
                    "Go to https://httpbin.org/headers and get headers",
                ],
                headless=True,
            )

            print(f"Success rate: {result['success_rate']:.0%}")

        asyncio.run(test_convenience())

    elif choice == "3":
        print("Testing branched tasks starting from a freshly captured storage state...")
        asyncio.run(test_branched_tasks())

    else:
        asyncio.run(test_sdk())


if __name__ == "__main__":
    main()

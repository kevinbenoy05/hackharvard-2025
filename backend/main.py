import asyncio
import os
import sys

import contextlib

# Suppress gRPC/ALTS warnings before any imports
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Note: QUIET and NON_INTERACTIVE can be set via environment variables
# to disable prompts/prints when running programmatically


@contextlib.contextmanager
def _suppress_native_stderr():
    """Temporarily silence C-level stderr (fd=2) to hide early absl/gRPC logs."""

    try:
        fd = sys.stderr.fileno()
    except Exception:
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

with _suppress_native_stderr():
    from browser_tasks import run_conversational_task, run_parallel_tasks, test_branched_tasks, test_sdk

# Native stderr restored automatically by context manager


def main() -> None:
    print("🚀 Parallel Browser SDK")
    print("Choose test option:")
    print("1. Test SDK with simple parallel tasks")
    print("2. Test convenience functions")
    print("3. Test branched tasks from freshly captured state")
    print("4. Test conversational agent (asks clarifying questions)")

    choice = input("Enter choice (1-4): ").strip()

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

    elif choice == "4":
        print("Testing conversational agent...\n")

        async def test_conversational() -> None:
            import warnings
            import logging

            # Suppress warnings
            warnings.filterwarnings("ignore")
            logging.getLogger().setLevel(logging.ERROR)

            initial_query = input("Enter your vague task description: ").strip()

            result = await run_conversational_task(
                initial_query=initial_query,
                max_steps=20,
                headless=True,
            )

            print("\n" + "="*80)
            print("📊 EXECUTION SUMMARY")
            print("="*80)
            print(f"\n📝 Original query: {result['original_query']}")
            print(f"🎯 Refined task: {result['refined_task']}")
            
            # Show parallelization info if available
            if result.get('parallelization_info'):
                parallel_info = result['parallelization_info']
                if parallel_info.get('should_split'):
                    print(f"🚀 Deployed {len(result['task_descriptions'])} parallel agents")
                    print(f"📋 Parallelization strategy: {parallel_info.get('reason', 'N/A')}")
                else:
                    print(f"✨ Single agent execution: {parallel_info.get('reason', 'N/A')}")
            
            print(f"✅ Success rate: {result['execution_result']['success_rate']:.0%}")
            print(f"⏱️  Total execution time: {result['execution_result']['total_time']:.2f}s")

            # Display results from all agents
            if result['execution_result']['execution_results']:
                print("\n💡 RESULTS:")
                for i, exec_result in enumerate(result['execution_result']['execution_results'], 1):
                    if len(result['execution_result']['execution_results']) > 1:
                        print(f"\n--- Agent {i} ---")
                    
                    if exec_result.success:
                        # Extract the final result text
                        final_text = exec_result.result.final_result()
                        print(final_text)
                    else:
                        print(f"❌ Failed: {exec_result.error}")

            print("\n" + "="*80 + "\n")

        asyncio.run(test_conversational())

    else:
        asyncio.run(test_sdk())


if __name__ == "__main__":
    main()

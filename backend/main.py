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
                headless=False,
            )

            print(f"Success rate: {result['success_rate']:.0%}")

        asyncio.run(test_convenience())

    elif choice == "3":
        print("Testing branched tasks starting from a freshly captured storage state...")
        asyncio.run(test_branched_tasks())

    elif choice == "4":
        print("Testing conversational agent with voice...\n")

        async def test_conversational() -> None:
            import warnings
            import logging
            from browser_tasks import get_voice_input, speak_text

            # Suppress warnings
            warnings.filterwarnings("ignore")
            logging.getLogger().setLevel(logging.ERROR)

            # Get initial query via voice
            initial_query = await get_voice_input("Describe your task", debug=True)
            print(f"📝 You said: {initial_query}\n")

            result = await run_conversational_task(
                initial_query=initial_query,
                max_steps=20,
                headless=False,
                enable_parallel_agents=True,
                use_existing_chrome=True,  # NEW: Use your logged-in Chrome!
                cdp_url="http://localhost:9222",  # NEW: CDP port
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
            results_text_parts = []
            if result['execution_result']['execution_results']:
                print("\n💡 RESULTS:")
                for i, exec_result in enumerate(result['execution_result']['execution_results'], 1):
                    if len(result['execution_result']['execution_results']) > 1:
                        print(f"\n--- Agent {i} ---")

                    if exec_result.success:
                        # Extract the final result text
                        final_text = exec_result.result.final_result()
                        print(final_text)
                        results_text_parts.append(final_text)
                    else:
                        error_msg = f"Agent {i} failed: {exec_result.error}"
                        print(f"❌ {error_msg}")
                        results_text_parts.append(error_msg)

            print("\n" + "="*80 + "\n")

            # Speak the results
            if results_text_parts:
                if results_text_parts:
                    print("🔊 Speaking results...\n")
                    combined_results = " ".join(results_text_parts)
                    # FIX #2: Truncate the result if it's too long for the text-to-speech API.
                    max_audio_length = 4096  # API character limit
                    if len(combined_results) > max_audio_length:
                        print(f"⚠️  Result text is too long ({len(combined_results)} chars), truncating for audio playback.")
                        combined_results = combined_results[:max_audio_length]

                    await speak_text(combined_results)

        asyncio.run(test_conversational())

    else:
        asyncio.run(test_sdk())


if __name__ == "__main__":
    main()

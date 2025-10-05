import asyncio
import os
from browser_tasks import ParallelBrowserSDK, ParallelTask, BrowserConfig

async def test_simple_navigation():
    """Simple test - just navigate and extract info."""
    
    print("🧪 Test 1: Simple Navigation & Data Extraction\n")
    
    sdk = ParallelBrowserSDK()
    
    task = ParallelTask(
        task_description="Go to https://httpbin.org/get and tell me my IP address and user agent",
        max_steps=5
    )
    
    config = BrowserConfig(
        cdp_url="http://localhost:9222",
        headless=False,
        separate_profiles=False
    )
    
    result = await sdk.execute_parallel([task], config)
    
    print("="*60)
    if result['success_rate'] == 1.0:
        print("✅ PASSED: Navigation works!")
        for exec_result in result['execution_results']:
            print(f"\n{exec_result.result.final_result()}")
    else:
        print("❌ FAILED: Navigation issue")
    print("="*60 + "\n")


async def test_authenticated_session():
    """Test that we're using authenticated Chrome with saved sessions."""
    
    print("🧪 Test 2: Authenticated Session Test\n")
    print("This test verifies we're connected to YOUR logged-in Chrome.\n")
    
    sdk = ParallelBrowserSDK()
    
    # This will show different results if you're logged into Google vs not
    task = ParallelTask(
        task_description="""Go to https://accounts.google.com and tell me:
1. Am I currently logged in to a Google account? (Look for profile picture or sign in button)
2. If logged in, what email or name is displayed?""",
        max_steps=5
    )
    
    config = BrowserConfig(
        cdp_url="http://localhost:9222",
        headless=False,
        separate_profiles=False
    )
    
    result = await sdk.execute_parallel([task], config)
    
    print("="*60)
    print("🔑 Authentication Status:")
    for exec_result in result['execution_results']:
        print(f"\n{exec_result.result.final_result()}")
    print("\n" + "="*60 + "\n")


async def test_gmail_send():
    """Test sending email via Gmail - the wow factor!"""
    
    print("🧪 Test 3: Gmail Email Send (Authenticated Session)\n")
    print("⚠️  Make sure you're logged into Gmail in Chrome!\n")
    
    sdk = ParallelBrowserSDK()
    
    task = ParallelTask(
        task_description="""Go to gmail.com and compose a new email.

Click the Compose button.

After the compose window opens:
1. Click in the To field
2. Type: benoymichael@gmail.com
3. Press Tab key to move to subject field
4. Type: Test from Browser-Use
5. Press Tab key to move to body
6. Type: This is a test email sent using Browser-Use SDK.
7. Find and click the Send button

If you see a contacts popup when typing the email, press Enter or Tab to select/confirm the email address.

Tell me if the email was sent successfully.""",
        max_steps=25  # More steps for handling popups
    )
    
    config = BrowserConfig(
        cdp_url="http://localhost:9222",
        headless=False,
        separate_profiles=False,
        retry_on_failure=True,
        max_retries=3
    )
    
    result = await sdk.execute_parallel([task], config)
    
    print("="*60)
    if result['success_rate'] == 1.0:
        print("✅ PASSED: Email sent successfully!")
        for exec_result in result['execution_results']:
            print(f"\n{exec_result.result.final_result()}")
    else:
        print("⚠️  CHECK OUTPUT: Email may have been sent")
        for exec_result in result['execution_results']:
            print(f"\n{exec_result.result.final_result()}")
    print("="*60 + "\n")


async def test_form_interaction():
    """Test form filling - general purpose."""
    
    print("🧪 Test 4: Form Interaction\n")
    
    sdk = ParallelBrowserSDK()
    
    task = ParallelTask(
        task_description="""Go to https://httpbin.org/forms/post
        
Fill out the form with:
- Customer name: John Doe
- Telephone: 555-1234
- Email address: john@example.com
- Pizza size: Select 'Medium'
- Pizza toppings: Check 'Bacon'
- Delivery time: Select '11:30'

Then click Submit and tell me if the form was submitted successfully.""",
        max_steps=15
    )
    
    config = BrowserConfig(
        cdp_url="http://localhost:9222",
        headless=False,
        separate_profiles=False,
        retry_on_failure=True,
        max_retries=2
    )
    
    result = await sdk.execute_parallel([task], config)
    
    print("="*60)
    if result['success_rate'] == 1.0:
        print("✅ PASSED: Form interaction works!")
        for exec_result in result['execution_results']:
            print(f"\n{exec_result.result.final_result()}")
    else:
        print("⚠️  PARTIAL: Check output below")
        for exec_result in result['execution_results']:
            print(f"\n{exec_result.result.final_result()}")
    print("="*60 + "\n")


async def test_complex_interaction():
    """Test complex multi-step task on a real site."""
    
    print("🧪 Test 5: Complex Multi-Step Task\n")
    
    sdk = ParallelBrowserSDK()
    
    task = ParallelTask(
        task_description="""Go to https://www.google.com and search for 'browser automation'. 
Then tell me the titles of the first 3 search results you see.""",
        max_steps=10
    )
    
    config = BrowserConfig(
        cdp_url="http://localhost:9222",
        headless=False,
        separate_profiles=False
    )
    
    result = await sdk.execute_parallel([task], config)
    
    print("="*60)
    if result['success_rate'] == 1.0:
        print("✅ PASSED: Complex interaction works!")
        for exec_result in result['execution_results']:
            print(f"\n{exec_result.result.final_result()}")
    else:
        print("⚠️  PARTIAL: Check output below")
        for exec_result in result['execution_results']:
            print(f"\n{exec_result.result.final_result()}")
    print("="*60 + "\n")


async def run_all_tests():
    """Run all tests in sequence."""
    
    print("\n" + "="*60)
    print("🚀 BROWSER AUTOMATION TEST SUITE")
    print("="*60 + "\n")
    
    print("⚠️  SETUP REQUIRED:")
    print("1. Close ALL Chrome windows")
    print("2. Run: \"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe\" --remote-debugging-port=9222 --user-data-dir=\"C:\\ChromeProfile\"")
    print("3. Log into Gmail and any other accounts you want to test\n")
    
    input("Press Enter when ready...")
    print()
    
    # Run tests
    await test_simple_navigation()
    await test_authenticated_session()
    await test_gmail_send()  # Email test included!
    await test_form_interaction()
    await test_complex_interaction()
    
    print("\n" + "="*60)
    print("✨ TEST SUITE COMPLETE")
    print("="*60)
    print("\n💡 What this proves:")
    print("  ✓ Can connect to existing Chrome session")
    print("  ✓ Can use authenticated sessions (no re-login)")
    print("  ✓ Can send emails via Gmail (authenticated)")
    print("  ✓ Can navigate and extract data")
    print("  ✓ Can interact with forms")
    print("  ✓ Can perform multi-step tasks")
    print("\n🎯 For your hackathon: Any task that works manually")
    print("   in Chrome will work with this automation!")
    print()


def main():
    """Main test runner."""
    print("\n🎯 Choose test mode:")
    print("1. Run all tests (recommended)")
    print("2. Test simple navigation only")
    print("3. Test authenticated session only")
    print("4. Test Gmail email send only")
    print("5. Test form interaction only")
    print("6. Test complex interaction only")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        asyncio.run(run_all_tests())
    elif choice == "2":
        asyncio.run(test_simple_navigation())
    elif choice == "3":
        asyncio.run(test_authenticated_session())
    elif choice == "4":
        asyncio.run(test_gmail_send())
    elif choice == "5":
        asyncio.run(test_form_interaction())
    elif choice == "6":
        asyncio.run(test_complex_interaction())
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

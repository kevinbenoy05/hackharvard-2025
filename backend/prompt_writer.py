
import dotenv
from typing import Optional, Literal
from openai import OpenAI
from pydantic import BaseModel, Field

OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

class TaskAnalysis(BaseModel):
    """Analysis of whether a task needs browser automation."""
    is_feasible: bool = Field(description="Can this be automated reliably with browser automation?")
    requires_browser: bool = Field(description="Does this task actually need browser automation, or can it be done another way?")
    task_type: Literal["browser_automation", "simple_search", "llm_only", "api_call", "not_feasible"] = Field(
        description="What type of task is this?"
    )
    reasoning: str = Field(description="Why this classification was chosen")
    alternative_suggestion: Optional[str] = Field(
        default=None,
        description="If task doesn't need browser, what's a better way to handle it?"
    )
    confidence: float = Field(description="Confidence score 0-1 for this assessment")


class SimplePromptOptimizer:
    """
    Converts vague user requests into simple, clear browser automation prompts.
    Also identifies if browser automation is actually needed.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        actual_api_key = OPENAI_API_KEY if api_key is None else api_key
        if not actual_api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        self.client = OpenAI(api_key=actual_api_key)
    
    def analyze_task(self, user_request: str, verbose: bool = True) -> TaskAnalysis:
        """
        Analyze if task needs browser automation or can be handled differently.
        
        Returns TaskAnalysis with feasibility and task type.
        """
        
        if verbose:
            print("🔍 Analyzing task requirements...\n")
        
        system_prompt = """You are an expert at determining if tasks need browser automation.

Classify tasks into these categories:

1. **browser_automation** - Tasks that REQUIRE interacting with a website
   Examples:
   - "Send an email via Gmail"
   - "Order something on Amazon"
   - "Post on Twitter"
   - "Check my bank balance"
   - "Fill out a form on website X"

2. **simple_search** - Tasks that just need a Google search (can use browser, but very simple)
   Examples:
   - "What's the weather in Boston?"
   - "Who won the game last night?"
   - "When does the movie release?"
   - "What time does the store close?"

3. **llm_only** - Tasks that don't need browser at all - just LLM knowledge
   Examples:
   - "Explain quantum physics"
   - "Write a poem"
   - "Summarize this text"
   - "What's 2+2?"

4. **api_call** - Tasks better handled by API calls, not browser
   Examples:
   - "Get stock price for AAPL" (use finance API)
   - "What's the weather?" (use weather API)
   - "Translate this text" (use translation API)

5. **not_feasible** - Tasks that can't be automated reliably
   Examples:
   - "Solve this CAPTCHA"
   - "Pass 2FA verification"
   - "Do my homework for me" (ethical concerns)

IMPORTANT:
- The user has a logged-in Chrome browser (authentication is handled)
- If it can be done with browser automation, mark it feasible
- If it's just a Google search, suggest using search API instead
- Be practical and honest

Return your analysis."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this request:\n\n{user_request}"}
                ],
                response_format=TaskAnalysis
            )
            
            analysis = completion.choices[0].message.parsed
            
            if verbose:
                if analysis.is_feasible:
                    print(f"✅ Feasible: Yes")
                else:
                    print(f"❌ Feasible: No")
                
                print(f"🔖 Task Type: {analysis.task_type}")
                print(f"💭 Reasoning: {analysis.reasoning}")
                print(f"📊 Confidence: {analysis.confidence:.0%}")
                
                if analysis.alternative_suggestion:
                    print(f"💡 Suggestion: {analysis.alternative_suggestion}")
                print()
            
            return analysis
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Error during analysis: {e}\n")
            
            # Return safe default
            return TaskAnalysis(
                is_feasible=True,
                requires_browser=True,
                task_type="browser_automation",
                reasoning="Could not analyze - defaulting to browser automation",
                confidence=0.5
            )
    
    def optimize(self, user_request: str, verbose: bool = True) -> dict:
        """
        Convert user request into clean, actionable browser automation prompt.
        
        Returns dict with:
        - is_feasible: bool - Can this be automated?
        - requires_browser: bool - Does this need browser automation?
        - task_type: str - Classification of task
        - reasoning: str - Why this classification
        - optimized_prompt: str - Clean prompt for browser.use (if applicable)
        - estimated_steps: int - Recommended max_steps (if applicable)
        - alternative_suggestion: str - Better approach if not browser task
        """
        
        # Step 1: Analyze the task
        analysis = self.analyze_task(user_request, verbose=verbose)
        
        # If not feasible, return early
        if not analysis.is_feasible:
            return {
                "is_feasible": False,
                "requires_browser": False,
                "task_type": analysis.task_type,
                "reasoning": analysis.reasoning,
                "optimized_prompt": None,
                "estimated_steps": None,
                "alternative_suggestion": analysis.alternative_suggestion
            }
        
        # If doesn't need browser, return with suggestion
        if not analysis.requires_browser:
            return {
                "is_feasible": True,
                "requires_browser": False,
                "task_type": analysis.task_type,
                "reasoning": analysis.reasoning,
                "optimized_prompt": None,
                "estimated_steps": None,
                "alternative_suggestion": analysis.alternative_suggestion
            }
        
        # Step 2: Generate browser automation prompt
        if verbose:
            print("🔧 Creating browser automation instructions...\n")
        
        system_prompt = """You are an expert at creating clear, simple browser automation instructions.

CONTEXT:
- The user is running automation on their PERSONAL Chrome browser
- They are ALREADY LOGGED IN to all services (Gmail, Amazon, etc.)
- DO NOT include login steps or authentication

YOUR JOB:
Convert the user's request into simple, NUMBERED steps that a browser automation agent can follow.

RULES:
1. Use simple, natural language (not technical jargon)
2. One action per step
3. Be specific about what to click/type
4. Keep it short and clear
5. For typing in fields that have autocomplete (like Gmail "To" field), add: "Then press Tab to confirm"
6. Don't include technical selectors or CSS classes
7. Don't include verification steps - just the main actions
8. Focus on the actual task, not setup
9. If the task is complex, break it down into manageable steps
10. eEns

EXAMPLE INPUT:
"Send an email to john@example.com with subject 'Hello' and body 'Test'"

EXAMPLE OUTPUT:
1. Go to gmail.com
2. Click the Compose button
3. In the To field, type: john@example.com
4. Then press Tab to confirm
5. In the Subject field, type: Hello
6. In the message body, type: Test
7. Click the Send button
8. Report if the email was sent successfully.

NOW CREATE SIMPLE STEPS FOR THE USER'S REQUEST."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request}
                ],
                temperature=0.3
            )
            
            optimized_prompt = response.choices[0].message.content.strip()
            
            # Estimate steps: count action lines
            action_lines = [
                line for line in optimized_prompt.split('\n') 
                if line.strip() and not line.startswith('Then') and not line.startswith('Note')
            ]
            step_count = len(action_lines)
            estimated_steps = max(15, int(step_count * 2))  # 2x buffer for retries
            
            if verbose:
                print("✅ Optimization complete!\n")
                print(f"📊 Generated {step_count} steps")
                print(f"💡 Recommended max_steps: {estimated_steps}\n")
            
            return {
                "is_feasible": True,
                "requires_browser": True,
                "task_type": analysis.task_type,
                "reasoning": analysis.reasoning,
                "optimized_prompt": optimized_prompt,
                "estimated_steps": estimated_steps,
                "alternative_suggestion": None
            }
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Error during optimization: {e}\n")
            
            return {
                "is_feasible": False,
                "requires_browser": True,
                "task_type": "browser_automation",
                "reasoning": f"Error during optimization: {str(e)}",
                "optimized_prompt": None,
                "estimated_steps": None,
                "alternative_suggestion": "Please try rephrasing your request"
            }


def optimize_for_browser_use(user_request: str, api_key: Optional[str] = None, verbose: bool = True) -> dict:
    """
    Convenience function - converts user request to browser automation prompt.
    
    Args:
        user_request: What the user wants to do
        api_key: OpenAI API key (optional)
        verbose: Print progress
    
    Returns:
        dict with:
        - is_feasible: bool
        - requires_browser: bool  
        - task_type: str
        - reasoning: str
        - optimized_prompt: str (if applicable)
        - estimated_steps: int (if applicable)
        - alternative_suggestion: str (if applicable)
    """
    optimizer = SimplePromptOptimizer(api_key=api_key)
    return optimizer.optimize(user_request, verbose=verbose)


# Test examples
if __name__ == "__main__":
    test_cases = [
        "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'See you at 3pm'",
        "What's the weather in Boston?",
        "Search Amazon for wireless headphones and tell me the top 3 products",
        "Explain quantum physics to me",
        "Order a pizza from Dominos",
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print("=" * 80)
        print(f"TEST CASE {i}: {test_case}")
        print("=" * 80)
        print()
        
        result = optimize_for_browser_use(test_case)
        
        print("-" * 80)
        print("RESULT:")
        print("-" * 80)
        print(f"✅ Feasible: {result['is_feasible']}")
        print(f"🌐 Needs Browser: {result['requires_browser']}")
        print(f"📋 Task Type: {result['task_type']}")
        print(f"💭 Reasoning: {result['reasoning']}")
        
        if result['alternative_suggestion']:
            print(f"💡 Better Approach: {result['alternative_suggestion']}")
        
        if result['optimized_prompt']:
            print()
            print("OPTIMIZED PROMPT:")
            print("-" * 80)
            print(result['optimized_prompt'])
            print()
            print(f"💡 Use with max_steps={result['estimated_steps']}")
        
        print("\n")

# vision rover notebook utils
import os
import re
import yaml
import base64
import asyncio
import platform
import nest_asyncio
from langchain import hub
from IPython import display
from dotenv import load_dotenv
from typing import List, Dict, Optional, Set
from playwright.async_api import Page
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain as chain_decorator
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate

# Load/save functions
def load_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

# Classes
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    thought: str  # ReAct reasoning step
    plan: str     # Current plan
    action: str
    args: Optional[List[str]]


class AgentState(TypedDict):
    page: Page  # The Playwright web page
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output with ReAct components
    scratchpad: List[BaseMessage]  # Enhanced to include reasoning and plan history
    observation: str  # The most recent response from a tool
    visited_urls: Set[str]  # Track visited URLs to avoid loops
    plan_history: List[str]  # Track how the plan evolves
    action_history: List[Dict]  # Track all actions and their outcomes

# Agent tools
async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    
    bbox_id = click_args[0]
    # Strip any extra quotes before converting to int
    bbox_id = bbox_id.strip("'\"")  # Remove both single and double quotes
    bbox_id = int(bbox_id)
    
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id = type_args[0]
    bbox_id = bbox_id.strip("'\"")  # Remove quotes
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args
    
    # Clean up potential quotes in target and direction
    target = target.strip("'\"")
    direction = direction.strip("'\"")

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        try:
            target_id = int(target)
            bbox = state["bboxes"][target_id]
            x, y = bbox["x"], bbox["y"]
            scroll_direction = (
                -scroll_amount if direction.lower() == "up" else scroll_amount
            )
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)
        except ValueError:
            return f"Failed to convert target '{target}' to integer"
        except IndexError:
            return f"No bounding box found with ID: {target_id}"
        except Exception as e:
            return f"Error scrolling: {str(e)}"

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

async def close_popups(state: AgentState):
    """
    Closes any visible modal or pop-up on the page.
    """
    page = state["page"]
    # Common selectors for modal close buttons and overlay elements
    modal_selectors = [
        "button[class*='close']",
        "[class*='modal']",
        "[class*='modal'] button",
        "[class*='CloseButton']",
        "[aria-label*='close']",
        ".modal-close",
        ".close-modal",
        ".modal .close",
        ".modal-backdrop",
        ".modal-overlay",
        "[class*='overlay']"
    ]

    for selector in modal_selectors:
        try:
            elements = page.locator(selector)
            visible_elements = await elements.element_handles()

            for element in visible_elements:
                if await element.is_visible():
                    try:
                        # Try clicking with JavaScript first (more reliable)
                        await element.evaluate("node => node.click()")
                    except Exception:
                        # If JavaScript click fails, try regular click
                        await element.click()

        except TimeoutError:
            continue
        except Exception as e:
            print(f"Error handling selector {selector}: {str(e)}")
            continue

    return "Modals closed"

## Decorators
with open("conf/js/mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            asyncio.sleep(3)
    screenshot = await page.screenshot()
    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    # Track the current URL in visited URLs
    current_url = state["page"].url
    visited_urls = state.get("visited_urls", set())
    visited_urls.add(current_url)
    return {**state, **marked_page, "visited_urls": visited_urls}

## Parsing and formatting
def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def parse_react_output(text: str) -> dict:
    """Parse the ReAct framework output from the LLM"""
    # Extracting thought
    thought_match = re.search(r"Thought:(.*?)(?=Plan:|$)", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    
    # Extracting plan
    plan_match = re.search(r"Plan:(.*?)(?=Action:|$)", text, re.DOTALL)
    plan = plan_match.group(1).strip() if plan_match else ""
    
    # Extracting action
    action_match = re.search(r"Action:\s*(\w+)(?:\s+(.+))?$", text, re.DOTALL)
    
    if not action_match:
        return {"thought": thought, "plan": plan, "action": "retry", "args": f"Could not parse LLM Output: {text}"}
    
    action = action_match.group(1).strip()
    action_input = action_match.group(2).strip() if action_match.group(2) else None
    
    if action_input is not None:
        action_input = [inp.strip().strip("[]") for inp in action_input.split(";")]
    
    return {
        "thought": thought,
        "plan": plan,
        "action": action,
        "args": action_input
    }

def update_scratchpad_react(state: AgentState):
    """Update the scratchpad with ReAct framework components"""
    old = state.get("scratchpad", [])
    plan_history = state.get("plan_history", [])
    action_history = state.get("action_history", [])
    visited_urls = state.get("visited_urls", set())
    
    # Start or continue the scratchpad content
    if old:
        txt = old[-1].content
    else:
        txt = "# ReAct Web Navigation Agent\n\n"
        txt += "## Memory\n"
        txt += "- **Task**: " + state["input"] + "\n\n"
    
    # Update with current step information
    prediction = state.get("prediction", {})
    current_step = len(action_history) + 1
    
    # Add thought, plan, and observation
    txt += f"\n## Step {current_step}\n"
    if "thought" in prediction and prediction["thought"]:
        txt += f"**Thought**: {prediction['thought']}\n\n"
    
    if "plan" in prediction and prediction["plan"]:
        txt += f"**Plan**: {prediction['plan']}\n\n"
        # Track plan history
        plan_history.append(prediction["plan"])
    
    if state.get("observation"):
        # Add action and observation
        action = prediction.get("action", "Unknown")
        args = prediction.get("args", [])
        txt += f"**Action**: {action} {args}\n"
        txt += f"**Observation**: {state['observation']}\n"
        
        # Track action history
        action_history.append({
            "step": current_step,
            "action": action,
            "args": args,
            "result": state['observation'],
            "url": state["page"].url
        })
    
    # Add visited URLs section
    txt += "\n## Visited URLs\n"
    for url in visited_urls:
        txt += f"- {url}\n"
    
    return {
        **state, 
        "scratchpad": [SystemMessage(content=txt)],
        "plan_history": plan_history,
        "action_history": action_history
    }

## Updates the LLM and agent setup
def create_react_prompt(state):
    """Create the prompt for the ReAct agent"""
    page_url = state["page"].url
    scratchpad_content = ""
    if state.get("scratchpad"):
        scratchpad_content = state["scratchpad"][0].content
    
    return {
        "input": state["input"],
        "page_url": page_url,
        "bbox_descriptions": state.get("bbox_descriptions", ""),
        "scratchpad_content": scratchpad_content
    }

## Orchestrator
def select_tool(state: AgentState):
    """Route to appropriate tool or end based on the action"""
    action = state["prediction"]["action"]
    
    # Check if we're trying to revisit a recently visited page
    if action in ["Click", "Type"] and len(state.get("action_history", [])) > 3:
        # Get the last few actions
        recent_actions = state["action_history"][-3:]
        current_url = state["page"].url
        
        # Check for action loops at the same URL
        action_types = [a["action"] for a in recent_actions]
        urls = [a["url"] for a in recent_actions]
        
        # If we're doing the same action repeatedly on the same page, try something else
        if (action in action_types and current_url in urls and 
            action_types.count(action) >= 2 and urls.count(current_url) >= 2):
            return "agent"  # Force a reconsideration
    
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

## Browser
# We will set headless=False so we can watch the agent navigate the web.
async def setup_browser():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",  # Prevent detection
            "--no-sandbox",  # Needed in some environments
            "--disable-dev-shm-usage",  # Reduce memory issues
            "--disable-gpu",  # Prevents GPU-related issues
            "--window-size=1280,720",  # Set a standard window size
            "--incognito",  # Open in an incognito session
        ]
    )
    page = await browser.new_page()
    _ = await page.goto("https://www.google.com")
    return browser, page

## Agent
async def call_agent(graph, question: str, page, max_steps: int = 150):
    """Run the agent with enhanced tracking and reporting using colorful output"""
    # ANSI Color codes
    COLORS = {
        "yellow": "\033[93m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "underline": "\033[4m",
        "reset": "\033[0m"
    }
    
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
            "visited_urls": set(),
            "plan_history": [],
            "action_history": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    
    final_answer = None
    steps = []
    
    async for event in event_stream:
        if "agent" not in event:
            continue
            
        pred = event["agent"].get("prediction") or {}
        thought = pred.get("thought", "")
        plan = pred.get("plan", "")
        action = pred.get("action", "")
        action_input = pred.get("args", [])
        current_url = event['agent']['page'].url
        
        display.clear_output(wait=False)
        
        # Format the step for display with colors
        step_info = (
            f"{COLORS['yellow']}{COLORS['bold']}Step {len(steps) + 1}:{COLORS['reset']}\n"
            f"{COLORS['green']}Thought: {thought[:150]}{'...' if len(thought) > 150 else ''}{COLORS['reset']}\n"
            f"{COLORS['blue']}Plan: {plan[:150]}{'...' if len(plan) > 150 else ''}{COLORS['reset']}\n"
            f"{COLORS['magenta']}Action: {action} {action_input}{COLORS['reset']}\n"
            f"{COLORS['cyan']}URL: {current_url}{COLORS['reset']}\n"
            f"---"
        )
        
        steps.append(step_info)
        print("\n".join(steps))
        
        if "ANSWER" in action:
            final_answer = action_input[0] if action_input else "No answer provided"
            print(f"\n{COLORS['red']}{COLORS['bold']}FINAL ANSWER: {COLORS['reset']}{COLORS['red']}{final_answer}{COLORS['reset']}\n")
            break
    
    # Return both the answer and action history for analysis
    action_history = event.get("agent", {}).get("action_history", [])
    plan_history = event.get("agent", {}).get("plan_history", [])
    visited_urls = event.get("agent", {}).get("visited_urls", set())
    
    # Print final results with colors
    print(f"\n{COLORS['bold']}Final Results:{COLORS['reset']}")
    print(f"{COLORS['red']}Answer: {final_answer}{COLORS['reset']}")
    print(f"{COLORS['yellow']}Total steps: {len(steps)}{COLORS['reset']}")
    print(f"{COLORS['cyan']}Visited URLs: {len(visited_urls)}{COLORS['reset']}")
    
    return {
        "answer": final_answer,
        "action_history": action_history,
        "plan_history": plan_history,
        "visited_urls": visited_urls,
        "steps": len(steps)
    }
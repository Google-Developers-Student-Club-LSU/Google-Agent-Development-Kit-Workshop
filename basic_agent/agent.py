import asyncio
import os
from typing import Dict, List

from google.adk.agents.llm_agent import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.tools.google_api_tool import CalendarToolset
from google.genai import types

EXAM_EXTRACTION_PROMPT = (
    "You are an intelligent assistant that reads syllabi and tells students what to expect on the next exam. "
    "Pull out the format clues (MCQ versus short response), the grading weight, and any drop policy."
)


async def extract_exam_details_from_syllabus(syllabus_text: str) -> Dict[str, str]:
    """Highlight exam format/weight/drop clues while referencing the preloaded LLM prompt."""

    #tip: validate params jus tincase llm halucinates 
    if not syllabus_text:
        return {
            "status": "error",
            "error_message": (
                "No syllabus text provided. Send the document text or an image, "
                "and this tool will read it for you."
            ),
        }
    await asyncio.sleep(0)
    summary_parts = []
    lower = syllabus_text.lower()
    if "mcq" in lower or "multiple choice" in lower:
        summary_parts.append("Format: multiple choice questions are included.")
    if "short response" in lower or "short answer" in lower:
        summary_parts.append("Format: includes short response questions.")
    if "lowest" in lower and "dropped" in lower:
        summary_parts.append("Drop policy: lowest score may be dropped.")
    weight_mention = next(
        (token for token in ("20%", "25%", "30%", "50%") if token in syllabus_text),
        None,
    )
    if weight_mention:
        summary_parts.append(f"Weight: {weight_mention} of the final grade.")
    exam_lines = [
        line
        for line in syllabus_text.splitlines()
        if "exam" in line.lower() or "test" in line.lower()
    ][:3]
    return {
        "status": "success",
        "prompt": EXAM_EXTRACTION_PROMPT,
        "exam_summary": (
            " ".join(summary_parts)
            if summary_parts
            else "No clear format or weight was detected."
        ),
        "exam_lines": " | ".join(line.strip() for line in exam_lines),
    }


agent_description = (
    "Workshop agent that reads a syllabus, highlights exam format/weight, and delegates reminders to the Google Calendar toolset."
)
agent_instruction = (
    "Call extract_exam_details_from_syllabus so the preloaded prompt (EXAM_EXTRACTION_PROMPT) is applied and share that summary. "
    "Use the Calendar toolset to schedule any exam reminders the user requests."
)

planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
)

calendar_client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
calendar_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
if not (calendar_client_id and calendar_client_secret):
    raise RuntimeError(
        "Provide GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET to authenticate the calendar tools."
    )

calendar_toolset = CalendarToolset(
    client_id=calendar_client_id, client_secret=calendar_client_secret
)

tools: List = [extract_exam_details_from_syllabus, calendar_toolset]

root_agent = Agent(
    model="gemini-2.0-flash",
    name="syllabus_exam_planner",
    description=agent_description,
    instruction=agent_instruction,
    planner=planner,
    tools=tools,
)

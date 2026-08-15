import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from itertools import repeat
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal, TypedDict

import pymupdf
from dotenv import load_dotenv
from langchain_chroma import Chroma

# LangChain components
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from unstructured.chunking.title import chunk_by_title

from llms import get_llm


class File(TypedDict):
    path: str
    content: str

class Action(BaseModel):
    type: Literal["tool call", "final answer"]
    tool: Literal["list_files", "read_file"] | None
    args: dict[str, Any]
    final_answer: str | None
    summary: str | None

class Response(BaseModel):
    observation: str | None
    thought: str | None
    action: Action

class ExecutorResult(BaseModel):
    decision: Literal["continue", "replan"]
    observations: str
    
class ListFilesArgs(BaseModel):
    pass

class ReadFilesArgs(BaseModel):
    path: str

class Command(BaseModel):
    name: Literal["pytest", "python", "ruff", "mypy"]
    args: list[str]
    
@dataclass
class Tool:
    func: Callable
    args_schema: type[BaseModel]

class Plan(BaseModel):
    instructions: list[str]

@dataclass
class AgentState:
    goal: str
    current_plan: Plan
    completed_steps: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    files_seen: set[str] = field(default_factory=set)
    current_step: str = ""
    step: int = 0
    debug: bool = False
    

llm = get_llm("openrouter")
root = Path("/home/user/home/ensimag/mini-rag-project")

ReAct_SYSTEM_PROMPT = """You are a read-only coding assistant working on the current project.

Your job is to answer questions about the codebase, explain how it works, identify likely
bugs, and suggest precise improvements. You can inspect files, but you cannot edit files,
execute code, or run tests. Never claim that you performed an action that your tools do not
support.

Available tools:

1. `list_files`
   - Purpose: list every file in the project.
   - Arguments: `{}`
   - Result: a list of file paths.

2. `read_file`
   - Purpose: read the complete contents of one file.
   - Arguments: `{"path": "<path>"}` where `path` is a project-relative path or a path
     returned by `list_files`.
   - Result: `{"path": "<path>", "content": "<file contents>"}`.

Operating rules:

- Base your answer only on the user's request and information actually present in the
  conversation or returned by a tool. Do not invent files, code, test results, or behavior.
- Before calling a tool, inspect the previous observations and files already seen.
- Do not call a tool again if its result is already available in the provided state,
  unless there is a specific reason to refresh or verify that information.
- If previous steps already identified a relevant file, use that information directly
  instead of calling `list_files` again.
- Read the files needed to support the answer before drawing conclusions about their
  contents. Avoid unnecessary tool calls and inspect one file per tool call.
- After each tool result, reassess whether more evidence is needed. When enough evidence is
  available, return a final answer that directly addresses the user and cites relevant file
  paths or code elements.
- If the available tools cannot verify something, state that limitation clearly and provide
  the best supported conclusion.
- When returning a final answer, also provide a concise `summary` of the useful facts
  discovered while completing the current step. This summary will be stored in the agent
  state and provided to later execution steps.
- The summary is intended for subsequent execution steps, not for the user.
It must contain only new, reusable information discovered during this step.
Do not repeat the current instruction.
Do not describe the reasoning process.
- Do not include detailed reasoning, raw tool outputs, or conversational filler in the summary.
- For tool calls, `summary` must be null.
- observation` must contain at most 2 short sentences.
- thought` must contain at most 2 short sentences.
- summary` must contain at most 3 short sentences.
- Do not repeat information between `observation`, `thought`, `final_answer`, and `summary`.

Every response must exactly follow this JSON-compatible schema:

{
  "observation": string | null,
  "thought": string | null,
  "action": {
    "type": "tool call" | "final answer",
    "tool": "list_files" | "read_file" | null,
    "args": object,
    "final_answer": string | null,
    "summary": string | null
  }
}

For a tool call, return this structure:

{
  "observation": "Concise summary of the facts established so far, or null",
  "thought": "Brief explanation of why this tool call is the next useful step",
  "action": {
    "type": "tool call",
    "tool": "list_files",
    "args": {},
    "final_answer": null,
    "summary": null
  }
}

For `read_file`, use the same structure with `"tool": "read_file"` and
`"args": {"path": "<path>"}`.

For the final answer, return this structure:

{
  "observation": "Concise summary of the relevant findings, or null",
  "thought": "Brief explanation that enough evidence is available to answer",
  "action": {
    "type": "final answer",
    "tool": null,
    "args": {},
    "final_answer": "Complete answer to the user",
    "summary": "Concise facts useful to subsequent execution steps"
  }
}

Do not add fields, omit fields, or use any other value for the enums. Request only one tool
call per response. Return data matching the schema, not Markdown or explanatory text around
it."""

PLANNER_SYSTEM_PROMPT ="""You are a planning agent for an autonomous software engineering agent. 
Your role is to transform the user's goal into a short sequence of clear, high-level instructions that 
can be executed one after another by another agent. Rules: 
    - Do not execute tools yourself. 
    - Do not attempt to solve the task directly. 
    - Each instruction must describe a goal, not a specific tool call. 
    - Instructions should be ordered logically. 
    - Each instruction should be independently understandable by the executor. 
    - Avoid unnecessary steps. 
    - Do not assume facts about the repository that have not been observed. 
    - Include a final verification step when the task involves modifying code. 
    - The plan may later be changed if execution reveals new information. 
    - Prefer 2 to 4 steps for ordinary code-analysis tasks.
    - Do not create separate steps for tightly related operations that can reasonably
    be completed together.
    - Each step must contribute new information or work.
    - Avoid steps whose purpose substantially overlaps with another step.
    - If the goal is trivial and can clearly be completed in a single execution step,
    return a plan containing exactly one instruction.
    Example: User goal: "Fix the failing tests related to document deletion.
" Good plan: 
    1. Identify the failing tests and collect the relevant error information. 
    2. Locate and inspect the code responsible for the failure. 
    3. Determine the root cause of the bug. 
    4. Apply an appropriate correction. 
    5. Run the relevant tests again and verify the fix.
The executor is read-only.

It can:
- list project files;
- read project files.

It cannot:
- modify files;
- execute code;
- run tests;
- run shell commands.

Only create steps that can be completed using these capabilities."""

FINAL_RESPONSE_SYSTEM_PROMP = """You are the final response generator of an autonomous agent.

The execution phase has finished.
Use the original goal and the observations collected during execution to produce
a clear, concise final answer for the user.

Do not call tools.
Do not invent information that is not supported by the observations.
Do not mention internal planning steps unless they are useful to the answer."""

ROOT = Path("/home/user/home/ensimag/mini-rag-project")

IGNORED_DIRS = {
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "chroma_db",
    "uploads",
    "node_modules",
    "dist",
    "build",
    "pymupdf-venv",
}

ALLOWED_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
}

ALLOWED_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "requirements.txt",
}

def list_files() -> list[Path]:
    paths: list[Path] = []

    for current_dir, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        current_path = Path(current_dir)

        for filename in files:
            path = current_path / filename

            if (
                path.suffix not in ALLOWED_EXTENSIONS
                and path.name not in ALLOWED_FILENAMES
            ):
                continue

            paths.append(path.relative_to(ROOT))
    return paths

def read_file(path: str) -> File:
    """reads complete content of a file"""
    with open(path) as f:
        return {"path": path, "content": f.read()}
    
def run_command(command: Command):
    name = command.name
    args = command.args
    result = subprocess.run(
                            [name, *args],
                            capture_output=True,
                            text=True
            )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
    
def apply_patch(source_path: Path, patch: str):
    output_path = source_path.with_name(f"{source_path.stem}_modified{source_path.suffix}")
    shutil.copy2(source_path, output_path)
    result = subprocess.run(
                            ["patch", str(output_path)],
                            input=patch,
                            text=True,
                            capture_output=True
            )
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(result.stderr)
    return output_path
        
    
TOOLS = {
    "list_files": Tool(func=list_files, args_schema=ListFilesArgs),
    "read_file": Tool(func=read_file, args_schema=ReadFilesArgs)
}

def invoke_structured(
    structured_llm,
    messages: list[BaseMessage],
    expected_type: type[BaseModel],
    stage: str,
    debug: bool = False,
) -> BaseModel:
    """Invoke a structured LLM and retain useful diagnostics on parse failures."""
    failure_details = "No response returned"

    for attempt in range(1, 3):
        envelope = structured_llm.invoke(messages)

        if not isinstance(envelope, dict):
            parsed = envelope
            parsing_error = None
            raw = None
        else:
            parsed = envelope.get("parsed")
            parsing_error = envelope.get("parsing_error")
            raw = envelope.get("raw")

        if isinstance(parsed, expected_type):
            return parsed

        metadata = getattr(raw, "response_metadata", {}) or {}
        raw_content = getattr(raw, "content", None)
        invalid_tool_calls = getattr(raw, "invalid_tool_calls", None)
        failure_details = (
            f"finish_reason={metadata.get('finish_reason')!r}, "
            f"parsing_error={parsing_error!r}, "
            f"invalid_tool_calls={invalid_tool_calls!r}, "
            f"content={str(raw_content)[:500]!r}"
        )

        if attempt == 1:
            debug_print(
                debug,
                f"[{stage}] Invalid structured response; retrying once. {failure_details}",
            )

    raise TypeError(
        f"LLM did not return {expected_type.__name__} during {stage}. {failure_details}"
    )

def debug_print(enabled: bool, message: str):
    if enabled:
        print(message)

def print_response_debug(enabled: bool, stage: str, response: Response):
    """Print the relevant parts of a ReAct decision without dumping tool contents."""
    debug_print(enabled, f"[{stage}][OBSERVATION] {response.observation}")
    debug_print(enabled, f"[{stage}][THOUGHT] {response.thought}")
    debug_print(enabled, f"[{stage}][ACTION] {response.action.model_dump_json()}")

def ReAct_agent_loop(query: str, debug: bool = False):
    structured_llm = llm.with_structured_output(
        Response,
        method="json_schema",
        include_raw=True,
        strict=True,
    )
    local_history: list[BaseMessage] = [SystemMessage(content=ReAct_SYSTEM_PROMPT)]
    local_history.append(HumanMessage(content=query))
    result = invoke_structured(
        structured_llm,
        local_history,
        Response,
        "REACT initial response",
        debug,
    )
    print_response_debug(debug, "REACT][TURN 1", result)
    action: Action = result.action
    local_history.append(AIMessage(result.model_dump_json()))
    turn = 1
    while action.type == 'tool call':
        if action.tool == None:
            raise TypeError("No tool was called")
        tool = TOOLS[action.tool]
        func = tool.func
        args = action.args
        debug_print(debug, f"[REACT][TURN {turn}][TOOL CALL] {action.tool} args={args}")
        val_args = tool.args_schema.model_validate(args)
        tool_result = func(**val_args.model_dump())
        local_history.append(HumanMessage(
        content=f"""
                Tool `{action.tool}` returned:

                {tool_result}
                """
                    ))
        result = invoke_structured(
            structured_llm,
            local_history,
            Response,
            "REACT tool result",
            debug,
        )
        turn += 1
        print_response_debug(debug, f"REACT][TURN {turn}", result)
        action = result.action
        local_history.append(AIMessage(result.model_dump_json()))
    debug_print(debug, f"[REACT][FINAL ANSWER] {action.final_answer}")
    return action.final_answer
        
def executor_agent_loop(state: AgentState):
    instructions = state.current_plan.instructions
    debug_print(state.debug, f"[EXECUTOR][START] Executing {len(instructions)} planned steps")
    for step, instruction in enumerate(instructions, start=1):
        state.step = step
        state.current_step = instruction
        debug_print(state.debug, f"[EXECUTOR][STEP {step}/{len(instructions)}][START] {instruction}")
        debug_print(state.debug, f"[EXECUTOR][STEP {step}/{len(instructions)}][FILES SEEN] {sorted(state.files_seen)}")
        debug_print(state.debug, f"[EXECUTOR][STEP {step}/{len(instructions)}][PREVIOUS OBSERVATIONS] {state.observations}")
        ReAct_loop(state)
        state.completed_steps.append(instruction)
        debug_print(state.debug, f"[EXECUTOR][STEP {step}/{len(instructions)}][COMPLETE]")
    final_message = f"""Original goal:
                    {state.goal}

                    Completed steps:
                    {state.completed_steps}

                    Observations:
                    {state.observations}

                    All planned steps have now been completed.

                    Produce the final answer to the user's original goal.
                    Base your answer only on the information contained in the observations.
                    Do not describe the planning process unless relevant."""
    debug_print(state.debug, "[EXECUTOR][FINAL SYNTHESIS][START]")
    result = llm.invoke([SystemMessage(content=FINAL_RESPONSE_SYSTEM_PROMP), HumanMessage(content=final_message)])
    debug_print(state.debug, f"[EXECUTOR][FINAL SYNTHESIS][RESULT] {result.content}")
    return result.content
 
def ReAct_loop(state: AgentState):
    files_read_this_step: set[str] = set()
    message = f"""Goal:
                    {state.goal}
                    current plan:
                    {state.current_plan.instructions}
                    completed steps:
                    {state.completed_steps}
                    files seen:
                    {state.files_seen}
                    observations:
                    {state.observations}
                    current step:
                    {state.current_step}"""
    instruction = state.current_step
    structured_llm = get_llm("openrouter").with_structured_output(
        Response,
        method="json_schema",
        include_raw=True,
        strict=True,
    )
    local_history: list[BaseMessage] = [SystemMessage(content=ReAct_SYSTEM_PROMPT)]
    local_history.append(HumanMessage(message))
    local_history.append(HumanMessage(content=instruction))
    result = invoke_structured(
        structured_llm,
        local_history,
        Response,
        f"EXECUTOR step {state.step} initial response",
        state.debug,
    )
    print_response_debug(state.debug, f"EXECUTOR][STEP {state.step}][TURN 1", result)
    action: Action = result.action
    local_history.append(AIMessage(result.model_dump_json()))
    turn = 1
    while action.type == 'tool call':
        if action.tool == None:
            raise TypeError("No tool was called")
        debug_print(
            state.debug,
            f"[EXECUTOR][STEP {state.step}][TURN {turn}][TOOL CALL] "
            f"{action.tool} args={action.args}"
        )
        tool = TOOLS[action.tool]
        func = tool.func
        args = action.args
        val_args = tool.args_schema.model_validate(args)
        if action.tool == 'read_file':
            path = val_args.path
            if path in files_read_this_step:
                raise ValueError(
                    f"File {path} was requested twice during plan step {state.step}."
                )
            if path in state.files_seen:
                debug_print(
                    state.debug,
                    f"[EXECUTOR][STEP {state.step}][TURN {turn}][DUPLICATE READ] "
                    f"{path} was read during an earlier plan step; reading it again because "
                    "this step has a separate ReAct history."
                )
            files_read_this_step.add(path)
            state.files_seen.add(path)
        tool_result = func(**val_args.model_dump())
        if isinstance(tool_result, dict) and "content" in tool_result:
            result_description = f"{len(tool_result['content'])} characters"
        elif isinstance(tool_result, list):
            result_description = f"{len(tool_result)} items"
        else:
            result_description = type(tool_result).__name__
        debug_print(
            state.debug,
            f"[EXECUTOR][STEP {state.step}][TURN {turn}][TOOL RESULT] "
            f"{action.tool}: {result_description}"
        )
        local_history.append(HumanMessage(
        content=f"""
                Tool `{action.tool}` returned:

                {tool_result}
                """
                    ))
        result = invoke_structured(
            structured_llm,
            local_history,
            Response,
            f"EXECUTOR step {state.step} tool result",
            state.debug,
        )
        turn += 1
        print_response_debug(state.debug, f"EXECUTOR][STEP {state.step}][TURN {turn}", result)
        action = result.action
        local_history.append(AIMessage(result.model_dump_json()))
    debug_print(state.debug, f"[EXECUTOR][STEP {state.step}][FINAL ANSWER] {action.final_answer}")
    if action.summary == None:
        raise ValueError("No observation was returned by the llm")
    state.observations.append(action.summary)
    debug_print(state.debug, f"[EXECUTOR][STEP {state.step}][SUMMARY SAVED] {action.summary}")
       
def planner_agent_loop(query: str, debug: bool = False):
    debug_print(debug, f"[PLANNER][START] Query: {query}")
    structured_llm = get_llm("openrouter").with_structured_output(
        Plan,
        method="json_schema",
        include_raw=True,
        strict=True,
    )
    
    message = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=query)]
    debug_print(debug, "[PLANNER][LLM CALL] Requesting an execution plan")
    result = invoke_structured(structured_llm, message, Plan, "PLANNER", debug)
    debug_print(debug, f"[PLANNER][PLAN RECEIVED] {len(result.instructions)} steps")
    for step, instruction in enumerate(result.instructions, start=1):
        debug_print(debug, f"[PLANNER][PLAN][STEP {step}/{len(result.instructions)}] {instruction}")
    agent_state = AgentState(goal=query, current_plan=result, debug=debug)
    debug_print(debug, "[PLANNER][HANDOFF] Initial state created; starting executor")
    return executor_agent_loop(agent_state)
    
DEBUG = False

if __name__ == "__main__":
    answer = planner_agent_loop(
        "Find where the ReAct agent loop is defined and explain its role.",
        debug=DEBUG,
    )
    print(answer)

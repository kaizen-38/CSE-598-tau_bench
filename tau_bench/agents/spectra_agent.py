"""SPECTRA agent — Slot-Provenance Execution with Critic-Triggered Recovery And loop detection.

Wraps the ReAct action loop with four interception layers that target the
empirically observed failure modes from Phase 1/2 error analysis:

  1. LoopStateDetector  (LSD)  — RC1 loops + RC2 phantom tools  [deterministic]
  2. ArgumentGroundingGuard (AGG) — RC1 placeholder / hallucinated args  [deterministic]
  3. SurrenderInterceptor (SISS) — RC4 premature transfer_to_human  [deterministic]
  4. ResponseConsistencyAuditor (RCA) — RC3 wrong outputs / synthesis errors  [1 LLM call]

The base agent system prompt (ReAct instruction + wiki + tools) is left
*completely unchanged*. All interception happens via synthetic API-error
messages that the model naturally incorporates into its reasoning.
"""

import json
from typing import Optional, List, Dict, Any, Tuple, Set

from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.agents.loop_detector import LoopStateDetector
from tau_bench.agents.arg_guard import ArgumentGroundingGuard
from tau_bench.agents.response_auditor import ResponseConsistencyAuditor
from tau_bench.agents.surrender_interceptor import SurrenderInterceptor
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


def _extract_tool_names(tools_info: List[Dict[str, Any]]) -> Set[str]:
    names: Set[str] = set()
    for tool in tools_info:
        if "function" in tool and "name" in tool["function"]:
            names.add(tool["function"]["name"])
        elif "name" in tool:
            names.add(tool["name"])
    names.add(RESPOND_ACTION_NAME)
    names.add("transfer_to_human_agents")
    return names


class SpectraAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.prompt = (
            wiki
            + "\n#Available tools\n"
            + json.dumps(tools_info)
            + REACT_INSTRUCTION
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.tools_info = tools_info

        tool_names = _extract_tool_names(tools_info)
        self.loop_detector = LoopStateDetector(tool_names)
        self.arg_guard = ArgumentGroundingGuard()
        self.response_auditor = ResponseConsistencyAuditor(
            model=model, provider=provider, temperature=0.0
        )
        self.surrender_interceptor = SurrenderInterceptor(
            max_retries=2, min_tool_calls=2
        )

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        action_str = message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(
            name=action_parsed["name"], kwargs=action_parsed["arguments"]
        )
        cost = (
            res._hidden_params.get("response_cost", 0.0)
            if hasattr(res, "_hidden_params")
            else 0.0
        )
        return message.model_dump(), action, cost

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info: Dict[str, Any] = {}
        tool_call_count = 0

        self.loop_detector.reset()
        self.surrender_interceptor.reset()

        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            total_cost += cost if cost is not None else 0.0

            is_tool_call = (
                action.name != RESPOND_ACTION_NAME
                and action.name != "transfer_to_human_agents"
            )

            # ── Guard 1: Loop & phantom-tool detection ──────────────
            if is_tool_call:
                loop_result = self.loop_detector.check(
                    action.name, action.kwargs
                )
                if loop_result.kind != "ok":
                    messages.extend(
                        [
                            message,
                            {
                                "role": "user",
                                "content": "API output: " + loop_result.message,
                            },
                        ]
                    )
                    continue

            # ── Guard 2: Argument grounding (placeholder detection) ─
            if is_tool_call:
                guard_result = self.arg_guard.check(
                    action.name, action.kwargs
                )
                if not guard_result.grounded:
                    messages.extend(
                        [
                            message,
                            {
                                "role": "user",
                                "content": "API output: " + guard_result.message,
                            },
                        ]
                    )
                    continue

            # ── Guard 3: Surrender interception ─────────────────────
            if action.name == "transfer_to_human_agents":
                if self.surrender_interceptor.should_block(
                    tool_call_count, messages
                ):
                    messages.extend(
                        [
                            message,
                            {
                                "role": "user",
                                "content": (
                                    "API output: "
                                    + self.surrender_interceptor.get_retry_message()
                                ),
                            },
                        ]
                    )
                    continue

            # ── Guard 4: Response consistency audit ─────────────────
            if action.name == RESPOND_ACTION_NAME and tool_call_count > 0:
                original = action.kwargs.get(RESPOND_ACTION_FIELD_NAME, "")
                audited = self.response_auditor.audit(original, messages)
                if audited != original:
                    action = Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={RESPOND_ACTION_FIELD_NAME: audited},
                    )

            # ── Execute the (possibly corrected) action ─────────────
            env_response = env.step(action)
            obs = env_response.observation
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
                tool_call_count += 1

            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )

            if env_response.done:
                break

        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )

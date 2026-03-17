from typing import List, Dict, Any

from litellm import completion


_AUDITOR_SYSTEM_PROMPT = (
    "You are a factual accuracy auditor for a customer service agent. "
    "Your ONLY job is to verify that every factual claim in the agent's "
    "draft response matches the actual API/tool outputs from the "
    "conversation.\n\n"
    "Check all verifiable claims: prices, totals, dates, times, flight "
    "numbers, reservation IDs, order IDs, refund amounts, payment "
    "method charges, statuses, passenger names, and item details.\n\n"
    "Rules:\n"
    "- If ALL claims are correct, return the response EXACTLY as-is.\n"
    "- If ANY claim is wrong, return a corrected version where only the "
    "incorrect values are replaced with the actual API values.\n"
    "- Do NOT change tone, style, or structure.\n"
    "- Do NOT add information the agent omitted.\n"
    "- Output ONLY the response text, nothing else."
)


class ResponseConsistencyAuditor:
    """LLM-based response auditor (one call per task).

    Targets RC3 (WrongOutputs / synthesis errors) by extracting
    factual claims from the agent's draft response and cross-
    referencing them against actual tool/API outputs in the
    conversation history.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def audit(
        self,
        proposed_response: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        conversation_summary = self._extract_tool_outputs(messages)
        if not conversation_summary:
            return proposed_response

        audit_messages = [
            {"role": "system", "content": _AUDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "TOOL/API OUTPUTS FROM CONVERSATION:\n"
                    f"{conversation_summary}\n\n"
                    "AGENT'S DRAFT RESPONSE TO CUSTOMER:\n"
                    f"{proposed_response}\n\n"
                    "Return the verified (and corrected if needed) response:"
                ),
            },
        ]

        try:
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=audit_messages,
                temperature=self.temperature,
            )
            corrected = res.choices[0].message.content
            if corrected and corrected.strip():
                return corrected.strip()
        except Exception:
            pass

        return proposed_response

    @staticmethod
    def _extract_tool_outputs(messages: List[Dict[str, Any]]) -> str:
        """Pull all API/tool outputs from the conversation history."""
        outputs = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("API output:"):
                outputs.append(content)
        if not outputs:
            return ""
        return "\n---\n".join(outputs[-15:])

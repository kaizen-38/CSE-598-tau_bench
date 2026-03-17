from typing import List, Dict, Any


class SurrenderInterceptor:
    """Blocks premature transfer_to_human_agents calls.

    Targets RC4 (PrematureSurrender — 41.5% of all failures).
    When the agent proposes surrender, this interceptor checks
    whether the agent has made a genuine effort (enough tool
    calls, enough turns) before allowing the transfer.
    """

    def __init__(self, max_retries: int = 2, min_tool_calls: int = 2) -> None:
        self._surrender_count = 0
        self._max_retries = max_retries
        self._min_tool_calls = min_tool_calls

    def reset(self) -> None:
        self._surrender_count = 0

    def should_block(
        self,
        tool_call_count: int,
        messages: List[Dict[str, Any]],
    ) -> bool:
        if self._surrender_count >= self._max_retries:
            return False

        if tool_call_count < self._min_tool_calls:
            self._surrender_count += 1
            return True

        assistant_count = sum(
            1 for m in messages if m.get("role") == "assistant"
        )
        if assistant_count < 4:
            self._surrender_count += 1
            return True

        return False

    @staticmethod
    def get_retry_message() -> str:
        return (
            "The transfer to a human agent is not available at this time. "
            "Please try to resolve the customer's request using the "
            "available tools. Consider: (1) Have you looked up the user's "
            "information? (2) Have you retrieved the relevant reservation "
            "or order details? (3) Are there alternative approaches you "
            "haven't tried yet?"
        )

import json
import hashlib
from dataclasses import dataclass
from typing import Optional, Set, Dict


@dataclass
class LoopResult:
    kind: str  # 'ok', 'phantom', 'repeat'
    tool: Optional[str] = None
    args: Optional[dict] = None
    message: Optional[str] = None


class LoopStateDetector:
    """Deterministic loop and phantom-tool detector (zero LLM cost).

    Targets RC2 (PhantomToolLoop) and the loop variant of RC1
    (ActionToolFailureLoop). Uses a hash set to detect repeated
    (tool_name, args) pairs and a schema-membership gate to catch
    calls to non-existent tools (e.g. Qwen3's <think> parsed by
    vLLM Hermes as a tool call).
    """

    def __init__(self, registered_tools: Set[str]):
        self.registered = set(registered_tools)
        self._seen: Set[str] = set()
        self._repeat_counts: Dict[str, int] = {}

    def reset(self) -> None:
        self._seen.clear()
        self._repeat_counts.clear()

    def check(self, tool_name: str, args: dict) -> LoopResult:
        if tool_name not in self.registered:
            valid = ", ".join(sorted(self.registered))
            return LoopResult(
                kind="phantom",
                tool=tool_name,
                message=(
                    f"Error: '{tool_name}' is not a recognized tool. "
                    f"Available tools are: {valid}. "
                    "Please use one of the listed tools."
                ),
            )

        state_key = tool_name + "|" + json.dumps(args, sort_keys=True)[:200]
        key_hash = hashlib.md5(state_key.encode()).hexdigest()

        if key_hash in self._seen:
            self._repeat_counts[key_hash] = (
                self._repeat_counts.get(key_hash, 1) + 1
            )
            return LoopResult(
                kind="repeat",
                tool=tool_name,
                args=args,
                message=(
                    f"Error: This exact call to '{tool_name}' with the same "
                    "arguments was already attempted. Please modify your "
                    "arguments or try a different approach. If you need "
                    "information you don't have, ask the user for it."
                ),
            )

        self._seen.add(key_hash)
        return LoopResult(kind="ok")

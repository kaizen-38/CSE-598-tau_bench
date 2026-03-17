import re
from dataclasses import dataclass
from typing import Optional, List


_PLACEHOLDER_PATTERNS: List[re.Pattern] = [
    re.compile(r'^"?(string|integer|boolean|number|array|object)"?$', re.I),
    re.compile(r"^<[^>]+>$"),
    re.compile(r'^"?\.{2,}"?$'),
    re.compile(r'^"?placeholder"?$', re.I),
    re.compile(r'^"?example"?$', re.I),
    re.compile(r'^"?N/?A"?$', re.I),
    re.compile(r'^"?null"?$', re.I),
    re.compile(r'^"?undefined"?$', re.I),
    re.compile(r'^"?none"?$', re.I),
]


@dataclass
class GroundingResult:
    grounded: bool
    arg_name: Optional[str] = None
    value: Optional[str] = None
    message: Optional[str] = None


class ArgumentGroundingGuard:
    """Deterministic argument grounding check (zero LLM cost).

    Targets RC1 (DataFetchDeadEnd, AuthCredentialLoop) by catching
    schema placeholder values leaked into tool-call arguments.
    Uses regex fast-path to detect literal type names ("string",
    "integer"), XML-style placeholders (<user_id>), and common
    sentinel values.
    """

    def check(self, action_name: str, args: dict) -> GroundingResult:
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            stripped = value.strip().strip('"').strip("'")
            if not stripped:
                continue
            for pattern in _PLACEHOLDER_PATTERNS:
                if pattern.match(stripped):
                    return GroundingResult(
                        grounded=False,
                        arg_name=key,
                        value=stripped,
                        message=(
                            f"Error: The argument '{key}' has value "
                            f"'{stripped}' which is a schema placeholder, "
                            "not an actual value. Retrieve the real value "
                            "from the conversation or ask the user."
                        ),
                    )
            if isinstance(value, str) and value in ("[]", "{}", "[[]]"):
                return GroundingResult(
                    grounded=False,
                    arg_name=key,
                    value=value,
                    message=(
                        f"Error: The argument '{key}' is empty ({value}). "
                        "Please provide the actual values."
                    ),
                )
        return GroundingResult(grounded=True)

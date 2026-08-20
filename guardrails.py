import re

BLOCKED_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"jailbreak",
    r"bypass security",
    r"drop database",
    r"<script>",
]


class GuardrailEngine:

    @staticmethod
    def validate_input(query: str) -> tuple[bool, str]:
        """Validates character constraints and flags jailbreak patterns."""
        if not query or len(query.strip()) < 3:
            return False, "Query is too short or empty."

        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return (
                    False,
                    "Security Exception: Inappropriate or adversarial prompt pattern detected.",
                )

        return True, "Passed"

    @staticmethod
    def check_retrieval_confidence(
        similarity_score: float, threshold: float = 0.68
    ) -> bool:
        """Gates the LLM call if semantic relevance is below threshold."""
        return similarity_score >= threshold

    @staticmethod
    def verify_groundedness(answer: str, context: str) -> bool:
        """Lexical sanity check preventing off-context hallucinations."""
        if "insufficient information" in answer.lower():
            return True

        answer_tokens = set(re.findall(r"\w{3,}", answer.lower()))
        context_tokens = set(re.findall(r"\w{3,}", context.lower()))

        if not answer_tokens:
            return True

        overlap = answer_tokens.intersection(context_tokens)
        # At least 2 key terms must be supported by the retrieved context
        return len(overlap) >= 2
        
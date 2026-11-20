"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

REDACTED = "[REDACTED]"

# Kênh liên hệ chính thức (ground_truth trong data/pii_hallucination_samples.json)
# — được phép xuất hiện, không tính là PII.
PUBLIC_ALLOWLIST = {"support@vinbank.example"}


def _known_secret_pattern() -> str:
    """Regex khớp nguyên văn các secret trong data/protected/vinbank_secrets.json."""
    try:
        from core.config import DEMO_SECRETS
    except Exception:
        DEMO_SECRETS = []
    values = sorted({s for s in DEMO_SECRETS if s}, key=len, reverse=True)
    if not values:
        return r"(?!x)x"  # không khớp gì
    return "|".join(re.escape(v) + (r"(?::\d+)?" if "." in v else "") for v in values)


PII_PATTERNS = {
    # --- Secret / credential ---
    # "password is X", "password=X", "mật khẩu: X", "pwd: X"
    "password": re.compile(
        r"\b(?:password|passwd|pwd|passcode|mật\s*khẩu|mat\s*khau)\b"
        r"\s*(?:is|=|:|là|la)\s*[\"'`]?\S+?[\"'`]?(?=[\s,;]|\.(?:\s|$)|$)",
        re.IGNORECASE,
    ),
    # sk-..., sk_live_..., v.v.
    "api_key": re.compile(r"\bsk[-_][A-Za-z0-9_-]{6,}", re.IGNORECASE),
    # Giá trị secret đã biết (admin123, db.vinbank.internal...) dù không có "password is"
    "internal_secret": re.compile(_known_secret_pattern(), re.IGNORECASE),
    # Host nội bộ *.internal / *.local (kèm port)
    "internal_host": re.compile(
        r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:internal|local|corp)(?::\d{2,5})?\b",
        re.IGNORECASE,
    ),
    # --- PII ---
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}"),
    # CMND 9 số / CCCD 12 số — đứng riêng, không dính chữ số khác
    "national_id": re.compile(r"(?<!\d)(?:\d{12}|\d{9})(?!\d)"),
    # SĐT VN: 0xxxxxxxxx (10–11 số) hoặc +84…, cho phép dấu cách/chấm/gạch
    "phone": re.compile(
        r"(?<![\d+])(?:\+84|0)(?:\d{9,10}|\d{2,3}[\s.-]\d{3}[\s.-]\d{3,4})(?!\d)"
    ),
}


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response or ""

    def _sub(match: re.Match) -> str:
        # Giữ nguyên kênh liên hệ công khai (không phải PII khách hàng)
        if match.group(0).lower() in PUBLIC_ALLOWLIST:
            return match.group(0)
        return REDACTED

    # Thứ tự quan trọng: secret/password trước (để "password is admin123" bị che
    # trọn cụm), rồi mới tới các mẫu PII chung.
    for name, pattern in PII_PATTERNS.items():
        hits = [
            m for m in pattern.finditer(redacted)
            if m.group(0).lower() not in PUBLIC_ALLOWLIST
        ]
        if hits:
            issues.append(f"{name}: {len(hits)} found")
            redacted = pattern.sub(_sub, redacted)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

OUTPUT_BLOCK_MESSAGE = (
    "I'm sorry, I can't share that response. "
    "How else can I help with your VinBank account?"
)


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_issues: list[str] = []

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Regex redact PII / secret
        filtered = content_filter(response_text)
        if not filtered["safe"]:
            self.redacted_count += 1
            self.last_issues = filtered["issues"]
            response_text = filtered["redacted"]
            llm_response.content = types.Content(
                role="model", parts=[types.Part.from_text(text=response_text)]
            )

        # 2. (Optional) LLM-as-Judge — chặn hẳn nếu judge nói UNSAFE
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=OUTPUT_BLOCK_MESSAGE)],
                )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")

    # Đối chiếu với dataset PII của lab
    cases = load_lab_pii_dataset().get("pii_cases", [])
    passed = 0
    print(f"\nPII dataset (data/pii_hallucination_samples.json): {len(cases)} cases")
    for case in cases:
        result = content_filter(case["input_text"])
        found = {i.split(":")[0] for i in result["issues"]}
        ok = result["safe"] == case["expect_safe"] and set(
            case.get("expect_issue_types", [])
        ) <= found
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {case['id']} {sorted(found)}")
    print(f"  → {passed}/{len(cases)} PASS")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()

"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct
"""

import re
import json

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    # Regex patterns cho từng loại PII — đã được định nghĩa sẵn, bạn chỉ cần dùng
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE":       r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted = value
        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[{pii_type}_REDACTED]", redacted)
        if redacted != value:
            return FailResult(error_message="PII detected", fix_value=redacted)
        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    @staticmethod
    def _repair(text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text).strip()
        try:
            return json.dumps(json.loads(text), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        # Remove trailing commas only outside JSON strings.
        text = re.sub(r'("(?:[^"\\]|\\.)*")|,\s*(?=[}\]])',
                      lambda m: m.group(1) or "", text)
        try:
            return json.dumps(json.loads(text), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        # literal_eval handles quoted Python dictionaries without corrupting apostrophes.
        import ast
        try:
            return json.dumps(ast.literal_eval(text), ensure_ascii=False)
        except (ValueError, SyntaxError, TypeError):
            return text

    def validate(self, value: str, metadata: dict):
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            try:
                parsed = json.loads(self._repair(value))
                return FailResult(error_message="JSON repaired", fix_value=json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                return FailResult(error_message="Unrepairable JSON", fix_value=json.dumps({"error": "invalid_json"}))


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    guard.configure(allow_metrics_collection=False)

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)

        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    guard.configure(allow_metrics_collection=False)

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)

        status = "✅ Pass" if result.validation_passed else "❌ Fail"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:60]}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    from contextlib import redirect_stdout
    from io import StringIO
    from pathlib import Path
    evidence = Path(__file__).parent.parent / "evidence"
    evidence.mkdir(exist_ok=True)
    for demo, filename in [(demo_pii_guard, "04_pii_demo_log.txt"),
                           (demo_json_guard, "04_json_demo_log.txt")]:
        output = StringIO()
        with redirect_stdout(output):
            demo()
        log = output.getvalue()
        print(log)
        (evidence / filename).write_text(log, encoding="utf-8")

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    main()

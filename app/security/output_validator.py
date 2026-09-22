import re

from app.security.pii_detector import redact_pii


PROJECT_PATTERN = re.compile(
    r"\bProject\s+[A-Z][A-Za-z0-9_-]*\b"
)


def redact_sensitive_data(text):
    # Step 1: General PII detection
    redacted_text, pii_findings = redact_pii(text)

    detected_terms = [
        finding["value"]
        for finding in pii_findings
    ]

    # Step 2: Detect confidential project names automatically
    project_matches = list(
        PROJECT_PATTERN.finditer(redacted_text)
    )

    # Replace from end to start so indexes stay valid
    for match in reversed(project_matches):
        detected_terms.append(match.group())

        redacted_text = (
            redacted_text[:match.start()]
            + "[REDACTED_PROJECT]"
            + redacted_text[match.end():]
        )

    return redacted_text, detected_terms

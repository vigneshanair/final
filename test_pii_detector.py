from app.security.pii_detector import redact_pii


text = """
Michael Thompson works for the company.

Employee ID: ABC-98765
Email: michael.thompson@company.com
Phone: 404-555-1289
"""

redacted, findings = redact_pii(text)

print("ORIGINAL:")
print(text)

print("\nDETECTED PII:")

for finding in findings:
    print(
        finding["type"],
        "->",
        finding["value"]
    )

print("\nREDACTED:")
print(redacted)

from app.security.output_validator import redact_sensitive_data


text = """
John Doe works on Project Falcon.
Employee ID: TEST-1042
Email: john@example.com
"""

safe_text, detected = redact_sensitive_data(text)

print("ORIGINAL:")
print(text)

print("\nSAFE OUTPUT:")
print(safe_text)

print("\nDETECTED:")
print(detected)

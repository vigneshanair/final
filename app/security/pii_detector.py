import re
import spacy


class PIIDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        )

        self.phone_pattern = re.compile(
            r"\b(?:\+?1[-.\s]?)?"
            r"(?:\(?\d{3}\)?[-.\s]?)"
            r"\d{3}[-.\s]?\d{4}\b"
        )

        self.employee_id_pattern = re.compile(
            r"\b[A-Z]{2,10}-\d{3,10}\b"
        )

    def detect(self, text):
        findings = []

        # Email detection
        for match in self.email_pattern.finditer(text):
            findings.append({
                "type": "EMAIL",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Phone number detection
        for match in self.phone_pattern.finditer(text):
            findings.append({
                "type": "PHONE",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Employee / identifier detection
        for match in self.employee_id_pattern.finditer(text):
            findings.append({
                "type": "EMPLOYEE_ID",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Person-name detection
        doc = self.nlp(text)

        for entity in doc.ents:
            if entity.label_ == "PERSON":
                findings.append({
                    "type": "PERSON",
                    "value": entity.text,
                    "start": entity.start_char,
                    "end": entity.end_char
                })

        return findings


def redact_pii(text):
    detector = PIIDetector()

    findings = detector.detect(text)

    # Remove duplicates
    unique = {}

    for finding in findings:
        key = (
            finding["start"],
            finding["end"],
            finding["type"]
        )
        unique[key] = finding

    findings = list(unique.values())

    # Replace from the end of the string backwards
    # so positions don't change while replacing.
    findings = sorted(
        findings,
        key=lambda x: x["start"],
        reverse=True
    )

    redacted = text

    for finding in findings:
        replacement = f"[REDACTED_{finding['type']}]"

        redacted = (
            redacted[:finding["start"]]
            + replacement
            + redacted[finding["end"]:]
        )

    return redacted, findings

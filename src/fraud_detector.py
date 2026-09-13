import re
from nemo.collections.asr.models import ASRModel


SUSPICIOUS_PATTERNS = {
    "Bank/Card impersonation": [
        "calling you from",
        "fraud watch",
        "security of your",
        "visa",
        "mastercard",
    ],

    "Request for financial information": [
        "account numbers",
        "card numbers",
        "confirm your cards",
        "confirm your",
        "which card",
        "visa or your mastercard",
    ],

    "Request for security credentials": [
        "security code",
        "authorized card holder",
    ],

    "Fraud/scare tactic": [
        "fraud",
        "computer related frauds",
        "held responsible",
        "fraud charge",
        "protection",
    ],
}


def detect_fraud(transcript):
    text = transcript.lower()

    detected = []
    score = 0

    for category, patterns in SUSPICIOUS_PATTERNS.items():
        matches = []

        for pattern in patterns:
            if re.search(re.escape(pattern), text):
                matches.append(pattern)

        if matches:
            detected.append({
                "category": category,
                "matches": matches
            })

            score += 20

    score = min(score, 100)

    if score >= 70:
        risk = "HIGH"
    elif score >= 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return score, risk, detected


print("Loading Parakeet...")

model = ASRModel.from_pretrained(
    "nvidia/parakeet-ctc-0.6b"
)

model = model.to("cuda")

print("Transcribing fraud_call.wav...")

output = model.transcribe([
    "audio/fraud_call.wav"
])

result = output[0]

if hasattr(result, "text"):
    transcript = result.text
else:
    transcript = str(result)


score, risk, indicators = detect_fraud(transcript)


print("\n========== TRANSCRIPT ==========\n")
print(transcript)

print("\n========== FRAUD DETECTION ==========\n")
print(f"Risk Score : {score}%")
print(f"Risk Level : {risk}")

print("\nSuspicious Indicators:")

for indicator in indicators:
    print(f"\n[{indicator['category']}]")

    for match in indicator["matches"]:
        print(f"  - {match}")

print("\n======================================")

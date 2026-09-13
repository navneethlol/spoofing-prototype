import torch
from nemo.collections.asr.models import ASRModel

print("Loading Parakeet...")

model = ASRModel.from_pretrained("nvidia/parakeet-ctc-0.6b")
model = model.to("cuda")

print("Transcribing fraud_call.wav...")

output = model.transcribe(["audio/fraud_call.wav"])

print("\n========== TRANSCRIPT ==========\n")

result = output[0]

if hasattr(result, "text"):
    print(result.text)
else:
    print(result)

print("\n================================")


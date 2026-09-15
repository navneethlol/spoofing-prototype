import torch
import librosa
import sys

from models.AASIST import Model


CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}


print("Loading AASIST...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(CONFIG).to(device)

checkpoint = torch.load(
    "models/weights/AASIST.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

print("AASIST loaded successfully!")
print("Device:", device)


audio_path = "../audio/fraud_call.wav"

print("\nLoading audio:", audio_path)

audio, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)

print("Sample rate:", sr)
print("Duration:", round(len(audio) / sr, 2), "seconds")


# Convert to tensor
audio = torch.tensor(audio, dtype=torch.float32)


# AASIST expects 64600 samples (~4 seconds)
nb_samp = CONFIG["nb_samp"]

if len(audio) < nb_samp:
    audio = torch.nn.functional.pad(
        audio,
        (0, nb_samp - len(audio))
    )
else:
    audio = audio[:nb_samp]


audio = audio.unsqueeze(0).to(device)


print("\nRunning AASIST...")

with torch.no_grad():
    output = model(audio)

print("\nRaw model output:")
print(output)

features, logits = output

print("\nClassification logits:")
print(logits)

probabilities = torch.softmax(logits, dim=1)

print("\nProbabilities:")
print(probabilities)

print("\nPredicted class:")
print(torch.argmax(probabilities, dim=1).item())

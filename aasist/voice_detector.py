import torch
import librosa
import numpy as np

from models.AASIST import Model


# AASIST configuration
CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}

AUDIO_PATH = "../audio/fake_voice.wav"
MODEL_PATH = "models/weights/AASIST.pth"

SAMPLE_RATE = 16000
CHUNK_SIZE = 64600


# ==============================
# LOAD MODEL
# ==============================

print("Loading AASIST...")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Model(CONFIG).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

print("AASIST loaded successfully!")
print("Device:", device)


# ==============================
# LOAD AUDIO
# ==============================

print("\nLoading audio...")

audio, sr = librosa.load(
    AUDIO_PATH,
    sr=SAMPLE_RATE,
    mono=True
)

duration = len(audio) / SAMPLE_RATE

print(f"Audio duration: {duration:.2f} seconds")


# ==============================
# SPLIT INTO CHUNKS
# ==============================

chunks = []

for start in range(0, len(audio), CHUNK_SIZE):

    chunk = audio[start:start + CHUNK_SIZE]

    # Ignore extremely short final chunks
    if len(chunk) < SAMPLE_RATE:
        continue

    # Pad short final chunk
    if len(chunk) < CHUNK_SIZE:
        chunk = np.pad(
            chunk,
            (0, CHUNK_SIZE - len(chunk))
        )

    chunks.append(chunk)


print(f"Chunks to analyze: {len(chunks)}")


# ==============================
# ANALYSIS
# ==============================

spoof_scores = []
bonafide_scores = []


print("\nRunning voice anti-spoofing analysis...\n")


for i, chunk in enumerate(chunks):

    audio_tensor = torch.tensor(
        chunk,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        _, logits = model(audio_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # AASIST:
    # Class 0 = Spoof
    # Class 1 = Bonafide

    spoof_probability = probabilities[0, 0].item()
    bonafide_probability = probabilities[0, 1].item()

    spoof_scores.append(spoof_probability)
    bonafide_scores.append(bonafide_probability)

    # Calculate RMS energy of the chunk
    rms = np.sqrt(np.mean(chunk ** 2))

    print(
        f"Chunk {i + 1:02d}: "
        f"Spoof={spoof_probability * 100:.2f}% | "
        f"Bonafide={bonafide_probability * 100:.2f}% | "
        f"RMS={rms:.4f}"
    )


# ==============================
# AGGREGATE RESULTS
# ==============================

average_spoof = np.mean(spoof_scores)
average_bonafide = np.mean(bonafide_scores)


# ==============================
# FINAL RESULT
# ==============================

print("\n========================================")
print("       VOICE AUTHENTICITY RESULT")
print("========================================")

print(
    f"Average Spoof Score    : "
    f"{average_spoof * 100:.2f}%"
)

print(
    f"Average Bonafide Score : "
    f"{average_bonafide * 100:.2f}%"
)


if average_spoof >= 0.5:

    print(
        "\nVerdict: POSSIBLE AI-GENERATED / "
        "SPOOFED VOICE"
    )

else:

    print(
        "\nVerdict: LIKELY AUTHENTIC HUMAN VOICE"
    )


print("========================================")

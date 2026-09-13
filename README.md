# Spoofing Prototype

AI-based prototype for detecting fraudulent and scam phone calls using speech recognition and fraud-pattern analysis.

## Overview

Fraudulent phone calls often use social engineering techniques such as:

- Impersonating banks or financial institutions
- Creating urgency or fear
- Requesting card/account information
- Asking for OTPs, PINs, or security codes
- Attempting to obtain sensitive financial information

This project explores an AI-assisted system that analyzes a phone call, converts the caller's speech into text, and identifies suspicious patterns in the conversation.

## Current Prototype

The current prototype processes a recorded call through the following pipeline:

```text
Audio Recording
      ↓
NVIDIA Parakeet ASR
      ↓
Speech-to-Text Transcript
      ↓
Fraud Pattern Detection
      ↓
Risk Score
      ↓
Risk Level


# Absher Anti-Spoof MVP (Visual + Vitality + Context)

This repository contains a lightweight MVP implementing a 3-layer verification pipeline:

- **Layer 1 (Physiological / vitality)**: rPPG-like signal proxy from the green channel → `vitality_score ∈ [0,1]`
- **Layer 2 (Visual anti-spoof)**: motion + texture + integrity → `visual_score ∈ [0,1]`
- **Layer 3 (Contextual)**: device/geo/behavior **simulation** → `context_score ∈ [0,1]`

Final fusion (as agreed):

`final_score = 0.40*vitality_score + 0.40*visual_score + 0.20*context_score`

Decision:

`final_output = 1 if final_score >= 0.75 else 0`

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

A local web UI will open. Upload a short face video (~8 seconds) and select contextual scenarios.

## Notes (MVP)
- Contextual signals are **simulated** because production Absher APIs are not available in the MVP environment.
- The scoring logic is fully implemented; swapping simulated inputs with real telemetry is a direct integration step.

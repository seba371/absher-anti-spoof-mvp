
import gradio as gr
from scoring import run_mvp

DEVICE_CHOICES = [
    "Trusted device (known / secure)",
    "New device (not seen before)",
    "Risky device (emulator / rooted / unknown)",
]
GEO_CHOICES = [
    "Same city / normal network",
    "Same country / slightly unusual",
    "Different country / suspicious network",
]
BEHAVIOR_CHOICES = [
    "Normal transaction behavior",
    "Somewhat unusual",
    "Highly unusual",
]

def format_report(result: dict) -> str:
    pct = lambda x: f"{x*100:.1f}%"
    out = []
    out.append("## Results")
    out.append(f"**Final decision:** {'✅ TRUSTED (1)' if result['final_output']==1 else '❌ NOT TRUSTED (0)'}")
    out.append(f"**Final score:** {pct(result['final_score'])}")
    out.append("")
    out.append("### Layer 1 — Physiological (Vitality)")
    out.append(f"- vitality_score: {pct(result['vitality_score'])}  (raw std={result['vitality_raw']:.3f})")
    out.append("")
    out.append("### Layer 2 — Visual Anti-Spoof")
    out.append(f"- motion_score: {pct(result['motion_score'])}  (raw={result['motion_raw']:.3f})")
    out.append(f"- texture_score: {pct(result['texture_score'])}  (raw={result['texture_raw']:.1f})")
    out.append(f"- integrity_score: {pct(result['integrity_score'])}  (raw={result['integrity_raw']:.1f})")
    out.append(f"- visual_score (0.40/0.35/0.25): {pct(result['visual_score'])}")
    out.append(f"- visual_output (>=75%): {result['visual_output']}")
    out.append("")
    out.append("### Layer 3 — Contextual (Simulated)")
    out.append(f"- device_score: {pct(result['device_score'])} (raw={result['device_raw']})")
    out.append(f"- geo_score: {pct(result['geo_score'])} (raw={result['geo_raw']})")
    out.append(f"- behavior_score: {pct(result['behavior_score'])} (raw={result['behavior_raw']})")
    out.append(f"- context_score (0.40/0.35/0.25): {pct(result['context_score'])}")
    out.append(f"- contextual_output (>=70%): {result['contextual_output']}")
    out.append("")
    out.append("### Final Fusion")
    out.append("- final_score = 0.40*vitality_score + 0.40*visual_score + 0.20*context_score")
    out.append("- final_output = 1 if final_score >= 75% else 0")
    return "\n".join(out)

def run(video_file, device, geo, behavior):
    if video_file is None:
        return "Please upload a short face video (ideally ~8 seconds)."
    try:
        res = run_mvp(video_file, device, geo, behavior)
        return format_report(res)
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Absher Anti-Spoof MVP") as demo:
    gr.Markdown("# Absher Anti-Spoof MVP (Visual + Vitality + Context)\nUpload a short face video (ideally ~8 seconds). Choose a simulated context scenario, then run the check.")

    with gr.Row():
        video = gr.Video(label="Upload video (face, ~8 seconds)", sources=["upload"], format="mp4")
    with gr.Row():
        device = gr.Dropdown(DEVICE_CHOICES, value=DEVICE_CHOICES[0], label="Device scenario (MVP simulation)")
        geo = gr.Dropdown(GEO_CHOICES, value=GEO_CHOICES[0], label="Geo/network scenario (MVP simulation)")
        behavior = gr.Dropdown(BEHAVIOR_CHOICES, value=BEHAVIOR_CHOICES[0], label="Behavior scenario (MVP simulation)")
    btn = gr.Button("Run verification")
    report = gr.Markdown()

    btn.click(run, inputs=[video, device, geo, behavior], outputs=[report])

if __name__ == "__main__":
    demo.launch()

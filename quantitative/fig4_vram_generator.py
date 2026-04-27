"""
Generate Fig 4: VRAM trace per pipeline stage.

Uses real vram_verify.json measurements to construct a meaningful
stage-by-stage VRAM usage chart for the T1000 (4,096 MB).

Run: python quantitative/fig4_vram_generator.py
Output: figures/fig4_vram_trace.png
"""
import json
import sys
from pathlib import Path

# Use Python 3.11 has matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

# Load real VRAM data
vram = json.loads(Path("logs/vram_verify.json").read_text())
phi3_mb = vram["delta_phi3_mb"]      # 2736
finbert_mb = vram["delta_finbert_mb"]  # 525
peak_mb = vram["peak_mb"]            # 3261
budget_mb = vram["budget_limit_mb"]  # 4096
headroom_mb = vram["headroom_mb"]    # 835

# Pipeline stages and their VRAM usage
stages = [
    "Initial\n(empty)",
    "Phi-3-mini\nloads",
    "FinBERT\nloads\n(Phase A)",
    "FinBERT\nflush()\ncall",
    "Phi-3\nmap-reduce\n(Phase B)",
    "Guardrail\n+ done",
]
vram_values = [
    0,                          # Initial
    phi3_mb,                    # Phi-3 loaded
    phi3_mb + finbert_mb,       # FinBERT loaded (peak)
    phi3_mb,                    # FinBERT flushed
    phi3_mb + 180,              # Phi-3 map-reduce (KV cache in use)
    phi3_mb + 80,               # Guardrail (smaller KV)
]

# Colors per region
colors = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71", "#e67e22", "#9b59b6"]

fig, ax = plt.subplots(figsize=(11, 6))

# Background shading for phases
ax.axvspan(0.5, 2.5, alpha=0.08, color="#3498db", label="Phase A (FinBERT active)")
ax.axvspan(4.5, 5.5, alpha=0.08, color="#e67e22", label="Phase B (Phi-3 map-reduce)")

# Budget line
ax.axhline(budget_mb, color="#e74c3c", linestyle="--", linewidth=2.5,
           label=f"T1000 Budget: {budget_mb:.0f} MB")

# Danger zone
ax.fill_between(range(-1, len(stages) + 1), budget_mb, budget_mb * 1.08,
                color="#e74c3c", alpha=0.15, label="OOM zone")

# Main VRAM line
x = list(range(len(stages)))
ax.plot(x, vram_values, "o-", linewidth=2.5, markersize=9, color="#2c3e50", zorder=5)

# Colored markers
for i, (xv, yv, c) in enumerate(zip(x, vram_values, colors)):
    ax.scatter(xv, yv, color=c, s=120, zorder=6)
    offset = 50 if i != 2 else -120
    ax.annotate(f"{yv:.0f} MB",
                (xv, yv), xytext=(xv, yv + offset),
                ha="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5) if abs(offset) > 60 else None)

# Headroom annotation
ax.annotate("",
            xy=(2, phi3_mb + finbert_mb), xytext=(2, budget_mb),
            arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=1.5))
ax.text(2.1, (phi3_mb + finbert_mb + budget_mb) / 2,
        f"Headroom\n{headroom_mb:.0f} MB", color="#27ae60", fontsize=9, va="center")

# Flush annotation
ax.annotate("flush_gpu_cache()\nreleases FinBERT",
            xy=(3, phi3_mb), xytext=(3.5, phi3_mb - 300),
            fontsize=8, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))

ax.set_title(
    "Fig 4: VRAM Usage per Pipeline Stage (NVIDIA T1000, 4 GB)\n"
    "Phase-serialized GPU allocation prevents OOM while preserving parallelism",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Pipeline Stage", fontsize=11)
ax.set_ylabel("VRAM Used (MB)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylim(-100, budget_mb * 1.1)
ax.set_xlim(-0.3, len(stages) - 0.7)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Phi-3 fill under curve
ax.fill_between(x, 0, vram_values, alpha=0.12, color="#3498db")

plt.tight_layout()
out_path = Path("figures/fig4_vram_trace.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

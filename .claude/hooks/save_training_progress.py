"""
PostToolUse hook: detects training metrics in Bash output and appends to training_progress.md
"""
import json
import sys
import re
from datetime import datetime
from pathlib import Path

data = json.load(sys.stdin)

if data.get("tool_name") != "Bash":
    sys.exit(0)

output = data.get("tool_response", {})
if isinstance(output, dict):
    text = output.get("stdout", "") + output.get("stderr", "")
elif isinstance(output, str):
    text = output
else:
    text = str(output)

# Only trigger if output looks like a training run
training_signals = ["val_ap", "val/ap", "AP@", "epoch=", "Epoch ", "train_loss", "val_loss", "test_ap", "test/ap"]
if not any(s.lower() in text.lower() for s in training_signals):
    sys.exit(0)

# Extract key metric lines
metric_lines = []
for line in text.splitlines():
    line_l = line.lower()
    if any(s.lower() in line_l for s in ["ap", "loss", "epoch", "precision", "recall", "f1", "auc"]):
        clean = line.strip()
        if clean:
            metric_lines.append(clean)

if not metric_lines:
    sys.exit(0)

# Append to training_progress.md
notes_path = Path(__file__).parent.parent.parent / "training_progress.md"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

entry = f"\n## {timestamp}\n" + "\n".join(f"- {l}" for l in metric_lines[:30]) + "\n"

with open(notes_path, "a", encoding="utf-8") as f:
    f.write(entry)

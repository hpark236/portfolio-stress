"""Ask a model for this month's portfolio and save it as a dated snapshot.

    .venv/bin/python build/ai_portfolio.py claude

The prompt is fixed (build/prompts/ai_portfolio.md) and contains nothing about the stress tests,
so the model is not steered toward the scenarios it will be judged on. Each run is saved once and
never edited: data/holdings/<name>/<YYYY-MM-DD>.json. Performance is only counted from that date,
because asking a model to pick stocks "as of" a past date lets it use what it knows happened later.
"""

import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = (ROOT / "build" / "prompts" / "ai_portfolio.md").read_text()

RUNNERS = {
    # Claude Code CLI, run from an empty directory so no project files are in context.
    "claude": lambda p: json.loads(subprocess.run(["claude", "-p", p, "--output-format", "json"], capture_output=True, text=True,
                                                  check=True, cwd=tempfile.mkdtemp()).stdout),
}


def main(name):
    today = date.today().isoformat()
    out = ROOT / "data" / "holdings" / f"{name}-forward" / f"{today}.json"
    if out.exists():
        print("already have", out)
        return
    raw = RUNNERS[name](PROMPT.replace("{date}", today))
    text = raw.get("result", "")
    body = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    model = next(iter(raw.get("modelUsage", {}) or {}), name)
    snap = {
        "name": f"Claude forward portfolio ({model})",
        "source": f"Fixed monthly prompt run through the Claude Code CLI on {today}, model {model}. Tracked only from this date.",
        "model": model,
        "note": body.get("note", ""),
        "holdings": [{"ticker": h["ticker"].upper().replace(".", "-"), "weight": float(h["weight"]), "reason": h.get("reason", "")} for h in body["holdings"]],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=1))
    print("saved", out, [(h["ticker"], h["weight"]) for h in snap["holdings"]])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "claude")

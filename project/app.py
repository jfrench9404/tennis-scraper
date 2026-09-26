"""Browser UI for locally hosted or Streamlit-hosted authorized tennis analysis."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Tennis Vision", layout="wide")
st.title("Tennis Vision")
st.caption("Upload video you own or are authorized to analyze. Court mapping is automatic when confidence allows.")
upload = st.file_uploader("Match or practice clip", type=["mp4", "mov", "mkv"])
weights = st.text_input("Detector weights", "yolo11x.pt", help="Downloads a generic model on first use. Use a racket-aware custom model for contact events.")

if upload and st.button("Analyze video", type="primary"):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / upload.name
        source.write_bytes(upload.getbuffer())
        result_dir = root / "result"
        command = [sys.executable, "-m", "tennis_vision", "--input", str(source), "--output", str(result_dir), "--weights", weights]
        with st.spinner("Tracking players, ball, court, and event candidates…"):
            completed = subprocess.run(command, cwd=Path(__file__).parent, text=True, capture_output=True)
        if completed.returncode:
            st.error(completed.stderr or completed.stdout)
        else:
            summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
            st.json(summary)
            st.video((result_dir / "annotated.mp4").read_bytes())
            st.download_button("Download event log", (result_dir / "events.jsonl").read_bytes(), "events.jsonl", "application/x-ndjson")
            st.download_button("Download consolidated shots", (result_dir / "shots.json").read_bytes(), "shots.json", "application/json")

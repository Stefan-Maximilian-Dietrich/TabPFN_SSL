# tests/test_local_smoke_run.py
import os
import subprocess
import sys
from pathlib import Path


def test_local_smoke_run_creates_outputs(tmp_path):
    # Schreibe Ergebnisse in temporären Ordner, damit das Repo nicht vollgemüllt wird
    env = dict(os.environ)
    env["RESULTS_DIR"] = str(tmp_path / "results")

    cmd = [sys.executable, "scripts/smoke_local.py"]
    subprocess.run(cmd, check=True, env=env)

    # Jetzt prüfen wir grob: wurden CSVs erzeugt?
    # model_train.py schreibt CSVs unter results_dir/.../ID_<seed>.csv :contentReference[oaicite:11]{index=11}
    csv_files = list(Path(env["RESULTS_DIR"]).rglob("*.csv"))
    assert len(csv_files) >= 1, "Smoke run produced no CSV output files."
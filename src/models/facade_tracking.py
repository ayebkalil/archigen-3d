"""MLflow runs for the Pix2Pix vs Stable Diffusion hybrid facade comparison.

Logging never decides whether a generation succeeded: callers get the result first
and a separate tracking status (``logged`` / ``failed`` / ``disabled``) afterwards.
"""
from __future__ import annotations

import io
import json
import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.furniture.tracking import code_version, get_client, tracking_enabled

EXPERIMENT = "facade-engine-comparison"
LOGGER = logging.getLogger(__name__)


def _experiment_id(client):
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment:
        return experiment.experiment_id
    try:
        return client.create_experiment(EXPERIMENT)
    except Exception:
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment:
            return experiment.experiment_id
        raise


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "save"):
        value.save(path, format="PNG")
    elif isinstance(value, (bytes, bytearray)):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def log_run(name: str, params: Dict[str, Any], metrics: Dict[str, Any], artifacts: Dict[str, Any],
            tags: Optional[Dict[str, str]] = None, status: str = "FINISHED", force: bool = False,
            tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    """One complete run. ``artifacts`` maps relative paths to images, bytes, text or JSON."""
    if not (force or tracking_enabled()):
        return {"status": "disabled"}
    run_id = None
    try:
        from mlflow.entities import Metric, Param
        client = get_client(tracking_uri)
        run = client.create_run(_experiment_id(client), tags={"mlflow.runName": name, **(tags or {})})
        run_id = run.info.run_id
        merged = {**code_version(), **params}
        client.log_batch(run_id, params=[Param(k, str(v)[:6000]) for k, v in merged.items() if v is not None])
        now = int(time.time() * 1000)
        values = [Metric(k, float(v), now, 0) for k, v in metrics.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        if values:
            client.log_batch(run_id, metrics=values)
        with tempfile.TemporaryDirectory(prefix="facade-mlflow-") as directory:
            root = Path(directory)
            for relative, value in artifacts.items():
                if value is not None:
                    _write(root / relative, value)
            client.log_artifacts(run_id, str(root))
        client.set_terminated(run_id, status)
        return {"status": "logged", "run_id": run_id, "experiment": EXPERIMENT}
    except Exception as exc:
        LOGGER.warning("Facade MLflow tracking failed (run=%s): %s", run_id, exc)
        try:
            if run_id:
                client.set_terminated(run_id, "FAILED")
        except Exception:
            pass
        return {"status": "failed", "run_id": run_id, "error": f"{type(exc).__name__}: {exc}"}

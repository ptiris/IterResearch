from __future__ import annotations

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="IterResearch Python Sandbox", version="1.0.0")


class RunRequest(BaseModel):
    code: str = Field(..., description="Python source code to execute")
    language: str = Field(default="python", description="Execution language")
    run_timeout: int = Field(default=50, ge=1, le=600, description="Timeout in seconds")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run_code(request: RunRequest) -> dict:
    if request.language.lower() != "python":
        raise HTTPException(status_code=400, detail="Only python language is supported")

    code = request.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="Empty code")

    start_time = time.time()
    stdout = ""
    stderr = ""

    # Keep the runtime environment minimal. The container should provide the isolation boundary.
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
    }

    with tempfile.TemporaryDirectory(prefix="iterresearch-sandbox-") as tmpdir:
        tmp_path = Path(tmpdir)
        script_path = tmp_path / "main.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=request.run_timeout,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            if completed.returncode != 0 and not stderr:
                stderr = f"Process exited with code {completed.returncode}"
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if stderr:
                stderr += "\n"
            stderr += "[PythonInterpreter Error] TimeoutError: Execution timed out."
        except Exception as exc:
            stderr = f"[PythonInterpreter Error] {type(exc).__name__}: {exc}"

    execution_time = round(time.time() - start_time, 6)
    return {
        "stdout": stdout,
        "stderr": stderr,
        "execution_time": execution_time,
    }

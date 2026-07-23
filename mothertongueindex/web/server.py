"""
MotherTongueIndex web server.

Serves the static site under web/ and exposes a small JSON API that runs the
real tokenizers from the mti core engine.

  GET  /                serve the website
  GET  /api/models      list known models and tiers
  POST /api/analyze     {text, models[], show} -> per-model metrics (+ tokens)

Run:
  python web/server.py            # http://localhost:8000
  python web/server.py --port 9000

Intended for local or trusted deployment. It runs tokenizers on arbitrary input;
if you expose it publicly, add your own auth and rate limiting (see SECURITY.md).
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the sibling `mti` package importable when run as `python web/server.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mti.analyze import analyze          # noqa: E402
from mti.registry import list_models, GROUPS  # noqa: E402

WEB_DIR = os.path.dirname(os.path.abspath(__file__))

# Import FastAPI/pydantic at module scope so the request model is defined in
# module globals. Defining it inside build_app plus `from __future__ import
# annotations` makes FastAPI fail to resolve the body type (it treats the
# parameter as a query field). Guard the import so the CLI can print a friendly
# message when the web extra is not installed.
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from typing import Optional, List

    class AnalyzeReq(BaseModel):
        text: str
        models: Optional[List[str]] = None
        show: bool = False

    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False


def build_app():
    if not _FASTAPI_OK:
        raise SystemExit(
            "FastAPI is not installed. Install the web extra:\n"
            "  pip install 'mothertongueindex[web]'\n"
            "  or: pip install fastapi 'uvicorn[standard]'"
        )

    app = FastAPI(title="MotherTongueIndex", version="0.1.0")

    @app.get("/api/models")
    def models():
        return {
            "models": [
                {"id": m.id, "display": m.display, "tier": m.tier, "note": m.note}
                for m in list_models()
            ],
            "groups": GROUPS,
        }

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeReq):
        results = analyze(req.text, req.models, want_tokens=req.show)
        return JSONResponse({"results": [r.as_dict() for r in results]})

    # Static site at root (index.html served for "/").
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
    return app


app = None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MotherTongueIndex web server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn not installed. pip install 'uvicorn[standard]'")

    global app
    app = build_app()
    print(f"MotherTongueIndex serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

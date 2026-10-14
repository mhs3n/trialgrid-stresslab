"""Launch the local M7 evidence API and built React viewer."""

import argparse
import shutil
import subprocess
from pathlib import Path

from trialgrid.api.app import PROJECT_ROOT, create_app, create_demo_artifact
from trialgrid.results.artifacts import ArtifactStore


def _ensure_frontend() -> Path:
    frontend = PROJECT_ROOT / "frontend"
    distribution = frontend / "dist"
    if distribution.is_dir():
        return distribution
    npm = shutil.which("npm")
    if npm is None:
        raise ValueError("npm is required to build the React viewer")
    if not (frontend / "package-lock.json").is_file():
        raise ValueError("frontend/package-lock.json is required for a repeatable build")
    if not (frontend / "node_modules").is_dir():
        subprocess.run([npm, "ci"], cwd=frontend, check=True)
    subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
    if not distribution.is_dir():
        raise ValueError("frontend build did not produce frontend/dist")
    return distribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("demo_results"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("the unauthenticated demo binds only to localhost")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    store = ArtifactStore(args.artifact_dir)
    artifact = create_demo_artifact(store)
    print(f"Verified demo artifact: {artifact.artifact_id}")
    if args.generate_only:
        return
    try:
        import uvicorn
    except ModuleNotFoundError:
        parser.error("install the demo extra: python -m pip install -e '.[demo]'")
    try:
        frontend = _ensure_frontend()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(create_app(store.root, frontend_dist=frontend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

import uvicorn

from .config import DataConfig, TrainingConfig
from .pipeline import run_pipeline
from .serving import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart marketing ranking showcase.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run data generation, training and evaluation.")
    pipeline_parser.add_argument("--output-dir", default="artifacts")
    pipeline_parser.add_argument("--requests", type=int, default=1500)
    pipeline_parser.add_argument("--candidates-per-request", type=int, default=12)
    pipeline_parser.add_argument("--seed", type=int, default=42)
    pipeline_parser.add_argument("--epochs", type=int, default=8)
    pipeline_parser.add_argument("--batch-size", type=int, default=256)

    serve_parser = subparsers.add_parser("serve", help="Serve ranking model with FastAPI.")
    serve_parser.add_argument("--artifact-dir", default="artifacts")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pipeline":
        metrics = run_pipeline(
            output_dir=args.output_dir,
            data_config=DataConfig(
                requests=args.requests,
                candidates_per_request=args.candidates_per_request,
                seed=args.seed,
            ),
            training_config=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
            ),
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    if args.command == "serve":
        app = create_app(f"{args.artifact_dir}/deep_model.pt")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    raise ValueError(f"Unsupported command: {args.command}")

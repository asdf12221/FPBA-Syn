"""Command line interface for FPBA-Syn."""

from __future__ import annotations

import argparse
import sys

from .config import load_config, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpba-syn", description="Generate synthetic remote-sensing imagery from a COCO dataset.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("check", "Validate paths and configuration"), ("generate", "Run background generation and foreground redraw")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="YAML configuration file")
        command.add_argument("--skip-models", action="store_true", help="Only check dataset paths")
        if name == "generate":
            command.add_argument("--phase1-only", action="store_true", help="Stop after background generation")
            command.add_argument("--max-tasks", type=int, default=None, help="Limit source images for a smoke test")
    prepare = sub.add_parser("prepare", help="Create SAM conditioning data from COCO annotations")
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--image-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--sam-checkpoint", required=True)
    prepare.add_argument("--category-ids", required=True, help="Comma-separated COCO category ids")
    prepare.add_argument("--train-ratio", type=float, default=0.9)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--caption-template", default="Aerial remote sensing image with {objects}")
    annotate = sub.add_parser("annotate", help="Augment a COCO file with generated images")
    annotate.add_argument("--source", required=True)
    annotate.add_argument("--generated-dir", required=True)
    annotate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from .prepare import prepare_conditioning

        prepare_conditioning(
            args.annotations, args.image_root, args.output, args.sam_checkpoint,
            [int(value) for value in args.category_ids.split(",") if value.strip()],
            train_ratio=args.train_ratio, seed=args.seed, device=args.device,
            caption_template=args.caption_template,
        )
        return
    if args.command == "annotate":
        from .annotate import augment_coco

        augment_coco(args.source, args.generated_dir, args.output)
        return
    config = load_config(args.config)
    errors = validate_config(config, check_models=not args.skip_models)
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(2)
    if args.command == "check":
        print(f"OK: {args.config}")
        return
    from .pipeline import SyntheticGenerator

    result = SyntheticGenerator(config, phase1_only=args.phase1_only, max_tasks=args.max_tasks).run()
    print(result)

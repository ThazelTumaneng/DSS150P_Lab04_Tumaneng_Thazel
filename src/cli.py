# src/cli.py
import argparse
import sys
from src.ingest import ingest_data
from src.validate import validate_data
from src.transform import run_transformations
from src.train import train_stroke_model
from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Stroke Risk Prediction Reproducible Pipeline CLI"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the complete end-to-end pipeline.",
    )
    parser.add_argument(
        "--step",
        choices=["ingest", "validate", "transform", "train", "evaluate"],
        help="Run an individual pipeline step.",
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.all:
        print("▶️ [Step 1/5] Ingesting raw dataset...")
        ingest_data()

        print("\n▶️ [Step 2/5] Validating data contract...")
        if not validate_data():
            print("❌ Pipeline halted: Data contract failure.")
            sys.exit(1)

        print("\n▶️ [Step 3/5] Transforming (Staging & Curated layers)...")
        run_transformations()

        print("\n▶️ [Step 4/5] Training model...")
        train_stroke_model()

        print("\n▶️ [Step 5/5] Evaluating model & generating metrics...")
        evaluate_model()

        print("\n🎉 Pipeline executed successfully!")
        return

    if args.step == "ingest":
        ingest_data()
    elif args.step == "validate":
        validate_data()
    elif args.step == "transform":
        run_transformations()
    elif args.step == "train":
        train_stroke_model()
    elif args.step == "evaluate":
        evaluate_model()


if __name__ == "__main__":
    main()
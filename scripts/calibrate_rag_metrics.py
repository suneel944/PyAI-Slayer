#!/usr/bin/env python3
"""Script to calibrate RAG metrics using labeled evaluation set."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.ai.rag_calibration import RAGCalibrator
from core.ai.rag_eval_set import RAGEvalSet, create_example_eval_set


def main():
    """Run calibration and print report."""
    # Load or create eval set
    eval_set_path = Path("data/rag_eval_set.json")
    if eval_set_path.exists():
        print(f"Loading eval set from {eval_set_path}...")
        eval_set = RAGEvalSet.from_json(eval_set_path)
    else:
        print("Creating example eval set...")
        eval_set = create_example_eval_set()
        # Save it for future use
        eval_set_path.parent.mkdir(parents=True, exist_ok=True)
        eval_set.to_json(eval_set_path)
        print(f"Saved eval set to {eval_set_path}")

    # Run calibration
    print("\nRunning calibration...")
    calibrator = RAGCalibrator(eval_set)
    report = calibrator.generate_report()

    print("\n" + report)

    # Save recommendations
    recommendations = calibrator.calibrate_thresholds()
    import json

    recommendations_path = Path("data/rag_calibration_recommendations.json")
    recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(recommendations_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    print(f"\nRecommendations saved to {recommendations_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Print the commands to finish the multi-target pipeline.
This script does not execute any training or evaluation commands.
"""

def main() -> None:
    commands = [
        "python src/evaluate_model.py --target delayed",
        "python src/evaluate_model.py --target delayed_30",
        "python src/evaluate_model.py --target delayed_top25",
        "python src/target_analysis.py",
        "python src/compare_targets.py",
    ]

    print("Run these commands in order:")
    for command in commands:
        print(command)


if __name__ == "__main__":
    main()

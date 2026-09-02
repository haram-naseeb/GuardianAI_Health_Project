"""Command-line demo for the Location Agent.

Usage:
    python -m location_agent --latitude 31.5204 --longitude 74.3587
"""

from __future__ import annotations

import argparse
import json
import logging

from .agent import LocationAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GuardianAI Location Agent demo"
    )
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--accuracy-m", type=float, default=None)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    agent = LocationAgent()
    result = agent.find_best_facility(
        latitude=args.latitude,
        longitude=args.longitude,
        accuracy_m=args.accuracy_m,
        timestamp=args.timestamp,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

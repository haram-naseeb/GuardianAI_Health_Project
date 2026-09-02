"""Run a real Vision Agent analysis against Gemini from the command line."""

import sys

from app.agents.vision_agent import VisionAgent


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python run_vision_test.py <path_to_image>")
        raise SystemExit(1)

    agent = VisionAgent()
    result = agent.analyze_image(sys.argv[1])
    agent.display_result(result)


if __name__ == "__main__":
    main()

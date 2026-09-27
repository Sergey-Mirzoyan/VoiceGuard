from __future__ import annotations

import argparse
import sys

from voiceguard.dsp.inspect import inspect_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VoiceGuard DSP Tools")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for inspect
    subparsers.add_parser("inspect", help="Generate visual inspection plot for a clip")

    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "inspect":
        return inspect_main(argv[1:])

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return inspect_main(argv[1:])

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

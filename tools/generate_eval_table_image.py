#!/usr/bin/env python3
import sys

from generate_eval_table_images import main as generate_main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--recalls" not in argv:
        sys.argv = [sys.argv[0], *argv, "--recalls", "R40"]
    generate_main()

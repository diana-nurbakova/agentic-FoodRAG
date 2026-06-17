#!/usr/bin/env python
"""Thin wrapper for the 'embed' pipeline stage. Runnable from anywhere."""
import sys

from foodrag.notebook import setup
from foodrag.pipeline.embed import main

if __name__ == "__main__":
    setup()  # chdir to repo root + load .env
    raise SystemExit(main(sys.argv[1:]))

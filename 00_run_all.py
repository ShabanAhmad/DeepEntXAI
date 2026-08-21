#!/usr/bin/env python3
"""
DEEPENTXAI_FINAL — top-level launcher.

Thin wrapper so the whole pipeline can be started from the package root:

    ~/miniconda3/envs/ENT/bin/python 00_run_all.py [--list|--only 04 05|--from 03|...]

All the real orchestration logic lives in 01_Code/00_run_all.py; this just forwards
to it with the same arguments. See README.md for the stage table.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(HERE, "01_Code", "00_run_all.py")
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")

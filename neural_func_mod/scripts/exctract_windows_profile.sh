#!/bin/bash
PYTHONPATH=src python3 -m memory_profiler src/data_pipeline/create_training_data_reduced.py  parallel2025-06-20_00-01-10 0.1 6 3 2


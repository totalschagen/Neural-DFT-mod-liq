#!/bin/bash
PYTHONPATH=src python3 src/training/chunk_window_train.py prop_run red_dens_prov 10 1 1 0 512 0.002 1 8

PYTHONPATH=src python3 src/training/chunk_window_train.py prop_run red_dens_prov 10 1 1 0 512 0.002 0 12 


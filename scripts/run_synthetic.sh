#!/bin/bash

# hartmann
bash scripts/run_hartmann.sh
bash scripts/run_hartmann_baseline.sh

# levy
bash scripts/run_levy10.sh
bash scripts/run_levy10_baseline.sh
bash scripts/run_levy20.sh
bash scripts/run_levy20_baseline.sh

# ackley
bash scripts/run_ackley.sh
bash scripts/run_ackley_baseline.sh

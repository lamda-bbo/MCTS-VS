#!/bin/bash

python3 plot.py --root_dir=saved_logs/hartmann6_logs/ --output_name=results/hartmann6.pdf
python3 plot.py --root_dir=saved_logs/levy10_logs/ --output_name=results/levy10.pdf
python3 plot.py --root_dir=saved_logs/levy20_logs/ --output_name=results/levy20.pdf


# python3 plot.py --root_dir=logs/sota_logs/ --output_name=sota.pdf
# python3 plot.py --root_dir=old_logs/logs1228/real_logs/ --output_name=real.pdf
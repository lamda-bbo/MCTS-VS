#!/bin/bash

for ((i=42; i<=44; i++))
do 
    python3 dropout.py \
    --seed $i
done
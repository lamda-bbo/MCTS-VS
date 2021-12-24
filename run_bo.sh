#!/bin/bash

seed_start=2021
seed_end=2023

for ((seed=$seed_start; seed<=seed_end; seed++))
do
    {
    python3 lamcts
    }
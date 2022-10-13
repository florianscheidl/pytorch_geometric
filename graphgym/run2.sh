#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python graphgym/liwich_main.py --cfg graphgym/configs/hetero_mutag_heteroconv.yaml --repeat 3 # node classification
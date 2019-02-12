#!/bin/bash

python near_duplicate_checker.py --max_num_candidates 20 \
                                 --max_num_references 20 \
                                 --top_k 5 \
                                 --input_filename '../data/metadata/nearest_neighbor_results_test.json' \
                                 --output_filename '../data/metadata/nearest_neighbor_results_test.json' \


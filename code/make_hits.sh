#!/bin/bash
echo "Generating hits"
python mturk.py --args $1 generate_hits all_candidate_keys --out_file_name hits_to_submit_full.json --num_pos_control 2 --num_neg_control 2 --images_per_hit 25
less hits_to_submit_full.json | jq [.[0,1,18,19]]  > hits_to_submit.json
echo "Submitting hits"
python mturk.py --args $1 submit_hits --out_file_name hits_submitted_sandbox.json  --hit_data_file hits_to_submit.json

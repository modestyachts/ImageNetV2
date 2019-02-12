#!/bin/bash
DATE=`date +%Y-%m-%d-%H:%M:%S-%Z`
if test "$#" -lt 2; then
    echo "Illegal number of parameters: Usage: ARGS.JSON  USERNAME  CANDIDATE_KEYS"
    exit
fi
echo "Args: "
cat $1
echo "Generating hits"
python mturk.py --args $1 generate_hits $3 --out_file_name hits_to_submit_full_$DATE.json --num_pos_control 3 --num_neg_control 3 --images_per_hit 48
less hits_to_submit_full_$DATE.json  > hits_to_submit_$DATE.json
python mturk.py --args $1  generate_hit_htmls --out_file_name hit_$DATE.html  --hit_data_file_name hits_to_submit_$DATE.json
echo "Submitting hits"
python mturk.py --args $1 submit_hits --out_file_name $2_hits_submitted_$DATE.json  --hit_data_file hits_to_submit_$DATE.json --live
cp $2_hits_submitted_$DATE.json ../data/mturk/hit_data_live/
git add ../data/mturk/hit_data_live/
git commit -m "$2 added a hit"
git push origin master


#!/bin/bash
DATE=`date +%Y-%m-%d-%H:%M:%S-%Z`
if test "$#" -lt 3; then
    echo "Illegal number of parameters: Usage: ARGS.JSON  USERNAME  CANDIDATE_KEYS"
    exit
fi
echo "Args: "
cat $1
echo "Generating hits"
echo "Submitting hits"

python mturk.py --args $1 submit_hits --out_file_name $2_hits_submitted_$DATE.json  --hit_data_file $3 --live
cp $2_hits_submitted_$DATE.json ../data/mturk/hit_data_live/
git add ../data/mturk/hit_data_live/
git commit -m "$2 added a hit"
git push origin master


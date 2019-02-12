python3 sample_dataset.py above_threshold --dataset_size 10000 --selection_frequency_threshold 0.7 --min_num_annotations 10 --wnid_thresholds_filename ../data/dataset_metadata/sampling_thresholds_a.json  --seed 368567421 --output_filename imagenetv2-a-16.json --starting_from imagenetv2-a-15.json

python3 initialize_dataset_review.py carry_over_reviews --dataset_filename imagenetv2-a-16.json --starting_from imagenetv2-a-15.json

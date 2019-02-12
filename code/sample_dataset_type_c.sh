python3 sample_dataset.py best --dataset_size 10000 --min_num_annotations 10 --seed 435927806 --output_filename imagenetv2-c-1.json --starting_from imagenetv2-c-0.json

python3 initialize_dataset_review.py carry-over-reviews --dataset_filename imagenetv2-c-1.json --starting_from imagenetv2-c-0.json

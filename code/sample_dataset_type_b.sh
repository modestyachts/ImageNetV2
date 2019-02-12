python3 sample_dataset.py wnid_histogram --dataset_size 10000 --min_num_annotations_candidates 10 --min_num_annotations_val 10 --min_num_val_images_per_wnid 20 --seed 453315496 --output_filename imagenetv2-b-0.json

python3 initialize_dataset_review.py carry_over_reviews --dataset_filename imagenetv2-b-1.json --starting_from imagenetv2-b-0.json

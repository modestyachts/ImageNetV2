# ImageNetV2
The ImageNetV2 dataset contains new test data for the [ImageNet](www.image-net.org) benchmark.
This repository provides associated code for assembling and working with ImageNetV2.
The actual test sets are stored [in a separate location](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main).

ImageNetV2 contains three test sets with 10,000 new images each.
Importantly, these test sets were sampled *after* a decade of progress on the original ImageNet dataset.
This makes the new test data independent of existing models and guarantees that the accuracy scores are not affected by adaptive overfitting.
We designed the data collection process for ImageNetV2 so that the resulting distribution is as similar as possible to the original ImageNet dataset.
Our paper ["Do ImageNet Classifiers Generalize to ImageNet?"](http://people.csail.mit.edu/ludwigs/papers/imagenet.pdf) describes ImageNetV2 and associated experiments in detail.

In addition to the three test sets, we also release our pool of candidate images from which the test sets were assembled.
Each image comes with rich metadata such as the corresponding Flickr search queries or the annotations from MTurk workers.

The aforementioned paper also describes CIFAR-10.1, a new test set for CIFAR-10.
It can be found in the following repository: https://github.com/modestyachts/CIFAR-10.1


# Using the Dataset
Before explaining how the code in this repository was used to assemble ImageNetV2, we first describe how to load our new test sets.

## Test Set Versions

There are currently three test sets in ImageNetV2:

- `Threshold0.7` was built by sampling ten images for each class among the candidates with selection frequency at least 0.7. 

- `MatchedFrequency` was sampled to match the MTurk selection frequency distribution of the original ImageNet validation set for each class. 

- `TopImages` contains the ten images with highest selection frequency in our candidate pool for each class.

In our code, we adopt the following naming convention:
Each test set is identified with a string of the form

`imagenetv2-<test-set-letter>-<revision-number>`

for instance, `imagenetv2-b-31`. The `Threshold0.7`, `MatchedFrequency`, and `TopImages` have test set letters `a`, `b`, and `c`, respectively.
The current revision numbers for the test sets are `imagenetv2-a-44`, `imagenetv2-b-33`, `imagenetv2-c-12`.
We refer to our paper for a detailed description of these test sets and the review process underlying the different test set revisions.


## Loading a Test Set

You can download the test sets from the following url: https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main. There is a link for each individual dataset and the ImageNet datasets must be decompressed before use. 

To load the dataset, you can use the [`ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) class in [PyTorch](https://pytorch.org/) on the extracted folder. 


For instance, the following code loads the `MatchedFrequency` dataset:

```python
from torchvision import datasets
datasets.ImageFolder(root='imagenetv2-matched-frequency')
```


# Dataset Creation Pipeline

The dataset creation process has several stages outlined below.
We describe the process here at a high level.
If you have questions about any individual steps, please contact Rebecca Roelofs (roelofs@cs.berkeley.edu) and Ludwig Schmidt (ludwig@berkeley.edu).


## 1. Downloading images from Flickr

In the first stage, we collected candidate images from the Flickr image hosting service.
This requires a [Flickr API key](https://www.flickr.com/services/api/misc.api_keys.html).

We ran the following command to search Flickr for images for a fixed list of wnids:

```
python flickr_search.py "../data/flickr_api_keys.json" \
                        --wnids "{wnid_list.json}" \
                        --max_images 200 \
                        --max_date_taken "2013-07-11"\
                        --max_date_uploaded "2013-07-11"\
                        --min_date_taken "2012-07-11"\
                        --min_date_uploaded "2012-07-11" 
```
We refer to the paper for more details on which Flickr search parameters we used to complete our candidate pool.

The script outputs search result metadata, including the Flickr URLs returned for each query. 
This search result metadata is written to `/data/search_results/`. 

We then stored the images to an Amazon S3 bucket using 
```
python download_images_from_flickr.py ../data/search_results/{search_result.json} --batch --parallel
```

## 2. Create HITs
Similar to the original ImageNet dataset, we used Amazon Mechanical Turk (MTurk) to filter our pool of candidates.
The main unit of work on MTurk is a HIT (Human Intelligence Tasks), which in our case consists of 48 images with a target class.
The format of our HITs was derived from the original ImageNet HITs.

To submit a HIT, we performed the following steps.
They require a configured [MTurk account](https://www.mturk.com/).
  1. Encrypt all image URLs.  This is necessary so that MTurk workers cannot identify whether an image is from the original validation set or our candidate pool by the source URL. 
    `python encrypt_copy_objects.py imagenet2candidates_mturk --strip_string ".jpg" --pywren`
  2. Run the image consistency check.  This checks that all of the new candidate images have been stored to S3 and have encrypted URLs. 
    `python image_consistency_check.py`
  3. Generate hit candidates. This outputs a list of candidates to `data/hit_candidates`
    `python generate_hit_candidates.py  --num_wnids 1000`
  4. Submit live HITs to MTurk. 
    `bash make_hits_live.sh sample_args_10.json <username> <latest_hit_candidate_file>`
  5. Wait for prompt, and check if HTML file in the code/ directory looks correct.
  6. Type in the word LIVE to confirm submitting the HITs to MTurk (this costs money).

The HIT metadata created by `make_hits_live.sh` is stored in `data/mturk/hit_data_live/`.

After a set of HITs was submitted, you can check their progress using
`python3 mturk.py show_hit_progress --live --hit_file ../data/mturk/hit_data_live/{hit.json}`

Additionally, we occasionally used the Jupyter notebook `inspect_hit.ipynb` to visually examine the HITs we created.
The code for this notebook is stored in `inspect_hit_notebook_code.py`.



## 3. Remove near duplicates
Next, we removed near-duplicates from our candidate pool.
We checked for near-duplicates both within our new test set and between our new test set and the original ImageNet dataset.

To find near-duplicates, we computed the 30 nearest neighbors for each candidate image in three different metrics: l2 distance on raw pixels, l2 distance on features extracted from a pre-trained [VGG](https://arxiv.org/abs/1409.1556) model (fc7), and [SSIM](http://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf) (structural similarity).

The fc7 metric requires that each image is featurized using the same pre-trained VGG model. The scripts `featurize.py`, `feaurize_test.py` and `featurize_candidates.py` were used to perform the fc7 featurization. 

Next, we computed the nearest neighbors for each image.
Each metric has a different starting script:
* `run_near_duplicate_checker_dssim.py`
* `run_near_duplicate_checker_l2.py`
* `run_near_duplicate_checker_fc7.py`

All three scripts use `near_duplicate_checker.py` for the underlying computation.

The script `test_near_duplicate_checker.sh` was used to run the unit tests for the near duplicate checker contained in `test_near_duplicate_checker.py`. 

Finally, we manually reviewed the nearest neighbor pairs using the notebook `review_near_duplicates.ipynb`. The file `review_near_duplicates_notebook_code.py` contains the code for this notebook. The review output is saved in `data/metadata/nearest_neighbor_reviews_v2.json`.
All near duplicates that we found are saved in `data/metadata/near_duplicates.json`.


## 4. Sample Dataset
After we created a labeled candidate pool, we sampled the new test sets.

We use a separate bash script to sample each version of the dataset, i.e `sample_dataset_type_{a}.sh`.
Each script calls `sample_dataset.py` and `initialize_dataset_review.py` with the correct arguments.
The file `dataset_sampling.py` contains helper functions for the sampling procedure. 

## 5. Review Final Dataset
For quality control, we added a final reviewing step to our dataset creation pipeline.

* `initialize_dataset_review.py` initializes the metadata needed for each dataset review round.
* `final_dataset_inspection.ipynb` is used to manually review dataset versions. 
* `final_dataset_inspection_notebook_code.py` contains the code needed for the `final_dataset_inspection.ipynb` notebook.

* `review_server.py` is the review server used for additional cleaning of the candidate pool.  The review server starts a web UI that allows one to browse all candidate images for a particular class.  In addition, a user can easily flag images that are problematic or near duplicates.

The review server can use local, downloaded images if started with the flag
`python3 review_server.py --use_local_images`.
In addition, you also need to launch a separate static file server for serving the images.
There is a script in `data` for starting the static file server `./start_file_server.sh`.

The local images can be downloaded using 
* `download_all_candidate_images_to_cache.py`
* `download_dataset_images.py`


# Data classes

Our code base contains a set of data classes for working with various aspects of ImageNetV2.

* `imagenet.py`: This file contains the `ImageNetData` class that provides metadata about ImageNet (a list of classes, etc.) and functionality for loading images in the original ImageNet dataset. The scripts `generate_imagenet_metadata_pickle.py` are used to assemble `generate_class_info_file.py` some of the metadata in the `ImageNetData` class.

* `candidate_data.py` contains the `CandidateData` class that provides easy access to all candidate images in ImageNetV2 (both image data and metadata). The metadata file used in this class comes from `generate_candidate_metadata_pickle.py`.

* `image_loader.py` provides a unified interface to loading image data from either ImageNet or ImageNetV2.

* `mturk_data.py` provides the `MTurkData` class for accessing the results from our MTurk HITs. The data used by this class is assembled via `generate_mturk_data_pickle`.

* `near_duplicate_data.py` loads and processes the information about near-duplicates in ImageNetV2. Some of the metadata is prepared with `generate_review_thresholds_pickle.py`.

* `dataset_cache.py` allows easy loading of our various test set revisions.

* `prediction_data.py` provides functionality for loading the predictions of various classification models on our three test sets.

The functionality provided by each data class is documented via examples in the `notebooks` folder of this repository.

# Evaluation Pipeline
Finally, we describe our evaluation pipeline for the PyTorch models.
The main file is `eval.py`, which can be invoked as follows:

`python eval.py --dataset $DATASET --models $MODELS`

where $DATASET is one of
- `imagenet-validation-original` (the original validation set)
- `imagenetv2-b-33` (our new `MatchedFrequency` test set)
- `imagenetv2-a-44` (our new `Threshold.7` test set)
- `imagenetv2-c-12` (our new `TopImages` test set).

The $MODELS parameter is a comma-separated list of model names in the [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) or [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) repositories.
Alternatively, $MODELS can also be `all`, in which case all models are evaluated.


# License

Unless noted otherwise in individual files, the code in this repository is released under the MIT license (see the `LICENSE` file).
The `LICENSE` file does *not* apply to the actual image data.
The images come from Flickr which provides corresponding license information.
They can be used the same way as the original ImageNet dataset.

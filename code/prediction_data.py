from collections import Counter
import json
import pathlib
from timeit import default_timer as timer

import numpy as np

import dataset_cache


default_datasets_to_load = ['imagenetv2-a-44', 'imagenetv2-b-33', 'imagenetv2-c-12', 'imagenet-annotatedval20k-0']


def load_predictions_for_dataset(dataset_name, imgnet):
    dataset_filepath = (pathlib.Path(__file__).parent /  f'../data/datasets/{dataset_name}.json').resolve()
    with open(dataset_filepath, 'r') as f:
        cur_data = json.load(f)
    cur_imgs = [x[0] for x in cur_data['image_filenames']]
    result = {x : {} for x in cur_imgs}
    predictions_directory = (pathlib.Path(__file__).parent /  f'../data/predictions/{dataset_name}/').resolve()
    files_loaded = []
    for p in predictions_directory.glob('*.npy'):
        files_loaded.append(p)
        cur_model = p.stem
        cur_pred = np.load(p)
        assert len(cur_pred) == len(cur_imgs)
        for ii, img in enumerate(cur_imgs):
            for tmp in cur_pred[ii, :]:
                if tmp < 0 or tmp > 999:
                    print(f'Invalid prediction {tmp} for image {img} and model {cur_model} in dataset {dataset_name}')
                    assert tmp >= 0 and tmp <= 999
            result[img][cur_model] = tuple([imgnet.class_info_by_cid[x].wnid for x in cur_pred[ii, :]])
    return result, files_loaded


class PredictionData:
    def __init__(self, imgnet, dataset_cache, datasets_to_load=None, verbose=True, load_label_annotations=True):
        if datasets_to_load is None:
            self.datasets_to_load = default_datasets_to_load
        else:
            self.datasets_to_load = datasets_to_load
        if verbose:
            print(f'Loading predictions for datasets {self.datasets_to_load} ...')
        start_time = timer()
        self.imgnet = imgnet
        self.dataset_cache = dataset_cache
        self.predictions_by_image = {}
        num_files_loaded = 0
        for dataset in self.datasets_to_load:
            dataset_predictions, files_loaded = load_predictions_for_dataset(dataset, self.imgnet)
            num_files_loaded += len(files_loaded)
            for img, preds in dataset_predictions.items():
                if img not in self.predictions_by_image:
                    self.predictions_by_image[img] = preds
                else:
                    for model, model_preds in preds.items():
                        if model in self.predictions_by_image[img]:
                            if model_preds != self.predictions_by_image[img][model]:
                                dataset_index = self.datasets_to_load.index(dataset)
                                print(f'WARNING: Inconsistency while loading prediction data for\n  Model {model}\n  Image {img}')
                                print(f'Previously loaded predictions ({self.datasets_to_load[:dataset_index]}):')
                                print('  ', end='')
                                print(self.predictions_by_image[img][model])
                                print(f'But now loaded for dataset {dataset}:\n  ', end='')
                                print(model_preds)
                            #assert model_preds == self.predictions_by_image[img][model]
                        else:
                            self.predictions_by_image[img][model] = model_preds
        self.top1_counters_by_image = {}
        for img, preds in self.predictions_by_image.items():
            self.top1_counters_by_image[img] = Counter([x[0] for x in preds.values()])
        end_time = timer()
        if verbose:
            print(f'    done, took {end_time - start_time:.2f} seconds (loaded from {num_files_loaded} prediction files)')
        if load_label_annotations:
            self.reload_label_annotations(verbose=verbose)
        
    def reload_label_annotations(self, verbose=True):
        annotations_filepath = (pathlib.Path(__file__).parent /  f'../data/metadata/label_annotations.json').resolve()
        with open(annotations_filepath, 'r') as f:
            self.label_annotations = json.load(f)
        if verbose:
            print(f'Loaded label annotations from {annotations_filepath}')

    def get_model_list(self, imgs):
        model_union = set()
        model_intersection = (self.predictions_by_image[imgs[0][0]].keys())
        for img, _ in imgs:
            model_union |= set(self.predictions_by_image[img].keys())
            model_intersection &= set(self.predictions_by_image[img].keys())
        if len(model_union) != len(model_intersection):
            print(f'WARNING: some images in the dataset have predictions for different model sets.')
            print(f'Found {len(model_union)} different models in total, but only {len(model_intersection)} models have predictions for all images in the dataset.')
            print(f'The models present in only some images are: {sorted(list(model_union - model_intersection))}')
        cur_models = list(model_intersection)
        return cur_models
    
    def compute_dataset_accuracies(self, dataset_name, verbose=True):
        imgs = self.dataset_cache.get_dataset_images(dataset_name)
        assert len(imgs) > 0
        for img, _ in imgs:
            assert img in self.predictions_by_image
        cur_models = self.get_model_list(imgs)
        top1_accuracy = {x: 0 for x in cur_models}
        top5_accuracy = {x: 0 for x in cur_models}
        for img, wnid in imgs:
            for model in cur_models:
                if wnid == self.predictions_by_image[img][model][0]:
                    top1_accuracy[model] += 1
                if wnid in self.predictions_by_image[img][model]:
                    top5_accuracy[model] += 1
        for model in cur_models:
            top1_accuracy[model] /= len(imgs)
            top5_accuracy[model] /= len(imgs)
        return top1_accuracy, top5_accuracy
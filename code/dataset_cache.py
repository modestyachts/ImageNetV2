import json
import pathlib


def is_valid_dataset_name(name):
    dataset_filepath = (pathlib.Path(__file__).parent /  f'../data/datasets/{name}.json').resolve()
    return dataset_filepath.is_file()


class DatasetCache:
    def __init__(self, imgnet):
        self.imgnet = imgnet
        self.loaded_datasets = {}
        self.loaded_datasets_by_wnid = {}

    def load_dataset(self, dataset_name):
        assert dataset_name not in self.loaded_datasets
        assert dataset_name not in self.loaded_datasets_by_wnid
        assert is_valid_dataset_name(dataset_name)
        dataset_filepath = (pathlib.Path(__file__).parent /  f'../data/datasets/{dataset_name}.json').resolve()
        print(f'Loading dataset {dataset_name} from {dataset_filepath}')
        with open(dataset_filepath, 'r') as f:
            self.loaded_datasets[dataset_name] = json.load(f)
        self.loaded_datasets_by_wnid[dataset_name] = {x: set([]) for x in self.imgnet.class_info_by_wnid.keys()}
        for img, img_wnid in self.loaded_datasets[dataset_name]['image_filenames']:
            self.loaded_datasets_by_wnid[dataset_name][img_wnid].add(img)
    
    def get_dataset_by_wnid(self, dataset_name):
        if dataset_name not in self.loaded_datasets_by_wnid:
            self.load_dataset(dataset_name)
        return self.loaded_datasets_by_wnid[dataset_name]
    
    def get_dataset_images(self, dataset_name):
        if dataset_name not in self.loaded_datasets_by_wnid:
            self.load_dataset(dataset_name)
        return self.loaded_datasets[dataset_name]['image_filenames']
    
    def get_dataset_size(self, dataset_name):
        if dataset_name not in self.loaded_datasets_by_wnid:
            self.load_dataset(dataset_name)
        return len(self.loaded_datasets[dataset_name]['image_filenames'])
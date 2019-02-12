import json
import pathlib

import click
import tqdm

import candidate_data
import image_loader
import imagenet



@click.command()
@click.argument('datasets', type=str)
@click.option('--include_val', type=bool, is_flag=True)
def download_images(datasets, include_val):
    imgnet = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(exclude_blacklisted_candidates=False)
    loader = image_loader.ImageLoader(imgnet, cds)
    
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    assert len(all_wnids) == 1000

    for dataset in datasets.split(','):
        print(f'Downloading images for dataset {dataset} ...')
    
        dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / (dataset + '.json')
        dataset_filepath = dataset_filepath.resolve()
        assert dataset_filepath.is_file()
        with open(dataset_filepath, 'r') as f:
            data = json.load(f)
        
        dataset_by_wnid = {x: [] for x in all_wnids}
        for img, wnid in data['image_filenames']:
            dataset_by_wnid[wnid].append(img)
        for cur_wnid in tqdm.tqdm(all_wnids):
            images_to_download = dataset_by_wnid[cur_wnid]
            #if include_val:
            #    images_to_download.extend(imgnet.val_imgs_by_wnid[cur_wnid])
            loader.load_image_bytes_batch(images_to_download, size='scaled_500', verbose=False)

    if include_val:
        print('Downloading all validation images ...')
        for cur_wnid in tqdm.tqdm(all_wnids):
            images_to_download = imgnet.val_imgs_by_wnid[cur_wnid]
            loader.load_image_bytes_batch(images_to_download, size='scaled_500', verbose=False)


if __name__ == "__main__":
    download_images()
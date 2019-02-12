import json
import pathlib

import click
import tqdm

import candidate_data
import image_loader
import imagenet

imgnet = imagenet.ImageNetData()
cds = candidate_data.CandidateData(exclude_blacklisted_candidates=False)
loader = image_loader.ImageLoader(imgnet, cds)
    
all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
assert len(all_wnids) == 1000

print('Downloading all candidate images ...')
for cur_wnid in tqdm.tqdm(all_wnids):
    images_to_download = cds.candidates_by_wnid[cur_wnid]
    images_to_download = [x['id_ours'] for x in images_to_download]
    loader.load_image_bytes_batch(images_to_download, size='scaled_500', verbose=False)


if __name__ == "__main__":
    download_images()
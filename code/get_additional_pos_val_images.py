import argparse
import json
import random
import sys

import candidate_data
import imagenet
import mturk_data
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--num_new_images', type=int, default=None)
parser.add_argument('--treshold', type=int, default=None)
args = parser.parse_args()
treshold = args.treshold
num_new_images = args.num_new_images
assert treshold is not None
assert num_new_images is not None

rng = random.Random(674517721)
hit_size = 48

imgnet = imagenet.ImageNetData()
cds = candidate_data.CandidateData(load_metadata_from_s3=False, exclude_blacklisted_candidates=True)
mturk = mturk_data.MTurkData(live=True,
                             load_assignments=True,
                             source_filenames_to_ignore=mturk_data.main_collection_filenames_to_ignore)
with open("../data/metadata/wnid_to_most_similar_wnids.json") as f:
    wnid_to_most_similar_wnids = json.load(f)

pos_val_images_by_wnid = {}
for wnid in imgnet.class_info_by_wnid.keys():
    pos_val_images_by_wnid[wnid] = set()

for hit in mturk.hits.values():
    cur_wnid = hit['wnid']
    for pvi in hit['images_pos_control']:
        pos_val_images_by_wnid[cur_wnid].add(pvi)
        assert mturk.image_num_assignments[pvi][cur_wnid] >= 10

for wnid, pos_val_images in pos_val_images_by_wnid.items():
    for img in pos_val_images:
        assert imgnet.wnid_by_val_filename[img] == wnid
        assert img in imgnet.val_imgs_by_wnid[wnid]

wnids_to_augment = [x[0] for x in pos_val_images_by_wnid.items() if len(x[1]) < treshold]
print('{} wnids currently have less than {} pos val images'.format(len(wnids_to_augment), treshold))

result = {}
for wnid in wnids_to_augment:
    result[wnid] = {}
    cur_num = len(pos_val_images_by_wnid[wnid])
    remaining_images = []
    for img in imgnet.val_imgs_by_wnid[wnid]:
        appears_as_pos_val = False
        if img in mturk.hits_of_image:
            for cur_hit in mturk.hits_of_image[img]:
                if cur_hit['wnid'] == wnid:
                    assert img in cur_hit['images_pos_control']
                    appears_as_pos_val = True
                else:
                    assert img in cur_hit['images_neg_control']
        if not appears_as_pos_val:
            remaining_images.append(img)
    assert len(remaining_images) + len(pos_val_images_by_wnid[wnid]) == 50
    assert len(set(remaining_images) & pos_val_images_by_wnid[wnid]) == 0
    if len(remaining_images) < num_new_images:
        print('ERROR: wnid {} has only {} images remaining'.format(wnid, len(remaining_images)))
        assert False
    result[wnid]['pos_control'] = rng.sample(remaining_images, num_new_images)
    num_candidates = (hit_size - num_new_images) // 2
    if len(cds.candidates_by_wnid[wnid]) < num_candidates:
        print('Warning: wnid {} has less than {} candidates ({}), so cannot do 1/3 1/3 1/3 for this wnid'.format(
                wnid, num_candidates, len(cds.candidates_by_wnid[wnid])))
        num_candidates = len(cds.candidates_by_wnid[wnid])
    result[wnid]['candidates'] = [x['id_ours'] for x in rng.sample(cds.candidates_by_wnid[wnid], num_candidates)]
    num_neg_val = hit_size - num_new_images - num_candidates
    neg_wnid = wnid_to_most_similar_wnids[wnid][1]
    result[wnid]['neg_control'] = rng.sample(imgnet.val_imgs_by_wnid[neg_wnid], num_neg_val)

for wnid, imgs in result.items():
    assert len(imgs['pos_control']) == num_new_images
    assert len(imgs['pos_control'] + imgs['neg_control'] + imgs['candidates']) == hit_size
    for img in imgs['pos_control']:
        assert imgnet.wnid_by_val_filename[img] == wnid
    neg_wnid = wnid_to_most_similar_wnids[wnid][1]
    for img in imgs['neg_control']:
        assert imgnet.wnid_by_val_filename[img] == neg_wnid
    for img in imgs['candidates']:
        assert cds.all_candidates[img]['wnid'] == wnid

filename = 'additional_pos_val_images.json'
with open(filename, 'w') as f:
    json.dump(result, f, indent=2)
print('Wrote new pos val images for {} wnids to {}'.format(len(result), filename))
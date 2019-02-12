import datetime
import getpass
import json
import os
import pathlib
import sys
from timeit import default_timer as timer
import traceback

import click
import responder

import candidate_data
import dataset_cache
import imagenet
import mturk_data
import near_duplicate_data
import prediction_data
import problematic_images
import urllib
import utils

api = responder.API(static_dir='review_ui/static', templates_dir='review_ui/templates')

imgnet = imagenet.ImageNetData()
cds = candidate_data.CandidateData(exclude_blacklisted_candidates=False)
mturk = mturk_data.MTurkData(live=True,
                             load_assignments=True,
                             source_filenames_to_ignore=mturk_data.main_collection_filenames_to_ignore)
ndc = near_duplicate_data.NearDuplicateData(imgnet=imgnet,
                                            candidates=cds,
                                            mturk_data=mturk,
                                            load_review_thresholds=False,
                                            verbose=True)
dataset_cache = dataset_cache.DatasetCache(imgnet)
preds = prediction_data.PredictionData(imgnet=imgnet,
                                       dataset_cache=dataset_cache,
                                       verbose=True,
                                       datasets_to_load=prediction_data.default_datasets_to_load)
problematic = problematic_images.ProblematicImages()

valid_filenames = set(imgnet.get_all_image_names()) | set(cds.all_candidates.keys())


def is_valid_dataset_name(name):
    dataset_filepath = (pathlib.Path(__file__).parent /  f'../data/datasets/{name}.json').resolve()
    return dataset_filepath.is_file()


use_local_image_files = False

def get_image_url(img):
    if use_local_image_files:
        if img in cds.all_candidates:
            #file_path = f'static/imagenet2candidates_mturk/{img}.jpg'
            file_path = f'http://127.0.0.1:5043/cache/imagenet2candidates_mturk/{img}.jpg'
        else:
            #file_path = f'static/imagenet_validation_flat/{img}'
            file_path = f'http://127.0.0.1:5043/cache/imagenet_validation_flat/{img}'
        return file_path
    else:
        encrypted_filename = utils.encrypt_string_with_magic(img)
        return 'https://s3-us-west-2.amazonaws.com/imagenet2datav2/encrypted/' + urllib.parse.quote(encrypted_filename) + '.jpg'


def get_val_data(wnid):
    val_data = {}
    for x in imgnet.val_imgs_by_wnid[wnid]:
        if x in mturk.image_num_assignments and wnid in mturk.image_num_assignments[x]:
            num_assignments = mturk.image_num_assignments[x][wnid] 
            sel_frequency = mturk.image_fraction_selected[x][wnid]
        else:
            num_assignments = 0
            sel_frequency = 0.0
        val_data[x] = {'url': get_image_url(x),
                       'is_problematic': x in problematic.problematic_images,
                       'num_assignments': num_assignments,
                       'selection_frequency': sel_frequency}
    return val_data


@api.route('/wnid_browser')
def wnid_browser(req, resp):
    resp.content = api.template('wnid_browser.html')


@api.route('/predictions_browser')
def predictions_browser(req, resp):
    resp.content = api.template('predictions_browser.html')


def is_string_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def parse_predictions_request_parameters(req):
    assert 'datasets' in req.params
    datasets = req.params['datasets'].split(',')
    for d in datasets:
        assert is_valid_dataset_name(d)
        if d not in preds.datasets_to_load:
            print(f'ERROR: have not loaded dataset {d}')
        assert d in preds.datasets_to_load
    assert 'starting_wnid' in req.params
    starting_wnid = req.params['starting_wnid']
    assert starting_wnid in imgnet.class_info_by_wnid.keys()
    assert 'num_wnids' in req.params
    num_wnids = req.params['num_wnids']
    assert is_string_int(num_wnids)
    num_wnids = int(num_wnids)
    assert num_wnids >= 0
    if 'include_reviewed_wnids' in req.params:
        include_reviewed_wnids = req.params['include_reviewed_wnids'].lower()
        assert include_reviewed_wnids in ['true', 'false']
        if include_reviewed_wnids == 'true':
            include_reviewed_wnids = True
        else:
            include_reviewed_wnids = False
    else:
        include_reviewed_wnids = True
    return datasets, starting_wnid, num_wnids, include_reviewed_wnids


# Parameters: datasets, starting_wnid, num_wnids, include_reviewed_wnids
@api.route('/get_predictions')
def get_predictions(req, resp):
    try:
        datasets, starting_wnid, num_wnids, include_reviewed_wnids = parse_predictions_request_parameters(req)
        
        all_wnids = sorted(imgnet.class_info_by_wnid.keys())
        cur_index = all_wnids.index(starting_wnid)
        num_wnids_added = 0
        num_fully_annotated_wnids = 0

        response = {'prediction_data': {}}
        while cur_index < len(imgnet.class_info_by_wnid.keys()) and num_wnids_added < num_wnids:
            cur_wnid = all_wnids[cur_index]
            cur_imgs = set()
            for d in datasets:
                cur_imgs |= dataset_cache.get_dataset_by_wnid(d)[cur_wnid]
            if not include_reviewed_wnids:
                all_annotated = True
                for img in cur_imgs:
                    cur_preds = preds.top1_counters_by_image[img]
                    if len(cur_preds) > 0:
                        if img not in preds.label_annotations:
                            all_annotated = False
                            break
                        for pred in cur_preds.keys():
                            if pred not in preds.label_annotations[img] or preds.label_annotations[img][pred] == 'unreviewed':
                                all_annotated = False
                                break
            if not include_reviewed_wnids and all_annotated:
                num_fully_annotated_wnids += 1
                cur_index += 1
                continue
            print('num_fully_annotated_wnids', num_fully_annotated_wnids)
            wnid_data = {}
            for img in cur_imgs:
                wnid_data[img] = {'predictions': {},
                                  'url': get_image_url(img)}
                cur_preds = preds.top1_counters_by_image[img]
                for pred, freq in cur_preds.items():
                    if img not in preds.label_annotations:
                        state = 'unreviewed'
                    else:
                        if pred in preds.label_annotations[img]:
                            state = preds.label_annotations[img][pred]
                        else:
                            state = 'unreviewed'
                    wnid_data[img]['predictions'][pred] = {'num_models': freq, 'state': state}
            response['prediction_data'][cur_wnid] = wnid_data
            num_wnids_added += 1
            cur_index += 1
        
        response['num_fully_annotated_wnids_skipped'] = num_fully_annotated_wnids
        resp.media = response
    except:
        traceback.print_exc()
        raise


@api.route('/update_predictions')
async def update_predictions(req, resp):
    data = await req.media()
    annotations_filepath = (pathlib.Path(__file__).parent /  f'../data/metadata/label_annotations.json').resolve()
    with open(annotations_filepath, 'r') as f:
        cur_annotations = json.load(f)

    allowed_annotations = ['correct', 'wrong', 'unclear', 'dontknow', 'unreviewed']

    num_annotations_updated = 0
    num_images_updated = 0
    for img, annotations in data.items():
        assert img in valid_filenames
        for wnid, anno in annotations.items():
            assert wnid in imgnet.class_info_by_wnid
            assert anno in allowed_annotations
        if img not in cur_annotations:
            cur_annotations[img] = annotations
            num_annotations_updated += len(annotations)
            num_images_updated += 1
        else:
            updated_image = False
            for wnid, state in annotations.items():
                if wnid not in cur_annotations[img] or cur_annotations[img][wnid] != state:
                    cur_annotations[img][wnid] = state
                    updated_image = True
                    num_annotations_updated += 1
            if updated_image:
                num_images_updated += 1
    with open(annotations_filepath, 'w') as f:
        json.dump(cur_annotations, f, indent=2, sort_keys=True)
    print(f'Updated {num_annotations_updated} annotations total across {num_images_updated} images', end='')
    start = timer()
    if num_annotations_updated > 0:
        preds.reload_label_annotations(verbose=False)
    end = timer()
    print(f' (took {end - start:.2f} seconds to reload the annotation data)')
    
    resp.media = {'success': True,
                  'num_images_updated': num_images_updated,
                  'num_annotations_updated': num_annotations_updated}


@api.route('/mark_images_as_problematic')
async def mark_images_as_problematic(req, resp):
    try:
        data = await req.media()
        entries_to_add = {}
        for img in data['to_add']:
            assert img in valid_filenames
            if img in problematic.problematic_images:
                print(f'WARNING: image {img} is already in the list of problematic images, skipping to add')
                continue
            entries_to_add[img] = {'time_marked': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   'server_username': getpass.getuser(),
                                   'reason': 'marked as problematic in the review server'}
        entries_to_remove = []
        for img in data['to_remove']:
            assert img in valid_filenames
            if img not in problematic.problematic_images:
                print(f'WARNING: image {img} is not in the list of problematic images, skipping to remove')
                continue
            entries_to_remove.append(img)
        problematic.update_data(entries_to_add, entries_to_remove)
        print(f'Marked {len(entries_to_add)} images as problematic and removed {len(entries_to_remove)} from the list of problematic images')
        resp.media = {'success': True,
                      'num_added': len(entries_to_add),
                      'num_removed': len(entries_to_remove)}
    except:
        traceback.print_exc()
        raise


@api.route('/find_wnids')
def find_wnids(req, resp):
    if not 'q' in req.params:
        resp.media = []
        return
    search_term = req.params['q']
    found_wnids = imgnet.search_classes(search_term)
    response = []
    for info in found_wnids:
        response.append({'wnid': info.wnid,
                         'gloss': info.gloss,
                         'wikipedia_pages': info.wikipedia_pages,
                         'synset': info.synset})
    resp.media = response


@api.route('/get_all_wnid_info')
def get_all_wnid_info(req, resp):
    response = {}
    for wnid, info in imgnet.class_info_by_wnid.items():
        response[wnid] = ({'wnid': info.wnid,
                           'gloss': info.gloss,
                           'wikipedia_pages': info.wikipedia_pages,
                           'synset': info.synset})
    resp.media = response


@api.route('/update_candidates')
async def update_candidates(req, resp):
    data = await req.media()
    with open('../data/metadata/candidate_blacklist.json', 'r') as f:
        blacklist = json.load(f)
    with open('../data/metadata/near_duplicates.json', 'r') as f:
        near_duplicates = json.load(f)

    num_added_to_blacklist = 0
    num_removed_from_blacklist = 0
    ndc_set_sizes = []

    for key, value in data.items():
        if key == 'added_to_blacklist' and len(value) > 0:
            assert len(value) == len(set(value))
            for cid in value:
                assert cid in cds.all_candidates
                if cid in blacklist:
                    print(f'    Warning: candidate {cid} is already on the blacklist')
                else:
                    blacklist[cid] = 'invalid image (added in the review server)'
                    num_added_to_blacklist += 1
        elif key == 'removed_from_blacklist' and len(value) > 0:
            assert len(value) == len(set(value))
            for cid in value:
                assert cid in cds.all_candidates
                if cid in blacklist:
                    del blacklist[cid]
                    num_removed_from_blacklist += 1
                else:
                    print(f'    Warning: candidate {cid} is not on the blacklist (cannot remove)')
        elif key == 'near_duplicate_sets' and len(value) > 0:
            for cur_set in value:
                if len(cur_set) == 1:
                    print(f'    Warning: near-duplicate set with size 1, ignoring {cur_set}')
                else:
                    assert len(cur_set) == len(set(cur_set))
                    root_cid = cur_set[0]
                    if root_cid not in near_duplicates:
                        near_duplicates[root_cid] = []
                    for other_cid in cur_set[1:]:
                        if other_cid not in near_duplicates[root_cid]:
                            near_duplicates[root_cid].append(other_cid)
                    near_duplicates[root_cid] = list(sorted(set(near_duplicates[root_cid])))
                    ndc_set_sizes.append(len(cur_set))

    reload_ndc = False 
    if num_added_to_blacklist > 0 or num_removed_from_blacklist > 0:
        with open('../data/metadata/candidate_blacklist.json', 'w') as f:
            json.dump(blacklist, f, indent=2, sort_keys=True)
        print(f'    Added {num_added_to_blacklist} candidates to the blacklist and removed {num_removed_from_blacklist}', end='')
        cds.reload_blacklist()
        reload_ndc = True
    if len(ndc_set_sizes) > 0:
        with open('../data/metadata/near_duplicates.json', 'w') as f:
            json.dump(near_duplicates, f, indent=2, sort_keys=True)
        print(f'    Added {len(ndc_set_sizes)} near-duplicate sets (sizes {ndc_set_sizes})', end='')
        reload_ndc = True
    if reload_ndc:
        start_time = timer()
        ndc.reload_near_duplicate_data(verbose=False)
        reload_time = timer() - start_time
        print(f' (took {reload_time:.2f} seconds to reload the ndc data)')
    else:
        print()

    resp.media = {'success': True,
                  'num_added_to_blacklist': num_added_to_blacklist,
                  'num_removed_from_blacklist': num_removed_from_blacklist,
                  'near_duplicate_set_sizes': ndc_set_sizes}


@api.route('/wnid/{wnid}')
def get_wnid_data(req, resp, *, wnid):
    try:
        assert wnid in imgnet.class_info_by_wnid
        cand_data = {}
        for x in cds.candidates_by_wnid[wnid]:
            cid = x['id_ours']
            if cid in mturk.image_num_assignments and wnid in mturk.image_num_assignments[cid]:
                num_assignments = mturk.image_num_assignments[cid][wnid] 
                sel_frequency = mturk.image_fraction_selected[cid][wnid]
            else:
                num_assignments = 0
                sel_frequency = 0.0
            cand_data[cid] = {'url': get_image_url(cid),
                              'num_assignments': num_assignments,
                              'date_taken': x['date_taken'],
                              'selection_frequency': sel_frequency,
                              'is_blacklisted': cid in cds.blacklist,
                              'is_problematic': cid in problematic.problematic_images,
                              'is_near_duplicate': ndc.is_near_duplicate[cid]}
        val_data = get_val_data(wnid)
        class_info = imgnet.class_info_by_wnid[wnid]
        resp.media = {'wnid': wnid,
                      'gloss': class_info.gloss,
                      'synset': class_info.synset,
                      'wikipedia_pages': class_info.wikipedia_pages,
                      'val_imgs': val_data,
                      'candidates': cand_data}
    except:
        traceback.print_exc()
        raise


@click.command()
@click.option('--use_local_images', is_flag=True)
def main_function(use_local_images):
    global use_local_image_files
    use_local_image_files = use_local_images
    api.run()


if __name__ == '__main__':
    main_function()

from collections import Counter
from collections import namedtuple
from collections import OrderedDict
import concurrent
import hashlib
import io
import json
import os
import pathlib
import pickle
import tarfile
import time
import urllib

import boto3
import imageio
import numpy as np

import utils

# Batches the given set of candidates on S3 and returns a dictionary from
# candidate id to batch id. The batching is done via pywren.
def batch_candidates(candidates,
                     batch_size,
                     prefix='imagenet2candidates_batches',
                     bucket='imagenet2datav2',
                     verbose=True):
    import pywren
    from pywren import wrenconfig as wc

    def hash_ids(ids):
        return hashlib.sha256((','.join(ids)).encode()).hexdigest()

    def create_and_store_batch(batch_ids):
        batch_key = hash_ids(batch_ids)
        full_key = os.path.join(prefix, batch_key) + '.pickle'
        data = {}
        for cur_id in batch_ids:
            cur_key = 'imagenet2candidates_scaled/' + cur_id + '.jpg'
            data[cur_id], _ = utils.get_s3_object_bytes_with_backoff(cur_key, bucket=bucket)
        client = utils.get_s3_client()
        client.put_object(Key=full_key, Bucket=bucket, Body=pickle.dumps(data))
        return (batch_key, batch_ids)
    candidates = candidates.copy()
    cids = [c['id_ours'] for c in candidates]
    batches = list(utils.chunks(cids, batch_size))
    num_batches = len(batches)
    if verbose:
        print('Creating {} batches of size {} ...'.format(num_batches, batch_size))
    
    pywren_config = wc.default()
    pywren_config['runtime']['s3_bucket'] = 'imagenet2pywren'
    pywren_config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-imagenet2pywren.tar.gz'
    pwex = pywren.default_executor(config=pywren_config)
    print(f"Mapping over {len(batches)} images")
    futures = pwex.map(create_and_store_batch,
                       batches,
                       exclude_modules=["site-packages"])
    ALWAYS = 3
    done, not_done = pywren.wait(futures, ALWAYS)
    while len(not_done) > 0:
        done, not_done = pywren.wait(futures, ALWAYS)
        time.sleep(1)
    print('done')
    result = {}
    for res in done:
        actual_res = res.result()
        for cid in actual_res[1]:
            result[cid] = actual_res[0]
    print(len(result))
    for cand in candidates:
        assert cand['id_ours'] in result
        cand['batch'] = result[cand['id_ours']]
    return candidates


def load_blacklist():
    blacklist_filepath = pathlib.Path(__file__).parent / '../data/metadata/candidate_blacklist.json'
    blacklist_filepath = blacklist_filepath.resolve()
    with open(blacklist_filepath, 'r') as f:
        blacklist = json.load(f)
    return blacklist


def load_data():
    blacklist = load_blacklist()
    json_dir = pathlib.Path(__file__).parent / '../data/search_results'
    json_dir = json_dir.resolve()
    json_filenames = []
    for p in json_dir.glob('*.json'):
        json_filenames.append(p)
    json_filenames = sorted(json_filenames)
    json_data = []
    for p in json_filenames:
        with open(p, 'r') as f:
            json_data.append((str(p.name), json.load(f)))
    return str(json_dir), json_data, blacklist


class CandidateData:
    def __init__(self,
                 exclude_blacklisted_candidates=True,
                 cache_on_local_disk=True,
                 cache_root_path=None,
                 bucket='imagenet2datav2',
                 max_num_threads=20,
                 num_tries=4,
                 initial_delay=1.0,
                 delay_factor=2.0,
                 load_metadata_from_s3=False,
                 verbose=True):
        self.blacklist_excluded = exclude_blacklisted_candidates
        self.bucket = bucket
        self.cache_on_local_disk = cache_on_local_disk
        self.max_num_threads = max_num_threads
        self.num_tries = num_tries
        self.initial_delay = initial_delay
        self.delay_factor = delay_factor
        if self.cache_on_local_disk:
            if cache_root_path is None:
                self.cache_root_path = pathlib.Path(__file__).parent /  '../data/cache'
            else:
                self.cache_root_path = pathlib.Path(cache_root_path)
            self.cache_root_path = self.cache_root_path.resolve()
        else:
            assert cache_root_path is None
            self.cache_root_path = None
        if load_metadata_from_s3:
            key = 'metadata/candidate_metadata_2018-12-13_04-35-19_UTC.pickle'
            num_replicas = 10
            pickle_bytes = self.get_s3_file_bytes(key, verbose=verbose, num_replicas=num_replicas)
            pickle_dict = pickle.loads(pickle_bytes)
            data_source = 's3://' + bucket + '/' + key
            json_data = pickle_dict['json_data']
            self.blacklist = pickle_dict['blacklist']
            if verbose:
                print('Using pickled JSON data stored by {} from {} locally'.format(pickle_dict['username'], pickle_dict['json_dir']))
        else:
            data_source, json_data, self.blacklist = load_data()

        self.all_candidates = OrderedDict()
        self.duplicates = []
        num_blacklist_ignored = 0
        for p, cur_candidates in json_data:
            for candidate in cur_candidates:
                assert 'id_ours' in candidate
                assert 'wnid' in candidate
                assert 'url' in candidate
                cur_id = candidate['id_ours']
                candidate['source_filename'] = p
                if cur_id in self.blacklist and exclude_blacklisted_candidates:
                    num_blacklist_ignored += 1
                    continue
                if cur_id in self.all_candidates:
                    self.duplicates.append(candidate)
                else:
                    self.all_candidates[cur_id] = candidate
        self.duplicate_counts = Counter([x['id_ours'] for x in self.duplicates])
        self.candidates_by_wnid = {}
        for c in self.all_candidates.values():
            cur_wnid = c['wnid']
            if cur_wnid not in self.candidates_by_wnid:
                self.candidates_by_wnid[cur_wnid] = []
            self.candidates_by_wnid[cur_wnid].append(c)
        if verbose:
            print('Loaded {} unique candidates from {} search result JSON file(s).'.format(
                    len(self.all_candidates), len(json_data)))
            print('    {}/...'.format(data_source))
            for p in json_data[:10]:
                print('        {}'.format(p[0]))
            if len(json_data) > 10:
                print('        ...')
            print('    There were {} duplicate occurences.'.format(len(self.duplicates)))
            print('    Ignored {} candidate entries because they are on the blacklist (blacklist size: {}).'.format(
                    num_blacklist_ignored, len(self.blacklist)))
    
    def reload_blacklist(self):
        assert not self.blacklist_excluded
        self.blacklist = load_blacklist()

    def get_s3_file_bytes(self, remote_filename, verbose=True, num_replicas=1):
        return utils.get_s3_file_bytes(remote_filename,
                                       bucket=self.bucket,
                                       cache_on_local_disk=self.cache_on_local_disk,
                                       cache_root_path=self.cache_root_path,
                                       verbose=verbose,
                                       num_replicas=num_replicas)

    def get_s3_batched_objects(self, s3_resources, verbose=True):
        return utils.get_s3_batched_objects(
                s3_resources,
                bucket=self.bucket,
                cache_on_local_disk=self.cache_on_local_disk,
                cache_root_path=self.cache_root_path,
                verbose=verbose,
                max_num_threads=self.max_num_threads,
                num_tries=self.num_tries,
                initial_delay=self.initial_delay,
                delay_factor=self.delay_factor)

    def load_image(self, candidate_id, size=None, force_rgb=False, verbose=True):
        return self.load_image_batch([candidate_id], size=size, force_rgb=force_rgb, verbose=verbose)[candidate_id]
    
    def load_image_batch(self, candidate_ids, size=None, force_rgb=False, verbose=True):
        tmp_res = self.load_image_bytes_batch(candidate_ids, size=size, verbose=verbose)
        res = {}
        for k, v in tmp_res.items():
            img = imageio.imread(v)
            if force_rgb:
                res[k] = utils.make_rgb(img)
            else:
                res[k] = img
        return res

    def load_image_bytes(self, candidate_id, size=None, verbose=True):
        return self.load_image_bytes_batch([candidate_id], size=size, verbose=verbose)[candidate_id]
    
    def load_image_bytes_batch(self, candidate_ids, size=None, verbose=True):
        resource_descriptions = self.get_image_resource_descriptions(candidate_ids, size=size)
        result = self.get_s3_batched_objects(resource_descriptions, verbose=verbose)
        return result
    
    def get_image_resource_descriptions(self, candidate_ids, size=None, min_num_images_for_batch_loading=100000):
        if len(candidate_ids) == 0:
            return {}
        for cid in candidate_ids:
            assert cid in self.all_candidates
        result = {}
        if size == 'scaled_256':
            cids_by_batch = {}
            cids_without_batch = []
            for cid in candidate_ids:
                if 'batch' in self.all_candidates[cid]:
                    cur_batch = self.all_candidates[cid]['batch']
                    if cur_batch not in cids_by_batch:
                        cids_by_batch[cur_batch] = []
                    cids_by_batch[cur_batch].append(cid)
                else:
                    cids_without_batch.append(cid)
            for cur_batch, cids in cids_by_batch.items():
                if len(cids) >= min_num_images_for_batch_loading:
                    pickle_key = 'imagenet2candidates_batches/' + cur_batch + '.pickle'
                    for cid in cids:
                        result[cid] = utils.S3BatchResource(pickle_key, cid, 'pickle_dict', 1000 * 50 * 1000)
                else:
                    for cid in cids:
                        img_key = 'imagenet2candidates_scaled/' + cid + '.jpg'
                        result[cid] = utils.S3BatchResource(img_key, None, 'object_bytes', 50 * 1000)
            for cid in cids_without_batch:
                img_key = 'imagenet2candidates_scaled/' + cid + '.jpg'
                result[cid] = utils.S3BatchResource(img_key, None, 'object_bytes', 150 * 1000)
        elif size == 'scaled_500':
            for cid in candidate_ids:
                img_key = 'imagenet2candidates_mturk/' + cid + '.jpg'
                result[cid] = utils.S3BatchResource(img_key, None, 'object_bytes', 150 * 1000)
        elif size == 'original':
            for cid in candidate_ids:
                img_key = 'imagenet2candidates_original/' + cid + '.jpg'
                result[cid] = utils.S3BatchResource(img_key, None, 'object_bytes', 2 * 1000 * 1000)
        else:
            raise NotImplementedError()
        assert len(result) == len(candidate_ids)
        for cid in candidate_ids:
            assert cid in result
        return result
    
    def load_features(self, candidate_id, verbose=True):
        return self.load_features_batch([candidate_id], verbose=verbose)[candidate_id]

    def load_features_batch(self, candidate_ids, verbose=True):
        resource_descriptions = self.get_feature_resource_descriptions(candidate_ids)
        result = self.get_s3_batched_objects(resource_descriptions, verbose=verbose)
        return result
    
    def get_feature_resource_descriptions(self, candidate_ids):
        if len(candidate_ids) == 0:
            return {}
        for cid in candidate_ids:
            assert cid in self.all_candidates
        result = {}
        for cid in candidate_ids:
            features_key = 'imagenet2candidates_featurized/' + cid + '.npy'
            result[cid] = utils.S3BatchResource(features_key, None, 'numpy_bytes', 8 * 4096)
        assert len(result) == len(candidate_ids)
        for cid in candidate_ids:
            assert cid in result
        return result

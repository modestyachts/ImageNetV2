from collections import namedtuple
import concurrent
import io
import json
import os
import pathlib
import pickle
import tarfile

import boto3
import imageio
import numpy as np

import utils


ClassInfo = namedtuple('ClassInfo', ['wnid', 'cid', 'synset', 'wikipedia_pages', 'gloss'])


class UnknownDataPartError(Exception):
    pass


class ImageNetData:
    def __init__(self,
                 cache_on_local_disk=True,
                 cache_root_path=None,
                 bucket='imagenet2datav2',
                 max_num_threads=20,
                 num_tries=4,
                 initial_delay=1.0,
                 delay_factor=2.0,
                 verbose=True):
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

        metadata_filename = 'metadata/imagenet_metadata_2018-09-14_01-26-58_UTC.pickle'
        num_replicas = 10
        metadata_file_bytes = self.get_s3_file_bytes(metadata_filename, verbose=verbose, num_replicas=num_replicas)
        tmp_metadata = pickle.loads(metadata_file_bytes)
        self.train_imgs_by_wnid = tmp_metadata['train_imgs_by_wnid']
        self.val_imgs_by_wnid = tmp_metadata['val_imgs_by_wnid']
        self.wnid_by_val_filename = tmp_metadata['wnid_by_val_img_filename']
        self.test_filenames = tmp_metadata['test_filenames']
        self.test_batches = tmp_metadata['test_batches']
        self.test_batch_by_filename = tmp_metadata['test_batch_by_filename']
        
        tmp_class_info = tmp_metadata['class_info_by_wnid']
        self.class_info_by_wnid = {}
        self.class_info_by_cid = {}
        for tmpci in tmp_class_info.values():
            tci = ClassInfo(wnid=tmpci['wnid'],
                            cid=tmpci['cid'],
                            synset=tmpci['synset'],
                            wikipedia_pages=tmpci['wikipedia_pages'],
                            gloss=tmpci['gloss'])
            self.class_info_by_wnid[tci.wnid] = tci
            self.class_info_by_cid[tci.cid] = tci
        assert len(self.class_info_by_cid) == 1000
        assert len(self.class_info_by_wnid) == 1000
        for wnid in self.class_info_by_wnid.keys():
            assert wnid in self.train_imgs_by_wnid

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
    
    def search_classes(self, search_term):
        if search_term in self.class_info_by_wnid.keys():
            return [self.class_info_by_wnid[search_term]]
        result = []
        for cinfo in self.class_info_by_wnid.values():
            added = False
            for synset_term in cinfo.synset:
                if search_term in synset_term.lower():
                    if not added:
                        result.append(cinfo)
                        added = True
            if search_term in cinfo.gloss.lower():
                if not added:
                    result.append(cinfo)
                    added = True
        return result
    
    def get_wordnet_url(self, wnid):
        assert wnid in self.class_info_by_wnid
        return 'http://wordnet-rdf.princeton.edu/pwn30/' + wnid[1:] + '-n'

    def get_dataset_part(self, filename):
        if filename.startswith('ILSVRC2012_val_'):
            return 'val'
        elif filename.startswith('ILSVRC2012_test_'):
            return 'test'
        elif filename.startswith('n'):
            parts = filename.split('_')
            assert parts[0] in self.val_imgs_by_wnid
            return 'train'
        else:
            raise UnknownDataPartError('Invalid filename "{}"'.format(filename))
    
    def is_imagenet_filename(self, filename):
        try:
            self.get_dataset_part(filename)
            return True
        except UnknownDataPartError:
            return False

    def load_image_bytes(self, filename, size=None, verbose=True):
        return self.load_image_bytes_batch([filename], size=size, verbose=verbose)[filename]

    def load_image(self, filename, size=None, force_rgb=False, verbose=True):
        return self.load_image_batch([filename], size=size, force_rgb=force_rgb, verbose=verbose)[filename]
    
    def load_image_batch(self, filenames, size=None, force_rgb=False, verbose=True):
        tmp_res = self.load_image_bytes_batch(filenames, size=size, verbose=verbose)
        res = {}
        for k, v in tmp_res.items():
            img = imageio.imread(v)
            if force_rgb:
                res[k] = utils.make_rgb(img)
            else:
                res[k] = img
        return res

    def get_image_resource_descriptions(self, filenames, size=None):
        assert len(set(filenames)) == len(filenames)
        filenames_by_part = {'val': [], 'train': [], 'test': []}
        for fn in filenames:
            part = self.get_dataset_part(fn)
            filenames_by_part[part].append(fn)
        descriptions = self.get_train_image_resource_descriptions(filenames_by_part['train'], size=size)
        descriptions.update(self.get_val_image_resource_descriptions(filenames_by_part['val'], size=size))
        descriptions.update(self.get_test_image_resource_descriptions(filenames_by_part['test'], size=size))
        assert len(filenames) == len(descriptions)
        for fn in filenames:
            assert fn in descriptions
        return descriptions
    
    def get_feature_resource_descriptions(self, filenames):
        assert len(set(filenames)) == len(filenames)
        filenames_by_part = {'val': [], 'train': [], 'test': []}
        for fn in filenames:
            part = self.get_dataset_part(fn)
            filenames_by_part[part].append(fn)
        descriptions = self.get_train_feature_resource_descriptions(filenames_by_part['train'])
        descriptions.update(self.get_val_feature_resource_descriptions(filenames_by_part['val']))
        descriptions.update(self.get_test_feature_resource_descriptions(filenames_by_part['test']))
        assert len(filenames) == len(descriptions)
        for fn in filenames:
            assert fn in descriptions
        return descriptions
    
    def load_image_bytes_batch(self, filenames, size=None, verbose=True):
        resource_descriptions = self.get_image_resource_descriptions(filenames, size=size)
        result = self.get_s3_batched_objects(resource_descriptions, verbose=verbose)
        return result

    def load_features(self, filename, verbose=True):
        return self.load_features_batch([filename], verbose=verbose)[filename]

    def load_features_batch(self, filenames, verbose=True):
        resource_descriptions = self.get_feature_resource_descriptions(filenames)
        result = self.get_s3_batched_objects(resource_descriptions, verbose=verbose)
        return result
    
    def get_wnid_from_train_filename(self, filename):
        stem = pathlib.Path(filename).stem
        stem_parts = stem.split('_')
        assert len(stem_parts) == 2
        wnid = stem_parts[0]
        assert wnid in self.train_imgs_by_wnid
        assert filename in self.train_imgs_by_wnid[wnid]
        return wnid

    def get_train_image_resource_descriptions(self, filenames, size=None, min_num_images_for_batch_loading=100):
        if len(filenames) == 0:
            return {}
        if size != 'scaled_256':
            raise NotImplementedError()
        result = {}
        filenames_by_wnid = {}
        for fn in filenames:
            cur_wnid = self.get_wnid_from_train_filename(fn)
            if cur_wnid not in filenames_by_wnid:
                filenames_by_wnid[cur_wnid] = []
            filenames_by_wnid[cur_wnid].append(fn)
        for wnid, cur_filenames in filenames_by_wnid.items():
            if len(cur_filenames) >= min_num_images_for_batch_loading:
                tarball_key = 'imagenet-train/' + wnid + '-scaled.tar'
                for fn in cur_filenames:
                    result[fn] = utils.S3BatchResource(tarball_key, wnid + '/' + fn, 'tarball', 50 * 1300 * 1000)
            else:
                for fn in cur_filenames:
                    img_key = 'imagenet-train-individual/' + wnid + '/' + fn
                    result[fn] = utils.S3BatchResource(img_key, None, 'object_bytes', 50 * 1000)
        assert len(result) == len(filenames)
        for fn in filenames:
            assert fn in result
        return result
    
    def get_train_feature_resource_descriptions(self, filenames):
        if len(filenames) == 0:
            return {}
        result = {}
        filenames_by_wnid = {}
        for fn in filenames:
            cur_wnid = self.get_wnid_from_train_filename(fn)
            if cur_wnid not in filenames_by_wnid:
                filenames_by_wnid[cur_wnid] = []
            filenames_by_wnid[cur_wnid].append(fn)
        for wnid, cur_filenames in filenames_by_wnid.items():
            batch_key = 'imagenet-train-featurized/' + wnid + '-fc7.pkl'
            for fn in cur_filenames:
                stem = pathlib.Path(fn).stem
                result[fn] = utils.S3BatchResource(batch_key, stem, 'pickle_dict', 32 * 1300 * 1000)
        assert len(result) == len(filenames)
        for fn in filenames:
            assert fn in result
        return result
    
    def get_val_image_resource_descriptions(self, filenames, size=None, min_num_images_for_batch_loading=20):
        if len(filenames) == 0:
            return {}
        result = {}
        if size == 'scaled_256':
            filenames_by_wnid = {}
            for fn in filenames:
                cur_wnid = self.wnid_by_val_filename[fn]
                if cur_wnid not in filenames_by_wnid:
                    filenames_by_wnid[cur_wnid] = []
                filenames_by_wnid[cur_wnid].append(fn)
            for wnid, cur_filenames in filenames_by_wnid.items():
                if len(cur_filenames) >= min_num_images_for_batch_loading:
                    tarball_key = 'imagenet-validation/val-' + wnid + '-scaled.tar'
                    for fn in cur_filenames:
                        result[fn] = utils.S3BatchResource(tarball_key, wnid + '/' + fn, 'tarball', 50 * 50 * 1000)
                else:
                    for fn in cur_filenames:
                        img_key = 'imagenet-validation-individual/' + wnid + '/' + fn
                        result[fn] = utils.S3BatchResource(img_key, None, 'object_bytes', 50 * 1000)
        elif size == 'scaled_500':
            for fn in filenames:
                img_key = 'imagenet_validation_flat/' + fn
                result[fn] = utils.S3BatchResource(img_key, None, 'object_bytes', 150 * 1000)
        else:
            raise NotImplementedError()
        assert len(result) == len(filenames)
        for fn in filenames:
            assert fn in result
        return result
    
    def get_val_feature_resource_descriptions(self, filenames):
        if len(filenames) == 0:
            return {}
        result = {}
        filenames_by_wnid = {}
        for fn in filenames:
            cur_wnid = self.wnid_by_val_filename[fn]
            if cur_wnid not in filenames_by_wnid:
                filenames_by_wnid[cur_wnid] = []
            filenames_by_wnid[cur_wnid].append(fn)
        for wnid, cur_filenames in filenames_by_wnid.items():
            batch_key = 'imagenet-validation-featurized/val-' + wnid + '-fc7.pkl'
            for fn in cur_filenames:
                stem = pathlib.Path(fn).stem
                result[fn] = utils.S3BatchResource(batch_key, stem, 'pickle_dict', 32 * 50 * 1000)
        assert len(result) == len(filenames)
        for fn in filenames:
            assert fn in result
        return result

    def get_test_image_resource_descriptions(self, filenames, size=None, min_num_images_for_batch_loading=100):
        if len(filenames) == 0:
            return {}
        result = {}
        if size == 'scaled_256':
            filenames_by_batch = {}
            for fn in filenames:
                cur_batch = self.test_batch_by_filename[fn]
                if cur_batch not in filenames_by_batch:
                    filenames_by_batch[cur_batch] = []
                filenames_by_batch[cur_batch].append(fn)
            for batch, cur_filenames in filenames_by_batch.items():
                if len(cur_filenames) >= min_num_images_for_batch_loading:
                    pickle_key = 'imagenet-test-batches/' + batch + '.pickle'
                    for fn in cur_filenames:
                        result[fn] = utils.S3BatchResource(pickle_key, fn, 'pickle_dict', 50 * 1000 * 1000)
                else:
                    for fn in cur_filenames:
                        img_key = 'imagenet-test-scaled/' + fn
                        result[fn] = utils.S3BatchResource(img_key, None, 'object_bytes', 50 * 1000)
        elif size == 'scaled_500':
            for fn in filenames:
                img_key = 'imagenet-test/' + fn
                result[fn] = utils.S3BatchResource(img_key, None, 'object_bytes', 150 * 1000)
        else:
            raise NotImplementedError()
        assert len(result) == len(filenames)
        for fn in filenames:
            assert fn in result
        return result
    
    def get_test_feature_resource_descriptions(self, filenames):
        if len(filenames) == 0:
            return {}
        result = {}
        for fn in filenames:
            img_key = 'imagenet-test-featurized/' + fn + '.npy'
            result[fn] = utils.S3BatchResource(img_key, None, 'numpy_bytes', 8 * 4096)
        assert len(result) == len(filenames)
        for fn in filenames:
            assert fn in result
        return result

    def get_all_train_image_names(self, include_extension=True):
        result = []
        wnids = sorted(list(self.train_imgs_by_wnid.keys()))
        for wnid in wnids:
            result.extend(sorted(self.train_imgs_by_wnid[wnid]))
        if not include_extension:
            result = [pathlib.Path(name).stem for name in result]
        return result
    
    def get_all_val_image_names(self, include_extension=True):
        result = []
        wnids = sorted(list(self.val_imgs_by_wnid.keys()))
        for wnid in wnids:
            result.extend(sorted(self.val_imgs_by_wnid[wnid]))
        if not include_extension:
            result = [pathlib.Path(name).stem for name in result]
        return result
    
    def get_all_image_names(self):
        return self.get_all_train_image_names() + self.get_all_val_image_names() + self.test_filenames
    
    def get_wnid_of_image(self, filename):
        part = self.get_dataset_part(filename)
        if part == 'train':
            return self.get_wnid_from_train_filename(filename)
        elif part == 'val':
            return self.wnid_by_val_filename[filename]
        elif part == 'test':
            raise ValueError('wnid of test images is not known (filename {})'.format(filename))
        else:
            raise ValueError('Unknown dataset part {}'.format(part))


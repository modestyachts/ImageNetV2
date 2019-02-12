import io
import math
import pathlib

import imageio
import PIL

import imagenet
import utils

class ImageLoader:
    def __init__(self,
                 imagenet_loader,
                 candidate_loader,
                 cache_on_local_disk=True,
                 cache_root_path=None,
                 bucket='imagenet2datav2',
                 max_num_threads=40,
                 num_tries=4,
                 initial_delay=1.0,
                 delay_factor=math.sqrt(2.0)):
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
        self.imagenet_loader = imagenet_loader
        self.candidate_loader = candidate_loader
    
    def get_s3_batched_objects(self, s3_resources, verbose=True, download_callback=None):
        return utils.get_s3_batched_objects(
                s3_resources,
                bucket=self.bucket,
                cache_on_local_disk=self.cache_on_local_disk,
                cache_root_path=self.cache_root_path,
                verbose=verbose,
                max_num_threads=self.max_num_threads,
                num_tries=self.num_tries,
                initial_delay=self.initial_delay,
                delay_factor=self.delay_factor,
                download_callback=download_callback)

    def load_image_bytes(self, filename, size=None, verbose=True):
        return self.load_image_bytes_batch([filename], size=size, verbose=verbose)[filename]
    
    def load_image_bytes_batch(self, filenames, size=None, verbose=True, download_callback=None):
        resource_descriptions = self.get_image_resource_descriptions(filenames, size=size)
        result = self.get_s3_batched_objects(resource_descriptions,
                                             verbose=verbose,
                                             download_callback=download_callback)
        return result

    def load_image(self, filename, size=None, force_rgb=False, verbose=True, loader='imread'):
        return self.load_image_batch([filename], size=size, force_rgb=force_rgb, verbose=verbose, loader=loader)[filename]
    
    def load_image_batch(self,
                         filenames,
                         size=None,
                         force_rgb=False,
                         verbose=True,
                         loader='imread',
                         download_callback=None):
        tmp_res = self.load_image_bytes_batch(filenames,
                                              size=size,
                                              verbose=verbose,
                                              download_callback=download_callback)
        res = {}
        for k, v in tmp_res.items():
            if loader == 'imread':
                img = imageio.imread(v)
                if force_rgb:
                    res[k] = utils.make_rgb(img)
                else:
                    res[k] = img
            elif loader == 'pillow':
                img = PIL.Image.open(io.BytesIO(v))
                img.load()
                if force_rgb:
                    res[k] = img.convert('RGB')
                else:
                    res[k] = img
            else:
                raise ValueError('Unknown image loader {}'.format(loader))
        return res
    
    def get_image_resource_descriptions(self, filenames, size=None):
        imagenet_filenames = []
        candidate_filenames = []
        for fn in filenames:
            if self.imagenet_loader.is_imagenet_filename(fn):
                imagenet_filenames.append(fn)
            else:
                if fn not in self.candidate_loader.all_candidates:
                    print(fn)
                assert fn in self.candidate_loader.all_candidates
                candidate_filenames.append(fn)
        result = self.imagenet_loader.get_image_resource_descriptions(imagenet_filenames, size=size)
        result.update(self.candidate_loader.get_image_resource_descriptions(candidate_filenames, size=size))
        assert len(result) == len(filenames)
        return result
    
    def get_wnid_of_image(self, filename):
        if self.imagenet_loader.is_imagenet_filename(filename):
            return self.imagenet_loader.get_wnid_of_image(filename)
        else:
            if filename not in self.candidate_loader.all_candidates:
                raise ValueError('{} is not an ImageNet image and not a candidate'.format(filename))
            return self.candidate_loader.all_candidates[filename]['wnid']
    
    def get_feature_resource_descriptions(self, filenames):
        imagenet_filenames = []
        candidate_filenames = []
        for fn in filenames:
            if self.imagenet_loader.is_imagenet_filename(fn):
                imagenet_filenames.append(fn)
            else:
                assert fn in self.candidate_loader.all_candidates
                candidate_filenames.append(fn)
        result = self.imagenet_loader.get_feature_resource_descriptions(imagenet_filenames)
        result.update(self.candidate_loader.get_feature_resource_descriptions(candidate_filenames))
        assert len(result) == len(filenames)
        return result
    
    def load_features(self, filename, verbose=True):
        return self.load_features_batch([filename], verbose=verbose)[filename]
    
    def load_features_batch(self, filenames, verbose=True):
        resource_descriptions = self.get_feature_resource_descriptions(filenames)
        result = self.get_s3_batched_objects(resource_descriptions, verbose=verbose)
        return result

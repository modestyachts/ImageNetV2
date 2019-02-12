from collections import namedtuple
import concurrent
import concurrent.futures
import hashlib
import io
import json
import math
import numpy as np
import os
import pickle
import PIL.Image
import pathlib
import random
import statistics
import tarfile
import threading
import time
from time import sleep
from timeit import default_timer as timer
import urllib.request

import boto3
import botocore
from botocore.client import Config
import numpy
import scipy.stats

import aes

salt_store = None
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_salt():
    global salt_store
    if salt_store is None:
        salt_store = get_s3_file_bytes('salt.txt', cache_on_local_disk=False, verbose=False).decode('utf-8')
    return salt_store


def encrypt_string(x):
    salt = get_salt()
    cipher = aes.AESCipher(salt[:32].encode(), salt[32:48].encode())
    return cipher.encrypt(x.encode()).decode()


def decrypt_string(x):
    salt = get_salt()
    cipher = aes.AESCipher(salt[:32].encode(), salt[32:48].encode())
    return cipher.decrypt(x.encode()).decode()


def decrypt_s3_key(s3_key, bucket, prefix="encrypted"):
    s3_key = s3_key.replace("{0}/".format(prefix), "")
    base_key = s3_key.split(".")[0]
    return decrypt_string_with_magic(base_key.encode())


def encrypt_string_with_magic(string, magic="DEADBEEF", max_key_length=99):
    len_string = len(string)
    if (len_string > max_key_length):
        raise Exception("key too long")
    base_key = string
    base_key += " "*(max_key_length - len(string))
    base_key = magic + base_key
    new_base_key = encrypt_string(base_key)
    return new_base_key


def decrypt_string_with_magic(string, magic="DEADBEEF", max_key_length=99):
    old_key = decrypt_string(string).strip()
    magic_str = old_key[:len(magic)]
    assert magic == magic_str
    old_key = old_key[len(magic):]
    return old_key


def encrypt_s3_copy_key(s3_key, bucket, encrypt_out, strip_string, max_key_length=99):
    base_key = os.path.basename(s3_key)
    suffix = base_key.split(".")[1].lower()
    if suffix == "jpeg":
        suffix = "jpg"
    base_key = base_key.replace(strip_string, "")
    if len(base_key) > max_key_length:
        raise Exception("Key length too long")
    new_base_key = encrypt_string_with_magic(base_key, max_key_length=max_key_length)
    new_key = os.path.join(encrypt_out, new_base_key) + "." + suffix
    client = get_s3_client()
    print(s3_key, new_key)
    if (not key_exists(bucket=bucket, key=new_key)):
        return client.copy_object(Bucket=bucket, Key=new_key, CopySource="{0}/{1}".format(bucket, s3_key))
    else:
        return None


def hash_worker_id(worker_id):
    return hashlib.sha256((worker_id + get_salt()).encode()).hexdigest()


def hash_dataset_bytes(img_data, show_progress_bar=False):
    import tqdm
    sorted_keys = list(sorted(img_data.keys()))
    h = hashlib.sha256()
    if show_progress_bar:
        enumerable = tqdm.tqdm(sorted_keys)
    else:
        enumerable = sorted_keys
    for key in enumerable:
        h.update(img_data[key])
    return h.hexdigest()


def clopper_pearson(k, n, alpha=0.05):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi


def list_all_keys(prefix, bucket='imagenet2datav2'):
    client = get_s3_client()
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix)
    if (objects.get('Contents') == None):
        return []
    keys = list(map(lambda x: x['Key'], objects.get('Contents', [] )))
    truncated = objects['IsTruncated']
    next_marker = objects.get('NextMarker')
    while truncated:
        objects = client.list_objects(Bucket=bucket, Prefix=prefix,
                                      Delimiter=prefix, Marker=next_marker)
        truncated = objects['IsTruncated']
        next_marker = objects.get('NextMarker')
        keys += list(map(lambda x: x['Key'], objects['Contents']))
    return list(filter(lambda x: len(x) > 0, keys))


def load_urls(release='fall11'):
    data_path = os.path.join(os.path.dirname(__file__), '../data/urls/')
    filename = release + '_urls.txt'
    file_path = os.path.abspath(os.path.join(data_path, filename))
    print('Loading URLs from {} ...'.format(file_path))

    urls = {}
    with open(file_path, 'r', encoding='latin_1') as f:
        for line in f:
            parts = line.split(maxsplit=1)
            assert parts[0] not in urls
            parts2 = parts[0].split('_')
            assert len(parts2[0]) == 9
            assert parts2[0].startswith('n')
            assert parts2[1].isdigit()
            urls[parts[0]] = parts[1]
    if release == 'fall11':
        assert len(urls) == 14197122
    elif release == 'winter11':
        assert len(urls) == 12184113
    else:
        print('WARNING: release "{}" is unknown, not checking the file size.'.format(release))
    return urls


def list_files_in_s3_tarball(tarball_key, bucket='imagenet2datav2'):
    read_bytes, _ = get_s3_object_bytes_with_backoff(tarball_key, bucket=bucket)
    tarball = tarfile.open(fileobj=io.BytesIO(read_bytes))
    result = []
    for member in tarball.getmembers():
        if member.isfile():
            result.append(member.name)
    return result


def get_s3_object_bytes_with_backoff(key, *,
                                     bucket='imagenet2datav2',
                                     num_tries=40,
                                     initial_delay=1.0,
                                     delay_factor=2.0,
                                     num_replicas=1,
                                     thread_local=None):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, 'get_object_client'):
            thread_local.get_object_client = get_s3_client()
        client = thread_local.get_object_client
    delay = initial_delay
    num_tries_left = num_tries

    if num_replicas > 1: 
        replicas_counter_len = len(str(num_replicas))
        format_string = '_replica{{:0{}d}}-{{}}'.format(replicas_counter_len)
    while num_tries_left >= 1:
        try:
            if num_replicas > 1:
                cur_replica = random.randint(1, num_replicas)
                cur_key = key + format_string.format(cur_replica, num_replicas)
            else:
                cur_key = key
            read_bytes = client.get_object(Key=cur_key, Bucket=bucket)["Body"].read()
            return read_bytes, cur_key
        except:
            if num_tries_left == 1:
                raise Exception('get backoff failed ' + key + ' ' + str(delay))
            else:
                sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def put_s3_object_bytes_with_backoff(file_bytes, key, bucket='imagenet2datav2', num_tries=40, initial_delay=1.0, delay_factor=2.0):
    client = get_s3_client()
    delay = initial_delay
    num_tries_left = num_tries
    while num_tries_left >= 1:
        try:
            client.put_object(Key=key, Bucket=bucket, Body=file_bytes)
            return
        except:
            if num_tries_left == 1:
                print('put backoff failed' + key)
                raise Exception('put backoff failed ' + key + ' ' + str(len(file_bytes))+ ' ' + str(delay))
            else:
                sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def download_s3_file_with_backoff(key, local_filename, *,
                                  bucket='imagenet2datav2',
                                  num_tries=40,
                                  initial_delay=1.0,
                                  delay_factor=math.sqrt(2.0),
                                  num_replicas=1,
                                  thread_local=None):
    if thread_local is None:
        client = get_s3_client()
    else:
        if not hasattr(thread_local, 'download_client'):
            thread_local.download_client = get_s3_client()
        client = thread_local.download_client
    delay = initial_delay
    num_tries_left = num_tries
    
    if num_replicas > 1:
        replicas_counter_len = len(str(num_replicas))
        format_string = '_replica{{:0{}d}}-{{}}'.format(replicas_counter_len)
    while num_tries_left >= 1:
        try:
            if num_replicas > 1:
                cur_replica = random.randint(1, num_replicas)
                cur_key = key + format_string.format(cur_replica, num_replicas)
            else:
                cur_key = key
            client.download_file(bucket, cur_key, local_filename)
            return cur_key
        except:
            if num_tries_left == 1:
                raise Exception('download backoff failed ' + ' ' + str(key) + ' ' + str(delay))
            else:
                sleep(delay)
                delay *= delay_factor
                num_tries_left -= 1


def download_from_s3_if_not_local(remote_filename, local_filename=None, *,
                                  bucket='imagenet2datav2',
                                  verbose=True,
                                  num_replicas=1):
    if local_filename is None:
        local_filepath = pathlib.Path(__file__).parent /  '../data/cache' / remote_filename
    else:
        local_filepath = pathlib.Path(local_filename)
    local_filepath = local_filepath.resolve()
    if local_filepath.is_file():
        return
    local_filepath.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print('{} not available locally, downloading from S3 ... '.format(local_filepath), end='')
    cur_key = download_s3_file_with_backoff(remote_filename, str(local_filepath), bucket=bucket, num_replicas=num_replicas)
    if verbose:
        print('done', end='')
        if num_replicas > 1:
            print(' (downloaded replica {} )'.format(cur_key), end='')
        print()
    assert local_filepath.is_file()


def get_s3_file_bytes(key, *,
                      bucket='imagenet2datav2',
                      cache_on_local_disk=True,
                      cache_root_path=None,
                      verbose=True,
                      num_replicas=1):
    if cache_on_local_disk:
        if cache_root_path is None:
            cache_root_path = pathlib.Path(__file__).parent /  '../data/cache'
        cache_root_path = pathlib.Path(cache_root_path)
        cache_root_path = cache_root_path.resolve()
        local_filename = cache_root_path / key

        download_from_s3_if_not_local(key, local_filename, bucket=bucket, verbose=verbose, num_replicas=num_replicas)
        
        if verbose:
            print('Reading from local file {} ... '.format(local_filename), end='')
        with open(local_filename, 'rb') as f:
            result = f.read()
        if verbose:
            print('done')
    else:
        if verbose:
            print('Loading {} from S3 ... '.format(key), end='')
        result, cur_key = get_s3_object_bytes_with_backoff(key, bucket=bucket, num_replicas=num_replicas)
        if verbose:
            print('done', end='')
            if num_replicas > 1:
                print(' (retrieved replica {} )'.format(cur_key), end='')
            print()
    return result


def np_to_png(a, fmt='png', scale=1):
    a = np.uint8(a)
    f = io.BytesIO()
    tmp_img = PIL.Image.fromarray(a)
    tmp_img = tmp_img.resize((scale * 256, scale * 256), PIL.Image.NEAREST)
    tmp_img.save(f, fmt)
    return f.getvalue()


def make_rgb(input_image):
    if len(input_image.shape) == 3:
        if input_image.shape[2] == 4:
            return input_image[:,:,:3]
        elif input_image.shape[2] == 3:
            return input_image
        else:
            raise ValueError
    elif len(input_image.shape) == 2:
        return np.stack((input_image, input_image, input_image), axis=2)
    else:
        raise ValueError


def get_s3_tar_structure(prefix, bucket='imagenet2datav2', verbose=True):
    import pywren
    keys = list_all_keys(prefix, bucket=bucket)

    if verbose:
        print('Found {} keys:'.format(len(keys)))
        for k in keys[:10]:
            print('  ' + k)
        print('  ...')
        print('Starting PyWren ...')

    start = timer()
    pwex = pywren.default_executor()
    futures = pwex.map(list_files_in_s3_tarball, keys)
    results = pywren.get_all_results(futures)
    end = timer()

    if verbose:
        print('Done, took {} seconds'.format(end - start))

    assert len(results) == len(keys)

    final_res = {}
    for ii in range(len(keys)):
        final_res[keys[ii]] = results[ii]

    return final_res


def key_exists(bucket, key):
    # Return true if a key exists in s3 bucket
    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response['Error']['Code'] != '404':
            raise
        return False
    except:
        raise 


def get_s3_client():
    if ('imagenet2' in boto3.Session()._session.available_profiles):
        session = boto3.Session(profile_name='imagenet2')
    else:
        session = boto3.Session()
    config = Config(connect_timeout=250, read_timeout=250)
    return session.client('s3', config=config)


S3BatchResource = namedtuple('S3BatchResource', ['batch_object_key', 'key_in_batch', 'batch_type', 'size_hint'])

def get_s3_batched_objects(s3_resources,
                           bucket='imagent2datav2',
                           cache_on_local_disk=True,
                           cache_root_path=None,
                           verbose=True,
                           max_num_threads=20,
                           num_tries=40,
                           initial_delay=1.0,
                           delay_factor=math.sqrt(2.0),
                           download_callback=None):
    objects_in_batch = {}
    for obj, s3res in s3_resources.items():
        if s3res.batch_object_key not in objects_in_batch:
            objects_in_batch[s3res.batch_object_key] = []
        objects_in_batch[s3res.batch_object_key].append(obj)
    batch_keys = objects_in_batch.keys()
    batch_types = {}
    for obj, s3res in s3_resources.items():
        if s3res.batch_object_key not in batch_types:
            batch_types[s3res.batch_object_key] = s3res.batch_type
        else:
            assert batch_types[s3res.batch_object_key] == s3res.batch_type

    # TODO: sort batch_keys by size hint
    tmp_result = get_s3_object_bytes_parallel(batch_keys,
                                              bucket=bucket,
                                              cache_on_local_disk=cache_on_local_disk,
                                              cache_root_path=cache_root_path,
                                              verbose=verbose,
                                              max_num_threads=max_num_threads,
                                              num_tries=num_tries,
                                              initial_delay=initial_delay,
                                              delay_factor=delay_factor,
                                              download_callback=download_callback)

    result = {}
    for b in batch_keys:
        cur_type = batch_types[b]
        if cur_type == 'object_bytes':
            assert len(objects_in_batch[b]) == 1
            result[objects_in_batch[b][0]] = tmp_result[b]
        elif cur_type == 'numpy_bytes':
            assert len(objects_in_batch[b]) == 1
            result[objects_in_batch[b][0]] = np.load(io.BytesIO(tmp_result[b]))
        elif cur_type == 'pickle_dict':
            cur_dict = pickle.loads(tmp_result[b])
            for obj in objects_in_batch[b]:
                result[obj] = cur_dict[s3_resources[obj].key_in_batch]
        elif cur_type == 'tarball':
            tf = tarfile.open(fileobj=io.BytesIO(tmp_result[b]))
            for obj in objects_in_batch[b]:
                tarinfo = tf.getmember(s3_resources[obj].key_in_batch)
                assert tarinfo.isfile()
                result[obj] = tf.extractfile(tarinfo).read()
        else:
            raise NotImplementedError('Unknown batch resource type "{}"'.format(cur_type))
    assert len(result) == len(s3_resources)
    for obj in s3_resources.keys():
        assert obj in result
    return result



def get_s3_object_bytes_parallel(keys,
                                 bucket='imagent2datav2',
                                 cache_on_local_disk=True,
                                 cache_root_path=None,
                                 verbose=True,
                                 progress_bar=False,
                                 max_num_threads=20,
                                 num_tries=40,
                                 initial_delay=1.0,
                                 delay_factor=math.sqrt(2.0),
                                 download_callback=None):
    if cache_on_local_disk:
        if cache_root_path is None:
            cache_root_path = pathlib.Path(__file__).parent /  '../data/cache'
        cache_root_path = pathlib.Path(cache_root_path)
        cache_root_path = cache_root_path.resolve()

        missing_keys = []
        for key in keys:
            local_filepath = cache_root_path / key
            if not local_filepath.is_file():
                missing_keys.append(key)
                local_filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                if download_callback:
                    download_callback(1)

        tl = threading.local()
        def cur_download_file(key):
            local_filepath = cache_root_path / key
            if verbose:
                print('{} not available locally, downloading from S3 ... '.format(local_filepath))
            download_s3_file_with_backoff(key,
                                          str(local_filepath),
                                          bucket=bucket,
                                          num_tries=num_tries,
                                          initial_delay=initial_delay,
                                          delay_factor=delay_factor,
                                          thread_local=tl)
            return local_filepath.is_file()

        if len(missing_keys) > 0:
            download_start = timer()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_threads) as executor:
                future_to_key = {executor.submit(cur_download_file, key): key for key in missing_keys}
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        success = future.result()
                        assert success
                        if download_callback:
                            download_callback(1)
                    except Exception as exc:
                        print('Key {} generated an exception: {}'.format(key, exc))
                        raise exc
            download_end = timer()
            if verbose:
                print('Downloading took {} seconds'.format(download_end - download_start))

        result = {}
        # TODO: parallelize this as well?
        for key in keys:
            local_filepath = cache_root_path / key
            if verbose:
                print('Reading from local file {} ... '.format(local_filepath), end='')
            with open(local_filepath, 'rb') as f:
                result[key] = f.read()
            if verbose:
                print('done')
    else:
        tl = threading.local()
        def cur_get_object_bytes(key):
            if verbose:
                print('Loading {} from S3 ... '.format(key))
            return get_s3_object_bytes_with_backoff(key,
                                                    bucket=bucket,
                                                    num_tries=num_tries,
                                                    initial_delay=initial_delay,
                                                    delay_factor=delay_factor,
                                                    thread_local=tl)[0]
        
        download_start = timer()
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_threads) as executor:
            future_to_key = {executor.submit(cur_get_object_bytes, key): key for key in keys}
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result[key] = future.result()
                    if download_callback:
                        download_callback(1)
                except Exception as exc:
                    print('Key {} generated an exception: {}'.format(key, exc))
                    raise exc
        download_end = timer()
        if verbose:
            print('Getting object bytes took {} seconds'.format(download_end - download_start))
    return result


def get_labeled_candidates(imgnet, cds, mturk, threshold=0.7, wnid_threshold_delta=0.1):
    candidates_above_threshold_by_wnid = {}
    candidates_above_wnid_threshold_by_wnid = {}
    candidates = []
    pos_val_fraction_selected_by_wnid = {}

    assert len(set(imgnet.val_imgs_by_wnid.keys())) == 1000
    for wnid in imgnet.val_imgs_by_wnid.keys():
        candidates_above_threshold_by_wnid[wnid] = []
        candidates_above_wnid_threshold_by_wnid[wnid] = []
        pos_val_fraction_selected_by_wnid[wnid] = []
        for img in imgnet.val_imgs_by_wnid[wnid]:
            if img in mturk.image_fraction_selected and wnid in mturk.image_fraction_selected[img]:
                pos_val_fraction_selected_by_wnid[wnid].append(mturk.image_fraction_selected[img][wnid])
        cur_wnid_threshold = np.mean(pos_val_fraction_selected_by_wnid[wnid])
        for cc in cds.candidates_by_wnid[wnid]:
            cur_cid = cc['id_ours']
            if cur_cid in mturk.image_fraction_selected and wnid in mturk.image_fraction_selected[cur_cid] and mturk.image_fraction_selected[cur_cid][wnid] >= threshold:
                candidates_above_threshold_by_wnid[wnid].append(cc)
                candidates.append(cur_cid)
            if cur_cid in mturk.image_fraction_selected and wnid in mturk.image_fraction_selected[cur_cid] and mturk.image_fraction_selected[cur_cid][wnid] >= cur_wnid_threshold - wnid_threshold_delta:
                candidates_above_wnid_threshold_by_wnid[wnid].append(cc)
                candidates.append(cur_cid)
    return sorted(list(set(candidates)))



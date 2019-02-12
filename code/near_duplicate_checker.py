import boto3
import imageio
import io
from collections import defaultdict
import numpy as np
import os
import pywren
from pywren import wrenconfig as wc
from skimage.measure import compare_ssim
import sys
import tarfile
import json
import argparse
from timeit import default_timer as timer
from collections import namedtuple
import numpy as np
import hashlib
import pickle
import time
import itertools
import logging
from six import reraise

import imagenet
import mturk_data
import utils
import candidate_data
import image_loader
import dataset_sampling

#f = open("log.txt", "w+")

#logging.basicConfig(stream=f, level=logging.DEBUG)

distance_metrics = ['l2', 'dssim', 'fc7']
BUCKET = "imagenet2datav2"
ALWAYS = 3

def get_all_reference_filenames(imgnt, cds, mturk):
    all_references = []
    all_references.extend(imgnt.get_all_train_image_names())
    all_references.extend(imgnt.get_all_val_image_names())
    all_references.extend(imgnt.test_filenames)
    #all_references.extend(utils.get_labeled_candidates(imgnt, cds, mturk))
    all_references.extend(dataset_sampling.get_histogram_sampling_ndc_candidates(imgnet=imgnt, cds=cds, mturk=mturk))
    #all_references.extend(list(cds.all_candidates.keys()))
    return all_references

def get_reference_filenames_for_wnid(wnid, imgnt):
  with open('../data/metadata/wnid_to_most_similar_wnids.json', 'r') as fp:
      similarity_dict = json.load(fp)
  similar_wnids = similarity_dict[wnid]
  filenames = []
  for wnid in similar_wnids[:6]:
      filenames.extend(imgnt.train_imgs_by_wnid[wnid])
  return filenames

def group_candidates_by_wnid(filenames, cds):
    wnid_to_cd = {}
    for f in filenames:
        wnid = cds.all_candidates[f]['wnid']
        if wnid in wnid_to_cd:
            wnid_to_cd[wnid].append(f)
        else:
            wnid_to_cd[wnid] = [f]
    return wnid_to_cd

def dssim_distance_function(image1, image2, window_size):
    return (1-compare_ssim(image1.reshape(256, 256, 3),
                                  image2.reshape(256, 256, 3),
                                  multichannel=True,
                                  win_size=window_size))/2

def l2_distance_function(X, Y, batch_size=250):
    X_chunks = list(utils.chunks(X, batch_size))
    Y_chunks = list(utils.chunks(Y, batch_size))
    X_chunk_idxs = list(utils.chunks(range(len(X)), batch_size))
    Y_chunk_idxs = list(utils.chunks(range(len(Y)), batch_size))
    D = np.zeros((len(X), len(Y)), dtype='float64')
    i = 0
    for x_chunk, x_chunk_idxs in zip(X_chunks, X_chunk_idxs):
      for y_chunk, y_chunk_idxs  in zip(Y_chunks, Y_chunk_idxs):
        i += 1
        x_ = np.vstack(x_chunk).astype('float64').reshape(len(x_chunk), -1)
        y_ = np.vstack(y_chunk).astype('float64').reshape(len(y_chunk), -1)
        idxs = list(itertools.product(x_chunk_idxs, y_chunk_idxs))
        D[x_chunk_idxs[0]:x_chunk_idxs[-1]+1, y_chunk_idxs[0]:y_chunk_idxs[-1]+1] = blas_l2_distance(x_, y_)
    return D

def blas_l2_distance(x, y):
    d = x.dot(y.T)
    d *= -2
    d += np.linalg.norm(x, axis=1)[:, np.newaxis]**2
    d += np.linalg.norm(y, axis=1)[:, np.newaxis].T**2
    return d


def compute_top_k(input_dict, top_k):
    """ Returns a list of tuples where the first tuple element is the image name
    and the second is the l2 distance."""
    all_results = {}
    for distance_metric, distance_dict in input_dict.items():
        top_k_results = {}
        for cd_name, ref_dists in distance_dict.items():
            top_k_results[cd_name] = []
            ref_filenames, ref_dists = zip(*ref_dists.items())
            top_indices = np.argsort(ref_dists)
            for jj in range(min(top_k, len(top_indices))):
                cur_index = top_indices[jj]
                top_k_results[cd_name].append(
                    (ref_filenames[cur_index], ref_dists[cur_index])
                )
        all_results[distance_metric] = top_k_results
    return all_results

def compute_l2_distances(candidate_image_dict, reference_image_dict, img_size):
    candidate_filenames, candidate_images = zip(*list(candidate_image_dict.items()))
    reference_filenames, reference_images = zip(*list(reference_image_dict.items()))
    dst_matrix = l2_distance_function(candidate_images,
                                      reference_images)
    result = {}
    for ii, cd_name in enumerate(candidate_filenames):
        distances = dst_matrix[ii, :]
        result[cd_name] = {}
        for jj, ref_name in enumerate(reference_filenames):
            result[cd_name][ref_name] = distances[jj]
    return result

def compute_dssim_distances(candidate_image_dict, reference_image_dict, window_size):
    result = {}
    for cd_name, cd_image in candidate_image_dict.items():
        result[cd_name] = {}
        for ref_name, ref_image in reference_image_dict.items():
            result[cd_name][ref_name] = dssim_distance_function(cd_image,
                                                                 ref_image, window_size)
    return result

def compute_hash(distance_measures, candidate_filenames, reference_filenames, top_k, window_size):
    sha256 = hashlib.sha256()
    sha256.update(str(','.join(sorted(distance_measures))).encode())
    sha256.update(str(len(candidate_filenames)).encode()) 
    for c in candidate_filenames:
      sha256.update(str(c).encode())
    sha256.update(str(len(reference_filenames)).encode()) 
    for r in reference_filenames:
      sha256.update(str(r).encode())
    sha256.update(str('top_k').encode())
    sha256.update(str(top_k).encode())
    sha256.update(str('window_size').encode())
    sha256.update(str(window_size).encode())
    return sha256.hexdigest()

def compute_nearest_neighbors(distance_measures, candidate_filenames, reference_filenames, top_k, window_size, cache, cache_root):
    cache_key = compute_hash(distance_measures, candidate_filenames, reference_filenames, top_k, window_size)
    full_key = f"{cache_root}/{cache_key}"
    timing_info = {}
    if cache:
        if utils.key_exists(BUCKET, full_key):
            load_start = timer()
            ret_value = pickle.loads(utils.get_s3_object_bytes_with_backoff(full_key)[0])
            load_end = timer()
            compute_start = compute_end = timer()
            timing_info['load_start'] = load_start
            timing_info['load_end'] = load_end
            timing_info['compute_start'] = compute_start
            timing_info['compute_end'] = compute_end
            timing_info['cached'] = True
            return ret_value, timing_info

    imgnt = imagenet.ImageNetData(cache_on_local_disk=True, verbose=False, cache_root_path='/tmp/imagenet2_cache')
    cds = candidate_data.CandidateData(cache_on_local_disk=True, load_metadata_from_s3=True, verbose=False, cache_root_path='/tmp/imagenet2_cache')
    loader = image_loader.ImageLoader(imgnt, cds, cache_on_local_disk=True, num_tries=4, cache_root_path='/tmp/imagenet2_cache')
    load_start = timer()
    if ('l2' in distance_measures) or ('dssim' in distance_measures):
        candidate_image_dict = loader.load_image_batch(candidate_filenames, size='scaled_256', force_rgb=True, verbose=False)
        reference_image_dict = loader.load_image_batch(reference_filenames, size='scaled_256', force_rgb=True, verbose=False)
    if 'fc7' in distance_measures:
        candidate_feature_dict = loader.load_features_batch(candidate_filenames, verbose=False)
        reference_feature_dict = loader.load_features_batch(reference_filenames, verbose=False)
    load_end = timer()

    compute_start = timer()
    result = {}
    for distance_measure in distance_measures:
        if distance_measure == 'l2':
            result['l2'] = compute_l2_distances(candidate_image_dict, reference_image_dict, 196608)
        elif distance_measure == 'dssim':
            result['dssim'] = compute_dssim_distances(candidate_image_dict, reference_image_dict, window_size)
        elif distance_measure == 'fc7':
            result['fc7'] = compute_l2_distances(candidate_feature_dict, reference_feature_dict, 4096)
        else:
            raise ValueError('Unknown distance measure')
    compute_end = timer()
    timing_info = {}
    timing_info['load_start'] = load_start
    timing_info['load_end'] = load_end
    timing_info['compute_start'] = compute_start
    timing_info['compute_end'] = compute_end
    timing_info['cached'] = False

    res = compute_top_k(result, top_k)
    if cache:
        utils.put_s3_object_bytes_with_backoff(pickle.dumps(res), full_key)

    return res, timing_info

def merge_result(result, top_k):
    '''Result is a list of key-value dictionaries. key is a image name, value is a list of tuples.'''
    merged_result = defaultdict(list)
    for image_dict in result:
        for k, v in image_dict.items():
            merged_result[k].extend(v)

    for k, v in merged_result.items():
        v.sort(key=lambda tup: tup[1])
        merged_result[k] = v[0:top_k]
    return merged_result

def get_cd_ref_pairs(cd_keys, ref_filenames, cd_chunk_size, ref_chunk_size):
    cd_ref_pairs = []
    for cd_keys_chunk in utils.chunks(cd_keys, cd_chunk_size):
        for ref_filenames_chunk in utils.chunks(ref_filenames, ref_chunk_size):
            cd_ref_pairs.append((cd_keys_chunk, ref_filenames_chunk))
    return cd_ref_pairs


def wait_for_futures(futures, print_frequency=100, raise_exception=True):
    results = []
    retrieved = {}
    call_id_to_failed_future = {}
    done, not_dones = pywren.wait(futures, ALWAYS)
    while len(not_dones) > 0:
        done, not_dones = pywren.wait(futures, ALWAYS)
        for finished_future in done:
            #print('finished future')
            if finished_future not in retrieved:
                try:
                    if "stdout" in finished_future.run_status:
                        if len(finished_future.run_status["stdout"]) > 0:
                            print(finished_future.run_status["stdout"])
                    #print('Adding finished future to result')
                    results.append(finished_future.result())
                    retrieved[finished_future] = True
                    if len(retrieved) % print_frequency == 0:
                        timing_results = [timing_info for _, timing_info in results]
                        summarize_timing_infos(timing_results)
                except:
                    if finished_future._traceback is not None:
                        print("Future exception traceback was ", finished_future._traceback)
                    print('future call_id {} failed'.format(finished_future.call_id))
                    retrieved[finished_future] = True
                    call_id_to_failed_future[int(finished_future.call_id)] = finished_future
                    if raise_exception:
                        reraise(finished_future._traceback[0], finished_future._traceback[1], finished_future._traceback[2])
    return results, call_id_to_failed_future

def get_near_duplicates_chunks(cd_ref_pairs,
                               imgnt,
                               cds,
                               top_k,
                               dssim_window_size,
                               use_pywren,
                               return_ndc_results,
                               distance_metrics=['l2', 'fc7', 'dssim'],
                               cache=True,
                               cache_root="ndc_cache"):
    pywren_config = wc.default()
    pywren_config["runtime"]["s3_bucket"] = "imagenet2pywren"
    pywren_config["runtime"]["s3_key"] = "pywren.runtime/pywren_runtime-3.6-imagenet2pywren.meta.json"

    def get_results(cd_ref_pair):
        candidates, references = cd_ref_pair
        result = compute_nearest_neighbors(distance_metrics,
                                         candidates,
                                         references,
                                         top_k,
                                         dssim_window_size,
                                         cache,
                                         cache_root)
        if return_ndc_results:
            return result
        else:
            return None, result[1]
    
    if use_pywren:
        pwex = pywren.standalone_executor(config=pywren_config)
        print("pywren config", pwex.config)
        print('Number of pywren calls', len(cd_ref_pairs))
        extra_env = {"AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"], 
                     "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"], 
                     "OMP_NUM_THREADS": "1", 
                     "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"]}
        futures = pwex.map(get_results, cd_ref_pairs, exclude_modules=["site-packages"], extra_env=extra_env)
        all_results = []
        print('Waiting for futures')
        results, call_id_to_failed_future = wait_for_futures(futures, print_frequency=100, raise_exception=False)
        print('Got {} results'.format(len(results)))
        print('{} futures failed'.format(len(call_id_to_failed_future)))
        all_results.extend(results)
        failed_cd_ref_pairs = []
        for call_id, failed_future in call_id_to_failed_future.items():
            failed_cd_ref_pairs.append(cd_ref_pairs[int(call_id)])
        if len(failed_cd_ref_pairs) > 0:
            # Retry failed futures
            print('Retrying {} failed futures'.format(len(failed_cd_ref_pairs)))
            futures = pwex.map(get_results, failed_cd_ref_pairs, exclude_modules=["site-packages"], extra_env=extra_env)
            results, call_id_to_failed_future = wait_for_futures(futures, print_frequency=1, raise_exception=True)
            all_results.extend(results)
        #assert len(all_results) == len(cd_ref_pairs)
        print('Retrieved {} results'.format(len(all_results)))
    else:
        all_results = []
        for cd_ref_pair in cd_ref_pairs:
            all_results.append(get_results(cd_ref_pair))
    assert len(all_results) == len(cd_ref_pairs)
    timing_results = [timing_info for _, timing_info in all_results]
    
    start = timer()
    print('Beginning merge') 
    merged_results = {}
    for d in distance_metrics:
        merged_results[d] = {}
    if return_ndc_results:
        dist_results = [dist_res for dist_res, _ in all_results]
        separated_results = {}
        for d in distance_metrics:
            separated_results[d] = []
        for r in dist_results:
            for d, distance_dict in r.items():
                separated_results[d].append(distance_dict)

        for d in distance_metrics:
            merged_results[d] = merge_result(separated_results[d], top_k)
    end = timer()
    print('Merge took {} seconds'.format(end-start))
    return merged_results, timing_results


def get_near_duplicates(candidate_filenames,
                        reference_filenames,
                        imgnt,
                        cds,
                        top_k,
                        dssim_window_size,
                        use_pywren,
                        ref_chunk_size,
                        cd_chunk_size,
                        return_ndc_results,
                        distance_metrics =['l2', 'dssim', 'fc7'],
                        cache=True,
                        cache_root="ndc_cache"):
    cd_ref_pairs = get_cd_ref_pairs(candidate_filenames, reference_filenames, cd_chunk_size, ref_chunk_size)
    return get_near_duplicates_chunks(cd_ref_pairs, imgnt, cds, top_k, dssim_window_size, use_pywren,
                                     return_ndc_results, distance_metrics, cache, cache_root)

def select_test_candidates(imgnt, cds, mturk, nn_results, args):
    result = {}
    cd_per_dist_counter = {}
    for d in distance_metrics:
        result[d] = []
        cd_per_dist_counter[d] =  0

    num_cds = 0
    #candidate_list = utils.get_labeled_candidates(imgnt, cds, mturk)
    candidate_list = dataset_sampling.get_histogram_sampling_ndc_candidates(imgnet=imgnt, cds=cds, mturk=mturk)
    grouped_cds = group_candidates_by_wnid(candidate_list, cds)
    
    sorted_candidate_list = []
    for wnid in grouped_cds:
        sorted_candidate_list.extend(list(sorted(grouped_cds[wnid])))
    sorted_candidate_list = sorted_candidate_list
    #candidate_list = list(cds.all_candidates.keys())
    for cd_id in sorted_candidate_list:
        cd_added = False
        if cd_id in nn_results:
            for d in distance_metrics:
                if d in nn_results[cd_id]:
                    #print('d is {}'.format(d))
                    #print('Top k is {}'.format(args.top_k))
                    #print('Len nearest neighbors is {}'.format(len(nn_results[cd_id][d])))
                    assert len(nn_results[cd_id][d]) == args.top_k
                else:
                    cd_per_dist_counter[d] += 1
                    if num_cds < args.max_num_candidates and d in args.metrics:
                        # print('Adding cd {} for metric {}'.format(cd_id, d))
                        result[d].append(cd_id)
                        cd_added = True
        else:
            for d in distance_metrics:
                cd_per_dist_counter[d] += 1
                if num_cds < args.max_num_candidates and d in args.metrics:
                    #print('Adding cd {} for metric {}'.format(cd_id, d))
                    result[d].append(cd_id)
                    cd_added = True
        if cd_added:
            num_cds += 1
    with open('../data/metadata/fc7_candidates.json', 'w') as f:
        json.dump(result['fc7'], f, indent=2)
    return result, cd_per_dist_counter

def compute_distances_for_all_references(candidates,
                                         distance_metric,
                                         imgnt,
                                         cds,
                                         mturk,
                                         args):
    references = get_all_reference_filenames(imgnt, cds, mturk)
    if args.max_num_references is not None:
        if len(references) > args.max_num_references:
            print('WARNING: Truncating references')
        references = references[:args.max_num_references]
    print()
    print('Checking {} candidates against {} references {}'.format(len(candidates),
            len(references), distance_metric))
    print()
    result, timing_infos = get_near_duplicates(candidates,
                                 references,
                                 imgnt,
                                 cds,
                                 top_k=args.top_k,
                                 dssim_window_size=args.dssim_window_size,
                                 use_pywren=args.use_pywren,
                                 ref_chunk_size=args.ref_chunk_size,
                                 cd_chunk_size=args.cd_chunk_size,
                                 return_ndc_results=args.return_ndc_results,
                                 distance_metrics=[distance_metric],
                                 cache=(not args.no_cache),
                                 cache_root=args.cache_root)

    summarize_timing_infos(timing_infos)
    assert len(result) == 1
    return result[distance_metric], timing_infos

def print_nn_stats(nn_result):
    for metric in distance_metrics:
        counter = 0
        for cd in nn_result:
            for d in nn_result[cd]:
                if d == metric:
                    counter += 1
        print('{} candidates for metric {}'.format(counter, metric))

def compute_distances_for_wnid_references(candidates, distance_metric, imgnt, cds, mturk, args):
    grouped_cds = group_candidates_by_wnid(candidates, cds)
    result = {}
    timing_infos_by_wnid = {}
    print('{} wnids for candidate chunk'.format(len(grouped_cds)))

    cd_ref_pairs = []
    for wnid in grouped_cds:
        references = []
        references.extend(get_reference_filenames_for_wnid(wnid, imgnt))
        references.extend(imgnt.test_filenames)
        references.extend(imgnt.get_all_val_image_names())
        #references.extend(utils.get_labeled_candidates(imgnt, cds, mturk))
        references.extend(dataset_sampling.get_histogram_sampling_ndc_candidates(imgnet=imgnt, cds=cds, mturk=mturk))
        candidates_for_wnid = grouped_cds[wnid]
        if args.max_num_references is not None:
            if len(references) > args.max_num_references:
                print('WARNING: Truncating references')
            references = references[:args.max_num_references]

        print('Checking {} candidates from wnid {} against {} references ({})'.format(len(candidates_for_wnid),
                                                                             wnid,
                                                                             len(references),
                                                                             distance_metric))
        cd_ref_pairs.extend(get_cd_ref_pairs(candidates_for_wnid, references, args.ref_chunk_size, args.cd_chunk_size))



    iresult, timing_infos = get_near_duplicates_chunks(cd_ref_pairs,
                                                       imgnt,
                                                       cds,
                                                       top_k=args.top_k,
                                                       dssim_window_size=args.dssim_window_size,
                                                       use_pywren=args.use_pywren,
                                                       return_ndc_results=args.return_ndc_results,
                                                       distance_metrics=[distance_metric])
    assert len(iresult) == 1
    print('Final summarization of timing')
    summarize_timing_infos(timing_infos)
    return iresult[distance_metric]

def save_ndc_result(result, nn_results, metric, args):
    for cd, nn_list in result.items():
        if cd in nn_results:
            if metric in nn_results[cd]:
                print('WARNING: Replacing {} nearest neighbors for {} and distance {}'.format(len(nn_list), cd,
                 metric))
                nn_results[cd][metric] = nn_list
            else:
                nn_results[cd][metric] = nn_list
        else:
            nn_results[cd] = {}
            nn_results[cd][metric] = nn_list
    with open(args.output_filename, 'wb') as fp:
        #json.dump(nn_results, fp, indent=2)
        pickle.dump(nn_results, fp)
    return nn_results


def summarize_timing_infos(timing_infos, use_tqdm=False):
    load_times_cached = []
    load_times_non_cached = []
    compute_times_cached = []
    compute_times_non_cached = []
    for ti in timing_infos:
        if ti['cached']:
            load_times_cached.append(ti['load_end'] - ti['load_start'])
            compute_times_cached.append(ti['compute_end'] - ti['compute_start'])
        else: 
            load_times_non_cached.append(ti['load_end'] - ti['load_start'])
            compute_times_non_cached.append(ti['compute_end'] - ti['compute_start'])
    print('Timing statistics: {} cached, {} non-cached'.format(len(load_times_cached), len(load_times_non_cached)))

    if len(load_times_cached) == 0:
        load_times_cached = [0.0]
    if len(compute_times_cached) == 0:
        compute_times_cached = [0.0]
    if len(load_times_non_cached) == 0:
        load_times_non_cached = [0.0]
    if len(compute_times_non_cached) == 0:
        compute_times_non_cached = [0.0]
    
    print('Load times (cached): min {:.3f}, max {:.3f}, avg {:.3f}, med {:.3f}'.format(np.min(load_times_cached),
                                                            np.max(load_times_cached),
                                                            np.mean(load_times_cached),
                                                            np.median(load_times_cached)))
    print('Compute times (cached): min {:.3f}, max {:.3f}, avg {:.3f}, med {:.3f}'.format(np.min(compute_times_cached),
                                                            np.max(compute_times_cached),
                                                            np.mean(compute_times_cached),
                                                            np.median(compute_times_cached)))
    print('Load times (non_cached): min {:.3f}, max {:.3f}, avg {:.3f}, med {:.3f}'.format(np.min(load_times_non_cached),
                                                            np.max(load_times_non_cached),
                                                            np.mean(load_times_non_cached),
                                                            np.median(load_times_non_cached)))
    print('Compute times (non_cached): min {:.3f}, max {:.3f}, avg {:.3f}, med {:.3f}'.format(np.min(compute_times_non_cached),
                                                            np.max(compute_times_non_cached),
                                                            np.mean(compute_times_non_cached),
                                                            np.median(compute_times_non_cached)))
    sys.stdout.flush()
    return 0


def main(args):
    imgnt = imagenet.ImageNetData(verbose=False)
    cds = candidate_data.CandidateData(verbose=False)
    filenames_to_ignore = [
        '2018-08-06_17:33_vaishaal.json',
        '2018-08-17_17:24_vaishaal.json',
        'vaishaal_hits_submitted_2018-08-17-18:28:33-PDT.json',
        'vaishaal_hits_submitted_2018-08-17-18:50:38-PDT.json',
        'vaishaal_hits_submitted_2018-08-17-19:28:24-PDT.json',
        'vaishaal_hits_submitted_2018-08-17-19:56:28-PDT.json',
        'vaishaal_hits_submitted_2018-08-25-09:47:26-PDT.json']
    mturk = mturk_data.MTurkData(live=True, load_assignments=True, source_filenames_to_ignore=filenames_to_ignore, verbose=False)
    
    with open(args.input_filename, 'rb') as fp:
        nn_results = pickle.load(fp)
    print('Current nearest neighbor statistics')
    print_nn_stats(nn_results)
    print()

    metric_to_cd, cd_to_dist_counter = select_test_candidates(imgnt, cds, mturk, nn_results, args)

    print('Remaining candidate distances to compute')
    for d, num_cds in cd_to_dist_counter.items():
        print('{} cds left for metric {}'.format(num_cds, d))
    print()
    print('Computing neighbors for ')
    for d in distance_metrics:
          print('{} candidates in metric {}.'.format(len(metric_to_cd[d]), d))
    print()

    for metric in args.metrics:
        if metric == 'l2' or metric == 'fc7':
            candidates = metric_to_cd[metric]
            result, _ = compute_distances_for_all_references(candidates, metric, imgnt, cds, mturk, args)
            if len(result) != len(candidates):
                print('WARNING: len(result) {} len(candidates) {}'.format(len(result), len(candidates)))
                #assert len(result) == len(candidates)
            nn_results = save_ndc_result(result, nn_results, metric, args)
    for metric in args.metrics:
        if metric == 'dssim':
            print('Computing distances for dssim')
            candidates = metric_to_cd[metric]
            result = compute_distances_for_wnid_references(candidates, metric, imgnt, cds, mturk, args)
            if len(result) != len(candidates):
                print('WARNING: len(result) {} len(candidates) {}'.format(len(result), len(candidates)))
                #assert len(result) == len(candidates)
            print('Saving results')
            start = timer()
            nn_results = save_ndc_result(result, nn_results, metric, args)
            end = timer()
            print('Saving the results took {} seconds'.format(end-start))
    num_candidates_left = {}
    for d in args.metrics:
       num_candidates_left[d] = cd_to_dist_counter[d]
    return num_candidates_left

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run near duplicate checker")
    parser.add_argument("--max_num_candidates", default=None, type=int)
    parser.add_argument("--max_num_references", default=None, type=int)
    parser.add_argument("--output_filename", default=None, type=str)
    parser.add_argument("--input_filename", default=None, type=str)
    parser.add_argument("--use_pywren", default=False, action="store_const", const=True)
    parser.add_argument("--return_ndc_results", default=True, action="store_const", const=True)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--dssim_window_size", default=35, type=int)
    parser.add_argument("--ref_chunk_size", default=150, type=int)
    parser.add_argument("--cd_chunk_size", default=150, type=int)
    parser.add_argument("--metrics", default=None, type=str)
    parser.add_argument("--no_cache", default=False, action="store_const", const=True)
    parser.add_argument("--cache_root", default="ndc_cache", type=str)

    args = parser.parse_args()
    assert args.input_filename is not None
    assert args.output_filename is not None
    args.metrics = args.metrics.split(",")
    for metric in args.metrics:
        assert metric in distance_metrics
    assert len(set(args.metrics)) == len(args.metrics)
    print('using pywren: ', args.use_pywren)
    print('cache : ', not args.no_cache)
    main(args)

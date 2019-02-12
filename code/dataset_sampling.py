from collections import OrderedDict
import copy
from datetime import datetime, timezone
import getpass
import json
import random

import numpy as np

import candidate_data
import imagenet
import mturk_data
import near_duplicate_data


standard_bins = [0.2, 0.4, 0.6, 0.8]


def build_histogram(freqs, bins):
    num_bins = len(bins) + 1
    result = np.zeros(num_bins)
    for f in freqs:
        found_bin = False
        for ii in range(num_bins - 1):
            if f < bins[ii]:
                result[ii] += 1
                found_bin = True
                break
        if not found_bin:
            result[-1] += 1
    return result


def round_histogram(hist, target_sum):
    fractional_hist = target_sum * hist / np.sum(hist)
    floor_hist = np.floor(fractional_hist)
    floor_sum = int(np.round(np.sum(floor_hist)))
    remainder_hist = fractional_hist - floor_hist
    remainder = target_sum - floor_sum
    top_buckets = list(reversed(sorted(enumerate(remainder_hist), key=lambda x:(x[1], x[0]))))
    result = np.copy(floor_hist).astype(np.int64)
    for ii in range(remainder):
        result[top_buckets[ii][0]] += 1
    return result


def get_time_string():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d_%H:%M:%S_%Z')


def sample_val_dummy(dataset_size, seed):
    num_classes = 1000
    assert dataset_size % num_classes == 0
    num_per_class = dataset_size // num_classes
    rng = random.Random(seed)
    imgnet = imagenet.ImageNetData()

    dataset_images = []
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    for wnid in all_wnids:
        images_for_wnid = list(sorted(imgnet.val_imgs_by_wnid[wnid]))
        cur_images = rng.sample(images_for_wnid, num_per_class)
        dataset_images.extend(sorted([(x, wnid) for x in cur_images]))
    assert len(dataset_images) == dataset_size

    result = {}
    result['sampling_function'] = 'sample_val_dummy'
    result['time_string'] = get_time_string()
    result['username'] = getpass.getuser()
    result['seed'] = seed
    result['image_filenames'] = dataset_images
    return result


def sample_val_annotated(dataset_size, min_num_annotations, seed):
    num_classes = 1000
    assert dataset_size % num_classes == 0
    num_per_class = dataset_size // num_classes
    rng = random.Random(seed)
    imgnet = imagenet.ImageNetData()
    mturk = mturk_data.MTurkData(
            live=True,
            load_assignments=True,
            source_filenames_to_ignore=mturk_data.main_collection_filenames_to_ignore)

    dataset_images = []
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    for wnid in all_wnids:
        valid_images_for_wnid = []
        for img in imgnet.val_imgs_by_wnid[wnid]:
            if img in mturk.image_num_assignments and wnid in mturk.image_num_assignments[img] and mturk.image_num_assignments[img][wnid] >= min_num_annotations:
                valid_images_for_wnid.append(img)
        valid_images_for_wnid = sorted(valid_images_for_wnid)
        assert len(valid_images_for_wnid) >= num_per_class
        cur_images = rng.sample(valid_images_for_wnid, num_per_class)
        dataset_images.extend(sorted([(x, wnid) for x in cur_images]))
    
    rng.shuffle(dataset_images)
    assert len(dataset_images) == dataset_size

    result = {}
    result['sampling_function'] = 'sample_val_annotated'
    result['min_num_annotations'] = min_num_annotations
    result['time_string'] = get_time_string()
    result['username'] = getpass.getuser()
    result['seed'] = seed
    result['image_filenames'] = dataset_images
    return result


def get_prev_dataset_by_wnid(starting_from, dataset_size, all_wnids, cds):
    num_classes = 1000
    num_per_class = dataset_size // num_classes
    if starting_from is None:
        prev_dataset_by_wnid = {}
        for wnid in all_wnids:
            prev_dataset_by_wnid[wnid] = []
    else:
        prev_filenames = starting_from['image_filenames']
        assert len(prev_filenames) <= dataset_size
        prev_dataset_by_wnid = {}
        for img, wnid in prev_filenames:
            assert wnid in all_wnids
            if wnid not in prev_dataset_by_wnid:
                prev_dataset_by_wnid[wnid] = []
            prev_dataset_by_wnid[wnid].append(img)
            assert cds.all_candidates[img]['wnid'] == wnid
        assert len(prev_dataset_by_wnid) <= num_classes
        for wnid in prev_dataset_by_wnid.keys():
            assert len(prev_dataset_by_wnid[wnid]) <= num_per_class
    return prev_dataset_by_wnid


def get_histogram_bin(x, bins):
    assert x >= 0.0 and x <= 1.0
    for ii, boundary in enumerate(bins):
        if x < boundary:
            return ii
    return len(bins)


def get_bin_boundaries(bins, ii):
    assert ii >= 0 and ii <= len(bins)
    if ii == 0:
        return 0.0, bins[0]
    elif ii == len(bins):
        return bins[-1], 1.0
    else:
        return bins[ii - 1], bins[ii]


def compute_wnid_histogram(*,
                           wnid,
                           imgnet,
                           mturk,
                           min_num_annotations_val,
                           min_num_val_images_per_wnid,
                           histogram_bins=None,
                           num_per_class):
    val_freqs = []
    for img in imgnet.val_imgs_by_wnid[wnid]:
        if img in mturk.image_num_assignments and wnid in mturk.image_num_assignments[img]:
            if mturk.image_num_assignments[img][wnid] >= min_num_annotations_val:
                val_freqs.append(mturk.image_fraction_selected[img][wnid])
    if len(val_freqs) < min_num_val_images_per_wnid:
        return None
    tmp_hist = build_histogram(val_freqs, histogram_bins)
    return round_histogram(tmp_hist, num_per_class)


def compute_candidate_wnid_histogram(*,
                                     wnid,
                                     cds,
                                     mturk,
                                     ndc,
                                     min_num_annotations,
                                     histogram_bins,
                                     include_blacklisted=False,
                                     include_near_duplicates=False):
    val_freqs = []
    for cand in cds.candidates_by_wnid[wnid]:
        cid = cand['id_ours']
        if not include_blacklisted and cid in cds.blacklist:
            continue
        if not include_near_duplicates and ndc.is_near_duplicate[cid]:
            continue
        if cid in mturk.image_num_assignments and wnid in mturk.image_num_assignments[cid]:
            if mturk.image_num_assignments[cid][wnid] >= min_num_annotations:
                cur_freq = mturk.image_fraction_selected[cid][wnid]
                val_freqs.append(cur_freq)
    return build_histogram(val_freqs, histogram_bins)
    

def compute_wnid_histograms(*,
                            imgnet,
                            mturk,
                            min_num_annotations_val,
                            min_num_val_images_per_wnid,
                            histogram_bins,
                            num_per_class):
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    success = True
    usable_val_imgs_by_wnid = {}
    wnid_histograms = {}
    for wnid in all_wnids:
        val_freqs = []
        usable_val_imgs_by_wnid[wnid] = []
        for img in imgnet.val_imgs_by_wnid[wnid]:
            if img in mturk.image_num_assignments and wnid in mturk.image_num_assignments[img]:
                if mturk.image_num_assignments[img][wnid] >= min_num_annotations_val:
                    val_freqs.append(mturk.image_fraction_selected[img][wnid])
                    usable_val_imgs_by_wnid[wnid].append(img)
        if len(val_freqs) < min_num_val_images_per_wnid:
            success = False
        tmp_hist = build_histogram(val_freqs, histogram_bins)
        wnid_histograms[wnid] = round_histogram(tmp_hist, num_per_class)
    return success, wnid_histograms, usable_val_imgs_by_wnid


def get_histogram_sampling_ndc_candidates(*,
                                          imgnet,
                                          cds,
                                          mturk,
                                          min_num_annotations_val=10,
                                          min_num_val_images_per_wnid=20,
                                          histogram_bins=standard_bins,
                                          num_per_class=10):
    result = []
    histograms_success, wnid_histograms, _ = compute_wnid_histograms(
            imgnet=imgnet,
            mturk=mturk,
            min_num_annotations_val=min_num_annotations_val,
            min_num_val_images_per_wnid=min_num_val_images_per_wnid,
            histogram_bins=histogram_bins,
            num_per_class=num_per_class)
    assert histograms_success
    for cid, cand in cds.all_candidates.items():
        if cid in cds.blacklist:
            continue
        wnid = cand['wnid']
        if cid not in mturk.image_fraction_selected:
            cur_freq = 0.0
        else:
            cur_freq = mturk.image_fraction_selected[cid][wnid]
        cur_bin = get_histogram_bin(cur_freq, histogram_bins)
        if wnid_histograms[wnid][cur_bin] > 0:
            result.append(cid)
    return list(sorted(set(result)))


def sample_best(*,
                dataset_size,
                min_num_annotations,
                near_duplicate_review_targets,
                seed,
                starting_from=None):
    rng = random.Random(seed)
    num_classes = 1000
    assert dataset_size % num_classes == 0
    for metric in near_duplicate_data.metric_names:
        assert metric in near_duplicate_review_targets
    assert len(near_duplicate_review_targets) == len(near_duplicate_data.metric_names)
    num_per_class = dataset_size // num_classes
    imgnet = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(load_metadata_from_s3=False,
                                       exclude_blacklisted_candidates=False)
    mturk = mturk_data.MTurkData(
            live=True,
            load_assignments=True,
            source_filenames_to_ignore=mturk_data.main_collection_filenames_to_ignore)
    ndc = near_duplicate_data.NearDuplicateData(
            imgnet=imgnet,
            candidates=cds,
            mturk_data=mturk,
            load_review_thresholds=True)
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))

    def is_cid_ok(cid, wnid):
        if cid in cds.blacklist:
            return False, 'blacklisted'
        if cid not in mturk.image_num_assignments:
            return False, 'few_assignments'
        if wnid not in mturk.image_num_assignments[cid]:
            return False, 'few_assignments'
        if mturk.image_num_assignments[cid][wnid] < min_num_annotations:
            return False, 'few_assignments'
        if ndc.is_near_duplicate[cid]:
            return False, 'near_duplicate'
        sufficiently_reviewed = True
        if cid not in ndc.review_threshold:
            sufficiently_reviewed = False
        else:
            for metric in near_duplicate_data.metric_names:
                if metric not in ndc.review_threshold[cid]:
                    sufficiently_reviewed = False
                elif ndc.review_threshold[cid][metric] <= near_duplicate_review_targets[metric]:
                    sufficiently_reviewed = False
        if not sufficiently_reviewed:
            return False, 'unreviewed'
        return True, None
    prev_dataset_by_wnid = get_prev_dataset_by_wnid(starting_from, dataset_size, all_wnids, cds)

    def sorting_fn(cid, wnid):
        in_prev_indicator = 0
        if cid in prev_dataset_by_wnid[wnid]:
            in_prev_indicator = 1
        return mturk.image_fraction_selected[cid][wnid], mturk.image_num_assignments[cid][wnid], in_prev_indicator, cid

    dataset_images = []
    sampling_candidates = {}
    exclusions = {}
    success = True
    carried_over_from_prev = {}
    for wnid in all_wnids:
        sampling_candidates[wnid] = []
        exclusions[wnid] = OrderedDict([('blacklisted', []),
                                        ('few_assignments', []),
                                        ('below_threshold', []),
                                        ('near_duplicate', []),
                                        ('unreviewed', [])])
        for cand in cds.candidates_by_wnid[wnid]:
            cid = cand['id_ours']
            cur_ok, cur_reason = is_cid_ok(cid, wnid)
            if cur_ok:
                sampling_candidates[wnid].append(cid)
            else:
                exclusions[wnid][cur_reason].append(cid)
        sampling_candidates[wnid] = list(reversed(sorted(sampling_candidates[wnid], key=lambda x: sorting_fn(x, wnid))))
        if len(sampling_candidates[wnid]) < num_per_class:
            success = False
            tmp_images = [(x, wnid) for x in sampling_candidates[wnid]]
            dataset_images.extend(tmp_images)
        else:
            tmp_images = [(x, wnid) for x in sampling_candidates[wnid][:num_per_class]]
            dataset_images.extend(tmp_images)
    
    rng.shuffle(dataset_images)
    if success:
        assert len(dataset_images) == dataset_size
    
    result = {}
    result['sampling_function'] = 'sample_best'
    result['target_size'] = dataset_size
    result['min_num_annotations'] = min_num_annotations
    result['near_duplicate_review_targets'] = near_duplicate_review_targets
    result['time_string'] = get_time_string()
    result['username'] = getpass.getuser()
    result['seed'] = seed
    result['image_filenames'] = dataset_images
    result['is_valid'] = success
    if starting_from is not None:
        result['starting_from'] = starting_from['output_filename']
    return success, result, sampling_candidates, exclusions, carried_over_from_prev


def sample_wnid_histogram(*,
                          dataset_size,
                          histogram_bins,
                          min_num_annotations_candidates,
                          min_num_annotations_val,
                          min_num_val_images_per_wnid,
                          near_duplicate_review_targets,
                          seed,
                          starting_from=None,
                          allow_upward_sampling=False):
    num_classes = 1000
    assert dataset_size % num_classes == 0
    for metric in near_duplicate_data.metric_names:
        assert metric in near_duplicate_review_targets
    assert len(near_duplicate_review_targets) == len(near_duplicate_data.metric_names)
    num_per_class = dataset_size // num_classes
    num_bins = len(histogram_bins) + 1
    rng = random.Random(seed)
    imgnet = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(load_metadata_from_s3=False,
                                       exclude_blacklisted_candidates=False)
    mturk = mturk_data.MTurkData(
            live=True,
            load_assignments=True,
            source_filenames_to_ignore=mturk_data.main_collection_filenames_to_ignore)
    ndc = near_duplicate_data.NearDuplicateData(
            imgnet=imgnet,
            candidates=cds,
            mturk_data=mturk,
            load_review_thresholds=True)
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))

    success = True
    histograms_success, wnid_histograms, usable_val_imgs_by_wnid = compute_wnid_histograms(
            imgnet=imgnet,
            mturk=mturk,
            min_num_annotations_val=min_num_annotations_val,
            min_num_val_images_per_wnid=min_num_val_images_per_wnid,
            histogram_bins=histogram_bins,
            num_per_class=num_per_class)
    if not histograms_success:
        success = False
    prev_dataset_by_wnid = get_prev_dataset_by_wnid(starting_from, dataset_size, all_wnids, cds)
    
    def is_cid_ok(cid, wnid):
        if cid in cds.blacklist:
            return False, 'blacklisted'
        if cid not in mturk.image_num_assignments:
            return False, 'few_assignments'
        if wnid not in mturk.image_num_assignments[cid]:
            return False, 'few_assignments'
        if mturk.image_num_assignments[cid][wnid] < min_num_annotations_candidates:
            return False, 'few_assignments'
        if ndc.is_near_duplicate[cid]:
            return False, 'near_duplicate'
        sufficiently_reviewed = True
        if cid not in ndc.review_threshold:
            sufficiently_reviewed = False
        else:
            for metric in near_duplicate_data.metric_names:
                if metric not in ndc.review_threshold[cid]:
                    sufficiently_reviewed = False
                elif ndc.review_threshold[cid][metric] <= near_duplicate_review_targets[metric]:
                    sufficiently_reviewed = False
        if not sufficiently_reviewed:
            return False, 'unreviewed'
        return True, None

    dataset_images = []
    sampling_candidates = {}
    exclusions = {}
    carried_over_from_prev = {}
    upward_sampled = {}
    for wnid in all_wnids:
        cur_target = wnid_histograms[wnid]
        exclusions[wnid] = {}
        for x in range(num_bins):
            exclusions[wnid][x] = OrderedDict([('blacklisted', []),
                                               ('few_assignments', []),
                                               ('below_threshold', []),
                                               ('near_duplicate', []),
                                               ('unreviewed', [])])
        carried_over_from_prev[wnid] = {x: [] for x in range(num_bins)}
        sampled_images_by_bin = {x: [] for x in range(num_bins)}
        prev_by_bin = {x: [] for x in range(num_bins)}
        for cid in prev_dataset_by_wnid[wnid]:
            cur_freq = mturk.image_fraction_selected[cid][wnid]
            cur_bin = get_histogram_bin(cur_freq, histogram_bins)
            cur_ok, cur_reason = is_cid_ok(cid, wnid)
            if cur_ok:
                prev_by_bin[cur_bin].append(cid)
            else:
                exclusions[wnid][cur_bin][cur_reason].append(cid)
        for cur_bin in range(num_bins):
            if len(prev_by_bin[cur_bin]) <= cur_target[cur_bin]:
                sampled_images_by_bin[cur_bin].extend(prev_by_bin[cur_bin])
                carried_over_from_prev[wnid][cur_bin].extend(prev_by_bin[cur_bin])
            else:
                cur_sample = rng.sample(prev_by_bin[cur_bin], cur_target[cur_bin])
                sampled_images_by_bin[cur_bin].extend(cur_sample)
                carried_over_from_prev[wnid][cur_bin].extend(cur_sample)
        
        sample_candidates_by_bin = {x: [] for x in range(num_bins)}
        unmodified_sample_candidates_by_bin = {x: [] for x in range(num_bins)}
        for cand in cds.candidates_by_wnid[wnid]:
            cid = cand['id_ours']
            if cid in mturk.image_fraction_selected and wnid in mturk.image_fraction_selected[cid]:
                cur_freq = mturk.image_fraction_selected[cid][wnid]
            else:
                cur_freq = 0.0
            cur_bin = get_histogram_bin(cur_freq, histogram_bins)
            cur_ok, cur_reason = is_cid_ok(cid, wnid)
            if cur_ok:
                already_used = False
                for tmp_bin in range(num_bins):
                    if cid in carried_over_from_prev[wnid][tmp_bin]:
                        already_used = True
                if not already_used:
                    sample_candidates_by_bin[cur_bin].append(cid)
            else:
                exclusions[wnid][cur_bin][cur_reason].append(cid)
        for cur_bin in range(num_bins):
            sample_candidates_by_bin[cur_bin] = list(sorted(sample_candidates_by_bin[cur_bin]))
            unmodified_sample_candidates_by_bin[cur_bin] = copy.deepcopy(sample_candidates_by_bin[cur_bin])
            num_remaining_to_sample = cur_target[cur_bin] - len(sampled_images_by_bin[cur_bin])
            if num_remaining_to_sample > len(sample_candidates_by_bin[cur_bin]):
                if not allow_upward_sampling:
                    success = False
                cur_sample = sample_candidates_by_bin[cur_bin]
                sample_candidates_by_bin[cur_bin] = []
            else:
                cur_sample = rng.sample(sample_candidates_by_bin[cur_bin], num_remaining_to_sample)
                sample_candidates_by_bin[cur_bin] = list(set(sample_candidates_by_bin[cur_bin]) - set(cur_sample))
            sampled_images_by_bin[cur_bin].extend(cur_sample)
        if allow_upward_sampling:
            upward_sampled[wnid] = []
            for cur_bin in range(num_bins):
                cur_upward_sampled = []
                num_remaining_to_sample = cur_target[cur_bin] - len(sampled_images_by_bin[cur_bin])
                if num_remaining_to_sample > 0:
                    assert len(sample_candidates_by_bin[cur_bin]) == 0
                for _ in range(num_remaining_to_sample):
                    found_bin = False
                    for next_bin in range(cur_bin + 1, num_bins):
                        if len(sample_candidates_by_bin[next_bin]) > 0:
                            sample_candidates_from_prev = set(sample_candidates_by_bin[next_bin]) & set(prev_dataset_by_wnid[wnid])
                            if len(sample_candidates_from_prev) > 0:
                                cur_sample = [list(sample_candidates_from_prev)[0]]
                                print(f'    upward sampled {cur_sample[0]} from the prev dataset')
                            else:
                                cur_sample = rng.sample(sample_candidates_by_bin[next_bin], 1)
                                print(f'    upward sampled {cur_sample[0]} randomly')
                            assert len(cur_sample) == 1
                            sampled_images_by_bin[cur_bin].extend(cur_sample)
                            sample_candidates_by_bin[next_bin] = list(set(sample_candidates_by_bin[next_bin]) - set(cur_sample))
                            cur_upward_sampled.append((cur_sample[0], next_bin))
                            found_bin = True
                            break
                    if not found_bin:
                        success = False
                upward_sampled[wnid].append(cur_upward_sampled)
        for cur_bin in range(num_bins):
            dataset_images.extend([x, wnid] for x in sampled_images_by_bin[cur_bin])
        sampling_candidates[wnid] = unmodified_sample_candidates_by_bin

    rng.shuffle(dataset_images)
    if len(dataset_images) > dataset_size:
        print(len(dataset_images), dataset_size)
    assert len(dataset_images) <= dataset_size
    if success:
        assert len(dataset_images) == dataset_size
    
    result = {}
    result['sampling_function'] = 'sample_wnid_histogram'
    result['target_size'] = dataset_size
    result['histogram_bins'] = histogram_bins
    result['min_num_annotations_candidates'] = min_num_annotations_candidates
    result['min_num_annotations_val'] = min_num_annotations_val
    result['min_num_val_images_per_wnid'] = min_num_val_images_per_wnid
    result['near_duplicate_review_targets'] = near_duplicate_review_targets
    result['time_string'] = get_time_string()
    result['username'] = getpass.getuser()
    result['seed'] = seed
    result['image_filenames'] = dataset_images
    result['is_valid'] = success
    result['allow_upward_sampling'] = allow_upward_sampling
    if starting_from is not None:
        result['starting_from'] = starting_from['output_filename']
    
    result_metadata = {}
    result_metadata['wnid_histograms'] = wnid_histograms
    result_metadata['usable_val_imgs_by_wnid'] = usable_val_imgs_by_wnid
    result_metadata['sampling_candidates'] = sampling_candidates
    result_metadata['exclusions'] = exclusions
    result_metadata['carried_over_from_prev'] = carried_over_from_prev
    result_metadata['upward_sampled'] = upward_sampled
    return success, result, result_metadata


def sample_above_threshold(*,
                           dataset_size,
                           selection_frequency_threshold,
                           min_num_annotations,
                           near_duplicate_review_targets,
                           seed,
                           starting_from=None,
                           wnid_thresholds=None):
    num_classes = 1000
    assert dataset_size % num_classes == 0
    for metric in near_duplicate_data.metric_names:
        assert metric in near_duplicate_review_targets
    assert len(near_duplicate_review_targets) == len(near_duplicate_data.metric_names)
    num_per_class = dataset_size // num_classes
    rng = random.Random(seed)
    imgnet = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(load_metadata_from_s3=False,
                                       exclude_blacklisted_candidates=False)
    mturk = mturk_data.MTurkData(
            live=True,
            load_assignments=True,
            source_filenames_to_ignore=mturk_data.main_collection_filenames_to_ignore)
    ndc = near_duplicate_data.NearDuplicateData(
            imgnet=imgnet,
            candidates=cds,
            mturk_data=mturk,
            load_review_thresholds=True)
    
    def is_cid_ok(cid, wnid):
        if cid in cds.blacklist:
            return False, 'blacklisted'
        if cid not in mturk.image_num_assignments:
            return False, 'few_assignments'
        if wnid not in mturk.image_num_assignments[cid]:
            return False, 'few_assignments'
        if mturk.image_num_assignments[cid][wnid] < min_num_annotations:
            return False, 'few_assignments'
        if wnid in wnid_thresholds:
            cur_threshold = wnid_thresholds[wnid]
        else:
            cur_threshold = selection_frequency_threshold
        if mturk.image_fraction_selected[cid][wnid] < cur_threshold:
            return False, 'below_threshold'
        if ndc.is_near_duplicate[cid]:
            return False, 'near_duplicate'
        sufficiently_reviewed = True
        for metric in near_duplicate_data.metric_names:
            if cid not in ndc.review_threshold or metric not in ndc.review_threshold[cid]:
                sufficiently_reviewed = False
            elif ndc.review_threshold[cid][metric] <= near_duplicate_review_targets[metric]:
                sufficiently_reviewed = False
        if not sufficiently_reviewed:
            return False, 'unreviewed'
        return True, None
    
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    if wnid_thresholds is not None:
        for wnid in wnid_thresholds.keys():
            assert wnid in all_wnids
    
    prev_dataset_by_wnid = get_prev_dataset_by_wnid(starting_from, dataset_size, all_wnids, cds)

    dataset_images = []
    sampling_candidates = {}
    exclusions = {}
    success = True
    carried_over_from_prev = {}
    for wnid in all_wnids:
        sampling_candidates[wnid] = []
        exclusions[wnid] = OrderedDict([('blacklisted', []),
                                        ('few_assignments', []),
                                        ('below_threshold', []),
                                        ('near_duplicate', []),
                                        ('unreviewed', [])])
        carried_over_from_prev[wnid] = []
        if wnid in prev_dataset_by_wnid:
            for cid in prev_dataset_by_wnid[wnid]:
                if is_cid_ok(cid, wnid)[0]:
                    carried_over_from_prev[wnid].append(cid)
        for cand in cds.candidates_by_wnid[wnid]:
            cid = cand['id_ours']
            cur_ok, cur_reason = is_cid_ok(cid, wnid)
            if cur_ok:
                if cid not in carried_over_from_prev[wnid]:
                    sampling_candidates[wnid].append(cid)
            else:
                exclusions[wnid][cur_reason].append(cid)
        sampling_candidates[wnid] = list(sorted(sampling_candidates[wnid]))
        remaining_to_sample = num_per_class - len(carried_over_from_prev[wnid])
        if len(sampling_candidates[wnid]) < remaining_to_sample:
            success = False
            tmp_images = [(x, wnid) for x in carried_over_from_prev[wnid]] + [(x, wnid) for x in sampling_candidates[wnid]]
            dataset_images.extend(tmp_images)
        else:
            new_images = rng.sample(sampling_candidates[wnid], remaining_to_sample)
            tmp_images = [(x, wnid) for x in carried_over_from_prev[wnid]] + [(x, wnid) for x in new_images]
            dataset_images.extend(tmp_images)
    
    rng.shuffle(dataset_images)
    if success:
        assert len(dataset_images) == dataset_size
    
    result = {}
    result['sampling_function'] = 'sample_above_threshold'
    result['target_size'] = dataset_size
    result['selection_frequency_threshold'] = selection_frequency_threshold
    result['min_num_annotations'] = min_num_annotations
    result['near_duplicate_review_targets'] = near_duplicate_review_targets
    result['time_string'] = get_time_string()
    result['username'] = getpass.getuser()
    result['seed'] = seed
    result['image_filenames'] = dataset_images
    result['is_valid'] = success
    if starting_from is not None:
        result['starting_from'] = starting_from['output_filename']
    if wnid_thresholds is not None:
        result['wnid_thresholds'] = wnid_thresholds
    return success, result, sampling_candidates, exclusions, carried_over_from_prev

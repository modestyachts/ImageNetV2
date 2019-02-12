import json
import pathlib
import statistics

import click

import dataset_sampling
import imagenet
import mturk_data


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dataset_size', required=True, type=int)
@click.option('--seed', required=True, type=int)
@click.option('--output_filename', required=True, type=str)
def dummy(dataset_size, seed, output_filename):
    output_filepath = pathlib.Path(__file__).parent / '../data/datasets' / output_filename
    output_filepath = output_filepath.resolve()
    assert not output_filepath.is_file()
    result = dataset_sampling.sample_val_dummy(dataset_size, seed)
    result['output_filename'] = output_filename
    with open(output_filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print('Wrote dataset to {}'.format(output_filepath))


@cli.command()
@click.option('--dataset_size', required=True, type=int)
@click.option('--min_num_annotations', required=True, type=int)
@click.option('--seed', required=True, type=int)
@click.option('--output_filename', required=True, type=str)
def val_annotated(dataset_size, min_num_annotations, seed, output_filename):
    output_filepath = pathlib.Path(__file__).parent / '../data/datasets' / output_filename
    output_filepath = output_filepath.resolve()
    assert not output_filepath.is_file()
    result = dataset_sampling.sample_val_annotated(dataset_size, min_num_annotations, seed)
    result['output_filename'] = output_filename
    with open(output_filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print('Wrote dataset to {}'.format(output_filepath))


@cli.command()
@click.option('--dataset_size', required=True, type=int)
@click.option('--min_num_annotations', required=True, type=int)
@click.option('--seed', required=True, type=int)
@click.option('--output_filename', required=True, type=str)
@click.option('--starting_from', default=None, type=str)
def best(dataset_size,
         min_num_annotations,
         seed,
         output_filename,
         starting_from):
    output_filepath = pathlib.Path(__file__).parent / '../data/datasets' / output_filename
    output_filepath = output_filepath.resolve()
    assert not output_filepath.is_file()

    if starting_from is not None:
        starting_from_filepath = pathlib.Path(__file__).parent / '../data/datasets' / starting_from
        starting_from_filepath = starting_from_filepath.resolve()
        assert starting_from_filepath.is_file()
        with open(starting_from_filepath, 'r') as f:
            starting_from_loaded = json.load(f)
    else:
        starting_from_loaded = None

    imgnet = imagenet.ImageNetData()
    mturk = mturk_data.MTurkData(
            live=True,
            load_assignments=True,
            source_filenames_to_ignore=mturk_data.main_collection_filenames_to_ignore)
    
    review_targets = {'l2' : 1.2e8, 'dssim' : 0.2205, 'fc7' : 1.32e4}
    
    success, result, sampling_candidates, exclusions, carried_over_from_prev = dataset_sampling.sample_best(
            dataset_size=dataset_size,
            min_num_annotations=min_num_annotations,
            near_duplicate_review_targets=review_targets,
            seed=seed,
            starting_from=starting_from_loaded)
    
    if not success:
        num_per_class = dataset_size // 1000
        print('Failed to sample a valid dataset.')
        print('The following wnids have fewer than {} candidates with at least {} annotations'.format(num_per_class, min_num_annotations))
        for wnid, cur_candidates in sampling_candidates.items():
            if len(cur_candidates) < num_per_class - len(carried_over_from_prev[wnid]):
                print('    {}: {} sampling candidates, plus {} carried over from the previous dataset  ({})'.format(
                        wnid,
                        len(cur_candidates),
                        len(carried_over_from_prev[wnid]),
                        ', '.join(imgnet.class_info_by_wnid[wnid].synset)))
                for reason, excluded_candidates in exclusions[wnid].items():
                    print('        {}: {} candidates'.format(reason, len(excluded_candidates)))
    
    avg_selection_frequency = 0.0
    for img, wnid in result['image_filenames']:
        avg_selection_frequency += mturk.image_fraction_selected[img][wnid]
    avg_selection_frequency /= len(result['image_filenames'])
    print(f'\nAverage selection frequency: {avg_selection_frequency:.2}')

    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    selection_frequencies_by_wnid = {x: [] for x in all_wnids}
    for img, wnid in result['image_filenames']:
        selection_frequencies_by_wnid[wnid].append(mturk.image_fraction_selected[img][wnid])
    min_selection_frequency_by_wnid = {x: min(selection_frequencies_by_wnid[x]) for x in all_wnids}
    avg_selection_frequency_by_wnid = {x: statistics.mean(selection_frequencies_by_wnid[x]) for x in all_wnids}

    show_worst_k = 20
    print('\nwnids with the smallest minimum selection frequencies:')
    for wnid, sel_freq in sorted(min_selection_frequency_by_wnid.items(), key=lambda x: (x[1], x[0]))[:show_worst_k]:
        synset = ', '.join(imgnet.class_info_by_wnid[wnid].synset)
        print(f'    {wnid}: {sel_freq:.3f}    ({synset})')
    print('\nwnids with the smallest average selection frequencies:')
    for wnid, sel_freq in sorted(avg_selection_frequency_by_wnid.items(), key=lambda x: (x[1], x[0]))[:show_worst_k]:
        synset = ', '.join(imgnet.class_info_by_wnid[wnid].synset)
        print(f'    {wnid}: {sel_freq:.3f}    ({synset})')

    result['output_filename'] = output_filename
    with open(output_filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print('\nWrote dataset to {}'.format(output_filepath))


@cli.command()
@click.option('--dataset_size', required=True, type=int)
@click.option('--min_num_annotations_candidates', required=True, type=int)
@click.option('--min_num_annotations_val', required=True, type=int)
@click.option('--min_num_val_images_per_wnid', required=True, type=int)
@click.option('--seed', required=True, type=int)
@click.option('--output_filename', required=True, type=str)
@click.option('--starting_from', default=None, type=str)
@click.option('--allow_upward_sampling', is_flag=True, type=bool)
def wnid_histogram(dataset_size,
                   min_num_annotations_candidates,
                   min_num_annotations_val,
                   min_num_val_images_per_wnid,
                   seed,
                   output_filename,
                   starting_from,
                   allow_upward_sampling):
    output_filepath = pathlib.Path(__file__).parent / '../data/datasets' / output_filename
    output_filepath = output_filepath.resolve()
    assert not output_filepath.is_file()

    if starting_from is not None:
        starting_from_filepath = pathlib.Path(__file__).parent / '../data/datasets' / starting_from
        starting_from_filepath = starting_from_filepath.resolve()
        assert starting_from_filepath.is_file()
        with open(starting_from_filepath, 'r') as f:
            starting_from_loaded = json.load(f)
    else:
        starting_from_loaded = None
    
    review_targets = {'l2' : 1.2e8, 'dssim' : 0.2205, 'fc7' : 1.32e4}
    histogram_bins = [0.2, 0.4, 0.6, 0.8]
    num_bins = len(histogram_bins) + 1
    success, result, results_metadata = dataset_sampling.sample_wnid_histogram(
        dataset_size=dataset_size,
        histogram_bins=histogram_bins,
        min_num_annotations_candidates=min_num_annotations_candidates,
        min_num_annotations_val=min_num_annotations_val,
        min_num_val_images_per_wnid=min_num_val_images_per_wnid,
        near_duplicate_review_targets=review_targets,
        seed=seed,
        starting_from=starting_from_loaded,
        allow_upward_sampling=allow_upward_sampling)
    imgnet = imagenet.ImageNetData()
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    if not success:
        total_num_problematic_bins = 0
        print(f'Failed to sample a valid dataset  ({len(result["image_filenames"])} instead of {dataset_size} images).')
        print('The following wnid bins have insufficient images (before potential upward sampling):')
        for wnid in all_wnids:
            cur_histogram = results_metadata['wnid_histograms'][wnid]
            cur_sampling_candidates = results_metadata['sampling_candidates'][wnid]
            cur_carried_over_from_prev = results_metadata['carried_over_from_prev'][wnid]
            cur_exclusions = results_metadata['exclusions'][wnid]

            problematic_bins = []
            for cur_bin in range(num_bins):
                if cur_histogram[cur_bin] > len(cur_sampling_candidates[cur_bin]) + len(cur_carried_over_from_prev[cur_bin]):
                    problematic_bins.append(cur_bin)
            total_num_problematic_bins += len(problematic_bins)
            if len(problematic_bins) > 0:
                print('wnid {} ({})'.format(wnid, ', '.join(imgnet.class_info_by_wnid[wnid].synset)))
                for cur_bin in problematic_bins:
                    cur_low, cur_high = dataset_sampling.get_bin_boundaries(histogram_bins, cur_bin)
                    cur_valid = len(cur_sampling_candidates[cur_bin]) + len(cur_carried_over_from_prev[cur_bin])
                    print('    bin ({} {}): target {}, currently have {}'.format(cur_low, cur_high, cur_histogram[cur_bin], cur_valid))
                    print('        {} sampling candidates'.format(len(cur_sampling_candidates[cur_bin])))
                    print('        {} carried over from previous dataset'.format(len(cur_carried_over_from_prev[cur_bin])))
                    for reason, excluded_candidates in cur_exclusions[cur_bin].items():
                        print('        {}: {} excluded candidates'.format(reason, len(excluded_candidates)))
                print()
        print('{} problematic bins in total'.format(total_num_problematic_bins))

    if allow_upward_sampling:
        num_upward_sampled = 0
        print('\nUpward sampled the following images:')
        for wnid in all_wnids:
            upward_sampled_for_wnid = results_metadata['upward_sampled'][wnid]
            has_upsampled_bins = False
            for cur_bin in range(num_bins):
                if len(upward_sampled_for_wnid[cur_bin]) > 0:
                    has_upsampled_bins = True
                    break
            if has_upsampled_bins:
                print('wnid {} ({})'.format(wnid, ', '.join(imgnet.class_info_by_wnid[wnid].synset)))
                for cur_bin in range(num_bins):
                    cur_upward_sampled = upward_sampled_for_wnid[cur_bin]
                    for cid, to_bin in cur_upward_sampled:
                        original_low, original_high = dataset_sampling.get_bin_boundaries(histogram_bins, cur_bin)
                        to_low, to_high = dataset_sampling.get_bin_boundaries(histogram_bins, to_bin)
                        print(f'    sampled {cid} belonging to bin ({original_low} {original_high}) from bin ({to_low} {to_high}) instead')
                        num_upward_sampled += 1
                print()
        print(f'\nUpwarded sampled {num_upward_sampled} images in total')
        if not success:
            print('The following wnid have insufficient images even after upward sampling:')
            num_per_class = dataset_size // 1000
            images_by_wnid = {}
            for img, wnid in result['image_filenames']:
                if wnid not in images_by_wnid:
                    images_by_wnid[wnid] = []
                images_by_wnid[wnid].append(img)
            for wnid in all_wnids:
                if len(images_by_wnid[wnid]) < num_per_class:
                    print('    wnid {}: {} / {} images  ({})'.format(wnid, len(images_by_wnid[wnid]), num_per_class, ', '.join(imgnet.class_info_by_wnid[wnid].synset)))
            
    result['output_filename'] = output_filename
    with open(output_filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print('Wrote dataset to {}'.format(output_filepath))


@cli.command()
@click.option('--dataset_size', required=True, type=int)
@click.option('--selection_frequency_threshold', required=True, type=float)
@click.option('--min_num_annotations', required=True, type=int)
@click.option('--seed', required=True, type=int)
@click.option('--output_filename', required=True, type=str)
@click.option('--starting_from', default=None, type=str)
@click.option('--wnid_thresholds_filename', default=None, type=str)
def above_threshold(dataset_size,
                    selection_frequency_threshold,
                    min_num_annotations,
                    seed,
                    output_filename,
                    starting_from,
                    wnid_thresholds_filename):
    output_filepath = pathlib.Path(__file__).parent / '../data/datasets' / output_filename
    output_filepath = output_filepath.resolve()
    assert not output_filepath.is_file()

    if starting_from is not None:
        starting_from_filepath = pathlib.Path(__file__).parent / '../data/datasets' / starting_from
        starting_from_filepath = starting_from_filepath.resolve()
        assert starting_from_filepath.is_file()
        with open(starting_from_filepath, 'r') as f:
            starting_from_loaded = json.load(f)
    
    if wnid_thresholds_filename is not None:
        wnid_thresholds_filepath = pathlib.Path(wnid_thresholds_filename)
        wnid_thresholds_filepath = wnid_thresholds_filepath.resolve()
        assert wnid_thresholds_filepath.is_file()
        with open(wnid_thresholds_filepath, 'r') as f:
            wnid_thresholds = json.load(f)
    else:
        wnid_thresholds = None

    review_targets = {'l2' : 1.2e8, 'dssim' : 0.2205, 'fc7' : 1.32e4}
    
    success, result, sampling_candidates, exclusions, carried_over_from_prev = dataset_sampling.sample_above_threshold(
            dataset_size=dataset_size,
            selection_frequency_threshold=selection_frequency_threshold,
            min_num_annotations=min_num_annotations,
            near_duplicate_review_targets=review_targets,
            seed=seed,
            starting_from=starting_from_loaded,
            wnid_thresholds=wnid_thresholds)
    
    if not success:
        imgnet = imagenet.ImageNetData()
        num_per_class = dataset_size // 1000
        print('Failed to sample a valid dataset.')
        print('The following wnids have fewer than {} candidates above threshold {} with at least {} annotations'.format(
                num_per_class,
                selection_frequency_threshold,
                min_num_annotations))
        for wnid, cur_candidates in sampling_candidates.items():
            if len(cur_candidates) < num_per_class - len(carried_over_from_prev[wnid]):
                print('    {}: {} sampling candidates, plus {} carried over from the previous dataset  ({})'.format(
                        wnid,
                        len(cur_candidates),
                        len(carried_over_from_prev[wnid]),
                        ', '.join(imgnet.class_info_by_wnid[wnid].synset)))
                for reason, excluded_candidates in exclusions[wnid].items():
                    print('        {}: {} candidates'.format(reason, len(excluded_candidates)))
    result['output_filename'] = output_filename
    with open(output_filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print('Wrote dataset to {}'.format(output_filepath))


if __name__ == "__main__":
    cli()
import json
import pathlib

import click

import imagenet

@click.group()
def cli():
    pass


def images_are_same(list1, list2):
    if len(list1) != len(list2):
        return False
    set1 = set(list1)
    set2 = set(list2)
    assert len(list1) == len(set1)
    assert len(list2) == len(set2)
    for img in set1:
        if img not in set2:
            return False
    return True


@cli.command()
@click.option('--dataset_filename', required=True, type=str)
@click.option('--starting_from', required=True, type=str)
def carry_over_reviews(dataset_filename, starting_from):
    dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / dataset_filename
    dataset_filepath = dataset_filepath.resolve()
    assert dataset_filepath.is_file()
    with open(dataset_filepath, 'r') as f:
        dataset = json.load(f)

    prev_dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / starting_from
    prev_dataset_filepath = prev_dataset_filepath.resolve()
    assert prev_dataset_filepath.is_file()
    with open(prev_dataset_filepath, 'r') as f:
        prev_dataset = json.load(f)
    assert dataset['starting_from'] == prev_dataset['output_filename']

    assert starting_from.endswith('.json')
    prev_review_filename = starting_from[:-5] + '_review.json'
    prev_review_filepath = pathlib.Path(__file__).parent / '../data/dataset_reviews' / prev_review_filename
    prev_review_filepath = prev_review_filepath.resolve()
    assert prev_review_filepath.is_file()
    with open(prev_review_filepath, 'r') as f:
        prev_review = json.load(f)

    imgnet = imagenet.ImageNetData()
    all_wnids = list(sorted(list(imgnet.class_info_by_wnid.keys())))
    assert len(all_wnids) == 1000

    prev_dataset_by_wnid = {}
    dataset_by_wnid = {}
    for wnid in all_wnids:
        prev_dataset_by_wnid[wnid] = []
        dataset_by_wnid[wnid] = []
    for img, wnid in dataset['image_filenames']:
        dataset_by_wnid[wnid].append(img)
    for img, wnid in prev_dataset['image_filenames']:
        prev_dataset_by_wnid[wnid].append(img)

    new_review = {}
    for wnid in all_wnids:
        new_review[wnid] = {}
        new_review[wnid]['problematic'] = False
        if prev_review[wnid]['problematic']:
            new_review[wnid]['reviewed'] = False
        else:
            if images_are_same(dataset_by_wnid[wnid], prev_dataset_by_wnid[wnid]):
                new_review[wnid]['reviewed'] = prev_review[wnid]['reviewed']
            else:
                new_review[wnid]['reviewed'] = False
    assert dataset_filename.endswith('.json')
    new_review_filename = dataset_filename[:-5] + '_review.json'
    new_review_filepath = pathlib.Path(__file__).parent / '../data/dataset_reviews' / new_review_filename
    new_review_filepath = new_review_filepath.resolve()
    assert not new_review_filepath.is_file()
    with open(new_review_filepath, 'w') as f:
        json.dump(new_review, f, indent=2, sort_keys=True)
    print('Wrote new review data to {}'.format(new_review_filepath))
    num_reviewed = len([x for x in new_review.items() if x[1]['reviewed']])
    num_problematic = len([x for x in new_review.items() if x[1]['problematic']])
    print('    {} reviewed wnids'.format(num_reviewed))
    print('    {} problematic wnids'.format(num_problematic))
    

if __name__ == "__main__":
    cli()
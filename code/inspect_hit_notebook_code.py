import statistics
from timeit import default_timer as timer

import ipywidgets as widgets

import mturk_utils
import utils


def generate_image_captions(hit_data, cur_assignments, cds):
    frequencies = {}
    for image_name in hit_data['images_all']:
        frequencies[image_name] = 0

    num_valid_assignments = 0
    for a in cur_assignments.values():
        if a['AssignmentStatus'] in ['Submitted', 'Approved']:
            num_valid_assignments += 1
            for img_name in a['Answer']:
                frequencies[img_name] += 1
    image_captions = {}
    for image_name in hit_data['images_all']:
        image_name_label = image_name
        if image_name.startswith('ILSVRC2012_val_'):
            if image_name in hit_data['images_neg_control']:
                image_explanation = 'negative control'
            else:
                assert image_name in hit_data['images_pos_control']
                image_explanation = 'positive control'
        else:
            candidate_data = cds.all_candidates[image_name]
            image_explanation = '{} search "{}"'.format(candidate_data['search_engine'], candidate_data['search_key'])
        image_captions[image_name] = [image_name_label, image_explanation, '{} / {}'.format(frequencies[image_name], num_valid_assignments)]
    return image_captions


def inspect_assignment(*, assignment_id, hit_data, cur_assignments, img_loader=None, cds=None):
    cur_a = cur_assignments[assignment_id]

    selected_filenames = cur_a['Answer']
    unselected_filenames = []
    for img_filename in hit_data['images_all']:
        if img_filename not in selected_filenames:
            unselected_filenames.append(img_filename)
   
    image_captions = generate_image_captions(hit_data, cur_assignments, cds)
    
    print('\n\nLoading images ...', end='')
    start = timer()
    image_data = img_loader.load_image_bytes_batch(hit_data['images_all'],
                                                   size='scaled_500',
                                                   verbose=False,
                                                   progress_bar=False)
    print(' done, took {:.1f} seconds\n'.format(timer() - start))

    num_cols = 3
    selected_label = widgets.Label(value='Selected images')
    selected_label.add_class('hit_heading')
    selected_grid = mturk_utils.show_image_grid(selected_filenames, image_captions, image_data, num_cols=num_cols)
    unselected_label = widgets.Label(value='Unselected images')
    unselected_label.add_class('hit_heading')
    unselected_grid = mturk_utils.show_image_grid(unselected_filenames, image_captions, image_data, num_cols=num_cols)
    return widgets.VBox([selected_label, selected_grid, unselected_label, unselected_grid])


def inspect_hit(*, cur_id=None, hits=None, hit_ids_to_uuid=None, client=None, imgnet=None, img_loader=None,
                cds=None):
    if '-' in cur_id:
        cur_uuid = cur_id
        cur_hit = hits[cur_uuid]
        cur_hit_id = cur_hit['hit_id']
    else:
        assert cur_id in hit_ids_to_uuid
        cur_hit_id = cur_id
        cur_uuid = hit_ids_to_uuid[cur_id]
        cur_hit = hits[cur_uuid]

    cur_assignments = mturk_utils.get_assignments_for_hit_from_aws(cur_hit_id, client, uuid=cur_uuid)

    hit_desc = mturk_utils.get_hit_description(cur_hit, cur_assignments, imgnet)
    print(hit_desc.title + '\n')
    print(hit_desc.wnid_info)
    print(hit_desc.creation)
    print('content hash: ' + hit_desc.all_images_hash)
    print(hit_desc.work_duration)
    print(hit_desc.work_delay)
    print(hit_desc.assignments_summary)
    for t in hit_desc.per_assignment_text:
        print('  ' + t)

    print('\n\nLoading images ...', end='')
    start = timer()
    image_data = img_loader.load_image_bytes_batch(cur_hit['images_all'],
                                                   size='scaled_500',
                                                   verbose=False,
                                                   progress_bar=False)
    print(' done, took {:.1f} seconds\n'.format(timer() - start))

    image_captions = generate_image_captions(cur_hit, cur_assignments, cds)
    return mturk_utils.show_hit_images(cur_hit, image_captions, image_data, num_cols=3), cur_hit

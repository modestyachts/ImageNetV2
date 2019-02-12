from collections import namedtuple
import hashlib
import io
import json
import math
import numpy as np
import os
import pathlib
import statistics
import tarfile
from timeit import default_timer as timer
import urllib.request
from xml.dom.minidom import parseString

import boto3
import botocore
import dateutil
import PIL.Image
import pytz

import aes
import imagenet
import utils

try:
    from tqdm import tqdm
    pass
except:
    pass

def get_assignments_for_hit_from_aws(hit_id, client, uuid):
    max_num_results = 100
    overall_result = client.list_assignments_for_hit(HITId=hit_id, MaxResults=max_num_results)
    assert overall_result['NumResults'] < max_num_results

    return_res = {}
    for res in overall_result['Assignments']:
        cur_res = {}
        assert res['HITId'] == hit_id
        if 'AssignmentStatus' in res and res['AssignmentStatus'] in ['Approved', 'Submitted', 'Rejected']:
            assert 'Answer' in res
            assert 'SubmitTime' in res
            assert 'AutoApprovalTime' in res
        if 'AssignmentStatus' in res and res['AssignmentStatus'] == 'Approved':
            assert 'ApprovalTime' in res
        if 'AssignmentStatus' in res and res['AssignmentStatus'] == 'Rejected':
            assert 'RejectionTime' in res
            assert 'RequesterFeedback' in res

        cur_res['WorkerId'] = utils.hash_worker_id(res['WorkerId'])
        cur_res['hit_uuid'] = uuid
        cur_res['AssignmentId'] = res['AssignmentId']
        required_keys = ['AssignmentStatus', 'AcceptTime']
        optional_keys = ['AutoApprovalTime', 'SubmitTime', 'ApprovalTime', 'RejectionTime', 'Deadline', 'RequesterFeedback']
        for key in required_keys:
            assert key in res
            cur_res[key] = res[key]
        for key in optional_keys:
            if key in res:
                cur_res[key] = res[key]
        if 'AssignmentStatus' in res and res['AssignmentStatus'] in ['Approved', 'Submitted', 'Rejected']:
            assert 'Answer' in res
            answer_xml = parseString(res['Answer'])
            answer = answer_xml.getElementsByTagName('FreeText')
            assert len(answer) == 1
            ans = ' '.join(t.nodeValue for t in answer[0].childNodes if t.nodeType == t.TEXT_NODE)
            cur_selected = ans.split('|')
            if len(cur_selected) > 0:
                if len(cur_selected[0]) > 100:
                    cur_selected_2 = []
                    for cur_sel in cur_selected:
                        cur_selected_2.append(utils.decrypt_string_with_magic(cur_sel))
                    cur_res['Answer'] = cur_selected_2
                else:
                    cur_res['Answer'] = cur_selected
            else:
                cur_res['Answer'] = []
        return_res[res['AssignmentId']] = cur_res
    return return_res


def get_hit_ids_and_annotations_from_aws(client):
    max_num_results = 100
    return_res = {}
    res = client.list_hits(MaxResults=max_num_results)
    for hit in res['HITs']:
        cur_annotation = None
        if 'RequesterAnnotation' in hit:
            cur_annotation = hit['RequesterAnnotation']
        assert hit['HITId'] not in return_res
        return_res[hit['HITId']] = cur_annotation
    next_token = None
    if 'NextToken' in res:
        next_token = res['NextToken']
    while next_token is not None:
        res = client.list_hits(MaxResults=max_num_results, NextToken=next_token)
        for hit in res['HITs']:
            cur_annotation = None
            if 'RequesterAnnotation' in hit:
                cur_annotation = hit['RequesterAnnotation']
            assert hit['HITId'] not in return_res
            return_res[hit['HITId']] = cur_annotation
        next_token = None
        if 'NextToken' in res:
            next_token = res['NextToken']
    return return_res


def load_local_hit_data(live=False, verbose=True, source_filenames_to_ignore=[], include_blacklisted_hits=False):
    start_time = timer()
    if live:
        json_dir = pathlib.Path(__file__).parent / '../data/mturk/hit_data_live'
        blacklist_filepath = pathlib.Path(__file__).parent / '../data/mturk/hit_blacklist_live.json'
    else:
        json_dir = pathlib.Path(__file__).parent / '../data/mturk/hit_data_sandbox'
        blacklist_filepath = pathlib.Path(__file__).parent / '../data/mturk/hit_blacklist_sandbox.json'
    blacklist_filepath = blacklist_filepath.resolve()
    with open(blacklist_filepath, 'r') as f:
        blacklist = json.load(f)
    json_dir = json_dir.resolve()
    json_filenames = []
    hit_data = {}
    num_filenames_ignored = 0
    for p in json_dir.glob('*.json'):
        if p.name not in source_filenames_to_ignore:
            json_filenames.append(p)
        else:
            num_filenames_ignored += 1
    assert num_filenames_ignored == len(source_filenames_to_ignore)
    json_filenames = sorted(json_filenames)
    blacklisted_hits = {}
    for p in json_filenames:
        with open(p, 'r') as f:
            cur_hit_data = json.load(f)
        for hit in cur_hit_data:
            assert 'uuid' in hit
            assert 'images_all' in hit
            cur_uuid = hit['uuid']
            assert cur_uuid not in hit_data
            hit['source_filename'] = p.name
            if include_blacklisted_hits or cur_uuid not in blacklist:
                hit_data[cur_uuid] = hit
            else:
                blacklisted_hits[cur_uuid] = hit
    end_time = timer()
    if verbose:
        print('Loaded {} HITs from {} hit data JSON file(s) in {:.0f} seconds.'.format(
                len(hit_data), len(json_filenames), end_time - start_time))
        print('    {}/...'.format(json_dir))
        for p in json_filenames[:3]:
            print('        {}'.format(p.name))
        if len(json_filenames) > 3:
            print('        ...')
        print('    Ignored {} filenames.'.format(num_filenames_ignored))
        print('    Ignored {} HITs from the blacklist (blacklist size {})'.format(len(blacklisted_hits),
                                                                                  len(blacklist)))
    hit_ids_to_uuid = {}
    for hit in hit_data.values():
        if 'hit_id' in hit:
            hit_ids_to_uuid[hit['hit_id']] = hit['uuid']
    return hit_data, hit_ids_to_uuid, json_dir, json_filenames, blacklisted_hits


def approve_assignments(assignment_messages, override_rejection=False, live=False):
    client = get_mturk_client(live=live)
    for a, msg in assignment_messages:
        client.approve_assignment(AssignmentId=a,
                                  RequesterFeedback=msg,
                                  OverrideRejection=override_rejection)


def reject_assignments(assignment_messages, live=False):
    client = get_mturk_client(live=live)
    for a, msg in assignment_messages:
        client.reject_assignment(AssignmentId=a,
                                 RequesterFeedback=msg)


def get_mturk_client(live=False):
    environments = {
        'live': {
            'endpoint': 'https://mturk-requester.us-east-1.amazonaws.com',
            'preview': 'https://www.mturk.com/mturk/preview',
            'manage': 'https://requester.mturk.com/mturk/manageHITs',
            'reward': '0.00'
        },
        'sandbox': {
            'endpoint': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com',
            'preview': 'https://workersandbox.mturk.com/mturk/preview',
            'manage': 'https://requestersandbox.mturk.com/mturk/manageHITs',
            'reward': '0.11'
        },
    }

    session = boto3.Session(profile_name='imagenet2')
    if live:
        mturk_environment = environments['live']
    else:
        mturk_environment = environments['sandbox']
    client = session.client(service_name='mturk',
                            region_name='us-east-1',
                            endpoint_url=mturk_environment['endpoint'])
    return client


def mturk_vs_local_consistency_check(live=False):
    client = get_mturk_client(live=live)
    _, local_hit_ids_to_uuid, _, _, _ = load_local_hit_data(live=live, include_blacklisted_hits=True)
    aws_hit_data = get_hit_ids_and_annotations_from_aws(client)
    print('Loaded {} HITs from MTurk.'.format(len(aws_hit_data)))
    num_errors = 0
    num_warnings = 0
    local_hit_ids_missing_remotely = []
    remote_hit_ids_missing_locally = []
    for hit_id in aws_hit_data:
        if aws_hit_data[hit_id] is None:
            print('HIT {} from MTurk does not have a UUID.'.format(hit_id))
            num_errors += 1
            continue
        if hit_id not in local_hit_ids_to_uuid:
            print('HIT {} from MTurk not among the local HITs'.format(hit_id))
            num_errors += 1
            remote_hit_ids_missing_locally.append(hit_id)
            continue
        if aws_hit_data[hit_id] != local_hit_ids_to_uuid[hit_id]:
            print('For HIT {}, the UUID on MTurk does not match the local UUID: {} vs {}'.format(hit_id,
                    aws_hit_data[hit_id], local_hit_ids_to_uuid[hit_id]))
            num_errors += 1
    for hit_id in local_hit_ids_to_uuid:
        if hit_id not in aws_hit_data:
            #print('Local HIT {} does not appear on MTurk.'.format(hit_id))
            num_warnings += 1
            local_hit_ids_missing_remotely.append(hit_id)
            continue
        if local_hit_ids_to_uuid[hit_id] != aws_hit_data[hit_id]:
            print('For HIT {}, the UUID on MTurk does not match the local UUID: {} vs {}'.format(hit_id,
                    aws_hit_data[hit_id], local_hit_ids_to_uuid[hit_id]))
            num_errors += 1
    if num_errors == 0:
        print('Consistency check passed.')
    else:
        print('There were {} errors during the consistency check.'.format(num_errors))
        if len(remote_hit_ids_missing_locally) > 0:
            print('Remote HIT ids that are missing locally:')
            print(json.dumps(remote_hit_ids_missing_locally, indent=2))
    print(f'There were {num_warnings} warnings during the consistency check.')
    print(f'    {len(local_hit_ids_missing_remotely)} local HIT ids were not found remotely')
    return num_errors, num_warnings, local_hit_ids_missing_remotely


def get_all_hit_assignments(*, live=False,
                               hit_source='aws',
                               local_mturk_ids_to_uuid=None,
                               verbose=True,
                               progress_bar=False,
                               backup_assignments=None):
    assert hit_source in ['aws', 'local']
    client = get_mturk_client(live=live)
    if hit_source == 'aws':
        hits_to_process = get_hit_ids_and_annotations_from_aws(client)
    else:
        if local_mturk_ids_to_uuid is None:
            _, local_mturk_ids_to_uuid, _, _, _ = load_local_hit_data(live=live, include_blacklisted_hits=True)
        hits_to_process = local_mturk_ids_to_uuid
    result = {}
    num_assignments = 0
    start_time = timer()
    to_iterate = hits_to_process.items()
    if progress_bar:
        to_iterate = tqdm(list(to_iterate))
    num_hits_from_backup = 0
    num_assignments_from_backup = 0
    for hit_id, uuid in to_iterate:
        if backup_assignments is not None and uuid in backup_assignments:
            result[uuid] = backup_assignments[uuid]
            num_hits_from_backup += 1
            num_assignments_from_backup += len(result[uuid])
        else:
            result[uuid] = get_assignments_for_hit_from_aws(hit_id, client, uuid)
        num_assignments += len(result[uuid])
    end_time = timer()
    if verbose:
        print('Retrieved {} assignments for {} HITs in {:.0f} seconds.'.format(num_assignments,
                len(hits_to_process), end_time - start_time))
        print(f'    {num_hits_from_backup} HITs (with a total of {num_assignments_from_backup} assignments) were obtained from the backup')
    return result


def parse_datetime_string(s):
    tmp_res = dateutil.parser.parse(s)
    if tmp_res.tzinfo is None or tmp_res.tzinfo.utcoffset(tmp_res) is None:
        return pytz.timezone('US/Pacific').localize(tmp_res)
    else:
        return tmp_res


def show_hit_images(hit_data, image_captions, image_data, num_cols=1):
    return show_image_grid(hit_data['images_all'], image_captions, image_data, num_cols=num_cols)


def show_image_grid(filenames, image_captions, image_data, num_cols=1, max_width='300px', max_height='300px', image_box_padding='12px'):
    import ipywidgets as widgets
    image_widgets = []
    for image_name in filenames:
        cur_img_widget = widgets.Image(value=image_data[image_name],
                layout=widgets.Layout(max_width=max_width, max_height=max_height))
        cur_img_widget.add_class('hit_image')

        cur_label_widgets = []
        if image_name in image_captions:
            for cur_label_text in image_captions[image_name]:
                if isinstance(cur_label_text, str):
                    cur_label = widgets.Label(value=cur_label_text, layout=widgets.Layout(margin='0px', height='20px'))
                    cur_label.add_class('image_grid_caption')
                    cur_label_widgets.append(cur_label)
                else:
                    # Here cur_label_text is assumed to be a widget
                    cur_label_widgets.append(cur_label_text)
        if len(cur_label_widgets) > 0:
            cur_label_vbox = widgets.VBox(cur_label_widgets)
            cur_label_vbox.layout.align_items = 'center'
            cur_label_vbox.layout.padding = '0px'
            cur_box = widgets.VBox([cur_img_widget, cur_label_vbox])
        else:
            cur_box = widgets.VBox([cur_img_widget])
        cur_box.layout.justify_content = 'flex-end'
        cur_box.layout.align_items = 'center'
        cur_box.layout.padding = image_box_padding
        image_widgets.append(cur_box)

    rows = []
    num_rows = int(math.ceil(len(image_widgets) / num_cols))
    for ii in range(num_rows):
        cur_num_cols = min(num_cols, len(image_widgets) - ii * num_cols)
        cur_row = []
        for jj in range(cur_num_cols):
            cur_index = ii * num_cols + jj
            cur_row.append(image_widgets[cur_index])
        rows.append(widgets.HBox(cur_row))
        rows[-1].layout.padding = '3px'

    return widgets.VBox(rows)


def get_hit_images_css():
    import ipywidgets as widgets
    return widgets.HTML("""
<style>
.hit_image {
    object-fit: contain;
}
.widget-vbox .hit_heading {
    font-size: 36px;
    line-height: 40px;
    height: 50px;
}
</style>
""")

    
def get_nn_review_css():
    import ipywidgets as widgets
    return widgets.HTML("""
<style>
.image_name {
    font-size: 8px;
    line-height: 12px;
    height: 12px;
}
</style>
""")


def get_dataset_inspection_css():
    import ipywidgets as widgets
    return widgets.HTML("""
<style>
.image_grid_caption {
    font-size: 8px;
    line-height: 12px;
    height: 12px;
}
.widget-vbox .wnid_heading {
    font-size: 36px;
    line-height: 40px;
    height: 50px;
    margin-top: 25px;
}
</style>
""")


def sort_hits_by_time(hits):
    tmp = [(parse_datetime_string(hit['time']), hit) for hit in hits]
    sorted_tmp = sorted(tmp, reverse=True)
    return sorted_tmp


HitDescription = namedtuple('HitDescription', ['title', 'wnid_info', 'creation', 'work_delay', 'work_duration', 'assignments_summary', 'per_assignment_text', 'all_images_hash'])


def get_hit_description(cur_hit, cur_assignments, imgnet=None, id_first=False):
    wnid = cur_hit['wnid']
    synset = imgnet.class_info_by_wnid[wnid].synset
    title = 'HIT {} (HIT id {} )'.format(cur_hit['uuid'], cur_hit['hit_id'])
    if imgnet is None:
        wnid_info = ''
    else:
        wnid_info = 'wnid {}, synset "{}"'.format(wnid, ', '.join(synset))
    cur_pos_control = cur_hit['images_pos_control']
    cur_neg_control = cur_hit['images_neg_control']
    hit_start_time = parse_datetime_string(cur_hit['time'])
    creation = 'created {}'.format(hit_start_time.strftime('%Y-%m-%d %H:%M:%S %z'))
    num_val_images = len(cur_pos_control) + len(cur_neg_control)
    durations = []
    delays = []
    per_assignment_text = []
    accuracies = []
    for a_id, a_data in cur_assignments.items():
        start_time = a_data['AcceptTime']
        end_time = a_data['SubmitTime']
        duration = (end_time - start_time).total_seconds()
        durations.append(duration)
        delay = (start_time - hit_start_time).total_seconds()
        delays.append(delay)

        num_pos_correct = 0
        for img in cur_pos_control:
            if img in a_data['Answer']:
                num_pos_correct += 1
        num_neg_correct = 0
        for img in cur_neg_control:
            if img not in a_data['Answer']:
                num_neg_correct += 1
        num_correct = num_pos_correct + num_neg_correct
        accuracies.append(num_correct / num_val_images)
        if id_first:
            per_assignment_text.append('{} : {:3.0f}% control ({}/{} pos, {}/{} neg), {} selected, {:.1f} sec, status "{}"'.format(
                    a_id,
                    100.0 * num_correct / num_val_images,
                    num_pos_correct, len(cur_pos_control),
                    num_neg_correct, len(cur_neg_control),
                    len(a_data['Answer']),
                    duration, a_data['AssignmentStatus']))
        else:
            per_assignment_text.append('{:3.0f}% control ({}/{} pos, {}/{} neg), {} selected, {:.1f} sec, status "{}", id {}'.format(
                    100.0 * num_correct / num_val_images,
                    num_pos_correct, len(cur_pos_control),
                    num_neg_correct, len(cur_neg_control),
                    len(a_data['Answer']),
                    duration, a_data['AssignmentStatus'], a_id))
    if len(durations) > 0:
        min_duration = min(durations)
        max_duration = max(durations)
        avg_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
    else:
        min_duration = 0.0
        max_duration = 0.0
        avg_duration = 0.0
        median_duration = 0.0
    if len(delays) > 0:
        min_delay = min(delays)
        max_delay = max(delays)
        avg_delay = statistics.mean(delays)
        median_delay = statistics.median(delays)
    else:
        min_delay = 0.0
        max_delay = 0.0
        avg_delay = 0.0
        median_delay = 0.0
    if len(accuracies) > 0:
        avg_accuracy = statistics.mean(accuracies)
    else:
        avg_accuracy = 0.0
    duration_string = 'work duration: min {:.0f}, max {:.0f}, avg {:.0f}, median {:.0f}'.format(
            min_duration, max_duration, avg_duration, median_duration)
    delay_string = 'work delay : min {:.0f}, max {:.0f}, avg {:.0f}, median {:.0f}'.format(
            min_delay, max_delay, avg_delay, median_delay)
    assignments_summary = '{} assigment(s), {:.0f}% avg accuracy on {} validation image(s)'.format(len(cur_assignments), 100.0 * avg_accuracy, num_val_images)
    all_images_str = '|'.join(cur_hit['images_all'])
    all_images_hash = hashlib.sha1(all_images_str.encode()).hexdigest()
    return HitDescription(title, wnid_info, creation, delay_string, duration_string, assignments_summary, per_assignment_text, all_images_hash)


def compute_assignment_accuracies(hit, assignment):
    cur_pos_control = hit['images_pos_control']
    cur_neg_control = hit['images_neg_control']
    num_val_images = len(cur_pos_control) + len(cur_neg_control)
    num_pos_correct = 0
    for img in cur_pos_control:
        if img in assignment['Answer']:
            num_pos_correct += 1
    num_neg_correct = 0
    for img in cur_neg_control:
        if img not in assignment['Answer']:
            num_neg_correct += 1
    num_correct = num_pos_correct + num_neg_correct
    return num_correct / num_val_images, num_pos_correct / len(cur_pos_control), num_neg_correct / len(cur_neg_control)


def compute_assignment_duration(assignment):
    start_time = assignment['AcceptTime']
    end_time = assignment['SubmitTime']
    duration = (end_time - start_time).total_seconds()
    return duration


def compute_assignment_delay(assignment, hit):
    hit_start_time = parse_datetime_string(hit['time'])
    start_time = assignment['AcceptTime']
    delay = (start_time - hit_start_time).total_seconds()
    return delay


def sort_assignments_by_accuracy(hits, assignments):
    tmp_res = []
    for uuid, assignment_dict in assignments.items():
        for ca in assignment_dict.values():
            cur_acc = compute_assignment_accuracies(hits[uuid], ca)
            tmp_res.append((ca, cur_acc[0], cur_acc[1], cur_acc[2]))
    return sorted(tmp_res, key=lambda x: x[1])
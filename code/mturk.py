import argparse
from collections import Counter,defaultdict
from datetime import datetime, date, time
import csv
import getpass
import inspect
import json
import sys
import time
import traceback
import uuid
from dateutil.tz import tzlocal

import boto3
import botocore
import click
import numpy as np
try:
    import tqdm
except:
    print('import tqdm failed')
from urllib.parse import quote

import candidate_data
import imagenet
import utils
import mturk_data
import mturk_utils

# This is hard coded in the html file
FIELDNAMES= ["synset", "gloss", "wiki"]
HTML_TEMPLATE =\
'''
<div class="col-xs-12 col-sm-6 col-md-4">
<div class="thumbnail"><label for="checkbox{checkboxnum}"><img alt="{img}" class="img-responsive center-block img-box" src="{url} " /> <input id="checkbox{checkboxnum}" name="selected" type="checkbox" value="{img}" /> </label></div>
</div>
'''

QUESTION_HEADER =\
'''
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
<HTMLContent><![CDATA[
'''
QUESTION_FOOTER = '''
</html>
]]>
</HTMLContent>
<FrameHeight>800</FrameHeight>
</HTMLQuestion>
'''


def _generate_hits(candidates, images_per_hit=25, pos_control=0, neg_control=0, seed=0):
    ''' Generates a list of dictionaries fully specifying the HITs '''
    #assert(neg_control == 0)
    c_data = candidate_data.CandidateData()
    imagenet_data = imagenet.ImageNetData()
    with open("../data/metadata/wnid_to_most_similar_wnids.json") as f:
        neg_ids = json.loads(f.read())
    grouped_by_class = defaultdict(list)
    np.random.seed(seed)
    hits = []
    print("Num Candidates ", len(candidates))
    for c in candidates:
        c_json = c_data.all_candidates[c]
        c_wnid = c_data.all_candidates[c]["wnid"]
        grouped_by_class[c_wnid].append(c_json)
    wiki_fail = False
    for k,v in grouped_by_class.items():
        class_info = imagenet_data.class_info_by_wnid[k]
        if (len(class_info.wikipedia_pages) == 0):
            print(f"no wikipedia page for {k}")
            wiki_fail = True
        hit_lines = list(utils.chunks(v, images_per_hit - pos_control - neg_control))
        tail_len = len(hit_lines[-1])

        if (tail_len != len(hit_lines[0]) and tail_len  < (images_per_hit - pos_control - neg_control)):
            idxs = np.random.choice(len(v) - tail_len, images_per_hit - tail_len -pos_control - neg_control, replace=False)
            for i in idxs:
                hit_lines[-1].append(v[i])

        for hit_line in hit_lines:
            hit_data = {}
            hit_data["wnid"] = k
            # list of image ids
            hit_data["images_to_label"] = []
            hit_data["images_pos_control"] = []
            hit_data["images_neg_control"] = []
            hit_data["images_all"] = []
            hit_data["user"] = getpass.getuser()
            hit_data["uuid"] = str(uuid.uuid4())
            hit_data["time"] = str(datetime.now(tzlocal()))
            hit_data["submitted"] = False
            hit_data["hit_id"] = ''
            hit_data["hit_type_id"] = ''

            val_imgs_dict  = imagenet_data.val_imgs_by_wnid
            wnid = k
            pos_class = val_imgs_dict[wnid]
            pos_extra = int(np.ceil((images_per_hit - pos_control - neg_control - len(hit_line))/2))
            neg_extra = int(np.floor((images_per_hit - pos_control - neg_control - len(hit_line))/2))
            if (len(hit_line) == images_per_hit - pos_control - neg_control):
                assert (pos_extra == 0)
                assert (neg_extra == 0)

            idxs_pos = np.random.choice(len(pos_class), pos_control + pos_extra, replace=False)
            if (wnid not in neg_ids):
                assert False

            neg_wnid = neg_ids[wnid][1]

            neg_class = val_imgs_dict[neg_wnid]
            idxs_neg = np.random.choice(len(neg_class), neg_control + neg_extra, replace=False)
            #idxs_neg = []
            pos_control_list = []
            neg_control_list = []

            for i in idxs_pos:
                pos_control_list.append(pos_class[i])
            for i in idxs_neg:
                neg_control_list.append(neg_class[i])


            for i,image in enumerate(hit_line):
                hit_data["images_to_label"].append(image['id_ours'])
                hit_data["images_all"].append(image['id_ours'])

            # right now this won't work
            for i,image in enumerate(pos_control_list):
                hit_data["images_pos_control"].append(image)
                hit_data["images_all"].append(image)

            for i,image in enumerate(neg_control_list):
                hit_data["images_neg_control"].append(image)
                hit_data["images_all"].append(image)
            np.random.shuffle(hit_data["images_all"])
            hits.append(hit_data)
    if (wiki_fail):
        assert False
    return hits


def generate_hit_html(hit_data, html_template_path, html_style_path, add_question_header=True):
    imagenet_data = imagenet.ImageNetData()
    with open(html_template_path, "r") as f:
        html_text = f.read()
    with open(html_style_path, "r") as f:
        style_text = f.read()
    htmls = {}
    for hit in hit_data:
        out_html = ''
        if (add_question_header):
            out_html += QUESTION_HEADER
        wnid = hit["wnid"]
        class_info = imagenet_data.class_info_by_wnid[wnid]
        synset = class_info.synset
        gloss = class_info.gloss
        wikipedia_pages = class_info.wikipedia_pages
        wikipedia_page = ", ".join(['<a href="{0}">{1}</a>'.format(x, x) for x in wikipedia_pages])
        synset = " or ".join(synset)
        image_html = ''
        for i,image in enumerate(hit["images_all"]):
            if (image in hit["images_to_label"]):
                # image is an id
                encrypted_image =  utils.encrypt_string_with_magic(image)
                image_decrypted =  utils.decrypt_string_with_magic(encrypted_image)
                assert(image_decrypted == image)
                encrypted_image_quoted = quote(encrypted_image)
                s3_link = "https://s3-us-west-2.amazonaws.com/imagenet2datav2/encrypted/{0}".format(encrypted_image_quoted)  + ".jpg"
                #print("S3 links ", s3_link)
            else:
                encrypted_image = utils.encrypt_string_with_magic(image)
                image_decrypted = utils.decrypt_string_with_magic(encrypted_image)
                assert(image_decrypted == image)
                encrypted_image_quoted= quote(encrypted_image)
                s3_link = "https://s3-us-west-2.amazonaws.com/imagenet2datav2/encrypted/{0}".format(encrypted_image_quoted) + ".jpg"
            html = HTML_TEMPLATE.format(img=encrypted_image, url=s3_link, checkboxnum=i)
            image_html += html
            image_html += "\n"
        html_body = html_text.format(image_data=image_html, synset=synset, gloss=gloss, wiki=wikipedia_page)
        out_html += html_body
        out_html += style_text
        if (add_question_header):
            out_html += QUESTION_FOOTER
        htmls[hit["uuid"]] = out_html
    return htmls

def _submit_hits(hit_data,
                hit_htmls,
                live=False,
               auto_approval_delay=24*3600*3,
               assignment_duration=3600,
               reward='0.30',
               title='Label new images for machine learning dataset',
               keywords='image, label, machine learning, artificial intelligence',
               description='Select images containing objects of the appropriate type',
               max_assignments=10,
               life_time=999999):
            client = mturk_utils.get_mturk_client(live)
            if (live):
                resp = input("You are about to submit a live hit, please type in the word LIVE (in all capitals): ")
                if (resp.strip() != "LIVE"):
                    exit()
            hit_type_response = client.create_hit_type(
            AutoApprovalDelayInSeconds=auto_approval_delay,
            AssignmentDurationInSeconds=assignment_duration,
            Reward=reward,
            Title=title,
            Keywords=keywords,
            Description=description)
            hit_type_id = hit_type_response["HITTypeId"]
            for hit in hit_data:
                try:
                    response = client.create_hit_with_hit_type(
                               HITTypeId=hit_type_id,
                               RequesterAnnotation=hit["uuid"],
                               UniqueRequestToken=hit["uuid"],
                               MaxAssignments=max_assignments,
                               LifetimeInSeconds=life_time,
                               Question=hit_htmls[hit['uuid']])
                    hit["hit_id"] = response['HIT']['HITId']
                    hit["hit_type_id"] = response['HIT']['HITTypeId']
                    hit["user"] = getpass.getuser()
                    hit["uuid"] = str(hit['uuid'])
                    hit["time"] = str(datetime.now(tzlocal()))
                    hit["submitted"] = True
                    print("\nCreated HIT: {}".format(response['HIT']['HITId']))
                except Exception as e:
                    print("hit {0} failed with error {1}, perhaps this hit already exists?".format(hit["uuid"], e))
                    traceback.print_exc()
            return hit_data


@click.group()
@click.option('--args', default=None)
@click.pass_context
def cli(ctx, args):
    ctx.obj = {'args' : args}


@click.group()
@click.pass_context
def hit(ctx):
    pass


@click.command()
@click.pass_context
@click.argument('candidates')
@click.option('--out_file_name', default=None)
@click.option('--images_per_hit', default=None, type=int)
@click.option('--num_pos_control', default=None, type=int)
@click.option('--num_neg_control', default=None, type=int)
@click.option('--seed', default=0, type=int)
@click.option('--cache_args', default=None, type=str)
def generate_hits(ctx, candidates, out_file_name, images_per_hit, num_pos_control, num_neg_control, seed, cache_args):
    ''' Generate a json list of hit info, used downstream in hit generation pipeline'''
    #TODO fix
    with open(candidates, "r") as f:
        print("filename", candidates)
        c_list = json.load(f)

    if (ctx.obj['args'] is not None):
        with open(ctx.obj['args']) as f:
            arg_dict  = json.load(f)
            generate_hits_args = arg_dict["generate_hits"]
    else:
        generate_hits_args= {}

    if (images_per_hit is not None): generate_hits_args['images_per_hit'] = images_per_hit
    if (num_pos_control is not None): generate_hits_args['pos_control'] = num_pos_control
    if (num_neg_control is not None): generate_hits_args['neg_control'] = num_neg_control
    if (seed is not None): generate_hits_args['seed'] = seed

    if (cache_args is not None):
      with open(cache_args, "w+") as f:
          f.write(json.dumps(arg_dict, indent=2))

    hits = _generate_hits(c_list, **generate_hits_args)
    if (out_file_name is not None):
        with open(out_file_name, "w+") as f:
            f.write(json.dumps(hits, indent=2))
    else:
        sys.stdout.write(json.dumps(hits, indent=2))

@click.command()
@click.pass_context
@click.option('--hit_data_file_name', default=None)
@click.option('--out_file_name', default=None)
@click.option('--index', default=0, type=int)
@click.option('--html_template', default="mturk/hit_template.html")
@click.option('--style_template', default="mturk/style.html")
def generate_hit_htmls(ctx, hit_data_file_name, out_file_name, index, html_template, style_template):
    ''' Debugging function to verify html'''
    if hit_data_file_name is None:
        stdin_text = click.get_text_stream('stdin')
        hit_data = json.loads(stdin_text.read())
        hit_id = hit_data[index]
    else:
        with open(hit_data_file_name) as f:
            hit_data = json.load(f)
            hit_id = hit_data[index]
    hit_html = generate_hit_html(hit_data, html_template, style_template, add_question_header=False)[hit_id['uuid']]
    if out_file_name is not None:
        with open(out_file_name, "w+") as f:
            f.write(hit_html)
    else:
        sys.stdout.write(hit_html)


def _list_hits(live, max_results):
    client = mturk_utils.get_mturk_client(live)
    hit_response = client.list_hits(MaxResults=max_results)
    all_hits = []
    while hit_response['NumResults'] > 0:
        all_hits += hit_response["HITs"]
        hit_response = client.list_hits(MaxResults=max_results, NextToken=hit_response['NextToken'])
    for hit in all_hits:
        hit['Expiration'] = str(hit['Expiration'])
        hit['CreationTime'] = str(hit['CreationTime'])
    return all_hits


def _delete_hits(hits, live):
    assert not live
    live_data = mturk_data.MTurkData(live=True, load_assignments=False, verbose=False)
    live_hit_ids = []
    for hit in live_data.hits.values():
        live_hit_ids.append(hit['hit_id'])
    live_hit_ids = set(live_hit_ids)
    for hit_id in hits:
        assert hit_id not in live_hit_ids
    client = mturk_utils.get_mturk_client(live)
    num_deleted = 0
    for hit_id in tqdm.tqdm(hits, desc='Deleting HITs'):
        try:
            client.update_expiration_for_hit(HITId=hit_id, ExpireAt=datetime(2018, 1, 1))
            response = client.delete_hit(HITId=hit_id)
            assert response['ResponseMetadata']['HTTPStatusCode'] == 200
            num_deleted += 1
        except Exception as e:
            print('ERROR while deleting the HIT with id {} (type {})'.format(
                    hit_id, type(e)))
            print('    ' + str(e))
    print('Deleted {} HITs'.format(num_deleted))


@click.command()
@click.option('--live', is_flag=True)
@click.option('--max_results', default=100)
@click.option('--out_file_name', default=None)
def list_hits(live, max_results, out_file_name):
    ''' outputs a json list'''
    all_hits = _list_hits(live, max_results)
    if (out_file_name is not None):
        with open(out_file_name) as f:
            f.write(json.dumps(all_hits, indent=2))
    else:
        sys.stdout.write(json.dumps(all_hits, indent=2))


@click.command()
@click.option('--hits_json', default=None)
@click.option('--live', is_flag=True)
def delete_hits(live, hits_json):
    ''' takes json from list hits as argument'''
    assert not live
    if hits_json is None:
        stdin_text = click.get_text_stream('stdin')
        hits = json.load(stdin_text)
    else:
        with open(hits_json) as f:
            hits = json.load(f)
    _delete_hits(hits, live=live)


@click.command()
@click.pass_context
@click.option('--assignment_file_name', default=None)
@click.option('--live', is_flag=True)
@click.option('--override_rejection', is_flag=True)
def approve_assignments(ctx,
                        assignment_file_name,
                        live,
                        override_rejection):
    with open(assignment_file_name, 'r') as f:
        assignments = json.load(f)
    mturk_utils.approve_assignments(assignments, override_rejection, live=live)
    print('Approved {} assignments.'.format(len(assignments)))


@click.command()
@click.pass_context
@click.option('--assignment_file_name', default=None)
@click.option('--live', is_flag=True)
def reject_assignments(ctx,
                       assignment_file_name,
                       live):
    with open(assignment_file_name, 'r') as f:
        assignments = json.load(f)
    mturk_utils.reject_assignments(assignments, live=live)
    print('Rejected {} assignments.'.format(len(assignments)))


@click.command()
@click.pass_context
@click.option('--live', is_flag=True)
def consistency_check(ctx,
                      live):
    mturk_utils.mturk_vs_local_consistency_check(live)


@click.command()
@click.pass_context
@click.option('--hit_file', required=True, multiple=True, type=str)
@click.option('--live', is_flag=True)
def show_hit_progress(ctx,
                      hit_file,
                      live):
    filenames = hit_file
    hits = []
    for fn in filenames:
        with open(fn, 'r') as f:
            cur_hits = json.load(f)
        hits.extend(cur_hits)
    print('Loaded {} HITs from {} files'.format(len(hits), len(filenames)))
    assignments = {}
    client = mturk_utils.get_mturk_client(live=live)
    for hit in tqdm.tqdm(hits, desc='Querying HITs'):
        assignments[hit['uuid']] = mturk_utils.get_assignments_for_hit_from_aws(
                hit['hit_id'], client, hit['uuid'])
    assignment_counts = []
    for a in assignments.values():
        cur_submitted = [x for x in a.values() if x['AssignmentStatus'] in ['Approved', 'Submitted']]
        assignment_counts.append(len(cur_submitted))
    counter = Counter(assignment_counts)
    for count, freq in counter.most_common():
        print('{} HITs have {} submitted assignments'.format(freq, count))


@click.command()
@click.pass_context
@click.option('--hit_data_file')
@click.option('--out_file_name', default=None)
@click.option('--index', default=0, type=int)
@click.option('--html_template', default="mturk/hit_template.html")
@click.option('--style_template', default="mturk/style.html")
@click.option('--live', is_flag=True)
@click.option('--auto_approval_delay', default=None)
@click.option('--assignment_duration', default=None)
@click.option('--reward', default=None)
@click.option('--title', default=None)
@click.option('--keywords', default=None)
@click.option('--description', default=None)
@click.option('--max_assignments', default=None)
@click.option('--life_time', default=None)
@click.option('--cache_args', default=None)
def submit_hits(ctx,
              hit_data_file,
              out_file_name,
              index,
              html_template,
              style_template,
              live,
              auto_approval_delay,
              assignment_duration,
              reward,
              title,
              keywords,
              description,
              max_assignments,
              life_time,
              cache_args):
    ''' Debugging function to verify html'''
    t = time.time()
    if (hit_data_file is not None):
        with open(hit_data_file, "r") as f:
            hit_data = json.loads(f.read())
    else:
        stdin_text = click.get_text_stream('stdin')
        hit_data = json.load(stdin_text)
    e = time.time()
    print(e - t)
    t = time.time()
    hit_htmls = generate_hit_html(hit_data, html_template, style_template)
    print(f"Launching {len(hit_htmls)} hits")
    e = time.time()
    print("HIT_HTML time", e - t)
    print("generated ")
    if (ctx.obj['args'] is not None):
        with open(ctx.obj['args']) as f:
            arg_dict  = json.load(f)
            submit_hits_args = arg_dict["submit_hits"]
    else:
        submit_hits_args = {}
    if (live is not None): submit_hits_args["live"] = live
    if (auto_approval_delay is not None): submit_hits_args["auto_approval_delay"] = auto_approval_delay
    if (assignment_duration is not None): submit_hits_args["assignment_duration"] = assignment_duration
    if (reward is not None): submit_hits_args["reward"] = reward
    if (title is not None): submit_hits_args["title"] = title
    if (keywords is not None): submit_hits_args["keywords"] = keywords
    if (description is not None): submit_hits_args["description"] = description
    if (max_assignments is not None): submit_hits_args["max_assignments"] = max_assignments
    if (life_time is not None): submit_hits_args["life_time"] = life_time

    if (cache_args is not None):
      with open(cache_args, "w+") as f:
          f.write(json.dumps(arg_dict))
    res = _submit_hits(hit_data, hit_htmls, **submit_hits_args)
    if (out_file_name is not None):
        with open(out_file_name, "w+") as f:
            f.write(json.dumps(res, indent=2))
    else:
        sys.stdout.write(json.dumps(res, indent=2))





cli.add_command(generate_hits)
cli.add_command(generate_hit_htmls)
cli.add_command(submit_hits)
cli.add_command(list_hits)
cli.add_command(delete_hits)
cli.add_command(approve_assignments)
cli.add_command(reject_assignments)
cli.add_command(consistency_check)
cli.add_command(show_hit_progress)


if __name__ == "__main__":
    cli() # pylint: disable=no-value-for-parameter







import datetime
import getpass
import json
import time
from timeit import default_timer as timer

import tqdm

import mturk_utils


client = mturk_utils.get_mturk_client(live=True)
max_num_results = 100

api_results = []

all_hit_ids = []

print('Retrieving HITs ...')
last_hit_count_print = 0
last_hit_count_print_time = timer()

res = client.list_hits(MaxResults=max_num_results)
cur_call = {
    'method': 'list_hits',
    'arguments': {'MaxResults': max_num_results},
    'result': res
}
api_results.append(cur_call)

for hit in res['HITs']:
    all_hit_ids.append(hit['HITId'])

next_token = None
if 'NextToken' in res:
    next_token = res['NextToken']
while next_token is not None:
    res = client.list_hits(MaxResults=max_num_results, NextToken=next_token)
    cur_call = {
        'method': 'list_hits',
        'arguments': {'MaxResults': max_num_results, 'NextToken': next_token},
        'result': res
    }
    api_results.append(cur_call)
    for hit in res['HITs']:
        all_hit_ids.append(hit['HITId'])
    if len(all_hit_ids) - last_hit_count_print > 2000:
        hits_per_second = (len(all_hit_ids) - last_hit_count_print) / (timer() - last_hit_count_print_time)
        print(f'    retrieved {len(all_hit_ids)} HITs so far ({hits_per_second:.1f} HITs per second)')
        last_hit_count_print = len(all_hit_ids)
        last_hit_count_print_time = timer()
    next_token = None
    if 'NextToken' in res:
        next_token = res['NextToken']

assert len(all_hit_ids) == len(set(all_hit_ids))
print(f'Retrieved {len(all_hit_ids)} HITs in total')
print('Retrieving corresponding assignments ...')

for hit_id in tqdm.tqdm(all_hit_ids):
    cur_result = client.list_assignments_for_hit(HITId=hit_id,
                                                 MaxResults=max_num_results)
    assert cur_result['NumResults'] < max_num_results
    cur_call = {
        'method': 'list_assignments_for_hit',
        'arguments': {
            'HITId': hit_id,
            'MaxRresults': max_num_results
        },
        'result': cur_result
    }
    api_results.append(cur_call)

time_string = datetime.datetime.now(datetime.timezone.utc).isoformat()

api_data = {
    'username': getpass.getuser(),
    'time': time_string,
    'results': api_results
}

def datetime_serializer(dt):
    assert type(dt) is datetime.datetime
    return dt.isoformat()

with open(f'api_data_{time_string}.json', 'w') as f:
    json.dump(api_data, f, sort_keys=True, indent=2, default=datetime_serializer)
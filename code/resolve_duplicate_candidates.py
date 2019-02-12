import copy
from datetime import datetime, timezone
import hashlib
import json
import pickle
import time

import candidate_data


cds = candidate_data.CandidateData(exclude_blacklisted_candidates=False, verbose=True)

all_cds = list(cds.all_candidates.values()) + cds.duplicates

for c in all_cds:
    assert c['search_engine'] == 'flickr'

cds_by_flickr_id = {}

for c in cds.all_candidates.values():
    cur_flickr_id = c['id_search_engine']
    if cur_flickr_id not in cds_by_flickr_id:
        cds_by_flickr_id[cur_flickr_id] = []
    cds_by_flickr_id[cur_flickr_id].append(c)


relevant_duplicates_by_flickr_id_wnid = {}
num_duplicates_skipped = 0
num_duplicates_proposed = 0
for c in cds.duplicates:
    cur_flickr_id = c['id_search_engine']
    cur_wnid = c['wnid']
    assert cur_flickr_id in cds_by_flickr_id
    if cur_wnid in [x['wnid'] for x in cds_by_flickr_id[cur_flickr_id]]:
        num_duplicates_skipped += 1
        continue
    if cur_flickr_id not in relevant_duplicates_by_flickr_id_wnid:
        relevant_duplicates_by_flickr_id_wnid[cur_flickr_id] = {}
    if cur_wnid in relevant_duplicates_by_flickr_id_wnid[cur_flickr_id]:
        num_duplicates_skipped += 1
    else:
        relevant_duplicates_by_flickr_id_wnid[cur_flickr_id][cur_wnid] = c
        num_duplicates_proposed += 1

assert num_duplicates_proposed + num_duplicates_skipped == len(cds.duplicates)

proposed_candidates = []
for tmp_dict in relevant_duplicates_by_flickr_id_wnid.values():
    for cur_c in tmp_dict.values():
        proposed_candidates.append(cur_c)
assert len(proposed_candidates) == num_duplicates_proposed

proposed_candidates_final = []
proposed_ids = set([])
for c in proposed_candidates:
    new_c = copy.deepcopy(c)
    new_id = hashlib.sha1((c['id_search_engine'] + ',' + c['search_engine'] + ',' + c['wnid']).encode()).hexdigest()
    assert new_id not in cds.all_candidates
    assert new_id not in proposed_ids
    proposed_ids.add(new_id)
    new_c['id_ours'] = new_id
    if 'batch' in new_c:
        del new_c['batch']
    proposed_candidates_final.append(new_c)

cur_flickr_id_wnid_pairs = set([])
for c in cds.all_candidates.values():
    to_add = (c['id_search_engine'], c['wnid'])
    assert to_add not in cur_flickr_id_wnid_pairs
    cur_flickr_id_wnid_pairs.add(to_add)
proposed_flickr_id_wnid_pairs = set([])
for c in proposed_candidates_final:
    cur_pair = (c['id_search_engine'], c['wnid'])
    assert cur_pair not in cur_flickr_id_wnid_pairs
    assert c['id_search_engine'] in cds_by_flickr_id
    assert cur_pair not in proposed_flickr_id_wnid_pairs
    proposed_flickr_id_wnid_pairs.add(cur_pair)

final_wnids = set([x['wnid'] for x in proposed_candidates_final])

output_filename = 'proposed_relevant_duplicate_candidates.json'
with open(output_filename, 'w') as f:
    json.dump(proposed_candidates_final, f, indent=2)

print('{} duplicates total'.format(len(cds.duplicates)))
print('Skipping {} duplicates, proposing to add {} as new candidates ({} wnids)'.format(num_duplicates_skipped,
                                                                                        num_duplicates_proposed,
                                                                                        len(final_wnids)))
print('Wrote proposed candidates to {}'.format(output_filename))



from datetime import datetime, timezone
import getpass
import pickle
import time

import mturk_utils
import utils

try:
    from tqdm import tqdm
    pass
except:
    pass

live = True

bucket = 'imagenet2datav2'

print('Running consistency check:')
num_errors, num_warnings, local_hit_ids_missing_remotely = mturk_utils.mturk_vs_local_consistency_check(live=live)
assert num_errors == 0
assert num_warnings == len(local_hit_ids_missing_remotely)

# TODO: handle the blacklist correctly (do not include in the HITs)
hits, mturk_ids_to_uuid, json_dir, json_filenames, blacklisted_hits = mturk_utils.load_local_hit_data(live=live,
                                                                                                      verbose=True,
                                                                                                      include_blacklisted_hits=True)
client = mturk_utils.get_mturk_client(live=live)

backup_s3_key = 'mturk_results/data_live_2018-12-04_17-24-42_UTC.pickle'
backup_bytes = utils.get_s3_file_bytes(backup_s3_key, verbose=True)
backup_data = pickle.loads(backup_bytes)

backup_assignments = {}
for hit_id in local_hit_ids_missing_remotely:
    cur_uuid = mturk_ids_to_uuid[hit_id]
    backup_assignments[cur_uuid] = backup_data['assignments'][cur_uuid]
print(f'Took assignment data for {len(backup_assignments)} HITs from the backup {backup_s3_key}')

assignments = mturk_utils.get_all_hit_assignments(live=live,
                                                  hit_source='local',
                                                  local_mturk_ids_to_uuid=mturk_ids_to_uuid,
                                                  verbose=True,
                                                  progress_bar=True,
                                                  backup_assignments=backup_assignments)

time_string = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%Z')
key = 'mturk_results/data_{}_'.format('live' if live else 'sandbox') + time_string + '.pickle'

pickle_dict = {}
pickle_dict['hits'] = hits
pickle_dict['blacklisted_hits'] = blacklisted_hits
pickle_dict['mturk_ids_to_uuid'] = mturk_ids_to_uuid
pickle_dict['assignments'] = assignments
pickle_dict['json_dir'] = str(json_dir)
pickle_dict['json_filenames'] = [str(fn) for fn in json_filenames]
pickle_dict['username'] = getpass.getuser()
pickle_dict['backup_s3_key'] = backup_s3_key
pickle_dict['local_hit_ids_missing_remotely'] = local_hit_ids_missing_remotely
pickle_dict['time_string'] = time_string
pickle_bytes = pickle.dumps(pickle_dict)

utils.put_s3_object_bytes_with_backoff(pickle_bytes, key, bucket=bucket)
print('Wrote metadata pickle file to s3://{}/{}'.format(bucket, key))
print('    {} bytes in total'.format(len(pickle_bytes)))
print('    HIT data JSON filenames:')
for fn in json_filenames:
    print('        {}'.format(fn.name))
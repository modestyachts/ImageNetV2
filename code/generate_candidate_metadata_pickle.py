from datetime import datetime, timezone
import getpass
import pickle
import time

import pywren
from pywren import wrenconfig as wc

import candidate_data
import utils

try:
    from tqdm import tqdm
    pass
except:
    pass

bucket = 'imagenet2datav2'
num_replicas = 10
use_pywren_for_replicas = False

json_dir, json_data, blacklist = candidate_data.load_data()

time_string = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%Z')
key = 'metadata/candidate_metadata_' + time_string + '.pickle'

pickle_dict = {}
pickle_dict['json_data'] = json_data
pickle_dict['json_dir'] = json_dir
pickle_dict['blacklist'] = blacklist
pickle_dict['username'] = getpass.getuser()
pickle_dict['time_string'] = time_string

pickle_bytes = pickle.dumps(pickle_dict)

utils.put_s3_object_bytes_with_backoff(pickle_bytes, key, bucket=bucket)

if num_replicas > 1:
    destinations = []
    replicas_counter_len = len(str(num_replicas))
    format_string = '_replica{{:0{}d}}-{{}}'.format(replicas_counter_len)
    for ii in range(num_replicas):
        destinations.append(key + format_string.format(ii + 1, num_replicas))
    
    if use_pywren_for_replicas:
        def s3_cp(dest):
            data, _ = utils.get_s3_object_bytes_with_backoff(key, bucket=bucket)
            utils.put_s3_object_bytes_with_backoff(data, dest, bucket=bucket)
            return dest
        
        pywren_config = wc.default()
        pywren_config['runtime']['s3_bucket'] = 'imagenet2pywren'
        pywren_config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-imagenet2pywren.tar.gz'
        pwex = pywren.default_executor(config=pywren_config)
        pbar = tqdm(total=len(destinations))
        futures = pwex.map(s3_cp, destinations, exclude_modules=['site-packages'])
        last_status = 0
        done, not_done = pywren.wait(futures)
        while len(not_done) > 0:
            ALWAYS = 3
            done, not_done = pywren.wait(futures, ALWAYS)
            pbar.update(len(done) - last_status)
            last_status = len(done)
            time.sleep(1)
        all_results = []
        for res in done:
            all_results.append(res.result())
    else:
        for dest in tqdm(destinations):
            utils.put_s3_object_bytes_with_backoff(pickle_bytes, dest, bucket=bucket)

print('Wrote metadata pickle file to s3://{}/{}'.format(bucket, key))
print('    {} bytes in total'.format(len(pickle_bytes)))
print('    Candidate JSON filenames:')
for fn, _ in json_data:
    print('        {}'.format(fn))
print('    Also stored {} replicas'.format(num_replicas))
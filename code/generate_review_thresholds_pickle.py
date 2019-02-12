from datetime import datetime, timezone
import getpass
import pickle
import time

import candidate_data
import imagenet
import mturk_utils
import near_duplicate_data
import utils

bucket = 'imagenet2datav2'

imgnet = imagenet.ImageNetData()
cds = candidate_data.CandidateData(exclude_blacklisted_candidates=False)

review_thresholds, missing_nn_results = near_duplicate_data.compute_review_thresholds(cds, imgnet)

time_string = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%Z')
key = 'review_thresholds/data_' + time_string + '.pickle'

pickle_dict = {}
pickle_dict['review_thresholds'] = review_thresholds
pickle_dict['missing_nn_results'] = missing_nn_results
pickle_dict['username'] = getpass.getuser()
pickle_dict['time_string'] = time_string
pickle_bytes = pickle.dumps(pickle_dict)

utils.put_s3_object_bytes_with_backoff(pickle_bytes, key, bucket=bucket)
print('Wrote review thresholds pickle file to s3://{}/{}'.format(bucket, key))
print('    {} bytes in total'.format(len(pickle_bytes)))
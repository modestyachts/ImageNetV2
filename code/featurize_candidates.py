import argparse
import io
import os
import pickle
import tarfile
import time
from timeit import default_timer as timer
import json
import boto3
import numpy as np
import skimage.transform
import torch
import torchvision.models as models
from torch.autograd import Variable
from torch import nn

import candidate_data
import imagenet
import featurize
import mturk_data
import utils


FEATURIZE_SIZE = (224, 224)
def featurize_candidates(bucket, prefix, batch_size, source_filename):
    imgnt = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(verbose=False)
    filenames_to_ignore = [
        '2018-08-06_17:33_vaishaal.json',
        '2018-08-17_17:24_vaishaal.json',
        'vaishaal_hits_submitted_2018-08-17-18:28:33-PDT.json',
        'vaishaal_hits_submitted_2018-08-17-18:50:38-PDT.json',
        'vaishaal_hits_submitted_2018-08-17-19:28:24-PDT.json',
        'vaishaal_hits_submitted_2018-08-17-19:56:28-PDT.json',
        'vaishaal_hits_submitted_2018-08-25-09:47:26-PDT.json']
    mturk = mturk_data.MTurkData(live=True, load_assignments=True, source_filenames_to_ignore=filenames_to_ignore, verbose=False)
    to_featurize = []
    to_featurize_keys = []
    client = utils.get_s3_client()
    i = 0
    #candidate_list = dataset_sampling.get_histogram_sampling_ndc_candidates(imgnet=imgnt, cds=cds, mturk=mturk)
    start = timer()
    with open('../data/metadata/fc7_candidates.json', 'r') as f:
        candidate_list = json.load(f)
    for k in candidate_list:
        key_name = os.path.join(prefix, str(k)+".npy")
        key_exists = utils.key_exists(bucket, key_name)
        if not key_exists:
            img = cds.load_image(k, size='original', verbose=False)
            img  = skimage.transform.resize(img, FEATURIZE_SIZE, preserve_range=True)
            to_featurize.append(img)
            to_featurize_keys.append(k)
            #if i > 250:
            #    break;
            i = i + 1
            print('Got candidate {}'.format(i))
    end = timer()
    print(f"Took {end-start} seconds to get remaining candidates.")
    print('Beginning featurization of {} items'.format(len(to_featurize_keys)))
    if len(to_featurize) > 0:
        to_featurize = np.stack(to_featurize, axis=0)
        print(f"input shape {to_featurize.shape}")
        batch_size = min(len(to_featurize), batch_size)
        features = featurize.vgg16_features(to_featurize, batch_size=batch_size)
        print(f"features shape {features.shape}")
        for i,f in enumerate(features):
            key_name = os.path.join(prefix, to_featurize_keys[i]+".npy")
            bio = io.BytesIO()
            np.save(bio, f)
            print("writing key {0}".format(key_name))
            utils.put_s3_object_bytes_with_backoff(bio.getvalue(), key_name)
    print(f"Took {end-start} seconds to get remaining candidates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("featurize candidate images if not exist")
    parser.add_argument("--bucket", default="imagenet2datav2")
    parser.add_argument("--prefix", default="imagenet2candidates_featurized")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--source_filename", type=str)
    args = parser.parse_args()
    featurize_candidates(args.bucket, args.prefix, args.batch_size, args.source_filename)

import argparse
import io
import pickle
import tarfile
import time
from timeit import default_timer as timer

import boto3
import numpy as np
import skimage.transform
import torch
import torchvision.models as models
from torch.autograd import Variable
from torch import nn

import candidate_data
import featurize
import imagenet
import utils


def featurize_and_upload_batch(to_featurize, to_featurize_keys, batch_size,  bucket, prefix, client):
    start = timer()
    to_featurize = np.stack(to_featurize, axis=0)
    print(f"input shape {to_featurize.shape}")
    batch_size = min(len(to_featurize), batch_size)
    end = timer()
    print('Stacking took ', end-start)
    start = timer()
    features = featurize.vgg16_features(to_featurize, batch_size=batch_size, use_gpu=True)
    end = timer()
    print('Featurization took ', end-start)
    print(f"features shape {features.shape}")
    start = timer()
    for i,f in enumerate(features):
	    key_name = os.path.join(prefix, f"{to_featurize_keys[i]}.npy")
	    bio = io.BytesIO()
	    np.save(bio, f)
	    client.put_object(Key=key_name, Bucket=bucket, Body=bio.getvalue(), ACL="bucket-owner-full-control")
    end = timer()
    print('Uploading took ', end-start)


FEATURIZE_SIZE = (224, 224)
def featurize_test_images(bucket, prefix, batch_size):
    imgnt = imagenet.ImageNetData()
    to_featurize = []
    to_featurize_keys = []
    client = utils.get_s3_client()
    start = timer()
    num_batches = 0
    for k in imgnt.test_filenames:
        key_name = os.path.join(prefix, f"{k}.npy")
        key_exists = utils.key_exists(bucket, key_name)
        if not key_exists:
            img = imgnt.load_image(k, size='scaled_256', force_rgb=True, verbose=False)
            img  = skimage.transform.resize(img, FEATURIZE_SIZE, preserve_range=True)
            to_featurize.append(img)
            to_featurize_keys.append(k)
            if len(to_featurize) >= batch_size:
                num_batches += 1
                featurize_and_upload_batch(to_featurize, to_featurize_keys, batch_size, bucket, prefix, client)
                end = timer()
                print('processing bach {} (size {}) took {} seconds'.format(num_batches, len(to_featurize), end-start))
                start = timer()
                to_featurize = []
                to_featurize_keys = []
    if len(to_featurize) > 0:
        featurize_and_upload_batch(to_featurize, to_featurize_keys, batch_size, bucket, prefix, client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("featurize test images if not exist")
    parser.add_argument("--bucket", default="imagenet2datav2")
    parser.add_argument("--prefix", default="imagenet-test-featurized")
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()
    featurize_test_images(args.bucket, args.prefix, args.batch_size)

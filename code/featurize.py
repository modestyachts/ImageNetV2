import io
import pickle
import sys
import tarfile
import time

import boto3
import imageio
import numpy as np
import skimage.transform
import torch
from torch.autograd import Variable
from torch import nn
import torchvision.models as models
import utils

def vgg16_features(images, batch_size=60, use_gpu=True):
    model = models.vgg16(pretrained=True)
    model.eval()
    results = []
    if (use_gpu):
        model = model.cuda()
    for i,images_chunk in enumerate(utils.chunks(images, batch_size)):
        if (len(images_chunk.shape) < 4):
            images_chunk = images[np.newaxis, :, :, :]
        images_chunk = images_chunk.astype('float32').transpose(0,3,1,2)
        images_torch = Variable(torch.from_numpy(images_chunk))
        if (use_gpu):
            images_torch = images_torch.cuda()
        x = model.features(images_torch)
        x = x.view(x.size(0), -1)
        fc7_net = torch.nn.Sequential(*list(model.classifier)[:-1])
        if (use_gpu):
            fc7_net = fc7_net.cuda()
        x = fc7_net(x)
        x = x.cpu().data.numpy()
        results.append(x)
        if (use_gpu):
            torch.cuda.empty_cache()
    return np.concatenate(results, axis=0)

def featurize_test(test_keys, batch_size=64):
    images = []
    for img_name in test_keys:
        try:
            file_bytes = utils.get_s3_file_bytes(img_name, verbose=False)
            image = imageio.imread(file_bytes)
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    print('Removing alpha channel for image', name)
                    image = image[:,:,:3]
            elif len(image.shape) == 2:
                image = np.stack((image,image,image), axis=2)
            if image.size!= 196608:
                print(img_name)
                raise
            image = skimage.transform.resize(image, (224, 224), preserve_range=True)
        except:
            print('Exception: '+ str(img_name) + str(sys.exc_info()[0]))
            raise
        images.append(image)
    images = np.stack(images, axis=0)
    print('Beginning featurization')
    features = vgg16_features(images, batch_size=batch_size)
    write_test_output(test_keys, features)
    return features

def write_test_output(test_keys, features, bucket="imagenet2datav2"):
    client = utils.get_s3_client()
    for idx in range(features.shape[0]):
        filename = test_keys[idx].split('.')[0].split('/')[1]
        key = 'imagenet-test-featurized-2/' + filename + '.npy'
        bio = io.BytesIO()
        np.save(bio, features[idx])
        bstream = bio.getvalue()
        client.put_object(Bucket=bucket, Key=key, Body=bstream, ACL="bucket-owner-full-control")
    print('Done writing features')


def featurize_s3_tarball(tarball_key, bucket="imagenet2datav2", batch_size=32):
    client = utils.get_s3_client()
    read_bytes = client.get_object(Key=tarball_key, Bucket=bucket)["Body"].read()
    tarball = tarfile.open(fileobj=io.BytesIO(read_bytes))
    images = []
    image_filenames = []
    for member in tarball.getmembers():
        f = tarball.extractfile(member)
        if (f != None):
            im = skimage.transform.resize(imageio.imread(f), (224, 224), preserve_range=True)
            if (len(im.shape) == 2):
                im = np.stack((im,im,im), axis=2)
            image_filenames.append(member.name)
            images.append(im)
    images = np.stack(images, axis=0)
    features = vgg16_features(images, batch_size=batch_size)
    write_output(tarball_key, features, image_filenames)
    return features,tarball_key

def write_output(tarball_key, features, image_filenames, bucket="imagenet2datav2",):
    client = utils.get_s3_client()
    dir_name, file_key = tarball_key.split('/')
    file_key = file_key.replace('-scaled.tar', '-fc7.pkl' )
    key = dir_name + '-featurized/' + file_key
    results = {}
    for idx in range(features.shape[0]):
        filename = image_filenames[idx].split('.')[0].split('/')[1]
        results[filename] = features[idx]
    tmp = pickle.dumps(results)
    print('Uploading {} to s3 '.format(key))
    client.put_object(Bucket=bucket, Key=key, Body=tmp, ACL="bucket-owner-full-control")

def featurize_all(keys):
    for img_class in keys:
        t0 = time.time()
        featurize_s3_tarball(img_class)
        t1 = time.time()
        print('Took {} seconds to upload features'.format(t1-t0))


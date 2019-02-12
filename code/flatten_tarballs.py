import io
import imagenet
import pywren
from pywren import wrenconfig as wc
import tarfile

import utils

def flatten_tarball(tarball_name, prefix, bucket="imagenet2datav2", verbose=False):
    tarball_bytes = utils.get_s3_file_bytes(tarball_name, cache_on_local_disk=False, verbose=verbose)
    tf = tarfile.open(fileobj=io.BytesIO(tarball_bytes))
    for member in tf.getmembers():
        if member.isfile():
            file_bytes = tf.extractfile(member).read()
            key = prefix + member.name
            utils.put_s3_object_bytes_with_backoff(file_bytes, key, bucket=bucket, delay_factor=10)
    return

def get_tarball_names(wnids, prefix):
    tarball_names = []
    for wnid in wnids:
        tarball_names.append(prefix + wnid + '-scaled.tar')
    return tarball_names

def main():
    imgnt = imagenet.ImageNetData()
    wnids = list(imgnt.train_imgs_by_wnid.keys())
    
    train_tarball_names = get_tarball_names(wnids, 'imagenet-train/')
    val_tarball_names = get_tarball_names(wnids, 'imagenet-validation/val-')

    def flatten_train_tarball(tarball_name):
        return flatten_tarball(tarball_name, prefix="imagenet-train-individual/")
    def flatten_val_tarball(tarball_name):
        return flatten_tarball(tarball_name, prefix="imagenet-validation-individual/")
    
    pwex = pywren.default_executor()
    futures = pwex.map(flatten_val_tarball, val_tarball_names)
    failed_wnids = []
    for future, wnid in zip(futures, wnids):
        try:
            future.result()
        except:
            failed_wnids.append(wnid)
            print('wnid failed', wnid)
    print(failed_wnids)
    results = pywren.get_all_results(futures)



if __name__ == "__main__":
   main() 

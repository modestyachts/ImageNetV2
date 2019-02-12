import boto3
import argparse
import utils
from pywren import wrenconfig as wc
import pywren

def encrypt_all_keys_in_prefix(bucket, prefix, encrypt_out, strip_string, use_pywren):
    keys = utils.list_all_keys(prefix)
    if (use_pywren):
        chunked_keys = utils.chunks(keys, 500)
        def pywren_job(key_chunk):
            for key in key_chunk:
                utils.encrypt_s3_copy_key(key, bucket, encrypt_out, strip_string)
            return 0
        config = wc.default()
        config['runtime']['s3_bucket'] = 'imagenet2datav2'
        config['runtime']['s3_key'] = 'pywren.runtime/pywren_runtime-3.6-imagenet2.tar.gz'
        pwex = pywren.default_executor(config=config)
        print(f"Submitting jobs for {len(keys)} keys")
        futures = pwex.map(pywren_job, chunked_keys, exclude_modules=["site-packages/"])
        pywren.wait(futures)
        [f.result() for f in futures]
    else:
        for key in keys:
            utils.encrypt_s3_copy_key(key, bucket, encrypt_out, strip_string)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="imagenet2datav2")
    parser.add_argument("--encrypt_out", default="encrypted")
    parser.add_argument("--strip_string", default="")
    parser.add_argument("--pywren", default=False, action='store_const', const=True)
    parser.add_argument("prefix")
    args = parser.parse_args()
    encrypt_all_keys_in_prefix(args.bucket, args.prefix, args.encrypt_out, args.strip_string, args.pywren)






import pywren
import json
import utils
import urllib
import argparse
import boto3
import ssl
import PIL
from PIL import Image
import io
import numpy as np
import skimage.transform
import imageio
import time
import candidate_data
import gc


MTURK_RESCALE_SIZE = 500
NDC_SIZE = 256
from PIL import Image, ImageFile

__all__ = ['fix_orientation']

# PIL's Error "Suspension not allowed here" work around:
# s. http://mail.python.org/pipermail/image-sig/1999-August/000816.html
ImageFile.MAXBLOCK = 1024*1024

# The EXIF tag that holds orientation data.
EXIF_ORIENTATION_TAG = 274

# Obviously the only ones to process are 3, 6 and 8.
# All are documented here for thoroughness.
ORIENTATIONS = {
    1: ("Normal", 0),
    2: ("Mirrored left-to-right", 0),
    3: ("Rotated 180 degrees", Image.ROTATE_180),
    4: ("Mirrored top-to-bottom", 0),
    5: ("Mirrored along top-left diagonal", 0),
    6: ("Rotated 90 degrees", Image.ROTATE_270),
    7: ("Mirrored along top-right diagonal", 0),
    8: ("Rotated 270 degrees", Image.ROTATE_90)
}

def fix_orientation(img, save_over=False):
    """
    `img` can be an Image instance or a path to an image file.
    `save_over` indicates if the original image file should be replaced by the new image.
    * Note: `save_over` is only valid if `img` is a file path.
    """
    path = None
    if not isinstance(img, Image.Image):
        path = img
        img = Image.open(path)
    elif save_over:
        raise ValueError("You can't use `save_over` when passing an Image instance.  Use a file path instead.")
    try:
        orientation = img._getexif()[EXIF_ORIENTATION_TAG]
    except (TypeError, AttributeError, KeyError):
        return (img , 0)
    if orientation in [3,6,8]:
        degrees = ORIENTATIONS[orientation][1]
        img = img.transpose(degrees)
        if save_over and path is not None:
            try:
                img.save(path, quality=95, optimize=1)
            except IOError:
                # Try again, without optimization (PIL can't optimize an image
                # larger than ImageFile.MAXBLOCK, which is 64k by default).
                # Setting ImageFile.MAXBLOCK should fix this....but who knows.
                img.save(path, quality=95)
        return (img, degrees)
    else:
        return (img, 0)

def quick_image_check(img_dict, bucket, prefix):
    url = img_dict['url']
    ext = url.split(".")[-1]
    key = img_dict["id_ours"]
    mturk_key = "{2}_mturk/{0}.{1}".format(key, ext, prefix)
    if (utils.key_exists(bucket, mturk_key)):
        return True
    else:
        return None

def write_images_to_s3(img_dicts, bucket, prefix):
    return [write_image_to_s3(x, bucket, prefix) for x in img_dicts]

def write_image_to_s3(img_dict, bucket, prefix):

    t = time.time()
    url = img_dict['url']
    ext = url.split(".")[-1]
    key = img_dict["id_ours"]
    mturk_key = "{2}_mturk/{0}.{1}".format(key, ext, prefix)
    if utils.key_exists(bucket, mturk_key):
        return img_dict
    gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)
    img_bytes = urllib.request.urlopen(url, context=gcontext).read()
    pil_image = Image.open(io.BytesIO(img_bytes))
    rotated_img, _ = fix_orientation(pil_image)
    np_image = np.array(rotated_img)
    np_image = utils.make_rgb(np_image)
    try:

        image_ndc = skimage.transform.resize(np_image, (NDC_SIZE, NDC_SIZE), preserve_range=True)
    except MemoryError:
        raise Exception(f"Image {img_dict} memory error")

    bigger_side = max(np_image.shape)
    scale_fac = MTURK_RESCALE_SIZE/bigger_side
    image_mturk = skimage.transform.rescale(np_image, scale=scale_fac, preserve_range=True)


    bio_mturk = io.BytesIO()
    bio_orig = io.BytesIO()
    bio_ndc = io.BytesIO()

    imageio.imwrite(uri=bio_orig, im=np_image, format="jpg", quality=90)
    try:
        imageio.imwrite(uri=bio_mturk, im=image_mturk, format="jpg", quality=90)
    except:
        raise Exception(f"Image {img_dict} error")
	
    imageio.imwrite(uri=bio_ndc, im=image_ndc, format="jpg", quality=90)

    client = utils.get_s3_client()
    ext = "jpg"
    backoff = 1
    while(True):
        try:
            client.put_object(Key="{2}_scaled/{0}.{1}".format(key, ext, prefix), Bucket=bucket, Body=bio_ndc.getvalue())
            client.put_object(Key="{2}_original/{0}.{1}".format(key, ext, prefix), Bucket=bucket, Body=bio_orig.getvalue())
            client.put_object(Key="{2}_mturk/{0}.{1}".format(key, ext, prefix), Bucket=bucket, Body=bio_mturk.getvalue())
            break
        except:
            time.sleep(backoff)
            backoff *= 2
    e = time.time()
    print("One image took ", e - t)
    img_dict["width"] = np_image.shape[1]
    img_dict["height"] = np_image.shape[0]
    return img_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download images to S3 from search_json")
    parser.add_argument("search_json")
    parser.add_argument("--bucket", default="imagenet2datav2", type=str)
    parser.add_argument("--parallel", default=False, action="store_const", const=True)
    parser.add_argument("--prefix", default="imagenet2candidates", type=str)
    parser.add_argument("--outfile", default=None, type=str)
    parser.add_argument("--batch", default=False, action="store_const", const=True)
    parser.add_argument("--quick_check", default=False, action="store_const", const=True)
    parser.add_argument("--chunk_size", default=10, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    args = parser.parse_args()

    with open(args.search_json) as fp:
        search_json = json.load(fp)

    img_dicts = search_json
    print(f"Total img dicts {len(img_dicts)}")
    with open("../data/metadata/candidate_blacklist.json") as f:
        blacklist = json.load(f)
    img_dicts = [x for x in img_dicts if x['id_ours'] not in blacklist]
    pwex = pywren.default_executor()
    def chunked_image_check(img_chunk):
        results = []
        for img in img_chunk:
            results.append(quick_image_check(img, args.bucket, args.prefix))
        return results

    not_done = 0
    if (args.quick_check):
        exists = {}
        chunked_img_dicts = list(utils.chunks(img_dicts, 100))
        print("Quick len", len(chunked_img_dicts))
        futures = pwex.map(chunked_image_check, chunked_img_dicts)
        pywren.wait(futures)
        for chunk, chunked_res  in zip(chunked_img_dicts, futures):
            for img, res in zip(chunk, chunked_res.result()):
                if (not res):
                    not_done += 1
                    exists[img['id_ours']] = "Too big"
                    print(img['id_ours'])
        with open("too_big.json", "w+") as f:
            f.write(json.dumps(exists, indent=2))

    print(f"{not_done} completed")
    print(f"{len(img_dicts)} images to download...")
    if (not args.parallel):
        [write_image_to_s3(x, args.bucket, args.prefix) for x in img_dicts]
    else:
        pwex = pywren.default_executor()
        print(pwex.config)
        print("submitting job to pywren")
        all_results = []
        chunked_dicts = list(utils.chunks(img_dicts, args.chunk_size))
        print(f"Submitting {len(chunked_dicts)} jobs to pywren")
        futures = pwex.map(lambda x: write_images_to_s3(x, args.bucket, args.prefix), chunked_dicts, exclude_modules=["site-packages"])
        pywren.wait(futures)
        for chunk, f in zip(chunked_dicts, futures):
            try:
                for elem, x in zip(chunk, f.result()):
                    all_results.append(x)
            except:
                print("storage exception for ", chunk)
                raise

        if (args.batch):
            print("Batching candidates")
            all_results = candidate_data.batch_candidates(all_results, args.batch_size)
        if (args.outfile is None):
            outfile = args.search_json
            print(f"Overwriting search json: {outfile}")
        with open(outfile, 'w+') as fp:
            json.dump(all_results, fp, indent=2)







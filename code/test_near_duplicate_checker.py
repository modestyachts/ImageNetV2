import datetime
import hashlib
import imageio
import io
import json
import os

import boto3
import numpy as np
import scipy
from skimage.transform import resize

import featurize
import near_duplicate_checker
import utils
import imagenet


def make_test_img(im_data, im_name, prefix, size, exact):
    sha1 = hashlib.sha1()
    try:
        wnid = im_data.get_wnid_from_train_filename(im_name)
    except:
        try:
            wnid = im_data.wnid_by_val_filename[im_name]
        except:
            wnid  = "TEST"
    new_photo = {}
    new_photo['id_ours'] = hashlib.sha1(("TEST_" + prefix + size + str(exact) + im_name).encode()).hexdigest()
    new_photo['id_search_engine'] = im_name
    new_photo['url'] = "TEST"
    new_photo['wnid'] = wnid
    new_photo['search_key'] = ""
    new_photo['search_engine'] = ""
    new_photo['uploader'] = "TEST"
    new_photo['date_upload_our_db'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_photo['date_taken'] = "TEST"
    new_photo['date_upload_search_engine'] = "TEST"
    new_photo['width'] = 256
    new_photo['height'] = 256
    new_photo['max_taken_date'] = "TEST"
    new_photo['max_uploaded_date'] = "TEST"
    new_photo['search_sort'] = "TEST"
    if (not exact):
        img = im_data.load_image(im_name, size=size)
    else:
        img = im_data.load_image_bytes(im_name, size=size)
    return new_photo, img


def make_test(references, mod_fn=lambda x : x + np.random.randn(*x.shape), prefix="GAUSSIAN", metrics=["l2"], control_images=1, num_extra_images=10, bucket="imagenet2datav2", size='scaled_256', exact=False):
    def test(top_k=5, seed=586724699, extra_pairs=[]):
            im_data = imagenet.ImageNetData()
            np.random.seed(seed)
            images = np.random.choice(references, control_images + num_extra_images, replace=False)
            extra_images = images[control_images:]
            images = images[:control_images]
            image_ids =  []
            img_info = []
            test_dataset = []
            client = utils.get_s3_client()
            true_dict = {}
            to_featurize = []
            to_featurize_keys = []
            for im_name in images:
                im_meta, img = make_test_img(im_data, im_name, prefix=prefix, size=size, exact=exact)
                true_dict[im_meta['id_ours']] = im_name
                img_info.append(im_meta)
                img_orig = img
                if not exact:
                    img = mod_fn(img)
                    img = resize(img, (256, 256), preserve_range=True)
                else:
                    im_bytes = img
                    img = imageio.imread(img)
                if 'fc7' in metrics:
                    key_name = os.path.join("imagenet2candidates_featurized", f"{im_meta['id_ours']}.npy")
                    im_resize = resize(img_orig, (224, 224), preserve_range=True)
                    to_featurize.append(im_resize.astype('float32'))
                    to_featurize_keys.append(key_name)
                bio = io.BytesIO()
                if not exact:
                    imageio.imwrite(uri=bio, im=img, format="jpg", quality=100)
                    bstream = bio.getvalue()

                else:
                    print("Exact bytes..")
                    bstream = im_bytes
                key = "imagenet2candidates_scaled/{0}.jpg".format(im_meta['id_ours'])
                print("uploading.. to {0}".format(key))
                client.put_object(Bucket=bucket, Key=key, Body=bstream)
            if len(to_featurize) > 0:
                to_featurize = np.stack(to_featurize, axis=0)
                batch_size = min(len(to_featurize), 32)
                features = featurize.vgg16_features(to_featurize, batch_size=batch_size, use_gpu=False)
                for i,f in enumerate(features):
                    key_name = to_featurize_keys[i]
                    bio = io.BytesIO()
                    np.save(bio, f)
                    print("writing features key {0}".format(key_name))
                    bstream = bio.getvalue()
                    print("feature hash ", hashlib.sha1(bstream).hexdigest())
                    client.put_object(Key=key_name, Bucket=bucket, Body=bstream)

            with open("../data/search_results/test_{0}_results.json".format(prefix), "w+") as f:
                f.write(json.dumps(img_info))
            candidates = [x['id_ours'] for x in img_info]
            extra_images = list(extra_images)
            print("extra pairs", extra_pairs)
            print("len extra_images", len(extra_images))
            for e,v in extra_pairs:
                true_dict[e] = v
                candidates.append(e)
                extra_images.append(v)
                print("len after append extra_images", len(extra_images))

            for e in extra_images:
                true_dict[e] = e

            for e in images:
                true_dict[e] = e





            reference_names = list(images) + list(extra_images)
            print(f"running near duplicate check on {candidates} vs {reference_names}")
            print(f"num references {len(references)}")
            res, t_info = near_duplicate_checker.get_near_duplicates(candidates, reference_names, top_k=top_k, dssim_window_size=35, use_pywren=False, ref_chunk_size=100, cd_chunk_size=100, distance_metrics=metrics)
            for m,val in res.items():
                for k,v in val.items():
                    true_answer = true_dict[k]
                    result = v[0][0]
                    if (true_answer != result):
                        print(m,val,k,v)
                    print(f"expected {true_answer}, got {result}")
                    print(v)
                    assert true_answer == result
                    print("Passed NDC for metric {0} for test {1}".format(m, prefix))
                    if (exact):
                        if (not np.isclose(v[0][1], 0)):
                            print(m, val, k, v)
                        assert np.isclose(v[0][1], 0)
            return res

    return test


if __name__ == "__main__":
    im_data = imagenet.ImageNetData()
    #image_names = im_data.get_all_val_image_names() + im_data.get_all_train_image_names() + im_data
    #image_names = im_data.get_all_val_image_names() + im_data.get_all_train_image_names() + im_data
    references = im_data.get_all_val_image_names()[:100]
    custom_test = make_test(references, mod_fn=lambda x: x + np.random.randn(*x.shape), metrics=['fc7'], size='scaled_256', num_extra_images=10, exact=False)
    custom_test(top_k=10, extra_pairs=[("n02085936_7394.JPEG", "n02085936_10397.JPEG")])
    custom_test(top_k=10)

    references = im_data.get_all_train_image_names()[:100]
    custom_test = make_test(references, mod_fn=lambda x: x + np.random.randn(*x.shape), metrics=['fc7'], size='scaled_256', num_extra_images=10, exact=False)
    custom_test(top_k=10, extra_pairs=[("n02085936_7394.JPEG", "n02085936_10397.JPEG")])
    custom_test(top_k=10)


    references = im_data.test_filenames
    exact_test = make_test(references, mod_fn=lambda x: x, metrics=['l2'], size='scaled_256', num_extra_images=100, exact=True, prefix="exact")
    exact_test(top_k=1)

    rescaled_test = make_test(references, mod_fn=lambda x: x, metrics=['l2', 'dssim', 'fc7'], size='scaled_500', num_extra_images=100, prefix="identity 3 metrics")
    rescaled_test(top_k=1)

    rescaled_test_with_noise = make_test(references, mod_fn=lambda x: x + np.random.randn(*x.shape), metrics=['l2', 'dssim', 'l2'], size='scaled_500', num_extra_images=100, prefix="gaussian noise 3 metrics")
    rescaled_test_with_noise(top_k=1)

    rescaled_val = make_test(references, mod_fn=lambda x: x, metrics=['l2', 'dssim', 'fc7'], size='scaled_500', num_extra_images=100)
    res = rescaled_val(top_k=2)
    for m,val in res.items():
        for k,v in val.items():
            if (m == 'l2'):
                assert np.abs(v[1][1]) > 1e4
            elif (m == 'ssim'):
                assert np.abs(v[1][1]) > 0.2
            elif (m == 'fc7'):
                #TODO: add an assertion here
                pass









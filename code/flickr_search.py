import flickrapi
import flickrapi.exceptions
import flickrapi.shorturl
import imagenet
import itertools
import json
from datetime import date
import datetime
import getpass
import utils
import time
import argparse
import logging
import hashlib
import pywren
import random
import urllib.request
import numpy as np
import pathlib
from pywren import wrenconfig as wc

PER_PAGE = 1000
MAX_TRIES = 5

logger = logging.getLogger(__name__)

def get_readable_time(unix_timestamp):
    return datetime.datetime.fromtimestamp(
        unix_timestamp
    ).strftime('%Y-%m-%d %H:%M:%S')

def get_image(url, cid, image_save_dir):
    try:
        pathlib.Path(image_save_dir).mkdir(parents=True, exist_ok=True) 
        image = urllib.request.urlopen(url).read()
        with open(image_save_dir + '/' + cid + '.' + url.split('.')[-1], 'wb') as image_file:
            image_file.write(image)
            image_file.close()
    except Exception as e:
        print(e)
    return

def flickr_search_synset(imgnt, wnids, api_key, api_secret, args):
    if args.use_additional_search_terms:
        with open("../data/metadata/wnid_to_parent_3.json", "r") as f:
            wnids_to_parent = json.load(f)
    if args.hand_selected_synset is not None:
        with open(args.hand_selected_synset, "r") as f:
            wnids_to_selected_synset = json.load(f)
    se = "flickr"
    res = []
    for wnid in wnids:
        num_backoffs = 0
        backoff = 1
        finished = False
        t = time.time()
        while num_backoffs < MAX_TRIES and not finished:
            try:
                synset = imgnt.class_info_by_wnid[wnid].synset
                flickr = flickrapi.FlickrAPI(api_key, api_secret, format='etree')
                if args.hand_selected_synset is not None:
                    if wnid in wnids_to_selected_synset:
                        synset = wnids_to_selected_synset[wnid]
                for search_key in synset:
                    if args.use_additional_search_terms:
                        additional_term = wnids_to_parent[wnid][0]
                        search_key = search_key + " " + additional_term
                    if args.concatenate_compounds:
                        if len(search_key.split(" ")) > 1:
                          search_key = search_key.replace(" ", "")
                    print('Search key {}'.format(search_key))
                    search_set = flickr.walk(text=search_key,
                             extras =
                             'date_upload,date_taken,o_dims,url_s,url_q,url_t,url_m,url_n,url_-,url_z,url_c,url_b,url_h,url_k,url_o',
                             sort = args.search_sort,
                             max_taken_date = args.max_date_taken,
                             max_uploaded_date = args.max_date_uploaded,
                             min_taken_date = args.min_date_taken,
                             min_uploaded_date = args.min_date_uploaded,
                             per_page=PER_PAGE)
                    num_images = 0
                    if args.max_images is None:
                        iterator = search_set
                    else:
                        iterator = itertools.islice(search_set, int(args.max_images / len(synset)))
                    total_num_images = 0
                    photo_urls = {}
                    url_counter = {}
                    url_types = ['url_s','url_q', 'url_t', 'url_m', 'url_n', 'url_-', 'url_z', 'url_c',
                        'url_b', 'url_h', 'url_k', 'url_o']
                    for key in url_types:
                        url_counter[key] = 0
                    good_url_types = ['url_o', 'url_k', 'url_h', 'url_b', 'url_c', 'url_z', 'url_-']
                    for photo in iterator:
                        total_num_images += 1
                        new_photo = {}
                        has_url=False
                        for key in url_types:
                            photo_urls[key] = photo.get(key)
                            if photo_urls[key]:
                               url_counter[key] += 1
                               has_url = True
                        if not has_url:
                            if photo.get('id'):
                                print(photo.get('id'))
                        url = None
                        for good_url_type in good_url_types:
                            if photo_urls[good_url_type]:
                                url = photo_urls[good_url_type]
                                photo_url_type = good_url_type
                                break
                        if url:
                            logging.debug('Photo title ' + str(photo.get('title')))
                            logger.debug('Url ' + str(url))
                            se_id = photo.get('id')
                            new_photo['id_ours'] = hashlib.sha1((se_id + se).encode()).hexdigest()
                            # Get image
                            if args.download_images:
                                get_image(url, new_photo['id_ours'], wnid)
                            
                            new_photo['id_search_engine'] = se_id
                            new_photo['url'] = url
                            new_photo['url_type'] = photo_url_type
                            new_photo['wnid'] = wnid
                            new_photo['search_key'] = search_key
                            new_photo['search_engine'] = se
                            new_photo['uploader'] = getpass.getuser()
                            new_photo['date_upload_our_db'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            new_photo['date_taken'] = photo.get('datetaken')
                            new_photo['date_upload_search_engine'] = get_readable_time(int(photo.get('dateupload')))
                            new_photo['width'] = photo.get('o_width')
                            new_photo['height'] = photo.get('o_height')
                            new_photo['max_taken_date'] = args.max_date_taken
                            new_photo['max_uploaded_date'] = args.max_date_uploaded
                            new_photo['min_taken_date'] = args.min_date_taken
                            new_photo['min_uploaded_date'] = args.min_date_uploaded
                            new_photo['search_sort'] = args.search_sort
                            new_photo['max_images'] = args.max_images
                            res.append(new_photo)
                            num_images = num_images + 1
                    print('PHOTOS FOUND: {}'.format(total_num_images))
                    print('PHOTOS ADDED: {}'.format(num_images))
                    finished = True
                    e = time.time()
                    print(f"{wnid} time took {e - t}")
            except flickrapi.exceptions.FlickrError:
                time.sleep(backoff)
                num_backoffs += 1
                backoff *= 2
    return res

def main(args):
    imgnt = imagenet.ImageNetData()
    with open(args.flickr_api_key_filename, 'r') as f:
        flickr_api_keys = json.load(f)
        api_key = flickr_api_keys[0]
        api_secret = flickr_api_keys[1]

    with open(args.wnids, 'r') as f:
        wnids = json.load(f)
    print('processing {} wnids'.format(len(wnids)))

    if not args.parallel:
        all_results = []
        for wnid in wnids:
            print("Flickr search for wnid {}".format(wnid))
            res = flickr_search_synset(imgnt, [wnid], api_key, api_secret, args) 
            all_results += res
    else:
        pywren_config = wc.default()
        pywren_config["runtime"]["s3_bucket"] = "imagenet2datav2"
        pywren_config["runtime"]["s3_key"] = "pywren.runtime/pywren_runtime-3.6-imagenet2.tar.gz"
        pwex = pywren.default_executor(config=pywren_config)
        pywren_func = lambda x: flickr_search_synset(imgnt, x, api_key, api_secret, args)
        pywren_args = list(utils.chunks(wnids, int(np.ceil(len(wnids)/args.num_serial_tasks))))
        num_images_per_wnid = {}
        with open('../data/metadata/flickr_' + args.min_date_uploaded + '_' + args.max_date_uploaded +
            '.json', 'r') as fp:
            num_images_per_wnid = json.load(fp)
        
        for ii, lst in enumerate(pywren_args):
            print("Map {} over {} wnids ".format(ii,len(lst)))
            unfinished_wnids = []
            for wnid in lst:
                if wnid not in num_images_per_wnid:
                    unfinished_wnids.append(wnid)
            print("Executing pywren call for {} wnids".format(len(unfinished_wnids)))
            futures = pwex.map(pywren_func, [[x] for x in unfinished_wnids])
            pywren.wait(futures)
            results = [f.result()[0] for f in futures]
            num_images = [f.result()[1] for f in futures] 
            for ii, wnid in enumerate(unfinished_wnids):
               num_images_per_wnid[wnid] = num_images[ii]  
            all_results = []
            for res in results:
                all_results += res
            with open('../data/metadata/flickr_' + args.min_date_uploaded + '_' + args.max_date_uploaded +
                '.json', 'w') as fp:
                json.dump(num_images_per_wnid, fp, indent=2)
    print('Got {} results'.format(len(all_results)))
    current_date = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) 
    out_file = '../data/search_results/'+ current_date + '_' + getpass.getuser() + '.json'
    with open(out_file, 'w+') as fp:
        json.dump(all_results, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate flickr image jsons")
    parser.add_argument("flickr_api_key_filename")
    parser.add_argument("--wnids", default="0", type=str)
    parser.add_argument("--max_images", default=None, type=int)
    parser.add_argument("--max_date_taken", default="2013-07-11", type=str)
    parser.add_argument("--max_date_uploaded", default="2013-07-11", type=str)
    parser.add_argument("--min_date_taken", default="2012-07-11", type=str)
    parser.add_argument("--min_date_uploaded", default="2012-07-11", type=str)
    parser.add_argument("--search_sort", default="date-posted-asc", type=str)
    parser.add_argument("--parallel", default=False, action="store_const", const=True)
    parser.add_argument("--download_images", default=False, action="store_const", const=True)
    parser.add_argument("--use_additional_search_terms", default=False, action="store_const", const=True)
    parser.add_argument("--hand_selected_synset", default=None, type=str)
    parser.add_argument("--concatenate_compounds", default=False, action="store_const", const=True)
    parser.add_argument("--num_serial_tasks", default=100, type=int)
    args = parser.parse_args()

    main(args)

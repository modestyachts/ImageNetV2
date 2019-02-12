import json
import os
import urllib.request

import imagenet
import utils

imgnet = imagenet.ImageNetData(load_class_info=False)

def lookup_wnid(wnid):
    url = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={0}'.format(wnid)
    return urllib.request.urlopen(url).read().decode().strip().split('\n')

gloss_bytes = utils.get_s3_file_bytes('metadata/gloss.txt', cache_on_local_disk=False)
gloss_string = gloss_bytes.decode('utf-8')
gloss_lines = gloss_string.split('\n')
gloss = {}
for line in gloss_lines:
    wnid = line[:9]
    cur_gloss = line[10:]
    gloss[wnid] = cur_gloss

tmpci2 = []
wnids = sorted(imgnet.train_imgs_by_wnid.keys())

for ii, wnid in enumerate(wnids):
    cur_dict = {}
    cur_dict['cid'] = ii
    cur_dict['wnid'] = wnid
    cur_dict['synset'] = lookup_wnid(wnid)
    cur_dict['gloss'] = gloss[wnid]
    tmpci2.append(cur_dict)

with open(os.path.join(os.getcwd(), '../data/metadata/class_info.json'), 'w') as f:
    json.dump(tmpci2, f, indent=2)

import pywren
from pywren import wrenconfig as wc
import candidate_data
import utils

pywren_config = wc.default()
pywren_config["runtime"]["s3_bucket"] = "imagenet2pywren"
pywren_config["runtime"]["s3_key"] = "pywren.runtime/pywren_runtime-3.6-imagenet2pywren.meta.json"
pwex = pywren.default_executor(config=pywren_config)
print("pywren config", pwex.config)

c_data = candidate_data.CandidateData()

all_cs = c_data.all_candidates

chunked_cs = list(utils.chunks(list(all_cs.keys()), 100))


def return_not_exists(lst):
    ret_lst = []
    for e in lst:
        key = "{0}/{1}.jpg".format("imagenet2candidates_scaled", e)
        exists = utils.key_exists(bucket="imagenet2datav2", key=key)
        print(exists, key)
        if (not exists):
            ret_lst.append(e)
    return ret_lst

def return_not_exists_encrypted(lst):
    ret_lst = []
    for e in lst:
        e = utils.encrypt_string_with_magic(e)
        key = "{0}/{1}.jpg".format("encrypted", e)
        exists = utils.key_exists(bucket="imagenet2datav2", key=key)
        print(exists, key)
        if (not exists):
            ret_lst.append(e)
    return ret_lst



futures0 = pwex.map(return_not_exists, chunked_cs)
futures1 = pwex.map(return_not_exists_encrypted, chunked_cs)

pywren.wait(futures0)
pywren.wait(futures1)

assert sum([len(f.result()) for f in futures0])  == 0

assert sum([len(f.result()) for f in futures1]) == 0

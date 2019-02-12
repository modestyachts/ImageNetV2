import imagenet
import json
import random
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def get_similarity_sorted_wnids(q_wnid):
    """Returns a list of sorted (wnid, dist) duples in order of 
    most similar to least similar to the query wnid."""
    q_synset = wn.synset_from_pos_and_offset(q_wnid[0], int(q_wnid[1:]))
    imgnet = imagenet.ImageNetData()
    wnids = list(imgnet.train_imgs_by_wnid.keys())
    similarity_dict = {}
    for wnid in wnids:
        cur_synset = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
        similarity_dict[wnid] = q_synset.path_similarity(cur_synset)

    sorted_wnids = [(k, similarity_dict[k]) for k in sorted(similarity_dict, 
                                                            key=similarity_dict.get, 
                                                            reverse=True)]
    return sorted_wnids

def get_top_k_wnids(q_wnid, top_k):
    """Get the top k closest wnids to the query wnid"""  
    sorted_wnids = get_similarity_sorted_wnids(q_wnid)
    closest_k_pairs = sorted_wnids[:top_k]
    return [wnid for wnid, dist in closest_k_pairs]

def get_all_top_k_wnids(wnids, top_k):
    result = {}
    for wnid in wnids:
        result[wnid] = get_top_k_wnids(wnid, top_k)
    return result

def get_farthest_wnid(wnid):
    sorted_wnids = get_similarity_sorted_wnids(wnid)
    farthest_dist = sorted_wnids[-1][1]
    farthest_wnids = [wnid for wnid, dist in sorted_wnids if dist == farthest_dist]
    return random.choice(farthest_wnids)

def get_all_negative_wnids(wnids):
    wnid_to_farthest_wnid = {}
    for wnid in wnids:
        wnid_to_farthest_wnid[wnid] = get_farthest_wnid(wnid)
    return wnid_to_farthest_wnid

def generate_opposite_class_json():
    imgnet = imagenet.ImageNetData()
    wnids = list(imgnet.train_imgs_by_wnid.keys())
    result = get_all_negative_wnids(wnids)
    with open('../data/metadata/wnid_to_farthest_wnid.json', 'w') as fp:
        json.dump(result, fp, indent=2)

def generate_top_k_wnids_json():
    top_k = 21
    imgnet = imagenet.ImageNetData()
    wnids = list(imgnet.train_imgs_by_wnid.keys())
    result = get_all_top_k_wnids(wnids, top_k)
    with open('../data/metadata/wnid_to_most_similar_wnids.json', 'w') as fp:
        json.dump(result, fp, indent=2)


if __name__ == "__main__":
  generate_top_k_wnids_json()
  #generate_opposite_class_json()

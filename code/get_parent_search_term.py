import json

import imagenet
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

def main():
    with open('../data/metadata/unprocessed_wnids.json', 'r') as f:
        bad_wnids = json.load(f)
    
    #with open('../data/metadata/wnid_to_parent_2.json', 'r') as f:
     #   wnid_to_parent = json.load(f)

    imgnt = imagenet.ImageNetData()
   
    wnid_to_parent = {}
    wnids_with_additional_search_terms = []
    wnids_with_no_additional_search_terms = []
    for wnid in bad_wnids:
        if wnid not in wnid_to_parent:
          wnid_to_parent[wnid] = []
        synset = imgnt.class_info_by_wnid[wnid].synset
        gloss = imgnt.class_info_by_wnid[wnid].gloss
        cur_synset = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
        gloss_list = gloss.split()
        for parent in cur_synset.hypernyms():
            inherited_hypernym = parent.hypernyms()
            for inherited_parent in inherited_hypernym:
                inherited_hypernym_list = inherited_parent.lemma_names()
            parent_list = parent.lemma_names()
        intersect = intersection(gloss_list, parent_list)
        if len(intersect) > 0:
          wnid_to_parent[wnid].extend(intersect)
          wnid_to_parent[wnid] = list(set(wnid_to_parent[wnid]))
          wnids_with_additional_search_terms.append(wnid)
        
        print('Wnid: ', wnid)
        print('Synset: ', synset)
        print('Gloss: ', gloss)
        print('Parent : ', parent_list)
        print('Parents parent: ', inherited_hypernym_list)
        print('Intersection' , intersect)
        print()
    
    with open('../data/metadata/wnid_to_parent_3.json', 'w') as f:
        json.dump(wnid_to_parent, f, indent=2)
    with open('../data/metadata/unprocessed_wnids_with_additional_search_terms.json', 'w') as f:
        json.dump(list(set(wnids_with_additional_search_terms)), f, indent=2)


if __name__ == "__main__":
    main()


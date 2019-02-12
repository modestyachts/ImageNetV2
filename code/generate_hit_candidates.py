import argparse
import candidate_data
from datetime import date
import datetime
import json
from near_duplicate_checker import print_nn_stats
import mturk_data
import pickle

thresholds = {
    'l2' : 1e8,
    'dssim' : 1e32,
    'fc7' : 1e4
}
def remove_blacklist_candidates(candidates, blacklist):
    new_candidates = []
    blacklist = set(blacklist)
    for cd in candidates:
        if cd not in blacklist:
            new_candidates.append(cd)
    return new_candidates


def remove_near_duplicates(candidates, near_duplicates, cd_filenames):
    new_candidates = []
    for cd in candidates:
        cd_has_imagenet_duplicate = False
        if cd in near_duplicates:
            for ref in near_duplicates[cd]:
                if ref not in cd_filenames:
                    cd_has_imagenet_duplicate = True
        if not cd_has_imagenet_duplicate:
            new_candidates.append(cd)
    return new_candidates


def remove_blacklisted_search_keys(candidates, blacklisted_search_keys, cds):
    new_candidates = []
    for cd in candidates:
        wnid = cds.all_candidates[cd]['wnid']
        if wnid not in blacklisted_search_keys:
            new_candidates.append(cd)
    return new_candidates


def filter_by_wnids(cd_lst, num_wnids):
    cd_data = candidate_data.CandidateData()
    wnids = { cd_data.all_candidates[x]['wnid'] for x in cd_lst }
    wnids = list(wnids)
    print("num_wnids", num_wnids)
    print("num_wnids", type(num_wnids))
    good_wnids = wnids[:num_wnids]
    good_cd_lst = [x for x in cd_lst if cd_data.all_candidates[x]['wnid'] in good_wnids]
    return good_cd_lst


def remove_sufficiently_labeled_candidates(candidates_for_hit, cds, mturk, target_num_assignments=10):
    new_candidates = []
    for cid in candidates_for_hit:
        cur_wnid = cds.all_candidates[cid]['wnid']
        if cid not in mturk.image_num_assignments:
            new_candidates.append(cid)
        elif cur_wnid not in mturk.image_num_assignments[cid]:
            new_candidates.append(cid)
        elif mturk.image_num_assignments[cid][cur_wnid] < target_num_assignments:
            new_candidates.append(cid)
    return new_candidates


def main(args):
    mturk = mturk_data.MTurkData(live=True, verbose=True, load_assignments=True)
    cds = candidate_data.CandidateData()
    cd_filenames = list(cds.all_candidates.keys())
    mturk_images = list(mturk.hits_of_image.keys())
    print('Current number of mturk images: {}'.format(len(mturk_images)))
    with open('../data/metadata/nearest_neighbor_results.pickle', 'rb') as f:
        nn_results = pickle.load(f)
    print('Current nearest neighbor statistics: ')
    print_nn_stats(nn_results)
    print()

    # Load existing reviews
    reviews = {}
    with open('../data/metadata/nearest_neighbor_reviews_v2.json', 'r') as f:
        reviews = json.load(f)
    print('Current review statistics: ')
    print_nn_stats(reviews)
    print()

    # Load existing near duplicates
    with open('../data/metadata/near_duplicates.json', 'r') as f:
        near_duplicates = json.load(f)
    print('Number of candidates with duplicates: {}'.format(len(near_duplicates)))
    print()

    # Load blacklisted search keywords
    with open('../data/metadata/blacklisted_search_keywords.json', 'r') as f:
        blacklisted_search_keywords = json.load(f)
    print('Blacklisted search keywords: {}'.format(blacklisted_search_keywords))
    
    # Load blacklisted candidates
    with open('../data/metadata/candidate_blacklist.json', 'r') as f:
        candidate_blacklist = json.load(f)

    candidates_for_hit = list(set(cd_filenames))
    candidates_for_hit = remove_sufficiently_labeled_candidates(candidates_for_hit, cds, mturk)
    print('{} possible unhit candidates'.format(len(candidates_for_hit)))
    candidates_for_hit = remove_blacklist_candidates(candidates_for_hit, candidate_blacklist)
    deduplicated_cds = remove_near_duplicates(candidates_for_hit, near_duplicates, cd_filenames)
    final_cds = remove_blacklisted_search_keys(deduplicated_cds, blacklisted_search_keywords, cds)

    final_cds = filter_by_wnids(final_cds, args.num_wnids)

    print('Saving {} candidates for hit'.format(len(final_cds)))
    current_date = str(datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) 
    with open('../data/hit_candidates/candidates_for_hit_' + current_date + '.json', 'w') as f:
        json.dump(final_cds, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate flickr image jsons")
    parser.add_argument("--num_wnids", default=350, type=int)
    args = parser.parse_args()
    main(args)

import hashlib
import json
import pathlib
import pickle
from timeit import default_timer as timer

try:
    import networkx as nx
except:
    pass
    #print('importing networkx failed')
try:
    import tqdm
except:
    pass
    #print('import tqdm failed')

import utils

metric_names = ['l2', 'fc7', 'dssim']


def check_review_data(review_data, cds, imgnet, imgnet_filenames=None):
    if imgnet_filenames is None:
        imgnet_filenames = set(imgnet.get_all_image_names())
    for candidate, ndcs in review_data.items():
        if candidate not in cds.all_candidates:
            print(f'Missing candidate {candidate}')
        assert candidate in cds.all_candidates
        assert len(ndcs) > 0
        for metric_name, ndc_results in ndcs.items():
            assert metric_name in metric_names
            assert 'references' in ndc_results
            cur_refs = ndc_results['references']
            assert len(cur_refs) == len(set(cur_refs))
            assert len(cur_refs) > 0
            for ref in cur_refs:
                assert ref in imgnet_filenames or ref in cds.all_candidates


def check_near_duplicates(ndc_data, cds, imgnet, imgnet_filenames=None):
    if imgnet_filenames is None:
        imgnet_filenames = set(imgnet.get_all_image_names())
    for candidate, ndcs in ndc_data.items():
        assert candidate in cds.all_candidates
        assert len(ndcs) > 0
        for ndc in ndcs:
            assert ndc in imgnet_filenames or ndc in cds.all_candidates


def compute_review_thresholds(cds, imgnet, verbose=True):
    review_data_filepath = (pathlib.Path(__file__).parent /  '../data/metadata/nearest_neighbor_reviews_v2.json').resolve()
    with open(review_data_filepath, 'r') as f:
        review_data = json.load(f)
    check_review_data(review_data, cds, imgnet)
    if verbose:
        print('Loaded review data from {}'.format(review_data_filepath))
        print('    Review data for {} candidates'.format(len(review_data)))
    print('Computing review thresholds ... ')
    start = timer()
    nn_data_filepath = (pathlib.Path(__file__).parent /  '../data/metadata/nearest_neighbor_results.pickle').resolve()
    with open(nn_data_filepath, 'rb') as f:
        nn_results = pickle.load(f)
    print('    Loaded nn results from {}'.format(nn_data_filepath))
    missing_nn_results = {x: set([]) for x in metric_names}
    review_threshold = {}
    for cand in cds.all_candidates.keys():
        if cand not in nn_results:
            review_threshold[cand] = {x: 0.0 for x in metric_names}
            for x in metric_names:
                missing_nn_results[x].add(cand)
            continue
        cur_thresholds = {}
        for cur_metric in metric_names:
            if cur_metric not in nn_results[cand]:
                cur_thresholds[cur_metric] = 0.0
                missing_nn_results[cur_metric].add(cand)
                continue
            if cand in review_data and cur_metric in review_data[cand]:
                if 'references' not in review_data[cand][cur_metric]:
                    print('"references" field missing for candidate {} and metric {}'.format(cand, cur_metric))
                    assert False
                reviewed_references = set(review_data[cand][cur_metric]['references'])
            else:
                reviewed_references = set([])
            min_unreviewed_distance = 1.0e100
            cur_flickr_id = cds.all_candidates[cand]['id_search_engine']
            # TODO: use a parameter instead of 30 here
            for ref_filename, dst in nn_results[cand][cur_metric][:30]:
                if ref_filename == cand:
                    continue
                if ref_filename in reviewed_references:
                    continue
                if ref_filename in cds.all_candidates:
                    ref_flickr_id = cds.all_candidates[ref_filename]['id_search_engine']
                    if ref_flickr_id == cur_flickr_id:
                        continue
                min_unreviewed_distance = dst
                break
            cur_thresholds[cur_metric] = min_unreviewed_distance
        review_threshold[cand] = cur_thresholds
    end = timer()
    print('done, took {} seconds'.format(end - start))
    return review_threshold, missing_nn_results


class NearDuplicateData:
    def __init__(self, *,
                 imgnet=None,
                 candidates=None,
                 mturk_data=None,
                 load_review_thresholds=False,
                 verbose=True,
                 bucket='imagenet2datav2'):
        assert imgnet is not None
        self.imgnet = imgnet
        self.imagenet_filenames = set(imgnet.get_all_image_names())
        assert candidates is not None
        assert not candidates.blacklist_excluded
        self.cds = candidates
        assert mturk_data is not None
        self.mturk = mturk_data
        
        review_data_filepath = (pathlib.Path(__file__).parent /  '../data/metadata/nearest_neighbor_reviews_v2.json').resolve()
        with open(review_data_filepath, 'r') as f:
            self.review_data = json.load(f)
        check_review_data(self.review_data, self.cds, self.imgnet, imgnet_filenames=self.imagenet_filenames)
        if verbose:
            print('Loaded review data from {}'.format(review_data_filepath))
            print('    Review info data {} candidates'.format(len(self.review_data)))
        
        ndc_resolution_override_filepath = (pathlib.Path(__file__).parent / '../data/metadata/near_duplicate_resolution_override.json').resolve()
        with open(ndc_resolution_override_filepath, 'r') as f:
            self.ndc_resolution_override_set = set(json.load(f))
        if verbose:
            print('Loaded near duplicate resolution override data from {}'.format(ndc_resolution_override_filepath))
            print('    {} resolution overrides'.format(len(self.ndc_resolution_override_set)))
        
        if load_review_thresholds:
            key = 'review_thresholds/data_2018-12-13_06-33-26_UTC.pickle'
            pickle_bytes = utils.get_s3_file_bytes(key, verbose=verbose)
            pickle_dict = pickle.loads(pickle_bytes)
            data_source = 's3://' + bucket + '/' + key
            self.review_threshold = pickle_dict['review_thresholds']
            print(f'Loaded review thresholds from {data_source}')
        
        self.reload_near_duplicate_data(verbose)
        
    def reload_near_duplicate_data(self, verbose):
        ndc_data_filepath = (pathlib.Path(__file__).parent / '../data/metadata/near_duplicates.json').resolve()
        with open(ndc_data_filepath, 'r') as f:
            self.ndc_data = json.load(f)
        check_near_duplicates(self.ndc_data, self.cds, self.imgnet, imgnet_filenames=self.imagenet_filenames)
        if verbose:
            print('Loaded near duplicate data from {}'.format(ndc_data_filepath))
            print('    Near duplicate data for {} candidates'.format(len(self.ndc_data)))
        
        self.graph = nx.Graph()
        # Adding edges for images sharing a Flickr id
        cids_by_flickr_id = {}
        for c in self.cds.all_candidates.values():
            cur_flickr_id = c['id_search_engine']
            if cur_flickr_id not in cids_by_flickr_id:
                cids_by_flickr_id[cur_flickr_id] = []
            cids_by_flickr_id[cur_flickr_id].append(c['id_ours'])
        for cds_group in cids_by_flickr_id.values():
            if len(cds_group) > 1:
                root_node = cds_group[0]
                for other_node in cds_group[1:]:
                    self.graph.add_edge(root_node, other_node)

        # Adding edges for near duplicates
        for c, ndcs in self.ndc_data.items():
            for ndc in ndcs:
                self.graph.add_edge(c, ndc)
        
        # Connected components
        if verbose:
            print('Computing connected components ... ', end='')
        start = timer()
        tmp_components = list(nx.connected_components(self.graph))
        end = timer()
        if verbose:
            print('done, took {} seconds'.format(end - start))
            print('There are {} non-singleton components'.format(len([x for x in tmp_components if len(x) > 1])))
        self.components = {}
        self.component_of_img = {}
        self.is_near_duplicate = {}
        num_imagenet_components = 0
        start = timer()

        for cid in self.cds.all_candidates.keys():
            if cid not in self.graph.nodes:
                self.is_near_duplicate[cid] = False

        if verbose:
            print('Processing components ... ', end='')
        overall_good_candidates_by_wnid = {}
        for comp in tmp_components:
            comp_name = hashlib.md5((','.join(sorted(list(comp)))).encode()).hexdigest()
            #comp_name = ','.join(sorted(list(comp)))
            assert comp_name not in self.components
            self.components[comp_name] = comp
            for img in comp:
                self.component_of_img[img] = comp
            if len(comp) < 2:
                print(comp)
            assert len(comp) >= 2
            comp_has_imagenet_filenames = False
            for img in comp:
                if img in self.imagenet_filenames:
                    comp_has_imagenet_filenames = True
            if comp_has_imagenet_filenames:
                # If the component contains an ImageNet image, all candidates in the component are near-duplicates.
                num_imagenet_components += 1
                for img in comp:
                    if img in self.cds.all_candidates:
                        self.is_near_duplicate[img] = True
            else:
                overrides = []
                for img in comp:
                    if img in self.ndc_resolution_override_set:
                        overrides.append(img)
                assert len(overrides) <= 1
                if len(overrides) == 1:
                    self.is_near_duplicate[overrides[0]] = False
                    for img in comp:
                        if img != overrides[0] and img in self.cds.all_candidates:
                            self.is_near_duplicate[img] = True
                else:
                    # We can take one candidate from this near duplicate set because it contains no ImageNet images.
                    # We pick the one for the wnid with the smallest number of good images.
                    good_frequency_threshold = 0.7
                    comp_by_wnid = {}
                    for img in comp:
                        assert img in self.cds.all_candidates
                        cur_wnid = self.cds.all_candidates[img]['wnid']
                        if cur_wnid not in comp_by_wnid:
                            comp_by_wnid[cur_wnid] = []
                        comp_by_wnid[cur_wnid].append(img)
                    wnid_scores = {}
                    for wnid in comp_by_wnid.keys():
                        wnid_scores[wnid] = 0.0
                        for img in comp_by_wnid[wnid]:
                            if img in self.cds.blacklist:
                                continue
                            if img not in self.mturk.image_fraction_selected:
                                continue
                            if wnid not in self.mturk.image_fraction_selected[img]:
                                continue
                            wnid_scores[wnid] = max(wnid_scores[wnid],
                                                    self.mturk.image_fraction_selected[img][wnid])
                    cur_wnids = [x[0] for x in wnid_scores.items() if x[1] >= good_frequency_threshold]
                    if len(cur_wnids) == 0:
                        cur_wnids = [x[0] for x in wnid_scores.items() if x[1] >= good_frequency_threshold - 0.1]
                        if len(cur_wnids) == 0:
                            cur_wnids = [max(wnid_scores.items(), key=lambda x: (x[1], x[0]))[0]]
                    assert len(cur_wnids) >= 1
                    cur_good_candidates_by_wnid = {}
                    for wnid in cur_wnids:
                        if wnid not in overall_good_candidates_by_wnid:
                            overall_good_candidates_by_wnid[wnid] = set()
                            for cand in self.cds.candidates_by_wnid[wnid]:
                                cid = cand['id_ours']
                                if cid in self.cds.blacklist:
                                    continue
                                if cid not in self.mturk.image_num_assignments:
                                    continue
                                if wnid not in self.mturk.image_num_assignments[cid]:
                                    continue
                                if self.mturk.image_fraction_selected[cid][wnid] >= good_frequency_threshold:
                                    overall_good_candidates_by_wnid[wnid].add(cid)
                        cur_good_candidates_by_wnid[wnid] = overall_good_candidates_by_wnid[wnid]
                    smallest_wnid = min(cur_good_candidates_by_wnid.items(), key=lambda x: (len(x[1]), x[0]))[0]
                    assert smallest_wnid in self.imgnet.class_info_by_wnid.keys()
                    wnid_cands = comp_by_wnid[smallest_wnid]
                    wnid_cands_with_scores = []
                    for cid in wnid_cands:
                        if cid in self.cds.blacklist:
                            score = 0.0
                        elif cid not in self.mturk.image_fraction_selected:
                            score = 0.0
                        elif smallest_wnid not in self.mturk.image_fraction_selected[cid]:
                            score = 0.0
                        else:
                            score = self.mturk.image_fraction_selected[cid][smallest_wnid]
                        wnid_cands_with_scores.append((cid, score))
                    comp_cand = max(wnid_cands_with_scores, key=lambda x: (x[1], x[0]))[0]
                    self.is_near_duplicate[comp_cand] = False
                    for img in comp:
                        if img != comp_cand and img in self.cds.all_candidates:
                            self.is_near_duplicate[img] = True
        end = timer()
        if verbose:
            print('done, took {} seconds'.format(end - start))
        if len(self.is_near_duplicate) != len(self.cds.all_candidates):
            print('{} near duplicate entries, but {} candidates'.format(len(self.is_near_duplicate), len(self.cds.all_candidates)))
        assert len(self.is_near_duplicate) == len(self.cds.all_candidates)
        for cand in self.cds.all_candidates.keys():
            assert cand in self.is_near_duplicate
        if verbose:
            print('{} non-singleton components contain at least one ImageNet image'.format(num_imagenet_components))
            num_near_duplicates = len([x for x in self.is_near_duplicate.values() if x])
            print('Currently {} candidates are marked as near-duplicates'.format(num_near_duplicates))
        

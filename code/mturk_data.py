import pathlib
import pickle

import mturk_utils
import utils

main_collection_filenames_to_ignore = [
    '2018-08-06_17:33_vaishaal.json',
    '2018-08-17_17:24_vaishaal.json',
    'vaishaal_hits_submitted_2018-08-17-18:28:33-PDT.json',
    'vaishaal_hits_submitted_2018-08-17-18:50:38-PDT.json',
    'vaishaal_hits_submitted_2018-08-17-19:28:24-PDT.json',
    'vaishaal_hits_submitted_2018-08-17-19:56:28-PDT.json',
    'vaishaal_hits_submitted_2018-08-25-09:47:26-PDT.json',
]

class MTurkData:
    def __init__(self, *,
                 live=False,
                 load_assignments=True,
                 assignment_source='s3',
                 source_filenames_to_ignore=[],
                 include_blacklisted_hits=False,
                 verbose=True,
                 cache_on_local_disk=True,
                 cache_root_path=None,
                 bucket='imagenet2datav2'):
        self.source_filenames_to_ignore = source_filenames_to_ignore
        self.include_blacklisted_hits = include_blacklisted_hits
        self.bucket = bucket
        self.cache_on_local_disk = cache_on_local_disk
        
        self.live = live
        if self.cache_on_local_disk:
            if cache_root_path is None:
                self.cache_root_path = pathlib.Path(__file__).parent /  '../data/cache'
            else:
                self.ache_root_path = pathlib.Path(cache_root_path)
            self.cache_root_path = self.cache_root_path.resolve()
        else:
            assert cache_root_path is None
            self.cache_root_path = None

        self.hits, self.mturk_ids_to_uuid, _, _, self.blacklisted_hits = mturk_utils.load_local_hit_data(
                live=self.live,
                verbose=verbose,
                source_filenames_to_ignore=source_filenames_to_ignore,
                include_blacklisted_hits=self.include_blacklisted_hits)
        self.hits_of_image = {}
        self.hits_for_wnid = {}
        for hit in self.hits.values():
            cur_wnid = hit['wnid']
            if cur_wnid not in self.hits_for_wnid:
                self.hits_for_wnid[cur_wnid] = []
            self.hits_for_wnid[cur_wnid].append(hit)
            for img in hit['images_all']:
                if img not in self.hits_of_image:
                    self.hits_of_image[img] = []
                self.hits_of_image[img].append(hit)
        if load_assignments:
            assert assignment_source in ['s3', 'mturk']
            if assignment_source == 'mturk':
                raise NotImplementedError
            else:
                key = 'mturk_results/data_live_2018-12-13_04-59-25_UTC.pickle'
                pickle_bytes = utils.get_s3_file_bytes(key, verbose=verbose)
                pickle_dict = pickle.loads(pickle_bytes)
                data_source = 's3://' + bucket + '/' + key
                all_assignments = pickle_dict['assignments']
                self.assignments = {}
                num_assignment_hits_ignored = 0
                for uuid, assignment_dict in all_assignments.items():
                    if uuid in self.hits:
                        self.assignments[uuid] = assignment_dict
                    else:
                        num_assignment_hits_ignored += 1
                num_hits_without_assignments = 0
                for uuid in self.hits.keys():
                    if uuid not in self.assignments:
                        self.assignments[uuid] = {}
                        num_hits_without_assignments += 1
                assert len(self.assignments) == len(self.hits)
                if verbose:
                    print('Using pickled JSON data stored by {} from {} locally'.format(pickle_dict['username'], pickle_dict['json_dir']))
                    print('    S3 source: {}'.format(data_source))
                    print('    Ignored assignment data for {} HITs'.format(num_assignment_hits_ignored))
                    print('    {} HITs do not have assignment data'.format(num_hits_without_assignments))
                self.assignment_time_string = pickle_dict['time_string']
            self.num_assignments = 0
            for cur_assignments in self.assignments.values():
                self.num_assignments += len(cur_assignments)
            self.image_fraction_selected = {}
            self.image_num_assignments = {}
            for img, cur_hits in self.hits_of_image.items():
                self.image_fraction_selected[img] = {}
                self.image_num_assignments[img] = {}
                for ch in cur_hits:
                    cur_wnid = ch['wnid']
                    if cur_wnid not in self.image_fraction_selected[img]:
                        self.image_fraction_selected[img][cur_wnid] = 0
                        assert cur_wnid not in self.image_num_assignments[img]
                        self.image_num_assignments[img][cur_wnid] = 0
                    cur_assignments = self.assignments[ch['uuid']]
                    for a in cur_assignments.values():
                        if a['AssignmentStatus'] in ['Submitted', 'Approved']:
                            self.image_num_assignments[img][cur_wnid] += 1
                            if img in a['Answer']:
                                self.image_fraction_selected[img][cur_wnid] += 1
            for img, wnid_dict in self.image_fraction_selected.items():
                for wnid in wnid_dict:
                    if self.image_num_assignments[img][wnid] == 0:
                        self.image_fraction_selected[img][wnid] = 0.0
                    else:
                        self.image_fraction_selected[img][wnid] /= self.image_num_assignments[img][wnid]

            self.num_valid_assignments_by_hit = {}
            for uuid, hit in self.hits.items():
                self.num_valid_assignments_by_hit[uuid] = 0
                cur_assignments = self.assignments[uuid]
                for a in cur_assignments.values():
                    if a['AssignmentStatus'] in ['Submitted', 'Approved']:
                        self.num_valid_assignments_by_hit[uuid] += 1

            self.hit_image_num_selected = {}
            self.hit_image_fraction_selected = {}
            for uuid, hit in self.hits.items():
                self.hit_image_num_selected[uuid] = {}
                self.hit_image_fraction_selected[uuid] = {}
                for img in hit['images_all']:
                    assert img not in self.hit_image_num_selected[uuid]
                    assert img not in self.hit_image_fraction_selected[uuid]
                    self.hit_image_num_selected[uuid][img] = 0
                    self.hit_image_fraction_selected[uuid][img] = 0
                    cur_assignments = self.assignments[uuid]
                    for a in cur_assignments.values():
                        if a['AssignmentStatus'] in ['Submitted', 'Approved'] and img in a['Answer']:
                            self.hit_image_num_selected[uuid][img] += 1
            for uuid, imgs in self.hit_image_num_selected.items():
                for img in imgs.keys():
                    if self.num_valid_assignments_by_hit[uuid] == 0:
                        self.hit_image_fraction_selected[uuid][img] = 0.0
                    else:
                        self.hit_image_fraction_selected[uuid][img] = self.hit_image_num_selected[uuid][img] / self.num_valid_assignments_by_hit[uuid]

            self.hits_by_worker = {}
            for uuid, cur_assignments in self.assignments.items():
                for ca in cur_assignments.values():
                    cur_worker = ca['WorkerId']
                    if cur_worker not in self.hits_by_worker:
                        self.hits_by_worker[cur_worker] = []
                    self.hits_by_worker[cur_worker].append(self.hits[uuid])

            self.assignments_flat = {}
            for cur_assignments in self.assignments.values():
                for ca_id, ca in cur_assignments.items():
                    assert ca_id not in self.assignments_flat
                    self.assignments_flat[ca_id] = ca
    
    def get_s3_file_bytes(self, remote_filename, verbose=True, num_replicas=1):
        return utils.get_s3_file_bytes(remote_filename,
                                       bucket=self.bucket,
                                       cache_on_local_disk=self.cache_on_local_disk,
                                       cache_root_path=self.cache_root_path,
                                       verbose=verbose,
                                       num_replicas=num_replicas)


        
    

from near_duplicate_checker import main
from collections import namedtuple
from timeit import default_timer as timer

Args = namedtuple('Args', ['top_k',
                           'input_filename',
                           'output_filename',
                           'return_ndc_results',
                           'metrics',
                           'cd_chunk_size',
                           'ref_chunk_size',
                           'use_pywren',
                           'max_num_candidates',
                           'max_num_references',
                           'dssim_window_size',
                           'no_cache',
                           'cache_root'])

args = Args(top_k=100,
            input_filename='../data/metadata/nearest_neighbor_results.pickle',
            output_filename='../data/metadata/nearest_neighbor_results.pickle',
            return_ndc_results=True,
            metrics=["fc7"],
            cd_chunk_size=1000,
            ref_chunk_size=1000,
            use_pywren=True,
            max_num_candidates=100000,
            max_num_references=None,
            dssim_window_size=35,
            no_cache=False,
            cache_root="ndc_cache")

i = 0
while True:
    start = timer()
    num_candidates_per_dist = main(args)
    end = timer()
    print('Iteration {} took {} seconds'.format(i, end-start))
    i += 1
    if num_candidates_per_dist['fc7'] == 0:
        break;


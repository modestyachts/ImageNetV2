import datetime
import math
from timeit import default_timer as timer

from IPython.display import display
from ipywidgets import widgets
import tqdm


def get_cd_metric_pairs(nn_results, reviews, thresholds, cds):
    cd_metric_pairs = []
    for cd, metric_dict in nn_results.items():
        if cd not in cds.blacklist: 
            cd_flickr_id = cds.all_candidates[cd]['id_search_engine']
            for metric in metric_dict.keys():
                cd_has_unreviewed_nn_below_threshold = False
                for _, nn_dist in enumerate(nn_results[cd][metric][:30]): 
                    nn = nn_dist[0]
                    dist = nn_dist[1]
                    nn_flickr_id = None
                    if nn in cds.all_candidates.keys():
                        nn_flickr_id = cds.all_candidates[nn]['id_search_engine']
                    if dist <= thresholds[metric] and nn != cd and cd_flickr_id != nn_flickr_id and (nn not in cds.blacklist):
                        # Candidate has a nearest neighbor below the current threshold,
                        # now check if it is unreviewed
                        if (cd in reviews) and (metric in reviews[cd]) and ('references' in
                            reviews[cd][metric]):
                                if nn not in reviews[cd][metric]['references']:
                                    cd_has_unreviewed_nn_below_threshold = True
                        else:
                            cd_has_unreviewed_nn_below_threshold = True
                if cd_has_unreviewed_nn_below_threshold:
                    cd_metric_pairs.append((cd, metric))
    return cd_metric_pairs

def get_num_near_duplicates(near_duplicates):
    num_near_duplicates = 0
    for key in near_duplicates:
        if len(near_duplicates[key]) > 0:
            num_near_duplicates += 1
    return num_near_duplicates

def get_near_duplicates_to_save(near_duplicates):
    near_duplicates_to_save = {}
    for key in near_duplicates:
        tmp_list = sorted(list(set(near_duplicates[key])))
        if len(tmp_list) > 0:
            near_duplicates_to_save[key] = tmp_list
    return near_duplicates_to_save

def parse_checkboxes(checkboxes):
    cur_selected = {}
    cur_unselected = {}
    for cd_image, metric_dict in checkboxes.items():
        cur_selected[cd_image] = {}
        cur_unselected[cd_image] = {}
        for metric, ref_to_box in metric_dict.items():
            cur_selected[cd_image][metric] = []
            cur_unselected[cd_image][metric] = []
            for ref_image, box in ref_to_box.items():
                if box.value:
                    cur_selected[cd_image][metric].append(ref_image)
                else:
                    cur_unselected[cd_image][metric].append(ref_image)
    return cur_selected, cur_unselected

def verify_checkboxes(checkboxes, cur_selected, cur_unselected, near_duplicates):
    for cd_image, metric_dict in cur_selected.items():
        for metric, selected_candidates in metric_dict.items():
            for cur_selected_candidate in selected_candidates:
                if cd_image not in near_duplicates:
                    near_duplicates[cd_image] = []
                if cur_selected_candidate not in near_duplicates[cd_image]:
                    near_duplicates[cd_image].append(cur_selected_candidate)

    for cd_image, metric_dict in cur_unselected.items():
        for metric, unselected_candidates in metric_dict.items():
            for cur_unselected_candidate in unselected_candidates:
                if cd_image in near_duplicates and cur_unselected_candidate in near_duplicates[cd_image]:
                    near_duplicates[cd_image].remove(cur_unselected_candidate)
    return near_duplicates

def parse_review_boxes(review_boxes, reviews, reviewer_name, thresholds): 
    for cd_image, metric_dict in review_boxes.items():
        for metric, ref_dict in metric_dict.items():
            box = ref_dict['box']
            if box.value:
                if cd_image not in reviews:
                    reviews[cd_image] = {}
                if metric not in reviews[cd_image]:
                    reviews[cd_image][metric] = {}
                reviews[cd_image][metric]['reviewer'] = reviewer_name
                reviews[cd_image][metric]['date'] = str(datetime.datetime.now())
                if 'references' not in reviews[cd_image][metric]:
                    reviews[cd_image][metric]['references'] = []
                for ref_name in ref_dict['references']:
                    reviews[cd_image][metric]['references'].append(ref_name)
                reviews[cd_image][metric]['references'] = list(set(sorted(reviews[cd_image][metric]['references'])))
                #reviews[cd_image][metric]['threshold'] = thresholds[metric]
    return reviews


def review_near_duplicates(cd_metric_pairs,
                           nn_results,
                           reviews,
                           near_duplicates,
                           top_k, 
                           thresholds,
                           offset, 
                           loader,
                           cds,
                           max_to_show=20):
    num_images_to_show = min(len(cd_metric_pairs) - offset, max_to_show)
    num_cols = top_k + 1
    num_rows = 25
    num_tabs = int(math.ceil(num_images_to_show / num_rows))
    checkboxes = {}
    review_boxes = {}
    image_width = '256px'
    image_height = '256px'

    img_filenames = []
    references_to_show = {}
    images_with_warnings = []
    for ii in range(num_images_to_show):
        cur_name, cur_metric = cd_metric_pairs[offset + ii]
        if cur_name not in references_to_show:
            references_to_show[cur_name] = {}
        img_filenames.append(cur_name)
        reviewed_references = []
        if cur_name in reviews:
            if cur_metric in reviews[cur_name]: 
                reviewed_references = reviews[cur_name][cur_metric]['references']
        cur_references = []
        cur_flickr_id = cds.all_candidates[cur_name]['id_search_engine']
        for ii, ref_name_dst in enumerate(nn_results[cur_name][cur_metric]):
            ref_name = ref_name_dst[0]
            dst = ref_name_dst[1]
            ref_flickr_id = None
            if ref_name in cds.all_candidates.keys():
                ref_flickr_id = cds.all_candidates[ref_name]['id_search_engine']
            if ref_name == cur_name:
                pass
            elif ref_name in cds.blacklist:
                pass
            elif dst > thresholds[cur_metric]:
                pass
            elif ref_name in reviewed_references:
                pass
            elif cur_flickr_id == ref_flickr_id:
                pass
            elif ii >= 30:
                pass
            else:
                cur_references.append((ref_name, dst))
        if len(cur_references) > num_cols - 1:
            images_with_warnings.append((cur_name, cur_metric, len(cur_references)))
            cur_references = cur_references[:num_cols - 1]
        references_to_show[cur_name][cur_metric] = cur_references
        img_filenames.extend([x[0] for x in cur_references])
    img_filenames = set(img_filenames)
    print('Loading image data ... ', end='')
    start = timer()
    img_data = loader.load_image_bytes_batch(img_filenames, size='scaled_256', verbose=False)
    end = timer()
    print('done, took {} seconds'.format(end - start))

    tab_contents = []
    for kk in tqdm.tqdm_notebook(range(num_tabs), desc='Setting up image tabs'):
        rows = []
        cur_num_rows = num_rows
        if kk == num_tabs - 1:
            cur_num_rows = int(math.ceil(num_images_to_show - (num_tabs - 1) * num_rows))
        for ii in range(cur_num_rows):
            cur_row = []
            cur_name, cur_metric = cd_metric_pairs[offset + kk*num_rows + ii]
            if cur_name not in checkboxes:
                checkboxes[cur_name] = {}
            assert cur_metric not in checkboxes[cur_name]
            #if cur_metric not in checkboxes[cur_name]:
            checkboxes[cur_name][cur_metric] = {}
            cur_img = widgets.Image(value=img_data[cur_name], layout=widgets.Layout(width=image_width, height=image_height))
            cur_label = widgets.Label(value=cur_name, layout=widgets.Layout(margin='0px', height='20px'))
            cur_label.add_class('image_name')
            review_label = widgets.Label(value='Metric {},  reviewed: '.format(cur_metric))
            is_reviewed = False
            review_checkbox = widgets.Checkbox(is_reviewed, 
                                               description='', 
                                               indent=False, 
                                               layout=widgets.Layout(width='100px', height='28'))
            if cur_name not in review_boxes:
                review_boxes[cur_name] = {}
            review_boxes[cur_name][cur_metric] = {'box': review_checkbox,
                                                  'references': [ref_name for ref_name, _ in references_to_show[cur_name][cur_metric]]}
            review_box = widgets.HBox([review_label, review_checkbox])
            cur_box = widgets.VBox([cur_img, cur_label, review_box])
            cur_box.layout.align_items = 'center'
            cur_box.layout.padding = '2px'
            cur_row.append(cur_box)
            for ref_name, dst in references_to_show[cur_name][cur_metric]:
                ref_img = widgets.Image(value=img_data[ref_name], layout=widgets.Layout(width=image_width, height=image_height))
                is_duplicate = False
                if cur_name in near_duplicates:
                    if ref_name in near_duplicates[cur_name]:
                        is_duplicate = True
                label_text = str(ref_name)
                dist_text = ' {:.2f}'.format(dst)
                ref_label = widgets.Label(value=label_text, layout=widgets.Layout(margin='0px', height='20px'))
                ref_label.add_class('image_name')
                ref_checkbox = widgets.Checkbox(is_duplicate, 
                                                description='', 
                                                indent=False, 
                                                layout=widgets.Layout(width='100px', height='28')) 
                checkbox_label = widgets.Label(value='Dst {},  duplicate:'.format(dist_text))
                ref_box = widgets.HBox([checkbox_label, ref_checkbox])
                checkboxes[cur_name][cur_metric][ref_name] = ref_checkbox
                cur_box = widgets.VBox([ref_img, ref_label, ref_box])
                cur_box.layout.align_items = 'center'
                cur_row.append(cur_box)
            cur_hbox = widgets.HBox(cur_row)
            rows.append(cur_hbox)
        tab_contents.append(widgets.VBox(rows))

    print('The following images have candidates below the threshold that were omitted:\n{}'.format(', '.join([r[0] for r in images_with_warnings])))

    tab = widgets.Tab()
    tab.children = tab_contents
    for i in range(len(tab.children)):
        tab.set_title(i, str(i))
    display(tab)
    return checkboxes, review_boxes



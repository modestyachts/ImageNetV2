from datetime import datetime, timezone
import getpass
import json
import pathlib
import pickle

from tqdm import tqdm

import utils

#train_metadata_filename = 'metadata/imagenet_train_tar_structure.json'
#train_metadata_file_bytes = utils.get_s3_file_bytes(train_metadata_filename, cache_on_local_disk=False)
#train_tar_structure = json.loads(train_metadata_file_bytes.decode('utf-8'))

train_tar_file_path = pathlib.Path('imagenet_train_tar_structure.json').resolve()
if train_tar_file_path.is_file():
    print('Loading train tar structure from {} ...'.format(train_tar_file_path))
    with open(train_tar_file_path, 'r') as f:
        train_tar_structure = json.load(f)
else:
    print('Computing train tar structure ...')
    train_tar_structure = utils.get_s3_tar_structure('imagenet-train/')
    with open(train_tar_file_path, 'w') as f:
        json.dump(train_tar_structure, f, indent=2)
assert len(train_tar_structure) == 1000

val_tar_file_path = pathlib.Path('imagenet_val_tar_structure.json').resolve()
if val_tar_file_path.is_file():
    print('Loading val tar structure from {} ...'.format(val_tar_file_path))
    with open(val_tar_file_path, 'r') as f:
        val_tar_structure = json.load(f)
else:
    print('Computing val tar structure ...')
    val_tar_structure = utils.get_s3_tar_structure('imagenet-validation/')
    with open(val_tar_file_path, 'w') as f:
        json.dump(val_tar_structure, f, indent=2)
assert len(val_tar_structure) == 1000

test_structure_file_path = pathlib.Path('imagenet_test_structure.json').resolve()
if test_structure_file_path.is_file():
    print('Loading test structure from {} ...'.format(test_structure_file_path))
    with open(test_structure_file_path, 'r') as f:
        test_structure = json.load(f)
else:
    print('Computing test structure ...')
    test_structure = utils.list_all_keys('imagenet-test/')
    with open(test_structure_file_path, 'w') as f:
        json.dump(test_structure, f, indent=2)
assert len(test_structure) == 100000

train_imgs_by_wnid = {}
for key, values in train_tar_structure.items():
    parts = key.split('/')
    assert parts[0] == 'imagenet-train'
    assert parts[1].startswith('n')
    wnid = parts[1][:9]
    assert wnid[1:].isdigit()
    image_names = []
    for val in values:
        assert val.startswith(wnid + '/')
        cur_name = val[len(wnid) + 1:]
        assert cur_name.startswith(wnid + '_')
        image_names.append(cur_name)
        tmp_path = pathlib.Path(cur_name)
        extension = tmp_path.suffix[1:]
        if extension not in ['JPEG']:
            print('WARNING: file {} has a non-standard extension {}.'.format(val, extension))
        stem = tmp_path.stem
        stem_parts = stem.split('_')
        assert len(stem_parts) == 2
        assert stem_parts[0] == wnid
        assert stem_parts[1].isdigit()
    train_imgs_by_wnid[wnid] = image_names
assert len(train_imgs_by_wnid) == 1000


val_imgs_by_wnid = {}
wnid_by_val_img_filename = {}
#'imagenet-validation/val-n01440764-scaled.tar'
#'n01440764/ILSVRC2012_val_00040358.JPEG'
for key, values in val_tar_structure.items():
    parts = key.split('/')
    assert parts[0] == 'imagenet-validation'
    assert parts[1].startswith('val-n')
    assert parts[1].endswith('-scaled.tar')
    wnid = parts[1][4:13]
    assert wnid[1:].isdigit()
    image_names = []
    for val in values:
        assert val.startswith(wnid + '/')
        cur_name = val[len(wnid) + 1:]
        assert cur_name.startswith('ILSVRC2012_val_')
        image_names.append(cur_name)
        tmp_path = pathlib.Path(cur_name)
        extension = tmp_path.suffix[1:]
        if extension not in ['JPEG']:
            print('WARNING: file {} has a non-standard extension {}.'.format(val, extension))
        stem = tmp_path.stem
        stem_parts = stem.split('_')
        assert len(stem_parts) == 3
        assert stem_parts[0] == 'ILSVRC2012'
        assert stem_parts[1] == 'val'
        assert stem_parts[2].isdigit()
        assert cur_name not in wnid_by_val_img_filename
        wnid_by_val_img_filename[cur_name] = wnid
    val_imgs_by_wnid[wnid] = image_names
assert len(val_imgs_by_wnid) == 1000
assert len(wnid_by_val_img_filename) == 50000

test_filenames = []
for filepath in test_structure:
    parts = filepath.split('/')
    assert parts[0] == 'imagenet-test'
    filename = parts[1]
    prefix = 'ILSVRC2012_test_'
    assert filename.startswith(prefix)
    assert filename.endswith('.JPEG')
    assert filename[len(prefix) : len(prefix) + 8].isdigit()
    test_filenames.append(filename)
assert len(test_filenames) == 100000


test_batches_file_path = pathlib.Path('imagenet_test_batches.json').resolve()
if test_batches_file_path.is_file():
    print('Loading test batches from {} ...'.format(test_batches_file_path))
    with open(test_batches_file_path, 'r') as f:
        test_batches = json.load(f)
else:
    raise NotImplementedError()
assert len(test_batches) == 100

test_batch_by_filename = {}
test_filenames_set = set(test_filenames)
for k, b in test_batches.items():
    assert len(b) == 1000
    for filename in b:
        assert filename in test_filenames_set
        test_batch_by_filename[filename] = k
assert len(test_batch_by_filename) == len(test_filenames)
for filename, b in test_batch_by_filename.items():
    assert filename in test_filenames
    assert filename in test_batches[test_batch_by_filename[filename]]


class_info_filepath = pathlib.Path(__file__).parent /  '../data/metadata/class_info.json'
if class_info_filepath.is_file():
    print('Loading class info from {} ...'.format(class_info_filepath))
    with open(class_info_filepath, 'r') as f:
        tmp_class_info = json.load(f)
else:
    raise Exception()
assert len(tmp_class_info) == 1000
wikipedia_info_filepath = pathlib.Path(__file__).parent /  '../data/metadata/wikipedia_pages.json'
if wikipedia_info_filepath.is_file():
    print('Loading wikipedia pages from {} ...'.format(wikipedia_info_filepath))
    with open(wikipedia_info_filepath, 'r') as f:
        wikipedia_info = json.load(f)
else:
    raise Exception()
assert len(wikipedia_info) == 1000
for wnid in train_imgs_by_wnid:
    assert wnid in wikipedia_info

class_info_by_wnid = {}
for tmpci in tmp_class_info:
    cur_wnid = tmpci['wnid']
    assert cur_wnid in train_imgs_by_wnid
    assert cur_wnid not in class_info_by_wnid
    tmpci['wikipedia_pages'] = wikipedia_info[cur_wnid]
    class_info_by_wnid[cur_wnid] = tmpci

overall_result = {}
overall_result['train_imgs_by_wnid'] = train_imgs_by_wnid
overall_result['val_imgs_by_wnid'] = val_imgs_by_wnid
overall_result['wnid_by_val_img_filename'] = wnid_by_val_img_filename
overall_result['test_filenames'] = test_filenames
overall_result['test_batches'] = test_batches
overall_result['test_batch_by_filename'] = test_batch_by_filename
overall_result['class_info_by_wnid'] = class_info_by_wnid

bucket = 'imagenet2datav2'
num_replicas = 10

time_string = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%Z')
key = 'metadata/imagenet_metadata_' + time_string + '.pickle'

pickle_bytes = pickle.dumps(overall_result)
utils.put_s3_object_bytes_with_backoff(pickle_bytes, key, bucket=bucket)

if num_replicas > 1:
    destinations = []
    replicas_counter_len = len(str(num_replicas))
    format_string = '_replica{{:0{}d}}-{{}}'.format(replicas_counter_len)
    for ii in range(num_replicas):
        destinations.append(key + format_string.format(ii + 1, num_replicas))
    for dest in tqdm(destinations):
        utils.put_s3_object_bytes_with_backoff(pickle_bytes, dest, bucket=bucket)

print('Wrote metadata pickle file to s3://{}/{}'.format(bucket, key))
print('    {} bytes in total'.format(len(pickle_bytes)))

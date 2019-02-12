import json
import pathlib

import click
import numpy as np
import torchvision.models
from tqdm import tqdm

import candidate_data
import eval_utils
import image_loader
import imagenet
import pretrainedmodels
import pretrainedmodels.utils as pretrained_utils
import torch
import os
import time

torch.backends.cudnn.deterministic = True

all_models = ['alexnet',
              'densenet121',
              'densenet161',
              'densenet169',
              'densenet201',
              'inception_v3',
              'resnet101',
              'resnet152',
              'resnet18',
              'resnet34',
              'resnet50',
              'squeezenet1_0',
              'squeezenet1_1',
              'vgg11',
              'vgg11_bn',
              'vgg13',
              'vgg13_bn',
              'vgg16',
              'vgg16_bn',
              'vgg19',
              'vgg19_bn']

extra_models = []
for m in pretrainedmodels.model_names:
    if m not in all_models:
        all_models.append(m)
        extra_models.append(m)



@click.command()
@click.option('--dataset', required=True, type=str)
@click.option('--models', required=True, type=str)
@click.option('--batch_size', default=32, type=str)
def eval(dataset, models, batch_size):
    dataset_filename = dataset
    if models == 'all':
        models = all_models
    else:
        models = models.split(',')
    for model in models:
        assert model in all_models

    dataset_filepath = pathlib.Path(__file__).parent / '../data/datasets' / (dataset_filename + '.json')
    print('Reading dataset from {} ...'.format(dataset_filepath))
    with open(dataset_filepath, 'r') as f:
        dataset = json.load(f)
    cur_imgs = [x[0] for x in dataset['image_filenames']]

    imgnet = imagenet.ImageNetData()
    cds = candidate_data.CandidateData(load_metadata_from_s3=False, exclude_blacklisted_candidates=False)
    loader = image_loader.ImageLoader(imgnet, cds)

    pbar = tqdm(total=len(cur_imgs), desc='Dataset download')
    img_data = loader.load_image_bytes_batch(cur_imgs, size='scaled_500', verbose=False, download_callback=lambda x:pbar.update(x))
    pbar.close()

    for model in tqdm(models, desc='Model evaluations'):
        if (model not in extra_models):
            tqdm.write('Evaluating {}'.format(model))
            resize_size = 256
            center_crop_size = 224
            if model == 'inception_v3':
                resize_size = 299
                center_crop_size = 299
            data_loader = eval_utils.get_data_loader(cur_imgs,
                                                     imgnet,
                                                     cds,
                                                     image_size='scaled_500',
                                                     resize_size=resize_size,
                                                     center_crop_size=center_crop_size,
                                                     batch_size=batch_size)
            pt_model = getattr(torchvision.models, model)(pretrained=True)
            if (torch.cuda.is_available()):
                pt_model = pt_model.cuda()
            pt_model.eval()
            tqdm.write('    Number of trainable parameters: {}'.format(sum(p.numel() for p in pt_model.parameters() if p.requires_grad)))

            predictions, top1_acc, top5_acc, total_time, num_images = eval_utils.evaluate_model(
                    pt_model, data_loader, show_progress_bar=True)
            tqdm.write('    Evaluated {} images'.format(num_images))
            tqdm.write('    Top-1 accuracy: {:.2f}'.format(100.0 * top1_acc))
            tqdm.write('    Top-5 accuracy: {:.2f}'.format(100.0 * top5_acc))
            tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))
            npy_out_filepath = pathlib.Path(__file__).parent / '../data/predictions' / dataset_filename / (model + '.npy')
            npy_out_filepath = npy_out_filepath.resolve()
            directory = os.path.dirname(npy_out_filepath)
            if not os.path.exists(directory):
                    os.makedirs(directory)
            if (os.path.exists(npy_out_filepath)):
                old_preds = np.load(npy_out_filepath)
                np.save(f'{npy_out_filepath}.{int(time.time())}', old_preds)
                print('checking old preds is same as new preds')
                if not np.allclose(old_preds, predictions):
                    diffs = np.round(old_preds - predictions, 4)
                    print('old preds != new preds')
                else:
                    print('old preds == new_preds!')
            np.save(npy_out_filepath, predictions)
            tqdm.write('    Saved predictions to {}'.format(npy_out_filepath))
        else:
            tqdm.write('Evaluating extra model {}'.format(model))
            if (model in {"dpn68b", "dpn92", "dpn107"}):
                pt_model = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet+5k')
            else:
                pt_model = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
            tf_img = pretrained_utils.TransformImage(pt_model)
            load_img = pretrained_utils.LoadImage()
            tqdm.write('    Number of trainable parameters: {}'.format(sum(p.numel() for p in pt_model.parameters() if p.requires_grad)))

            #print(pt_model)
            #print(load_img)
            dataset = eval_utils.ImageLoaderDataset(cur_imgs, imgnet, cds,
                      'scaled_500', transform=tf_img)

            data_loader = torch.utils.data.DataLoader(dataset,
                         batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)
            if (torch.cuda.is_available()):
                pt_model = pt_model.cuda()

            pt_model.eval()
            predictions, top1_acc, top5_acc, total_time, num_images = eval_utils.evaluate_model(
                    pt_model, data_loader, show_progress_bar=True)
            tqdm.write('    Evaluated {} images'.format(num_images))
            tqdm.write('    Top-1 accuracy: {:.2f}'.format(100.0 * top1_acc))
            tqdm.write('    Top-5 accuracy: {:.2f}'.format(100.0 * top5_acc))
            tqdm.write('    Total time: {:.1f}  (average time per image: {:.2f} ms)'.format(total_time, 1000.0 * total_time / num_images))
            npy_out_filepath = pathlib.Path(__file__).parent / '../data/predictions' / dataset_filename / (model + '.npy')
            npy_out_filepath = npy_out_filepath.resolve()
            directory = os.path.dirname(npy_out_filepath)
            if not os.path.exists(directory):
                    os.makedirs(directory)
            if (os.path.exists(npy_out_filepath)):
                old_preds = np.load(npy_out_filepath)
                np.save(f'{npy_out_filepath}.{int(time.time())}', old_preds)
                print('checking old preds is same as new preds')
                if not np.allclose(old_preds, predictions):
                    diffs = np.round(old_preds - predictions, 4)
                    print('old preds != new preds')
                else:
                    print('old preds == new_preds!')
            np.save(npy_out_filepath, predictions)
            tqdm.write('    Saved predictions to {}'.format(npy_out_filepath))







if __name__ == '__main__':
    eval()

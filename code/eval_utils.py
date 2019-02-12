import math
from timeit import default_timer as timer

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import tqdm

import image_loader


class ImageLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, imgnet, cds, size, verbose=False, transform=None):
        self.imgnet = imgnet
        self.cds = cds
        self.loader = image_loader.ImageLoader(imgnet, cds)
        self.size = size
        self.verbose = verbose
        self.filenames = list(filenames)
        self.transform = transform
        self.wnids = [self.loader.get_wnid_of_image(x) for x in self.filenames]
        self.class_ids = [self.imgnet.class_info_by_wnid[x].cid for x in self.wnids]
        for x in self.class_ids:
            assert x >= 0
            assert x < 1000
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        cur_fn = self.filenames[index]
        img = self.loader.load_image(cur_fn, size=self.size, verbose=self.verbose, loader='pillow', force_rgb=True)
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.class_ids[index])


def get_data_loader(imgs, imgnet, cds, image_size='scaled_500', batch_size=128, num_workers=0, resize_size=256, center_crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ImageLoaderDataset(imgs, imgnet, cds, image_size, transform=transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(center_crop_size),
                transforms.ToTensor(),
                normalize]))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return val_loader


def evaluate_model(model, data_loader, show_progress_bar=False, notebook_progress_bar=False):
    cudnn.benchmark = True

    num_images = 0
    num_top1_correct = 0
    num_top5_correct = 0
    predictions = []
    start = timer()
    with torch.no_grad():
        enumerable = enumerate(data_loader)
        if show_progress_bar:
            total = int(math.ceil(len(data_loader.dataset) / data_loader.batch_size))
            desc = 'Batch'
            if notebook_progress_bar:
                enumerable = tqdm.tqdm_notebook(enumerable, total=total, desc=desc)
            else:
                enumerable = tqdm.tqdm(enumerable, total=total, desc=desc)
        for ii, (img_input, target) in enumerable:
            img_input = img_input.cuda(non_blocking=True)
            _, output_index = model(img_input).topk(k=5, dim=1, largest=True, sorted=True)
            output_index = output_index.cpu().numpy()
            predictions.append(output_index)
            for jj, correct_class in enumerate(target.cpu().numpy()):
                if correct_class == output_index[jj, 0]:
                    num_top1_correct += 1
                if correct_class in output_index[jj, :]:
                    num_top5_correct += 1
            num_images += len(target)
    end = timer()
    predictions = np.vstack(predictions)
    assert predictions.shape == (num_images, 5)
    return predictions, num_top1_correct / num_images, num_top5_correct / num_images, end - start, num_images

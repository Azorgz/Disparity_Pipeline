# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
import glob
import time
from pprint import pprint

import torch

from utils.classes import ImageTensor
from utils.misc import time_fct
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                                 count_parameters)


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""

    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for sample in tqdm(zip(test_loader[0], test_loader[1]), total=len(test_loader[0])):
        image_vis = ImageTensor(sample[0])
        image_ir = ImageTensor(sample[1]).RGB('gray')
        focal = torch.Tensor([1739]).cuda()
        # focal = sample.get('focal', torch.Tensor(
        #     [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = time_fct(model)(image_vis, focal=focal)['metric_depth']
        pred_vis = time_fct(model)(image_vis, focal=focal)['metric_depth']
        pred_ir = time_fct(model)(image_ir, focal=focal)['metric_depth']
        time.sleep(0)
    return pred


def main(config):
    model = build_model(config)
    # test_loader = DepthDataLoader(config, 'online_eval').data
    test_loader = [sorted(glob.glob(config['dataset_vis'] + '/*.png') +
                          glob.glob(config['dataset_vis'] + '/*.jpg') +
                          glob.glob(config['dataset_vis'] + '/*.jpeg')),
                   sorted(glob.glob(config['dataset_ir'] + '/*.png') +
                          glob.glob(config['dataset_ir'] + '/*.jpg') +
                          glob.glob(config['dataset_ir'] + '/*.jpeg'))]
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True) / 1e6, 2)}M"
    return metrics


def infer_model(model_name, pretrained_resource, dataset=None, **kwargs):
    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "infer", **overwrite)
    config['dataset_vis'] = dataset[0]
    config['dataset_ir'] = dataset[1]
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default='zoedepth', type=str,
                        required=False, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False,
                        default="local::./Networks/Depth_anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt",
                        help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='/home/godeta/PycharmProjects/Datasets/Lynred/Day/master',
                        help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)
    datasets = [args.dataset + '/visible', args.dataset + '/infrared_corrected']
    infer_model(args.model, pretrained_resource=args.pretrained_resource,
                dataset=datasets, **overwrite_kwargs)

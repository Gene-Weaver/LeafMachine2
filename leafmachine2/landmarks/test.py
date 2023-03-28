from __future__ import print_function
# import argparse
import os, sys, time, shutil, math, cv2, inspect, itertools
# from parse import parse
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch import nn
# from torch.autograd import Variable
from torch.utils import data
# from torchvision import datasets
from torchvision import transforms
# import torchvision as tv
# from torchvision.models import inception_v3
import skimage.transform
from dataclasses import dataclass
from data import csv_collator
from data import ScaleImageAndLabel
from data import build_dataset
import losses
# import argparser
from models import unet_model
from metrics import Judge
from metrics import make_metric_plots
import utils

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import load_cfg, get_cfg_from_full_path, validate_dir

@dataclass
class Training_Opts():
    dataset: str = ''
    out: str = ''
    model: str = ''  # Checkpoint with the CNN model

    evaluate: bool = True
    cuda: bool = True
    imgsize: str = '256x256'
    radii: str = '4'
    taus: str = '-2' #Detection threshold between 0 and 1. If not selected, 25 thresholds in [0, 1] will be tested. tau=-1 means dynamic Otsu thresholding. tau=-2 means Beta Mixture Model-based thresholding.'
    n_points: int = None
    max_mask_pts: int = 100
    paint: bool = True
    force: bool = False
    seed: int = 4
    max_testset_size: int = float('inf')
    nThreads: int = 8
    ultrasmallnet: bool = False

    height: int = 256
    width: int = 256

    cfg: str = ''

    landmark: str = ''
    run_name: str = ''

    is_scripted: bool = False

    def __init__(self, landmark) -> None:
        dir_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # path_cfg_private = os.path.join(dir_home,'PRIVATE_DATA.yaml')
        # cfg_private = get_cfg_from_full_path(path_cfg_private)
        # if cfg_private['w_and_b']['w_and_b_key'] is not None:
        #     self.w_and_b_key = cfg_private['w_and_b']['w_and_b_key']

        path_to_config = dir_home
        cfg = load_cfg(path_to_config)
        self.cfg = cfg
        
        # if cfg_private['w_and_b']['landmark_project'] is not None:
        #     self.w_and_b_project = cfg_private['w_and_b']['landmark_project']

        # if cfg_private['w_and_b']['entity'] is not None:
        #     self.entity = cfg_private['w_and_b']['entity']
        
        self.dataset = cfg['leafmachine']['landmark_evaluate']['dir_images']
        self.out = cfg['leafmachine']['landmark_evaluate']['dir_save']
        validate_dir(self.out)
        self.model = cfg['leafmachine']['landmark_evaluate']['model']
        
        self.run_name = cfg['leafmachine']['landmark_evaluate']['run_name']
        # self.landmark = cfg['leafmachine']['landmark_evaluate']['landmark']
        self.landmark = landmark
        

        if cfg['leafmachine']['landmark_evaluate']['model_options']['use_gpu']:
            if torch.cuda.is_available():
                self.cuda = True
            else:
                self.cuda = False
        else:
            self.cuda = False

        if cfg['leafmachine']['landmark_evaluate']['model_options']['image_size'] is not None:
            imgsize = int(cfg['leafmachine']['landmark_evaluate']['model_options']['image_size'])
            self.imgsize = 'x'.join([str(imgsize), str(imgsize)])
            self.height = imgsize
            self.width = imgsize

        if cfg['leafmachine']['landmark_evaluate']['model_options']['radius'] is not None:
            self.radii = cfg['leafmachine']['landmark_evaluate']['model_options']['radius']

        if cfg['leafmachine']['landmark_evaluate']['model_options']['max_mask_pts'] is not None:
            self.max_mask_pts = cfg['leafmachine']['landmark_evaluate']['model_options']['max_mask_pts']

        ### Load model
        model_path = os.path.join(self.model, self.landmark)
        model_list = os.listdir(model_path)
        best_model = max(model_list, key=lambda filename: float(filename.split('-')[1].split('.')[0]))
        self.model = os.path.join(model_path, best_model)

        if 'ckpt' in best_model.split('.')[1]:
            self.is_scripted = False
        elif 'pt' in best_model.split('.')[1]:
            self.is_scripted = True


        # String/Int -> List
        if isinstance(self.taus, (list, range)):
            pass
        elif isinstance(self.taus, str) and ',' in self.taus:
            self.taus = [float(tau)
                         for tau in self.taus.replace('[', '').replace(']', '').split(',')]
        else:
            self.taus = [float(self.taus)]

        if isinstance(self.radii, (list, range)):
            pass
        elif isinstance(self.radii, str) and ',' in self.radii:
            self.radii = [int(r) for r in self.radii.replace('[', '').replace(']', '').split(',')]
        else:
            self.radii = [int(self.radii)]

def detect_landmark(landmark):
    opts = Training_Opts(landmark)

    # Prepare data
    testset, testset_loader, criterion_training, device, device_cpu = build_dataloaders(opts)

    # Prepare model
    try:
        model, bmm_tracker, tic = load_checkpoint_withDataParallel(opts, device)
    except:
        model, bmm_tracker, tic = load_checkpoint(opts, device)


    if testset.there_is_gt:
        # Prepare Judges that will compute P/R as fct of r and th
        judges = []
        for r, th in itertools.product(opts.radii, opts.taus):
            judge = Judge(r=r)
            judge.th = th
            judges.append(judge)

    # Empty output CSV (one per threshold)
    df_outs = [pd.DataFrame() for _ in opts.taus]

    # --force will overwrite output directory
    if opts.force:
        shutil.rmtree(opts.out)

    for batch_idx, (imgs, dictionaries) in tqdm(enumerate(testset_loader),
                                                total=len(testset_loader)):

        # Move to device
        imgs = imgs.to(device)

        # Pull info from this batch and move to device
        if testset.there_is_gt:
            target_locations = [dictt['locations'].to(device)
                                for dictt in dictionaries]
            target_count = [dictt['count'].to(device)
                            for dictt in dictionaries]

        target_orig_heights = [dictt['orig_height'].to(device)
                            for dictt in dictionaries]
        target_orig_widths = [dictt['orig_width'].to(device)
                            for dictt in dictionaries]

        # Lists -> Tensor batches
        if testset.there_is_gt:
            target_count = torch.stack(target_count)
        target_orig_heights = torch.stack(target_orig_heights)
        target_orig_widths = torch.stack(target_orig_widths)
        target_orig_sizes = torch.stack((target_orig_heights,
                                        target_orig_widths)).transpose(0, 1)
        origsize = (dictionaries[0]['orig_height'].item(),
                    dictionaries[0]['orig_width'].item())

        # Tensor -> float & numpy
        if testset.there_is_gt:
            target_count = target_count.item()
            target_locations = \
                target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
        target_orig_size = \
            target_orig_sizes[0].to(device_cpu).numpy().reshape(2)

        normalzr = utils.Normalizer(opts.height, opts.width)

        # Feed forward
        with torch.no_grad():
            est_maps, est_count = model.forward(imgs)

        # Convert to original size
        est_map_np = est_maps[0, :, :].to(device_cpu).numpy()
        est_map_np_origsize = \
            skimage.transform.resize(est_map_np,
                                    output_shape=origsize,
                                    mode='constant')
        orig_img_np = imgs[0].to(device_cpu).squeeze().numpy()
        orig_img_np_origsize = ((skimage.transform.resize(orig_img_np.transpose((1, 2, 0)),
                                                    output_shape=origsize,
                                                    mode='constant') + 1) / 2.0 * 255.0).\
            astype(np.float32).transpose((2, 0, 1))

        # Overlay output on original image as a heatmap
        orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_np_origsize,
                                                            map=est_map_np_origsize).\
            astype(np.float32)

        # Save estimated map to disk
        os.makedirs(os.path.join(opts.out, 'intermediate', 'estimated_map'),
                    exist_ok=True)
        cv2.imwrite(os.path.join(opts.out,
                                'intermediate',
                                'estimated_map',
                                dictionaries[0]['filename']),
                    orig_img_w_heatmap_origsize.transpose((1, 2, 0))[:, :, ::-1])

        # Tensor -> int
        est_count_int = int(round(est_count.item()))

        # The estimated map must be thresholded to obtain estimated points
        for t, tau in enumerate(opts.taus):
            if tau != -2:
                mask, _ = utils.threshold(est_map_np_origsize, tau)
            else:
                mask, _, mix = utils.threshold(est_map_np_origsize, tau)
                bmm_tracker.feed(mix)
            centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                            max_mask_pts=opts.max_mask_pts)

            # Save thresholded map to disk
            os.makedirs(os.path.join(opts.out,
                                    'intermediate',
                                    'estimated_map_thresholded',
                                    f'tau={round(tau, 4)}'),
                        exist_ok=True)
            cv2.imwrite(os.path.join(opts.out,
                                    'intermediate',
                                    'estimated_map_thresholded',
                                    f'tau={round(tau, 4)}',
                                    dictionaries[0]['filename']),
                        mask)

            # Paint red dots if user asked for it
            if opts.paint:
                # Paint a cross at the estimated centroids
                img_with_x_n_map = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                    points=centroids_wrt_orig,
                                                    color='red',
                                                    crosshair=True)
                # Save to disk
                os.makedirs(os.path.join(opts.out,
                                        'intermediate',
                                        'painted_on_estimated_map',
                                        f'tau={round(tau, 4)}'), exist_ok=True)
                cv2.imwrite(os.path.join(opts.out,
                                        'intermediate',
                                        'painted_on_estimated_map',
                                        f'tau={round(tau, 4)}',
                                        dictionaries[0]['filename']),
                            img_with_x_n_map.transpose((1, 2, 0))[:, :, ::-1])
                # Paint a cross at the estimated centroids
                img_with_x = utils.paint_circles(img=orig_img_np_origsize,
                                                points=centroids_wrt_orig,
                                                color='red',
                                                crosshair=True)
                # Save to disk
                os.makedirs(os.path.join(opts.out,
                                        'intermediate',
                                        'painted_on_original',
                                        f'tau={round(tau, 4)}'), exist_ok=True)
                cv2.imwrite(os.path.join(opts.out,
                                        'intermediate',
                                        'painted_on_original',
                                        f'tau={round(tau, 4)}',
                                        dictionaries[0]['filename']),
                            img_with_x.transpose((1, 2, 0))[:, :, ::-1])


            if opts.evaluate:
                target_locations_wrt_orig = normalzr.unnormalize(target_locations,
                                                                orig_img_size=target_orig_size)

                # Compute metrics for each value of r (for each Judge)
                for judge in judges:
                    if judge.th != tau:
                        continue
                    judge.feed_points(centroids_wrt_orig, target_locations_wrt_orig,
                                    max_ahd=criterion_training.max_dist)
                    judge.feed_count(est_count_int, target_count)

            # Save a new line in the CSV corresonding to the resuls of this img
            res_dict = dictionaries[0]
            res_dict['count'] = est_count_int
            res_dict['locations'] = str(centroids_wrt_orig.tolist())
            for key, val in res_dict.copy().items():
                if 'height' in key or 'width' in key:
                    del res_dict[key]
            df = pd.DataFrame(data={idx: [val] for idx, val in res_dict.items()})
            df = df.set_index('filename')
            df_outs[t] = df_outs[t].append(df)

    # Write CSVs to disk
    os.makedirs(os.path.join(opts.out, 'estimations'), exist_ok=True)
    for df_out, tau in zip(df_outs, opts.taus):
        df_out.to_csv(os.path.join(opts.out,
                                'estimations',
                                f'estimations_tau={round(tau, 4)}.csv'))

    os.makedirs(os.path.join(opts.out, 'intermediate', 'metrics_plots'),
                exist_ok=True)

    if opts.evaluate:

        with tqdm(total=len(judges), desc="Evaluating metrics") as pbar:
            # Output CSV where we will put all our metrics as a function of r and the threshold
            df_metrics = pd.DataFrame(columns=['r', 'th',
                                            'precision', 'recall', 'fscore', 'MAHD',
                                            'MAPE', 'ME', 'MPE', 'MAE',
                                            'MSE', 'RMSE', 'r', 'R2'])
            df_metrics.index.name = 'idx'

            for j, judge in enumerate(judges):
                # Accumulate precision and recall in the CSV dataframe
                df = pd.DataFrame(data=[[judge.r,
                                        judge.th,
                                        judge.precision,
                                        judge.recall,
                                        judge.fscore,
                                        judge.mahd,
                                        judge.mape,
                                        judge.me,
                                        judge.mpe,
                                        judge.mae,
                                        judge.mse,
                                        judge.rmse,
                                        judge.pearson_corr,
                                        judge.coeff_of_determination]],
                                columns=['r', 'th',
                                        'precision', 'recall', 'fscore', 'MAHD',
                                        'MAPE', 'ME', 'MPE', 'MAE',
                                        'MSE', 'RMSE', 'r', 'R2'],
                                index=[j])
                df.index.name = 'idx'
                df_metrics = df_metrics.append(df)
                pbar.update(1)

            # Write CSV of metrics to disk
            df_metrics.to_csv(os.path.join(opts.out, 'metrics.csv'))

            # Generate plots
            figs = make_metric_plots(csv_path=os.path.join(opts.out, 'metrics.csv'),
                                    taus=opts.taus,
                                    radii=opts.radii)
            for label, fig in figs.items():
                # Save to disk
                fig.savefig(os.path.join(opts.out,
                                        'intermediate',
                                        'metrics_plots',
                                        f'{label}.png'))



    # Save plot figures of the statistics of the BMM-based threshold
    if -2 in opts.taus:
        for label, fig in bmm_tracker.plot().items():
            fig.savefig(os.path.join(opts.out,
                                    'intermediate',
                                    'metrics_plots',
                                    f'{label}.png'))


    elapsed_time = int(time.time() - tic)
    print(f'It took {elapsed_time} seconds to evaluate all this dataset.')



def build_dataloaders(opts):
    # Tensor type to use, select CUDA or not
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device('cuda:0') if opts.cuda else device_cpu

    # Set seeds
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    # Data loading code
    try:
        testset = build_dataset(opts.dataset,
                                transforms=transforms.Compose([
                                    ScaleImageAndLabel(size=(opts.height, opts.width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]),
                                ignore_gt=not opts.evaluate,
                                max_dataset_size=opts.max_testset_size)
    except ValueError as e:
        print(f'E: {e}')
        exit(-1)
    testset_loader = data.DataLoader(testset,
                                     batch_size=1,
                                     num_workers=opts.nThreads,
                                     collate_fn=csv_collator)

    # Array with [height, width] of the new size
    resized_size = np.array([opts.height, opts.width])

    # Loss function
    criterion_training = losses.WeightedHausdorffDistance(resized_height=opts.height,
                                                           resized_width=opts.width,
                                                           return_2_terms=True,
                                                           device=device)

    return testset, testset_loader, criterion_training, device, device_cpu

def load_checkpoint(opts, device):
    with tqdm(total=1, desc="Loading checkpoint") as pbar:
        if os.path.isfile(opts.model):
            if opts.cuda:
                checkpoint = torch.load(opts.model, map_location=torch.device('cuda:0'))
            else:
                checkpoint = torch.load(
                    opts.model, map_location=lambda storage, loc: storage)
            # Model
            if opts.n_points is None:
                if 'n_points' not in checkpoint:
                    # Model will also estimate # of points
                    model = unet_model.UNet(3, 1,
                                            known_n_points=None,
                                            height=opts.height,
                                            width=opts.width,
                                            ultrasmall=opts.ultrasmallnet)

                else:
                    # The checkpoint tells us the # of points to estimate
                    model = unet_model.UNet(3, 1,
                                            known_n_points=checkpoint['n_points'],
                                            height=opts.height,
                                            width=opts.width,
                                            ultrasmall=opts.ultrasmallnet)
            else:
                # The user tells us the # of points to estimate
                model = unet_model.UNet(3, 1,
                                        known_n_points=opts.n_points,
                                        height=opts.height,
                                        width=opts.width,
                                        ultrasmall=opts.ultrasmallnet)

            # Parallelize
            model = model.to(device)

            # Load model in checkpoint
            if opts.cuda:
                state_dict = checkpoint['model']
            else:
                # remove 'module.' of DataParallel
                state_dict = OrderedDict()
                for k, v in checkpoint['model'].items():
                    name = k[7:]
                    state_dict[name] = v
            model.load_state_dict(state_dict)

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n\__ loaded checkpoint '{opts.model}' "
                  f"with {num_params} trainable parameters")
        else:
            print(f"\n\__  E: no checkpoint found at '{opts.model}'")
            exit(-1)

        # Set the module in evaluation mode
        model.eval()

        # Accumulative histogram of estimated maps
        bmm_tracker = utils.AccBetaMixtureModel()

        tic = time.time()

    return model, bmm_tracker, tic

def load_checkpoint_withDataParallel(opts, device):
    with tqdm(total=1, desc="Loading checkpoint") as pbar:
        if os.path.isfile(opts.model):
            if opts.cuda:
                checkpoint = torch.load(opts.model, map_location=torch.device('cuda:0'))
            else:
                checkpoint = torch.load(
                    opts.model, map_location=lambda storage, loc: storage)
            # Model
            if opts.n_points is None:
                if 'n_points' not in checkpoint:
                    # Model will also estimate # of points
                    model = unet_model.UNet(3, 1,
                                            known_n_points=None,
                                            height=opts.height,
                                            width=opts.width,
                                            ultrasmall=opts.ultrasmallnet)

                else:
                    # The checkpoint tells us the # of points to estimate
                    model = unet_model.UNet(3, 1,
                                            known_n_points=checkpoint['n_points'],
                                            height=opts.height,
                                            width=opts.width,
                                            ultrasmall=opts.ultrasmallnet)
            else:
                # The user tells us the # of points to estimate
                model = unet_model.UNet(3, 1,
                                        known_n_points=opts.n_points,
                                        height=opts.height,
                                        width=opts.width,
                                        ultrasmall=opts.ultrasmallnet)

            # Parallelize
            if opts.cuda:
                model = nn.DataParallel(model)
            model = model.to(device)

            # Load model in checkpoint
            if opts.cuda:
                state_dict = checkpoint['model']
            else:
                # remove 'module.' of DataParallel
                state_dict = OrderedDict()
                for k, v in checkpoint['model'].items():
                    name = k[7:]
                    state_dict[name] = v
            model.load_state_dict(state_dict)

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n\__ loaded checkpoint '{opts.model}' "
                  f"with {num_params} trainable parameters")
        else:
            print(f"\n\__  E: no checkpoint found at '{opts.model}'")
            exit(-1)

        # Set the module in evaluation mode
        model.eval()

        # Accumulative histogram of estimated maps
        bmm_tracker = utils.AccBetaMixtureModel()

        tic = time.time()

    return model, bmm_tracker, tic



if __name__ == '__main__':
    detect_landmark('lamina_width') # lamina_width midvein_trace

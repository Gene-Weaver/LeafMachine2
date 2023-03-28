from __future__ import print_function
import math, yaml, cv2, os, sys, inspect, time, shutil, wandb
from datetime import timedelta
from itertools import chain
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision as tv
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import skimage.transform
# from peterpy import peter
# from ballpark import ballpark
from dataclasses import dataclass

import losses
from models import unet_model
from metrics import Judge
import logger
# import argparser
import utils
import data
from data import csv_collator
from data import RandomHorizontalFlipImageAndLabel
from data import RandomVerticalFlipImageAndLabel
from data import ScaleImageAndLabel
from torch.utils.tensorboard import SummaryWriter
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import load_cfg, get_cfg_from_full_path, validate_dir
from launch_landmarks import launch


@dataclass
class Training_Opts():
    train_dir: str = ''
    val_dir: str = ''
    dir_save: str = ''
    run_name: str = 'landmark'
    save: str = 'saved_model.ckpt'
    landmark: str = ''

    visdom_env: str = "landmark_training"
    visdom_server: str = "localhost"
    visdom_port: int = 8989

    lr: float = 1e-3
    optimizer: str = 'adam' #['sgd', 'adam']
    batch_size: int = 32
    imgsize: str = '256x256'
    height: int = 256
    width: int = 256
    epochs: int = 50
    resume: str = ''
    log_interval: int = 60
    val_freq: int = 10
    max_mask_pts: int = 100
    radius: int = 2
    n_points: int = None
    lambdaa = 1 # weight that will increase the importance of estimating the right number of points

    nThreads: int = 8
    p: float = -1
    no_data_augm: bool = False
    drop_last_batch: bool = False
    seed: int = 4
    max_trainset_size: int = float('inf')
    max_valset_size: int = float('inf')
    replace_optimizer: bool = True
    paint: bool = False
    ultrasmallnet = False
    eval_batch_size: int = 1

    cuda: bool = True
    no_cuda: bool = False

    cfg: str = ''
    w_and_b_project: str = ''
    w_and_b_key: str = ''
    name: str = ''
    entity: str = ''

    n_gpus: int = 1
    n_machines: int = 1
    default_timeout_minutes: int = 30

    def __init__(self) -> None:
        dir_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path_cfg_private = os.path.join(dir_home,'PRIVATE_DATA.yaml')
        cfg_private = get_cfg_from_full_path(path_cfg_private)
        if cfg_private['w_and_b']['w_and_b_key'] is not None:
            self.w_and_b_key = cfg_private['w_and_b']['w_and_b_key']

        path_to_config = dir_home
        cfg = load_cfg(path_to_config)
        self.cfg = cfg
        
        if cfg_private['w_and_b']['landmark_project'] is not None:
            self.w_and_b_project = cfg_private['w_and_b']['landmark_project']

        if cfg_private['w_and_b']['entity'] is not None:
            self.entity = cfg_private['w_and_b']['entity']

        if cfg['leafmachine']['landmark_train']['landmark'] is not None:
            self.name = cfg['leafmachine']['landmark_train']['landmark']
        self.project = os.path.join(dir_home,self.w_and_b_project,self.w_and_b_project,self.name)

        self.train_dir = cfg['leafmachine']['landmark_train']['dir_train']
        self.val_dir = cfg['leafmachine']['landmark_train']['dir_val']
        self.dir_save = cfg['leafmachine']['landmark_train']['dir_save']
        self.run_name = cfg['leafmachine']['landmark_train']['run_name']
        self.landmark = cfg['leafmachine']['landmark_train']['landmark']

        self.batch_size = cfg['leafmachine']['landmark_train']['model_options']['batch_size']
        self.lr = cfg['leafmachine']['landmark_train']['model_options']['learning_rate']
        self.optim = cfg['leafmachine']['landmark_train']['model_options']['optimizer']
        self.epochs = cfg['leafmachine']['landmark_train']['model_options']['epochs']

        self.save_name = ''.join([self.run_name, '_', self.landmark])
        self.save_dir = os.path.join(self.dir_save,self.run_name,self.landmark)
        validate_dir(self.save_dir)

        if cfg['leafmachine']['landmark_train']['model_options']['number_of_points'] is not None:
            self.n_points = cfg['leafmachine']['landmark_train']['model_options']['number_of_points']
        else:
            self.n_points = None

        if cfg['leafmachine']['landmark_train']['model_options']['use_gpu']:
            if torch.cuda.is_available():
                self.cuda = True
            else:
                self.cuda = False
        else:
            self.cuda = False

        if cfg['leafmachine']['landmark_train']['model_options']['n_gpus'] is not None:
            self.n_gpus = cfg['leafmachine']['landmark_train']['model_options']['n_gpus']
        self.default_timeout_minutes = timedelta(self.default_timeout_minutes)

        if cfg['leafmachine']['landmark_train']['model_options']['visdom_env'] is not None:
            self.visdom_env = cfg['leafmachine']['landmark_train']['model_options']['visdom_env']
        
        if cfg['leafmachine']['landmark_train']['model_options']['visdom_server'] is not None:
            self.visdom_server = cfg['leafmachine']['landmark_train']['model_options']['visdom_server']

        if cfg['leafmachine']['landmark_train']['model_options']['visdom_port'] is not None:
            self.visdom_port = cfg['leafmachine']['landmark_train']['model_options']['visdom_port']

        if cfg['leafmachine']['landmark_train']['model_options']['image_size'] is not None:
            imgsize = int(cfg['leafmachine']['landmark_train']['model_options']['image_size'])
            self.imgsize = 'x'.join([str(imgsize), str(imgsize)])
            self.height = imgsize
            self.width = imgsize

        if cfg['leafmachine']['landmark_train']['model_options']['resume'] is not None:
            self.resume = cfg['leafmachine']['landmark_train']['model_options']['resume'] 

        if cfg['leafmachine']['landmark_train']['model_options']['log_interval'] is not None:
            self.log_interval = cfg['leafmachine']['landmark_train']['model_options']['log_interval']
        
        if cfg['leafmachine']['landmark_train']['model_options']['validation_frequency'] is not None:
            self.val_freq = cfg['leafmachine']['landmark_train']['model_options']['validation_frequency']
        
        if cfg['leafmachine']['landmark_train']['model_options']['max_mask_pts'] is not None:
            self.max_mask_pts = cfg['leafmachine']['landmark_train']['model_options']['max_mask_pts']
        
        if cfg['leafmachine']['landmark_train']['model_options']['weight_to_get_correct_number_of_points'] is not None:
            self.lambdaa = cfg['leafmachine']['landmark_train']['model_options']['weight_to_get_correct_number_of_points']




def train_landmarks(opts):
    # Setup W & B
    wandb.login(key=opts.w_and_b_key)
    wandb.init(project=opts.w_and_b_project,name=opts.name, entity=opts.entity, sync_tensorboard=False, settings=wandb.Settings(_disable_stats=True, _disable_meta=True))
    wandb.config = {
        "learning_rate": opts.lr,
        "epochs": opts.epochs,
        "batch_size": opts.batch_size
    }

    # Parse command line arguments
    # args = argparser.parse_command_args('training')

    writer = SummaryWriter()

    # Tensor type to use, select CUDA or not
    torch.set_default_dtype(torch.float32)
    device_cpu = torch.device('cpu')
    device = torch.device(f'cuda:{opts.n_gpus}') if opts.cuda else device_cpu

    # Create directory for checkpoint to be saved
    # if opts.save:
    #     os.makedirs(os.path.split(opts.save)[0], exist_ok=True)

    # Set seeds
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    # Visdom setup
    log = logger.Logger(server=opts.visdom_server,
                        port=opts.visdom_port,
                        env_name=opts.visdom_env)


    # Create data loaders (return data in batches)
    # trainset_loader, valset_loader, train_dir_temp, val_dir_temp = \
    trainset_loader, valset_loader = \
        data.get_train_val_loaders(train_dir=opts.train_dir,
                                max_trainset_size=opts.max_trainset_size,
                                collate_fn=csv_collator,
                                height=opts.height,
                                width=opts.width,
                                landmark=opts.landmark,
                                opts=opts,
                                seed=opts.seed,
                                batch_size=opts.batch_size,
                                drop_last_batch=opts.drop_last_batch,
                                num_workers=opts.nThreads,
                                val_dir=opts.val_dir,
                                max_valset_size=opts.max_valset_size)

    # opts.train_dir = train_dir_temp
    # opts.val_dir = val_dir_temp
    # Model
    with tqdm(total=1, desc='Building network') as progress_bar:
        model = unet_model.UNet(3, 1,
                                height=opts.height,
                                width=opts.width,
                                known_n_points=opts.n_points,
                                device=device,
                                ultrasmall=opts.ultrasmallnet)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" with {num_params} trainable parameters. ", end='')
        progress_bar.update()

    # try:
    #     model.to(f'cuda:{opts.n_gpus}')
    # except:
    model.to(f'cuda:{opts.n_gpus}')

    # model.to(device)

    '''if opts.n_gpus > 1:
        device_ids = list(range(opts.n_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
        model.to(f'cuda:{device_ids[0]}')
    else:
        # model = nn.DataParallel(model)
        model.to(device)'''

    # Loss functions
    loss_regress = nn.SmoothL1Loss()
    loss_loc = losses.WeightedHausdorffDistance(resized_height=opts.height,
                                                resized_width=opts.width,
                                                p=opts.p,
                                                return_2_terms=True,
                                                device=device)

    # Optimization strategy
    if opts.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                            lr=opts.lr,
                            momentum=0.9)
    elif opts.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                            lr=opts.lr,
                            amsgrad=True)

    start_epoch = 0
    lowest_mahd = np.infty

    # Restore saved checkpoint (model weights + epoch + optimizer state)
    if opts.resume:
        with tqdm(total=1, desc='Loading checkpoint') as progress_bar:
            if os.path.isfile(opts.resume):
                checkpoint = torch.load(opts.resume, map_location=torch.device(f'cuda:{opts.n_gpus}'))
                start_epoch = checkpoint['epoch']
                try:
                    lowest_mahd = checkpoint['mahd']
                except KeyError:
                    lowest_mahd = np.infty
                    print('W: Loaded checkpoint has not been validated. ', end='')
                if opts.cuda:
                    model = nn.DataParallel(model, device_ids=[opts.n_gpus])
                model.load_state_dict(checkpoint['model'])
                if not opts.replace_optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"\n\__ loaded checkpoint '{opts.resume}'"
                    f"(now on epoch {checkpoint['epoch']})")
                progress_bar.update()
            else:
                print(f"\n\__ E: no checkpoint found at '{opts.resume}'")
                exit(-1)

    running_avg = utils.RunningAverage(len(trainset_loader))

    normalzr = utils.Normalizer(opts.height, opts.width)

    # Time at the last evaluation
    tic_train = -np.infty
    tic_val = -np.infty

    epoch = start_epoch
    it_num = 0
    while epoch < opts.epochs:

        loss_avg_this_epoch = 0
        iter_train = tqdm(trainset_loader,
                        desc=f'Epoch {epoch} ({len(trainset_loader.dataset)} images)')

        # === TRAIN ===

        # Set the module in training mode
        model.train()

        for batch_idx, (imgs, dictionaries) in enumerate(iter_train):
            target_locations = [dictt['locations'] for dictt in dictionaries]
            target_counts = [dictt['count'] for dictt in dictionaries]
            target_orig_heights = [dictt['orig_height'] for dictt in dictionaries]
            target_orig_widths = [dictt['orig_width'] for dictt in dictionaries]

            # Pull info from this batch and move to device
            # try:
            #     imgs = imgs.to(f'cuda:{opts.n_gpus}')
            # except:
            # imgs = imgs.to(device)
            imgs = imgs.to(f'cuda:{opts.n_gpus}')

            # if opts.n_gpus > 1:
            #     device_ids = list(range(opts.n_gpus))
            #     imgs = nn.DataParallel(imgs, device_ids=device_ids)
            #     imgs = imgs.to(f'cuda:{device_ids[0]}')
            #     # imgs = imgs.cuda()
            # else:
            #     imgs = imgs.to(device)

            target_locations = [target_location.to(device) for target_location in target_locations]
            target_counts = [target_count.to(device) for target_count in target_counts]
            target_orig_heights = [target_orig_height.to(device) for target_orig_height in target_orig_heights]
            target_orig_widths = [target_orig_width.to(device) for target_orig_width in target_orig_widths]

            
            
            '''# Pull info from this batch and move to device
            if opts.n_gpus > 1:
                device_ids = list(range(opts.n_gpus))
                imgs = nn.DataParallel(imgs, device_ids=device_ids)
                imgs.to(f'cuda:{device_ids[0]}')
            else:
                imgs = imgs.to(device)'''
            


            '''# Pull info from this batch and move to device
            if opts.n_gpus > 1:
                device_ids = list(range(opts.n_gpus))
                imgs = nn.DataParallel(imgs, device_ids=device_ids)
                imgs.to(f'cuda:{device_ids[0]}')
            else:
                imgs = imgs.to(device)'''

            # imgs = imgs.to(device)
            '''target_locations = [dictt['locations'].to(device)
                                for dictt in dictionaries]
            target_counts = [dictt['count'].to(device)
                            for dictt in dictionaries]
            target_orig_heights = [dictt['orig_height'].to(device)
                                for dictt in dictionaries]
            target_orig_widths = [dictt['orig_width'].to(device)
                                for dictt in dictionaries]'''

            # Lists -> Tensor batches
            target_counts = torch.stack(target_counts)
            target_orig_heights = torch.stack(target_orig_heights)
            target_orig_widths = torch.stack(target_orig_widths)
            target_orig_sizes = torch.stack((target_orig_heights,
                                            target_orig_widths)).transpose(0, 1)

            # One training step
            optimizer.zero_grad()
            est_maps, est_counts = model.forward(imgs)
            term1, term2 = loss_loc.forward(est_maps,
                                            target_locations,
                                            target_orig_sizes)
            est_counts = est_counts.view(-1)
            target_counts = target_counts.view(-1)
            term3 = loss_regress.forward(est_counts, target_counts)
            term3 *= opts.lambdaa
            loss = term1 + term2 + term3
            loss.backward()
            optimizer.step()

            # Update progress bar
            running_avg.put(loss.item())
            iter_train.set_postfix(running_avg=f'{round(running_avg.avg/3, 1)}')

            # Log training error
            if time.time() > tic_train + opts.log_interval:
                tic_train = time.time()

                # Log training losses
                writer.add_scalar('Loss/Term1', term1, epoch)
                writer.add_scalar('Loss/Term2', term2, epoch)
                writer.add_scalar('Loss/Term3', term3 * opts.lambdaa, epoch)
                writer.add_scalar('Loss/Sum', loss / 3, epoch)
                writer.add_scalar('Loss/RunningAvg', running_avg.avg / 3, epoch)

                # Resize images to original size
                orig_shape = target_orig_sizes[0].data.to(device_cpu).numpy().tolist()
                orig_img_origsize = ((skimage.transform.resize(imgs[0].data.squeeze().to(device_cpu).numpy().transpose((1, 2, 0)),
                                                                output_shape=orig_shape,
                                                                mode='constant') + 1) / 2.0 * 255.0).\
                                    astype(np.float32).transpose((2, 0, 1))
                est_map_origsize = skimage.transform.resize(est_maps[0].data.unsqueeze(0).to(device_cpu).numpy().transpose((1, 2, 0)),
                                                            output_shape=orig_shape,
                                                            mode='constant').\
                                    astype(np.float32).transpose((2, 0, 1)).squeeze(0)

                # Overlay output on heatmap
                orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                    map=est_map_origsize).\
                                    astype(np.float32)

                # Send heatmap with circles at the labeled points to TensorBoard
                target_locs_np = target_locations[0].\
                                to(device_cpu).numpy().reshape(-1, 2)
                target_orig_size_np = target_orig_sizes[0].\
                                    to(device_cpu).numpy().reshape(2)
                target_locs_wrt_orig = normalzr.unnormalize(target_locs_np,
                                                            orig_img_size=target_orig_size_np)
                img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                points=target_locs_wrt_orig,
                                                color='white')
                writer.add_image('Training/Image with heatmap and labeled points', img_with_x, epoch)

                # Log training metrics to Weights & Biases
                wandb.log({
                    "train/term1": term1,
                    "train/term2": term2,
                    "train/term3": term3,
                    "train/loss": loss / 3,
                    "train/loss_running_avg": running_avg.avg / 3
                })

                '''# Log training losses
                log.train_losses(terms=[term1, term2, term3, loss / 3, running_avg.avg / 3],
                                iteration_number=epoch +
                                batch_idx/len(trainset_loader),
                                terms_legends=['Term1',
                                                'Term2',
                                                'Term3*%s' % opts.lambdaa,
                                                'Sum/3',
                                                'Sum/3 runn avg'])

                # Resize images to original size
                orig_shape = target_orig_sizes[0].data.to(device_cpu).numpy().tolist()
                orig_img_origsize = ((skimage.transform.resize(imgs[0].data.squeeze().to(device_cpu).numpy().transpose((1, 2, 0)),
                                                            output_shape=orig_shape,
                                                            mode='constant') + 1) / 2.0 * 255.0).\
                    astype(np.float32).transpose((2, 0, 1))
                est_map_origsize = skimage.transform.resize(est_maps[0].data.unsqueeze(0).to(device_cpu).numpy().transpose((1, 2, 0)),
                                                            output_shape=orig_shape,
                                                            mode='constant').\
                    astype(np.float32).transpose((2, 0, 1)).squeeze(0)

                # Overlay output on heatmap
                orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                    map=est_map_origsize).\
                    astype(np.float32)

                # Send heatmap with circles at the labeled points to Visdom
                target_locs_np = target_locations[0].\
                    to(device_cpu).numpy().reshape(-1, 2)
                target_orig_size_np = target_orig_sizes[0].\
                    to(device_cpu).numpy().reshape(2)
                target_locs_wrt_orig = normalzr.unnormalize(target_locs_np,
                                                            orig_img_size=target_orig_size_np)
                img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                points=target_locs_wrt_orig,
                                                color='white')
                log.image(imgs=[img_with_x],
                        titles=['(Training) Image w/ output heatmap and labeled points'],
                        window_ids=[1])'''

                # # Read image with GT dots from disk
                # gt_img_numpy = skimage.io.imread(
                #     os.path.join('/home/jprat/projects/phenosorg/data/plant_counts_dots/20160613_F54_training_256x256_white_bigdots',
                #                  dictionary['filename'][0]))
                # # dots_img_tensor = torch.from_numpy(gt_img_numpy).permute(
                # # 2, 0, 1)[0, :, :].type(torch.FloatTensor) / 255
                # # Send GT image to Visdom
                # viz.image(np.moveaxis(gt_img_numpy, 2, 0),
                #           opts=dict(title='(Training) Ground Truth'),
                #           win=3)

            it_num += 1

        # Never do validation?
        if (not opts.val_dir) or (not valset_loader) or (len(valset_loader) == 0) or (opts.val_freq == 0):

            # Time to save checkpoint?
            ckpt_path = os.path.join(opts.save_dir, ''.join([opts.save_name, '_epoch-', str(epoch+1), '.ckpt']))
            pt_path = os.path.join(opts.save_dir, ''.join([opts.save_name, '_epoch-', str(epoch+1), '.pt']))
            if opts.save and (epoch + 1) % opts.val_freq == 0:
                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'n_points': opts.n_points,
                            }, ckpt_path)
                # model_scripted = torch.jit.script(model) # Export to TorchScript
                # model_scripted.save(pt_path)
            epoch += 1
            continue

        # Time to do validation?
        if (epoch + 1) % opts.val_freq != 0:
            epoch += 1
            continue

        # === VALIDATION ===

        # Set the module in evaluation mode
        model.eval()

        judge = Judge(r=opts.radius)
        sum_term1 = 0
        sum_term2 = 0
        sum_term3 = 0
        sum_loss = 0
        iter_val = tqdm(valset_loader,
                        desc=f'Validating Epoch {epoch} ({len(valset_loader.dataset)} images)')
        for batch_idx, (imgs, dictionaries) in enumerate(iter_val):

            # Pull info from this batch and move to device
            imgs = imgs.to(device)
            target_locations = [dictt['locations'].to(device)
                                for dictt in dictionaries]
            target_counts = [dictt['count'].to(device)
                            for dictt in dictionaries]
            target_orig_heights = [dictt['orig_height'].to(device)
                                for dictt in dictionaries]
            target_orig_widths = [dictt['orig_width'].to(device)
                                for dictt in dictionaries]

            with torch.no_grad():
                target_counts = torch.stack(target_counts)
                target_orig_heights = torch.stack(target_orig_heights)
                target_orig_widths = torch.stack(target_orig_widths)
                target_orig_sizes = torch.stack((target_orig_heights,
                                                target_orig_widths)).transpose(0, 1)
            orig_shape = (dictionaries[0]['orig_height'].item(),
                        dictionaries[0]['orig_width'].item())

            # Tensor -> float & numpy
            target_count_int = int(round(target_counts.item()))
            target_locations_np = \
                target_locations[0].to(device_cpu).numpy().reshape(-1, 2)
            target_orig_size_np = \
                target_orig_sizes[0].to(device_cpu).numpy().reshape(2)

            normalzr = utils.Normalizer(opts.height, opts.width)

            if target_count_int == 0:
                continue

            # Feed-forward
            with torch.no_grad():
                est_maps, est_counts = model.forward(imgs)

            # Tensor -> int
            est_count_int = int(round(est_counts.item()))

            # The 3 terms
            with torch.no_grad():
                est_counts = est_counts.view(-1)
                target_counts = target_counts.view(-1)
                term1, term2 = loss_loc.forward(est_maps,
                                                target_locations,
                                                target_orig_sizes)
                term3 = loss_regress.forward(est_counts, target_counts)
                term3 *= opts.lambdaa
            sum_term1 += term1.item()
            sum_term2 += term2.item()
            sum_term3 += term3.item()
            sum_loss += term1 + term2 + term3

            # Update progress bar
            loss_avg_this_epoch = sum_loss.item() / (batch_idx + 1)
            iter_val.set_postfix(
                avg_val_loss_this_epoch=f'{loss_avg_this_epoch:.1f}-----')

            # The estimated map must be thresholed to obtain estimated points
            # BMM thresholding
            est_map_numpy = est_maps[0, :, :].to(device_cpu).numpy()
            est_map_numpy_origsize = skimage.transform.resize(est_map_numpy,
                                                            output_shape=orig_shape,
                                                            mode='constant')
            mask, _ = utils.threshold(est_map_numpy_origsize, tau=-1)
            # Obtain centroids of the mask
            centroids_wrt_orig = utils.cluster(mask, est_count_int,
                                            max_mask_pts=opts.max_mask_pts)

            # Validation metrics
            target_locations_wrt_orig = normalzr.unnormalize(target_locations_np,
                                                            orig_img_size=target_orig_size_np)
            judge.feed_points(centroids_wrt_orig, target_locations_wrt_orig,
                            max_ahd=loss_loc.max_dist)
            judge.feed_count(est_count_int, target_count_int)

            if time.time() > tic_val + opts.log_interval:
                tic_val = time.time()

                iteration_number = epoch + batch_idx / len(trainset_loader)
                writer.add_scalar('Training/Term1', term1, iteration_number)
                writer.add_scalar('Training/Term2', term2, iteration_number)
                writer.add_scalar('Training/Term3*{}'.format(opts.lambdaa), term3 * opts.lambdaa, iteration_number)
                writer.add_scalar('Training/Sum/3', loss / 3, iteration_number)
                writer.add_scalar('Training/Sum/3 running avg', running_avg.avg / 3, iteration_number)

                # Resize to original size
                orig_img_origsize = ((skimage.transform.resize(imgs[0].to(device_cpu).squeeze().numpy().transpose((1, 2, 0)),
                                                            output_shape=target_orig_size_np.tolist(),
                                                            mode='constant') + 1) / 2.0 * 255.0).\
                    astype(np.float32).transpose((2, 0, 1))
                est_map_origsize = skimage.transform.resize(est_maps[0].to(device_cpu).unsqueeze(0).numpy().transpose((1, 2, 0)),
                                                            output_shape=orig_shape,
                                                            mode='constant').\
                    astype(np.float32).transpose((2, 0, 1)).squeeze(0)

                # Overlay output on heatmap
                orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                    map=est_map_origsize).\
                    astype(np.float32)

                if not opts.paint:
                    # Send input and output heatmap (first one in the batch)
                    writer.add_image('Validation/Image w/ output heatmap', orig_img_w_heatmap_origsize, dataformats='CHW', global_step=epoch)
                else:
                    # Send heatmap with a cross at the estimated centroids to TensorBoard
                    img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                    points=centroids_wrt_orig,
                                                    color='green',
                                                    crosshair=True)
                    writer.add_image('Validation/Image w/ output heatmap and point estimations', img_with_x, dataformats='CHW', global_step=epoch)


                '''# Resize to original size
                orig_img_origsize = ((skimage.transform.resize(imgs[0].to(device_cpu).squeeze().numpy().transpose((1, 2, 0)),
                                                            output_shape=target_orig_size_np.tolist(),
                                                            mode='constant') + 1) / 2.0 * 255.0).\
                    astype(np.float32).transpose((2, 0, 1))
                est_map_origsize = skimage.transform.resize(est_maps[0].to(device_cpu).unsqueeze(0).numpy().transpose((1, 2, 0)),
                                                            output_shape=orig_shape,
                                                            mode='constant').\
                    astype(np.float32).transpose((2, 0, 1)).squeeze(0)

                # Overlay output on heatmap
                orig_img_w_heatmap_origsize = utils.overlay_heatmap(img=orig_img_origsize,
                                                                    map=est_map_origsize).\
                    astype(np.float32)

                # # Read image with GT dots from disk
                # gt_img_numpy = skimage.io.imread(
                #     os.path.join('/home/jprat/projects/phenosorg/data/plant_counts_dots/20160613_F54_validation_256x256_white_bigdots',
                #                  dictionary['filename'][0]))
                # # dots_img_tensor = torch.from_numpy(gt_img_numpy).permute(
                #     # 2, 0, 1)[0, :, :].type(torch.FloatTensor) / 255
                # # Send GT image to Visdom
                # viz.image(np.moveaxis(gt_img_numpy, 2, 0),
                #           opts=dict(title='(Validation) Ground Truth'),
                #           win=7)
                if not opts.paint:
                    # Send input and output heatmap (first one in the batch)
                    log.image(imgs=[orig_img_w_heatmap_origsize],
                            titles=['(Validation) Image w/ output heatmap'],
                            window_ids=[5])
                else:
                    # Send heatmap with a cross at the estimated centroids to Visdom
                    img_with_x = utils.paint_circles(img=orig_img_w_heatmap_origsize,
                                                    points=centroids_wrt_orig,
                                                    color='red',
                                                    crosshair=True )
                    log.image(imgs=[img_with_x],
                            titles=['(Validation) Image w/ output heatmap '
                                    'and point estimations'],
                            window_ids=[8])'''

        avg_term1_val = sum_term1 / len(valset_loader)
        avg_term2_val = sum_term2 / len(valset_loader)
        avg_term3_val = sum_term3 / len(valset_loader)
        avg_loss_val = sum_loss / len(valset_loader)

        # Log validation metrics
        log.val_losses(terms=(avg_term1_val,
                            avg_term2_val,
                            avg_term3_val,
                            avg_loss_val / 3,
                            judge.mahd,
                            judge.mae,
                            judge.rmse,
                            judge.mape,
                            judge.coeff_of_determination,
                            judge.pearson_corr \
                                if not np.isnan(judge.pearson_corr) else 1,
                            judge.precision,
                            judge.recall),
                    iteration_number=epoch,
                    terms_legends=['Term 1',
                                    'Term 2',
                                    'Term3*%s' % opts.lambdaa,
                                    'Sum/3',
                                    'AHD',
                                    'MAE',
                                    'RMSE',
                                    'MAPE (%)',
                                    'R^2',
                                    'r',
                                    f'r{opts.radius}-Precision (%)',
                                    f'r{opts.radius}-Recall (%)'])

        # Add validation metrics to Tensorboard
        writer.add_scalar('Validation/Term 1', avg_term1_val, epoch)
        writer.add_scalar('Validation/Term 2', avg_term2_val, epoch)
        writer.add_scalar('Validation/Term3*%s' % opts.lambdaa, avg_term3_val, epoch)
        writer.add_scalar('Validation/Sum/3', avg_loss_val / 3, epoch)
        writer.add_scalar('Validation/AHD', judge.mahd, epoch)
        writer.add_scalar('Validation/MAE', judge.mae, epoch)
        writer.add_scalar('Validation/RMSE', judge.rmse, epoch)
        writer.add_scalar('Validation/MAPE (%)', judge.mape, epoch)
        writer.add_scalar('Validation/R^2', judge.coeff_of_determination, epoch)
        writer.add_scalar('Validation/r', judge.pearson_corr if not np.isnan(judge.pearson_corr) else 1, epoch)
        writer.add_scalar(f'Validation/r{opts.radius}-Precision (%)', judge.precision, epoch)
        writer.add_scalar(f'Validation/r{opts.radius}-Recall (%)', judge.recall, epoch)

        # Log validation metrics to Weights & Biases
        wandb.log({
            "val/term1": avg_term1_val,
            "val/term2": avg_term2_val,
            "val/term3": avg_term3_val,
            "val/loss": avg_loss_val / 3,
            "val/mahd": judge.mahd,
            "val/mae": judge.mae,
            "val/rmse": judge.rmse,
            "val/mape": judge.mape,
            "val/coefficient_of_determination": judge.coeff_of_determination,
            "val/pearson_corr": judge.pearson_corr if not np.isnan(judge.pearson_corr) else 1,
            "val/precision": judge.precision,
            "val/recall": judge.recall,
            "epoch": epoch
        })

        # If this is the best epoch (in terms of validation error)
        if judge.mahd < lowest_mahd:
            # Keep the best model
            lowest_mahd = judge.mahd
            # TODO this = save_all_checkpoints
            # TODO add option to remove the epoch+1 for save_best_checkpoint_only
            ckpt_path = os.path.join(opts.save_dir, ''.join([opts.save_name, '_epoch-', str(epoch+1), '.ckpt']))
            pt_path = os.path.join(opts.save_dir, ''.join([opts.save_name, '_epoch-', str(epoch+1), '.pt']))
            if opts.save:
                torch.save({'epoch': epoch + 1,  # when resuming, we will start at the next epoch
                            'model': model.state_dict(),
                            'mahd': lowest_mahd,
                            'optimizer': optimizer.state_dict(),
                            'n_points': opts.n_points,
                            }, ckpt_path)
                print("Saved best checkpoint so far in %s " % ckpt_path)
                # model_scripted = torch.jit.script(model) # Export to TorchScript
                # model_scripted.save(pt_path)

        epoch += 1


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 01/01/2023 by William Weaver for LeafMachine2
"""

if __name__ == '__main__':
    # Create a TrainingOpts object from the parsed dictionary
    opts = Training_Opts()

    '''
    launch() allows multi-gpu training
    '''

    '''launch(
        train_landmarks,
        opts.n_gpus,
        opts.n_machines,
        machine_rank=0,
        dist_url="auto",
        opts=opts,
        timeout=opts.default_timeout_minutes,
    )'''
    
    train_landmarks(opts)
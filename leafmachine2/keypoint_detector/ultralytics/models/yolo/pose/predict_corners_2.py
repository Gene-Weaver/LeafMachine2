import os
import cv2
import sys
import inspect
import logging
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import gaussian_filter, laplace
import matplotlib.pyplot as plt
import torch
from ultralytics.cfg import DEFAULT_CFG, get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SingleImageDataset:
    """Custom dataset to handle a single numpy array as an image."""
    def __init__(self, image, filename, imgsz, transform=None):
        self.image = image
        self.filename = filename
        self.imgsz = imgsz
        self.transform = transform

        self.paths = [filename]
        self.im0 = [self._single_check(im) for im in [image]]
        self.imgsz = imgsz
        self.mode = 'image'
        # Generate fake paths
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im       
    def __len__(self):
        return 1  # always one image in this dataset

    def __getitem__(self, idx):
        # Apply transforms if any
        img = self.image
        if self.transform:
            img = self.transform(img)
        return [self.filename], img, None, "0"  # Mimicking a typical dataset output


class PosePredictor(DetectionPredictor):
    def __init__(self, model_path, dir_keypoint_overlay, save_keypoint_overlay, device='cuda', cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides)
        self.metadata = {}
        self.args.task = 'pose'
        self.setup_model_C(model_path)
        self.dir_keypoint_overlay = dir_keypoint_overlay
        self.save_keypoint_overlay = save_keypoint_overlay

        self.mapping = {
            'top_left': 0,
            'top_right': 1,
            'bottom_left': 2,
            'bottom_right': 3,
        }
        self.color_map = {
            'top_left': (0, 255, 0),  # Bright Green
            'top_right': (255, 0, 255),  # Magenta
            'bottom_left': (0, 0, 139),  # Dark Blue
            'bottom_right': (255, 0, 0),  # Bright Red
        }

        logging.info("PosePredictor initialized")

    def setup_model_C(self, model, verbose=True):
        self.model = AutoBackend(model or self.args.model,
                                 device=select_device(self.args.device, verbose=verbose),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

        logging.info(f"Model setup completed with model: {model}")
        logging.info(f"Model details: {self.model}")

    def setup_source_LM2(self, source, filename):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None

        # Handling source as an array of images or a single image
        if isinstance(source, np.ndarray):
            # Convert array to dataset format similar to handling a directory with one image
            # Assuming source is a single image or an array of images
            self.dataset = SingleImageDataset(source, filename, self.imgsz, self.transforms)
            self.source_type = 'array'
        else:
            # Handle source as a path or stream
            self.dataset = load_inference_source(source=source,
                                                imgsz=self.imgsz,
                                                vid_stride=self.args.vid_stride,
                                                stream_buffer=self.args.stream_buffer)
            self.source_type = self.dataset.source_type

        # Check if streaming is applicable
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                len(self.dataset) > 1000 or  # large number of images
                                                any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning('Stream warning: Adjust settings or switch to file-based processing.')

        # Initialize video path and writer if needed
        self.vid_path, self.vid_writer = [None] * len(self.dataset), [None] * len(self.dataset)

    @smart_inference_mode()
    def process_images(self, source=None, filename=None, *args, **kwargs):
        self.filename = filename
        self.setup_source_LM2(source, filename)

        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.batch = None
        self.run_callbacks('on_predict_start')

        if filename is None:
            for batch in self.dataset:
                self.process_images_run(batch, filename, source)
        else:
            self.process_images_run(self.dataset, filename, source)

        logging.info(f"Processing images completed for source: {source}, filename: {filename}")

        return self.metadata

    def process_images_run(self, batch, filename, img_rgb):
        self.run_callbacks('on_predict_batch_start')

        if filename is None:
            self.batch = batch
            path, im0s, vid_cap, s = batch
            file_key = os.path.splitext(os.path.basename(path[0]))[0]
            im = self.preprocess(im0s)
        else:
            file_key = filename
            im = self.preprocess_LM2(self.dataset.im0)

        preds = self.inference(im, self.args)

        if filename is None:
            results = self.postprocess(preds, im, im0s, img_rgb)
        else:
            results = self.postprocess(preds, im, self.dataset.im0, img_rgb)

        self.metadata[file_key] = {
            'keypoints': results['keypoints'],
            'angle': results['angle'],
            'tip': results['tip'],
            'base': results['base'],
            'keypoint_measurements': results['keypoint_measurements'],
        }

        logging.info(f"Image {file_key} processed with keypoints: {results['keypoints']}")

    def postprocess(self, preds, img, orig_imgs, img_rgb):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = {
            'img_name': None, 
            'keypoints': None, 
            'angle': None,
            'img_path': None,
            'tip': None,
            'base': None,
        }

        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

            if self.filename is None:
                img_path = self.batch[0][i]
                file_key = os.path.splitext(os.path.basename(img_path))[0]
            else:
                img_path = None
                file_key = self.filename

            res = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            pred_kpts_np = res.keypoints.xy.cpu().numpy()[0]

            angle, tip, base, keypoint_measurements = self.calc_angle(pred_kpts_np, img_rgb)

            self.rotate_image(-angle, orig_img, file_key)
            self.visualize_keypoints(orig_img, pred_kpts_np, file_key)

            results.update({
                'img_name': file_key,
                'img_path': img_path,
                'keypoints': pred_kpts_np,
                'angle': angle,
                'tip': tip,
                'base': base,
                'keypoint_measurements': keypoint_measurements,
            })

        return results

if __name__ == '__main__':
    img_path = "D:/Dropbox/LM2_Env/Image_Datasets/Herbarium_Sheet_Corners/Herbarium_Sheet_Corners_split/images/test"
    model_path = "D:/Dropbox/LeafMachine2/Herbarium_Sheet_Corners/corner_detector5/weights/best.pt"
    dir_keypoint_overlay = "D:/Dropbox/LM2_Env/Image_Datasets/Herbarium_Sheet_Corners/Herbarium_Sheet_Corners_split/test_corners"

    if not os.path.exists(img_path):
        print(f"Image path {img_path} does not exist.")
        exit()

    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        exit()

    if not os.path.exists(dir_keypoint_overlay):
        print(f"Creating {dir_keypoint_overlay}")
        os.makedirs(dir_keypoint_overlay, exist_ok=True)

    # Create dictionary for overrides
    overrides = {
        'model': model_path,
        'source': img_path,
        'name': 'corner_detector5',
        'boxes': False,
        'max_det': 1,
        'save_txt': True,
    }

    # Initialize PosePredictor
    pose_predictor = PosePredictor(model_path, dir_keypoint_overlay,
                                   save_keypoint_overlay=True,
                                   device='cuda', cfg=DEFAULT_CFG, overrides=overrides, _callbacks=None)
    pose_predictor.process_images(img_path)

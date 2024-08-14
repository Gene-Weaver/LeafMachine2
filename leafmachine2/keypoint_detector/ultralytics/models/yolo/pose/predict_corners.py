import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


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
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)
        return im       

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = self.image
        if self.transform:
            img = self.transform(img)
        return [self.filename], img, None, "0"


class PosePredictor(DetectionPredictor):
    def __init__(self, model_path, dir_keypoint_overlay, save_keypoint_overlay, device='cuda', cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides)
        self.metadata = {}
        self.args.task = 'pose'

        # Setup model
        self.setup_model(model_path)

        self.dir_keypoint_overlay = dir_keypoint_overlay
        self.save_keypoint_overlay = save_keypoint_overlay
        self.done_warmup = False

        self.mapping = {
            'top_left': 0,
            'top_right': 1,
            'bottom_left': 3,
            'bottom_right': 2,
        }
        self.color_map = {
            'top_left': (0, 255, 0),  # Bright Green
            'top_right': (255, 0, 255),  # Magenta
            'bottom_left': (0, 0, 139),  # Dark Blue
            'bottom_right': (255, 0, 0),  # Bright Red
        }

    def setup_model(self, model_path):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(model_path,
                                 device=select_device(self.args.device, verbose=True),
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=True)
        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def get_color_for_keypoint(self, key):
        for pattern, color in self.color_map.items():
            if pattern.endswith('*') and key.startswith(pattern.split('_*')[0]):
                return color
            elif key == pattern:
                return color
        return (255, 255, 255)  # Default to white if no match

    def visualize_keypoints(self, img, keypoints, img_path):
        pt_size = 30
        for key, idx in self.mapping.items():
            if idx < len(keypoints):
                point = keypoints[idx]
                color = self.get_color_for_keypoint(key)
                cv2.circle(img, (int(point[0]), int(point[1])), pt_size, color, -1)

        visual_path = os.path.join(self.dir_keypoint_overlay, os.path.basename(img_path))
        if self.save_keypoint_overlay:
            cv2.imwrite(visual_path, img)

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            
            img_path = self.batch[0][i]

            res = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            pred_kpts_np = res.keypoints.xy.cpu().numpy()[0]

            results.append(res)
            self.visualize_keypoints(orig_img, pred_kpts_np, img_path)
        return results

    def process_images_run(self, batch, filename, img_rgb):
        self.run_callbacks('on_predict_batch_start')

        if filename is None:
            self.batch = batch
            path, im0s, s = batch
            file_key = os.path.splitext(os.path.basename(path[0]))[0]
            im = self.preprocess(im0s)
        else:
            file_key = filename
            im = self.preprocess_LM2(self.dataset.im0)

        preds = self.model(im)  # Use the correct inference method

        if filename is None:
            results = self.postprocess(preds, im, im0s)
        else:
            results = self.postprocess(preds, im, self.dataset.im0)

        if results:
            self.metadata[self.filename] = {
                'keypoints': results[0].keypoints.xy.cpu().numpy()
            }

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

        return self.metadata

    def setup_source_LM2(self, source, filename):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None

        if isinstance(source, np.ndarray):
            self.dataset = SingleImageDataset(source, filename, self.imgsz, self.transforms)
            self.source_type = 'array'
        else:
            self.dataset = load_inference_source(source=source,
                                                 vid_stride=self.args.vid_stride)
            self.source_type = self.dataset.source_type

        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or
                                                  len(self.dataset) > 1000 or
                                                  any(getattr(self.dataset, 'video_flag', [False]))):
            LOGGER.warning('Stream warning: Adjust settings or switch to file-based processing.')

        self.vid_path, self.vid_writer = [None] * len(self.dataset), [None] * len(self.dataset)

    def preprocess_LM2(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        if not_tensor:
            img /= 255
        return img

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

    overrides = {
        'model': model_path,
        'source': img_path,
        'name': 'corner_detector5',
        'show_boxes': False,
        'max_det': 1,
        'visualize': False,
        'save_txt': True,
    }
    corner_predictor = PosePredictor(model_path, dir_keypoint_overlay, save_keypoint_overlay=True, 
                                     device='cuda', cfg=DEFAULT_CFG, overrides=overrides, _callbacks=None)
    results = corner_predictor.process_images(img_path)

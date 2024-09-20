# Ultralytics YOLO 🚀, AGPL-3.0 license
import os, cv2, sys, inspect, logging
import numpy as np
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw
from torchvision import transforms
import platform
from pathlib import Path
from scipy.ndimage import gaussian_filter, laplace
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
# from gtda.homology import VietorisRipsPersistence
# from gtda.plotting import plot_diagram

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))))
sys.path.append(currentdir)
# from detect import run
sys.path.append(parentdir)
# from machine.general_utils import Print_Verbose
# from leafmachine2.segmentation.detectron2.segment_leaves import keep_rows, get_string_indices, get_largest_polygon
# from leafmachine2.segmentation.detectron2.detector import Detector_LM2

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


class LeafWidthRefinement:
    def __init__(self, image_array):
        self.image = Image.fromarray(cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB))

    @staticmethod
    def calculate_rotation_angle(left_point, right_point):
        opposite = right_point[1] - left_point[1]  # y2 - y1
        adjacent = right_point[0] - left_point[0]  # x2 - x1
        angle = np.arctan2(opposite, adjacent) * 180 / np.pi
        return angle

    def rotate_image(self, angle):
        return self.image.rotate(-angle, expand=True)

    @staticmethod
    def rotate_point(center, point, angle):
        angle = np.radians(angle)
        ox, oy = center
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return int(qx), int(qy)

    @staticmethod
    def crop_stripe(image, center_y, height=10):
        top = max(0, int(center_y - height // 2))
        bottom = top + height
        return image.crop((0, top, image.width, bottom))
    
    def apply_gaussian_blur(self, data, sigma=12):
        return gaussian_filter(data, sigma=sigma)

    def apply_laplacian_filter(self, data):
        return laplace(data)

    def detect_zero_crossings(self, laplacian, threshold):
        signs = np.sign(laplacian)
        zero_crossings = np.where(np.diff(signs) != 0)[0]
        significant_changes = zero_crossings[np.where(np.abs(np.diff(laplacian))[zero_crossings] > threshold)]
        return significant_changes

    def process_image(self, left_point, right_point):
        show_plots = False
        
        angle = self.calculate_rotation_angle(left_point, right_point)
        rotated_image = self.rotate_image(-angle)
        
        original_width = self.image.width
        rotated_width = rotated_image.width
        width_difference = (rotated_width - original_width) / 2

        image_center = (self.image.width / 2, self.image.height / 2)
        new_left_point = self.rotate_point(image_center, left_point, angle)
        new_right_point = self.rotate_point(image_center, right_point, angle)

        # Adjust x-coordinates for width change after rotation
        new_left_point = (int(new_left_point[0] + width_difference), new_left_point[1])
        new_right_point = (int(new_right_point[0] + width_difference), new_right_point[1])

        new_y = (new_left_point[1] + new_right_point[1]) / 2
        cropped_image = self.crop_stripe(rotated_image, new_y)
        grayscale_image = cropped_image.convert('L')

        # Recalculate points relative to the cropped image
        crop_top = int(new_y - 5)
        adjusted_left_point = (new_left_point[0], new_left_point[1] - crop_top)
        adjusted_right_point = (new_right_point[0], new_right_point[1] - crop_top)

        # Plot points on the grayscale image
        draw = ImageDraw.Draw(cropped_image)
        point_radius = 5  # To ensure visibility
        draw.ellipse((adjusted_left_point[0] - point_radius, 5 - point_radius,
                      adjusted_left_point[0] + point_radius, 5 + point_radius), fill='red')
        draw.ellipse((adjusted_right_point[0] - point_radius, 5 - point_radius,
                      adjusted_right_point[0] + point_radius, 5 + point_radius), fill='red')
        

        LEFT = int(adjusted_left_point[0])
        RIGHT = int(adjusted_right_point[0])

        # Get middle
        MID = (LEFT + RIGHT) // 2

        # For grayscale_image, calc the avg column-wise value
        # column_values = np.array(grayscale_image)[:, 5]  # Assume stripe_height = 10 and we take the middle row
        column_values = np.sum(np.array(grayscale_image), axis=0)  # Sum values vertically across the entire height

        smoothed_values = self.apply_gaussian_blur(column_values)

        ## Calculate average intensity between LEFT and RIGHT
        avg_intensity = np.mean(smoothed_values[LEFT:RIGHT])
        std_deviation = np.std(smoothed_values[LEFT:RIGHT])

        # Define the thresholds for edge detection
        lower_threshold = avg_intensity - 300 #std_deviation * 10.0
        upper_threshold = avg_intensity + 300 #std_deviation * 10.0

        # Searching for true edges based on intensity deviation
        MID = (LEFT + RIGHT) // 2
        left_edge = next((i for i in range(MID, -1, -1) if smoothed_values[i] < lower_threshold or smoothed_values[i] > upper_threshold), MID)
        right_edge = next((i for i in range(MID, len(smoothed_values)) if smoothed_values[i] < lower_threshold or smoothed_values[i] > upper_threshold), MID)

        left_edge_reconstruct = self.rotate_point(image_center, (left_edge - width_difference, new_y), angle)
        right_edge_reconstruct = self.rotate_point(image_center, (right_edge - width_difference, new_y), angle)

        if show_plots:
            # # Determine the momentum at the initial edges
            # def get_momentum(index, direction):
            #     # Return the momentum at a given index with a direction (-1 for left, 1 for right)
            #     if 0 < index < len(smoothed_values) - 1:
            #         return np.sign(smoothed_values[index + direction] - smoothed_values[index])
            #     return 0

            # # Find the true edge by following the initial momentum until it changes
            # def find_true_edge(start_index, direction):
            #     current_momentum = get_momentum(start_index, direction)
            #     for i in range(start_index + direction, len(smoothed_values) if direction == 1 else -1, direction):
            #         if get_momentum(i, direction) != current_momentum or i == 0 or i == len(smoothed_values) - 1:
            #             return i - direction
            #     return start_index  # return start index if no change in momentum

            # # Adjusting the left and right edges to find the true edge
            # true_left_edge = find_true_edge(left_edge, -1)  # Continue to the left
            # true_right_edge = find_true_edge(right_edge, 1)  # Continue to the right


            # Draw the new edges
            draw = ImageDraw.Draw(cropped_image)
            draw.line([(left_edge, 0), (left_edge, cropped_image.height)], fill='cyan', width=3)
            draw.line([(right_edge, 0), (right_edge, cropped_image.height)], fill='green', width=3)

            # draw.line([(true_left_edge, 0), (left_edge, cropped_image.height)], fill='orange', width=3)
            # draw.line([(true_right_edge, 0), (right_edge, cropped_image.height)], fill='red', width=3)


            grayscale_image = np.array(grayscale_image)
            width_stripe = np.array(cropped_image)
            rotated_image = np.array(rotated_image)
            
            width_stripe = Image.fromarray(width_stripe)
            width_stripe.save('width_stripe.png')
            rotated_image = Image.fromarray(rotated_image)
            rotated_image.save('width_rotated.png')
            grayscale_image = Image.fromarray(grayscale_image)
            grayscale_image.save('width_grayscale.png')



            # Plotting the intensity profile and thresholds
            plt.figure(figsize=(10, 5))
            plt.plot(smoothed_values, label='Column Intensities')
            plt.axhline(y=avg_intensity, color='green', linestyle='--', label='Average Intensity')
            plt.axhline(y=lower_threshold, color='blue', linestyle='--', label='Lower Threshold')
            plt.axhline(y=upper_threshold, color='orange', linestyle='--', label='Upper Threshold')
            plt.axvline(x=left_edge, color='cyan', label='Left Edge', linestyle='-', linewidth=2)
            plt.axvline(x=right_edge, color='green', label='Right Edge', linestyle='-', linewidth=2)
            plt.title('Intensity Profile and Detected Edges')
            plt.xlabel('X Coordinate')
            plt.ylabel('Summed Intensity')
            plt.legend()
            plt.grid(True)
            plt.savefig('width_plot.png', bbox_inches='tight', pad_inches=0)


            

            # Draw these points on the original image for verification
            original_image = self.image  
            draw = ImageDraw.Draw(original_image)
            point_radius = 5
            draw.ellipse([left_edge_reconstruct[0] - point_radius, left_edge_reconstruct[1] - point_radius,
                        left_edge_reconstruct[0] + point_radius, left_edge_reconstruct[1] + point_radius], fill='cyan')
            draw.ellipse([right_edge_reconstruct[0] - point_radius, right_edge_reconstruct[1] - point_radius,
                        right_edge_reconstruct[0] + point_radius, right_edge_reconstruct[1] + point_radius], fill='red')

            original_image.save('width_reconstruct.png')


            # # Step 1: Smooth the intensity profile using Gaussian blur
            # smoothed_values = self.apply_gaussian_blur(column_values)

            # # Step 2: Apply Laplacian filter to detect second derivatives
            # laplacian_values = self.apply_laplacian_filter(smoothed_values)

            # threshold = np.std(laplacian_values) * 4.0

            # # Step 3: Detect zero crossings in the Laplacian output
            # edges = self.detect_zero_crossings(laplacian_values, threshold)

            # # Plotting results
            # plt.figure(figsize=(12, 6))
            # plt.plot(column_values, label='Original Intensities')
            # plt.plot(smoothed_values, label='Smoothed Intensities', linestyle='--')
            # plt.scatter(edges, smoothed_values[edges], color='red', label='Detected Edges', zorder=5)
            # plt.title('Edge Detection Using Gaussian Blur and Laplacian Filter')
            # plt.xlabel('X Coordinate')
            # plt.ylabel('Intensity')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig('cropped_leaf_image_PLOT_Blur.png', bbox_inches='tight', pad_inches=0)
        return np.array(grayscale_image), np.array(rotated_image), np.array(cropped_image)




class PosePredictor(DetectionPredictor):
    def __init__(self, model_path, dir_oriented_images, dir_keypoint_overlay, save_oriented_images, save_keypoint_overlay, device='cuda', cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides)
        self.metadata = {}
        self.args.task = 'pose'

        # Setup model
        self.setup_model(model_path)

        self.dir_oriented_images = dir_oriented_images
        self.dir_keypoint_overlay = dir_keypoint_overlay
        self.save_oriented_images = save_oriented_images
        self.save_keypoint_overlay = save_keypoint_overlay

        self.mapping = {
                'lamina_tip': 0,
                'apex_left': 1,
                'apex_center': 2,
                'apex_right': 3,
                'midvein_0': 4,
                'midvein_1': 5,
                'midvein_2': 6,
                'midvein_3': 7,
                'midvein_4': 8,
                'midvein_5': 9,
                'midvein_6': 10,
                'midvein_7': 11,
                'midvein_8': 12,
                'midvein_9': 13,
                'midvein_10': 14,
                'midvein_11': 15,
                'midvein_12': 16,
                'midvein_13': 17,
                'midvein_14': 18,
                'base_left': 19,
                'base_center': 20,
                'base_right': 21,
                'lamina_base': 22,
                'petiole_0': 23,
                'petiole_1': 24,
                'petiole_2': 25,
                'petiole_3': 26,
                'petiole_4': 27,
                'petiole_tip': 28,
                'width_left': 29,
                'width_right': 30,
            }
        self.color_map = {
            'lamina_tip': ((0, 255, 0), 10, 1),  # Bright Green, larger size 
            'apex_left': ((255, 141, 160), 5, 0),  # Light Pink 
            'apex_center': ((255, 0, 255), 5, 0),  # Magenta 
            'apex_right': ((255, 73, 103), 5, 0),  # Light Purple 
            'midvein_*': ((0, 255, 0), 5, 2),  # Medium Gray 
            'base_left': ((50, 200, 200), 5, 0),  # Light Blue 
            'base_center': ((0, 0, 255), 5, 0),  # Blue 
            'base_right': ((0, 100, 255), 5, 0),  # Dark Blue 
            'lamina_base': ((255, 0, 0), 10, 1),  # Bright Red, larger size 
            'petiole_*': ((0, 255, 255), 5, 2),  # Light Gray 
            'petiole_tip': ((0, 165, 255), 10, 1),  # Orange, larger size 
            'width_left': ((0, 0, 0), 5, 0),  # Black 
            'width_right': ((0, 0, 0), 5, 0)  # Black 
        }

    
    # def process_images(self, directory_path):
    #     results = []
    #     for filename in os.listdir(directory_path):
    #         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             image_path = os.path.join(directory_path, filename)
    #             result = self.infer_image(image_path)
    #             results.append((image_path, result))
    #     return results
    
    def setup_source_LM2(self, source, filename):
        """Sets up source and inference mode."""
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


    def preprocess_LM2(self, im):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img


    def process_images_run(self, batch, filename, img_rgb):
        self.metadata = {}
        
        self.run_callbacks('on_predict_batch_start')

        # Preprocess
        # with profilers[0]:
        if filename is None:
            self.batch = batch
            path, im0s, vid_cap, s = batch
            file_key = os.path.splitext(os.path.basename(path[0]))[0]
            im = self.preprocess(im0s) # for a dir
        else:
            file_key = filename
            im = self.preprocess_LM2(self.dataset.im0) # for a single array image


        # Inference
        # with profilers[1]:
        preds = self.inference(im)#, self.args)

        # Postprocess
        # with profilers[2]:
        if filename is None:
            results = self.postprocess(preds, im, im0s, img_rgb)
        else:
            results = self.postprocess(preds, im, self.dataset.im0, img_rgb)

        # Update metadata
        self.metadata[file_key] = {
            # 'img_path': results['img_path'],
            'keypoints': results['keypoints'],
            'angle': results['angle'],
            'tip': results['tip'],
            'base': results['base'],
            'keypoint_measurements': results['keypoint_measurements'],
        }
        # self.run_callbacks('on_predict_postprocess_end')


    @smart_inference_mode()
    def process_images(self, source=None, filename=None, *args, **kwargs):
        self.filename = filename
        # Setup source every time predict is called
        self.setup_source_LM2(source, filename)# if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        # if self.args.save or self.args.save_txt:
        #     (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        # self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.batch = None
        self.run_callbacks('on_predict_start')

        if filename is None:
            for batch in self.dataset:
                self.process_images_run(batch, filename, source)
        else:
            self.process_images_run(self.dataset, filename, source)
                

        return self.metadata

        #     # Visualize, save, write results
        #     n = len(im0s)
        #     for i in range(n):
        #         self.seen += 1
        #         self.results[i].speed = {
        #             'preprocess': profilers[0].dt * 1E3 / n,
        #             'inference': profilers[1].dt * 1E3 / n,
        #             'postprocess': profilers[2].dt * 1E3 / n}
        #         p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
        #         p = Path(p)

        #         if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
        #             s += self.write_results(i, self.results, (p, im, im0))
        #         if self.args.save or self.args.save_txt:
        #             self.results[i].save_dir = self.save_dir.__str__()
        #         if self.args.show and self.plotted_img is not None:
        #             self.show(p)
        #         if self.args.save and self.plotted_img is not None:
        #             self.save_preds(vid_cap, i, str(self.save_dir / p.name))

        #     self.run_callbacks('on_predict_batch_end')
        #     yield from self.results

        #     # Print time (inference-only)
        #     if self.args.verbose:
        #         LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # # Release assets
        # if isinstance(self.vid_writer[-1], cv2.VideoWriter):
        #     self.vid_writer[-1].release()  # release final video writer

        # # Print results
        # if self.args.verbose and self.seen:
        #     t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
        #     LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
        #                 f'{(1, 3, *im.shape[2:])}' % t)
        # if self.args.save or self.args.save_txt or self.args.save_crop:
        #     nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
        #     s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        # self.run_callbacks('on_predict_end')

    def get_color_for_keypoint(self, key):
        for pattern, (color, size, pt_type) in self.color_map.items():
            if pattern.endswith('*') and key.startswith(pattern[:-1]):
                return color, size, pt_type
            elif key == pattern:
                return color, size, pt_type
        return (255, 255, 255), 5, 0
            
    def calc_angle(self, keypoints, img_rgb):
        def calculate_distance(point1, point2):
            return np.linalg.norm(np.array(point1) - np.array(point2))
        
        def is_valid_point(point):
            """Check if a point is valid (not hidden)."""
            return isinstance(point, (list, np.ndarray)) and not np.all(np.isclose(point, [0.0, 0.0]))

        def calculate_valid_distance(point1, point2):
            if is_valid_point(point1) and is_valid_point(point2):
                return calculate_distance(point1, point2)
            else:
                return None

        def find_lowest_highest_points(points):
            """Find the lowest and highest valid points in a list based on their y-coordinate."""
            valid_points = [(i, p) for i, p in enumerate(points) if is_valid_point(p)]
            if not valid_points:
                return None, None  # If no valid points, return None
            # Sort by y-coordinate (index 1)
            sorted_by_y = sorted(valid_points, key=lambda x: x[1][1])
            return sorted_by_y[0], sorted_by_y[-1]  # Return (lowest, highest)
    
        do_width = False
        
        keypoint_measurements = {
            'distance_lamina': 0,
            'distance_width': 0,
            'distance_petiole': 0,
            'distance_midvein_span': 0,
            'distance_petiole_span': 0,
            'trace_midvein_distance': 0,
            'trace_petiole_distance': 0,
            'apex_angle': 0,
            'apex_is_reflex': False,
            'base_angle': 0,
            'base_is_reflex': False,
        }
        # Extract the coordinates of points 0 and 22
        lamina_tip = keypoints[0]
        apex_left = keypoints[1]
        apex_center = keypoints[2]
        apex_right = keypoints[3]
        midvein_0 = keypoints[4]
        midvein_1 = keypoints[5]
        midvein_2 = keypoints[6]
        midvein_3 = keypoints[7]
        midvein_4 = keypoints[8]
        midvein_5 = keypoints[9]
        midvein_6 = keypoints[10]
        midvein_7 = keypoints[11]
        midvein_8 = keypoints[12]
        midvein_9 = keypoints[13]
        midvein_10 = keypoints[14]
        midvein_11 = keypoints[15]
        midvein_12 = keypoints[16]
        midvein_13 = keypoints[17]
        midvein_14 = keypoints[18]
        base_left = keypoints[19]
        base_center = keypoints[20]
        base_right = keypoints[21]
        lamina_base = keypoints[22]
        petiole_0 = keypoints[23]
        petiole_1 = keypoints[24]
        petiole_2 = keypoints[25]
        petiole_3 = keypoints[26]
        petiole_4 =keypoints[27]
        petiole_tip = keypoints[28]
        width_left = keypoints[29]
        width_right = keypoints[30]
                
        midvein_points = [keypoints[i] for i in range(4, 19)]  # midvein_0 to midvein_14
        petiole_points = [keypoints[i] for i in range(23, 28)]  # petiole_0 to petiole_4

        ### Make sure that the orientation algorithm still gets points
        # Find the valid lowest and highest midvein points
        midvein_lowest, midvein_highest = find_lowest_highest_points(midvein_points)
        if midvein_lowest and midvein_highest:
            midvein_0_index, midvein_0 = midvein_lowest  # New midvein_0
            midvein_14_index, midvein_14 = midvein_highest  # New midvein_14
            # Reassign the keypoints based on the new found lowest and highest
            midvein_0 = midvein_points[midvein_0_index]
            midvein_14 = midvein_points[midvein_14_index]
        else:
            pass #raise ValueError("No valid midvein points found")

        # Find the valid lowest and highest petiole points
        petiole_lowest, petiole_highest = find_lowest_highest_points(petiole_points)
        if petiole_lowest and petiole_highest:
            petiole_0_index, petiole_0 = petiole_lowest  # New petiole_0
            petiole_4_index, petiole_4 = petiole_highest  # New petiole_4
            petiole_0 = petiole_points[petiole_0_index]
            petiole_4 = petiole_points[petiole_4_index]
        else:
            pass #raise ValueError("No valid petiole points found")

        


        # Check if lamina_tip, lamina_base, or petiole_tip are missing and reassign them
        if not is_valid_point(keypoints[0]):  # lamina_tip
            keypoints[0] = midvein_0
            lamina_tip = midvein_0
        if not is_valid_point(keypoints[22]):  # lamina_base
            keypoints[22] = midvein_14
            lamina_base = midvein_14
        if not is_valid_point(keypoints[28]):  # petiole_tip
            keypoints[28] = petiole_4
            petiole_tip = petiole_4



        # Calculate distances for specified points
        distance_lamina = calculate_valid_distance(lamina_tip, lamina_base)
        distance_width = calculate_valid_distance(width_left, width_right)
        distance_petiole = calculate_valid_distance(lamina_base, petiole_tip)
        distance_midvein_span = calculate_valid_distance(midvein_0, midvein_14)
        distance_petiole_span = calculate_valid_distance(petiole_0, petiole_4)


        # Calculate sequential distances for midvein points
        # Remove invalid points from midvein_points
        valid_midvein_points = [pt for pt in midvein_points if is_valid_point(pt)]
        midvein_distances = [calculate_distance(valid_midvein_points[i], valid_midvein_points[i + 1]) for i in range(len(valid_midvein_points) - 1)]
        trace_midvein_distance = sum(midvein_distances)

        # Calculate sequential distances for petiole points
        # Remove invalid points from petiole_points
        valid_petiole_points = [pt for pt in petiole_points if is_valid_point(pt)]
        petiole_distances = [calculate_distance(valid_petiole_points[i], valid_petiole_points[i + 1]) for i in range(len(valid_petiole_points) - 1)]
        trace_petiole_distance = sum(petiole_distances)

        def determine_angles(point1, center_point, point2, reference_point):
            # Convert points to numpy arrays
            p1 = np.array(point1)
            center = np.array(center_point)
            p2 = np.array(point2)
            ref = np.array(reference_point)

            # Create vectors from center point to point1 and point2
            vector1 = p1 - center
            vector2 = p2 - center

            # Dot product and magnitudes of vectors
            dot_prod = np.dot(vector1, vector2)
            mag1 = np.linalg.norm(vector1)
            mag2 = np.linalg.norm(vector2)

            # Safely clamp the value to the range [-1, 1] before calculating arccos
            # Check if either magnitude is zero before the division
            if mag1 == 0 or mag2 == 0:
                cos_theta = 0.0  # You can decide on the appropriate fallback value
            else:
                cos_theta = np.clip(dot_prod / (mag1 * mag2), -1.0, 1.0)
                
            # Calculate angle in radians and then convert to degrees
            angle_rad = np.arccos(cos_theta)
            angle_deg = np.degrees(angle_rad)

            # Determine if it is a reflex angle
            # Calculate vectors from the reference point to the center point
            ref_to_center = center - ref
            # Determine the direction of the vectors relative to the reference vector
            ref_dot1 = np.dot(ref_to_center, vector1)
            ref_dot2 = np.dot(ref_to_center, vector2)
            reflex_angle = (ref_dot1 > 0 and ref_dot2 > 0)

            # If it's a reflex angle, adjust by subtracting from 360 degrees
            if reflex_angle:
                angle_deg = 360 - angle_deg

            return angle_deg, reflex_angle

        apex_angle, apex_is_reflex = determine_angles(apex_left, apex_center, apex_right, midvein_7)
        base_angle, base_is_reflex = determine_angles(base_left, base_center, base_right, midvein_7)


        ###
        if do_width:
            Leaf_Width = LeafWidthRefinement(img_rgb)
            grayscale_image, width_stripe, rotated_image = Leaf_Width.process_image(width_left, width_right)


        keypoint_measurements = {
            'distance_lamina': distance_lamina,
            'distance_width': distance_width,
            'distance_petiole': distance_petiole,
            'distance_midvein_span': distance_midvein_span,
            'distance_petiole_span': distance_petiole_span,
            'trace_midvein_distance': trace_midvein_distance,
            'trace_petiole_distance': trace_petiole_distance,
            'apex_angle': apex_angle,
            'apex_is_reflex': apex_is_reflex,
            'base_angle': base_angle,
            'base_is_reflex': base_is_reflex,
        }


        # Calculate the differences in coordinates
        dx = lamina_base[0] - lamina_tip[0]  # Change in x
        dy = lamina_base[1] - lamina_tip[1]  # Change in y
        
        # Calculate the angle in radians between the line connecting p0 to p22 and the vertical axis
        # Note: np.arctan2 returns the angle in radians, and it takes (y, x) instead of (x, y)
        angle_rad = np.arctan2(dx, dy)
        
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        
        # The angle returned by arctan2 might need adjustment depending on your coordinate system's orientation
        # This assumes the origin (0,0) is at the top-left corner of the image, and y values increase downwards.
        return angle_deg, lamina_tip, lamina_base, keypoint_measurements
    
    def visualize_keypoints(self, img, keypoints, file_key):
        file_key = file_key + '.jpg'
        if self.save_keypoint_overlay:
            # Convert NumPy array (img) to a PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            for key, idx in self.mapping.items():
                point = keypoints[idx]
                color, size, pt_type = self.get_color_for_keypoint(key)
                # Draw circle in PIL: create an ellipse inside bounding box [(x1, y1), (x2, y2)]
                left_up_point = (int(point[0] - size), int(point[1] - size))
                right_down_point = (int(point[0] + size), int(point[1] + size))
                if pt_type == 0:
                    draw.ellipse([left_up_point, right_down_point], fill=color)
                elif pt_type == 1:
                    draw.ellipse([left_up_point, right_down_point], outline=color, width=3)
                elif pt_type == 2:
                    # Draw horizontal line
                    start_horiz = (int(point[0] - size), int(point[1]))
                    end_horiz = (int(point[0] + size), int(point[1]))
                    draw.line([start_horiz, end_horiz], fill=color, width=3)

                    # Draw vertical line
                    start_vert = (int(point[0]), int(point[1] - size))
                    end_vert = (int(point[0]), int(point[1] + size))
                    draw.line([start_vert, end_vert], fill=color, width=3)

            # Save visualization
            visual_path = os.path.join(self.dir_keypoint_overlay, file_key)
            pil_img.save(visual_path)


    def postprocess(self, preds, img, orig_imgs, img_rgb):

        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = {
            'img_name': None, 
            'keypoints' :None, 
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

            results['img_name'] = file_key
            results['img_path'] = img_path
            results['keypoints'] = pred_kpts_np
            results['angle'] = angle
            results['tip'] = tip
            results['base'] = base
            results['keypoint_measurements'] = keypoint_measurements

        return results
    

    def rotate_image(self, angle, orig_img, file_key):
        if self.save_oriented_images:
            file_key = file_key + '.jpg'

            # Calculate the center of the image and the image size
            image_center = tuple(np.array(orig_img.shape[1::-1]) / 2)
            height, width = orig_img.shape[:2]

            # Calculate the rotation matrix for the given angle
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

            # Calculate the sine and cosine of the rotation angle
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])

            # Calculate the new bounding dimensions of the image
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            # Adjust the rotation matrix to take into account translation
            rot_mat[0, 2] += bound_w / 2 - image_center[0]
            rot_mat[1, 2] += bound_h / 2 - image_center[1]

            # Perform the rotation, adjusting the canvas size
            rotated_img = cv2.warpAffine(orig_img, rot_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST)

            # Ensure the output directory exists
            if not os.path.exists(self.dir_oriented_images):
                os.makedirs(self.dir_oriented_images, exist_ok=True)

            # Construct the full path for saving the rotated image
            save_path = os.path.join(self.dir_oriented_images, file_key)

            # Save the rotated image
            if self.save_oriented_images:
                cv2.imwrite(save_path, rotated_img)
        
        # Display the rotated image
        # cv2.imshow('Rotated Image', rotated_img)
        # cv2.waitKey(0)  # Wait for a key press to close the image window
    
        # return rotated_img
            
    # def rotate_image(self, angle, orig_img, img_path):
    #     file_key = os.path.splitext(os.path.basename(img_path))[0]
    #     original_fname = os.path.basename(img_path)
    #     image_center = tuple(np.array(orig_img.shape[1::-1]) / 2)
    #     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #     bound_w, bound_h = orig_img.shape[1], orig_img.shape[0]
    #     rotated_img = cv2.warpAffine(orig_img, rot_mat, (bound_w, bound_h))

    #     # Ensure the output directory exists
    #     if not os.path.exists(self.dir_oriented_images):
    #         os.makedirs(self.dir_oriented_images, exist_ok=True)

    #     # Save the rotated image
    #     save_path = os.path.join(self.dir_oriented_images, original_fname)
    #     if self.save_oriented_images:
    #         cv2.imwrite(save_path, rotated_img)

    #     # Update metadata
    #     self.metadata[file_key].update({
    #         'rotated_image': save_path,
    #         'rotation_angle': angle,
    #         'rotated_keypoints': np.dot(rot_mat[:2, :2], self.metadata[file_key]['original_keypoints'].T + rot_mat[:2, 2]).T
    #     })
            
# def save_masks_color(save_individual_masks_color, use_efds_for_masks, overlay_data, cropped_overlay_size, leaf_type, output_path):
#     if len(overlay_data) > 0:
#         # unpack
#         overlay_poly, overlay_efd, overlay_rect, overlay_color = overlay_data

#         if use_efds_for_masks:
#             use_polys = overlay_efd
#         else:
#             use_polys = overlay_poly
        
#         if save_individual_masks_color:
#             # Create a black image
#             img = Image.new('RGB', (cropped_overlay_size[1], cropped_overlay_size[0]), color=(0, 0, 0))
#             draw = ImageDraw.Draw(img)

#             if use_polys != []:
#                 for i, poly in enumerate(use_polys):
#                     this_color = overlay_color[i]
#                     cls, this_color = next(iter(this_color.items()))
#                     # Set the color for the polygon based on its class    
#                     if leaf_type == 0:
#                         if 'leaf' in cls:
#                             color = [46, 255, 0]
#                         elif 'petiole' in cls:
#                             color = [0, 173, 255]
#                         elif 'hole' in cls:
#                             color = [209, 0, 255]
#                         else:
#                             color = [255, 255, 255]
#                     elif leaf_type == 1:
#                         if 'leaf' in cls:
#                             color = [0, 200, 255]
#                         elif 'petiole' in cls:
#                             color = [255, 140, 0]
#                         elif 'hole' in cls:
#                             color = [200, 0, 255]
#                         else:
#                             color = [255, 255, 255]
#                     # Draw the filled polygon on the image
#                     draw.polygon(poly, fill=tuple(color))
#             if leaf_type == 0:
#                 img.save(output_path)
#             elif leaf_type == 1:
#                 img.save(output_path)

# def save_masks_color_and_overlay(rotated_image_path, overlay_data, output_path):
#     # Load the rotated image
#     img = Image.open(rotated_image_path).convert("RGBA")

#     # Create a transparent overlay image
#     overlay = Image.new("RGBA", img.size, (255,255,255,0))
#     draw = ImageDraw.Draw(overlay)

#     # Unpack overlay data
#     overlay_poly, _, _, overlay_color = overlay_data

#     # Draw polygons with specified colors onto the overlay
#     for i, poly in enumerate(overlay_poly):
#         this_color = overlay_color[i]
#         cls, this_color = next(iter(this_color.items()))
#         color_BGR = this_color[0]  # Use the fill color
#         color = (color_BGR[2], color_BGR[1], color_BGR[0], color_BGR[3])
#         # Draw the filled polygon on the overlay
#         draw.polygon(poly, fill=color)

#     # Combine the original image with the overlay
#     combined = Image.alpha_composite(img, overlay)

#     # Convert combined image to RGB and save it
#     combined.convert("RGB").save(output_path)

# def segment_rotated_leaf(input_dir, output_dir, output_dir_overlay, logger=None):
#     if logger is None:
#         logger = logging.basicConfig(filename='example.log', level=logging.INFO, 
#                     format='%(asctime)s:%(levelname)s:%(message)s')
#     # Initialize parameters
#     THRESH = 0.7  
#     LEAF_TYPE = 0  

#     # Initialize the Detector_LM2 instance
#     DIR_MODEL = os.path.join(parentdir, 'leafmachine2', 'segmentation', 'models', 'Group3_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR')#'GroupB_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR')
#     detector = Detector_LM2(logger, DIR_MODEL, THRESH, LEAF_TYPE)

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Process each image in the input directory
#     for image_name in os.listdir(input_dir):
#         if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
#             image_path = os.path.join(input_dir, image_name)
#             output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}__seg.png")
#             output_path_overlay = os.path.join(output_dir_overlay, f"{os.path.splitext(image_name)[0]}__overlay.jpg")
            
#             # Read the image
#             img = cv2.imread(image_path)
#             height, width = img.shape[:2]
#             cropped_overlay_size = [height, width]
            
#             # Segment the image
#             out_polygons, out_bboxes, out_labels, out_color = detector.segment(img, generate_overlay=False, overlay_dpi=100, bg_color='black')
            
            
#             if len(out_polygons) > 0: # Success
#                 out_polygons, out_bboxes, out_labels, out_color = keep_rows(out_polygons, out_bboxes, out_labels, out_color, get_string_indices(out_labels))
            
#             overlay_color = []
#             overlay_poly = []
#             for i, polys in enumerate(out_polygons):
#                 color_rgb = tuple(map(lambda x: int(x*255), out_color[i]))
#                 fill_color = (color_rgb[0], color_rgb[1], color_rgb[2], 127)
#                 outline_color = (color_rgb[0], color_rgb[1], color_rgb[2])
#                 overlay_color.append({out_labels[i]: [fill_color, outline_color]})

#                 max_poly = get_largest_polygon(polys)#, value['height'], value['width'])
#                 overlay_poly.append(max_poly)

#             overlay_data = [overlay_poly, None, None, overlay_color]
#             save_masks_color(True, False, overlay_data, cropped_overlay_size, 0, output_path)
#             save_masks_color_and_overlay(image_path, overlay_data, output_path_overlay)

if __name__ == '__main__':
    do_segment = True
    # Define the paths
    # img_path = "D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_KEYPOINTS/GroundTruth_POINTS_V2/images/test"
    # img_path = "D:/Dropbox/PH/image_tests/LM2_viburnum_2000_NY/Plant_Components/Leaves_Whole"
    img_path = "D:/Dropbox/PH/image_tests/LM2_viburnum/Cropped_Images/By_Class/leaf_whole"
    # img_path = "D:/Dropbox/LM2_Env/Image_Datasets/SET_Diospyros/images_tiny"
    # model_path = "D:/Dropbox/LeafMachine2/leafmachine2/keypoint_detector/ultralytics/models/yolo/pose/trained_models/best_1280_sigma20.pt"
    model_path = "D:/Dropbox/LeafMachine2/KP_2024/uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2/weights/best.pt"
    
    # save_dir = "D:/Dropbox/LeafMachine2/KP_2024/OUTPUT2_uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2"
    # save_dir = "D:/Dropbox/PH/image_tests/LM2_viburnum_2000_NY/Key_Points"
    dir_oriented_images = "D:/Dropbox/PH/image_tests/LM2_viburnum/Key_PointsTEST"
    dir_keypoint_overlay = "D:/Dropbox/PH/image_tests/LM2_viburnum/Key_Points_PtsTEST"

    # save_dir_seg = "D:/Dropbox/LeafMachine2/KP_2024/OUTPUT8_uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2_SEG"
    # save_dir_seg = "D:/Dropbox/PH/image_tests/LM2_viburnum_2000_NY/Key_Points_SEG"
    dir_segment_oriented = "D:/Dropbox/PH/image_tests/LM2_viburnum/Key_Points_SEGTEST"
    # save_dir_overlay = "D:/Dropbox/LeafMachine2/KP_2024/OUTPUT8_uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2_OVERLAY"
    # save_dir_overlay = "D:/Dropbox/PH/image_tests/LM2_viburnum_2000_NY/Key_Points_OVERLAY"
    dir_segment_oriented_overlay = "D:/Dropbox/PH/image_tests/LM2_viburnum/Key_Points_OVERLAYTEST"


    if not os.path.exists(img_path):
        print(f"Image path {img_path} does not exist.")
        exit()
        
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        exit()
    
    if not os.path.exists(dir_oriented_images):
        print(f"Creating {dir_oriented_images}")
        os.makedirs(dir_oriented_images, exist_ok=True)

    if not os.path.exists(dir_keypoint_overlay):
        print(f"Creating {dir_keypoint_overlay}")
        os.makedirs(dir_keypoint_overlay, exist_ok=True)

    if not os.path.exists(dir_segment_oriented):
        print(f"Creating {dir_segment_oriented}")
        os.makedirs(dir_segment_oriented, exist_ok=True)
    
    if not os.path.exists(dir_segment_oriented_overlay):
        print(f"Creating {dir_segment_oriented_overlay}")
        os.makedirs(dir_segment_oriented_overlay, exist_ok=True)
    
    # Create dictionary for overrides
    overrides = {
        'model': model_path,
        'source': img_path,
        'name':'uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2',
        'boxes':False,
        'max_det':1,
        # 'visualize': True,
        'save_txt': True,
        # 'show':True
    }
    
    # Initialize PosePredictor
    pose_predictor = PosePredictor(model_path, dir_oriented_images, dir_keypoint_overlay, 
                                   save_oriented_images=True, save_keypoint_overlay=True, 
                                   device='cuda', cfg=DEFAULT_CFG, overrides=overrides, _callbacks=None)
    pose_predictor.process_images(img_path)

    # segment_rotated_leaf(img_path, dir_segment_oriented, dir_segment_oriented_overlay, logger=None)
    
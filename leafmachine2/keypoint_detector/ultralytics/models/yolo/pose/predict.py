# Ultralytics YOLO 🚀, AGPL-3.0 license
import os, cv2, sys, inspect, logging
import numpy as np

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))))
parentdir2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))
sys.path.append(parentdir2)
sys.path.append(currentdir)
sys.path.append(parentdir)

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from collections import defaultdict
from PIL import Image, ImageDraw

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))))
sys.path.append(currentdir)
# from detect import run
sys.path.append(parentdir)

'''Need these, but circular import with full LM2'''
from leafmachine2.segmentation.detectron2.segment_utils import get_largest_polygon, keep_rows, get_string_indices
from leafmachine2.segmentation.detectron2.detector import Detector_LM2

'''
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
'''
def save_masks_color_and_overlay(rotated_image_path, overlay_data, output_path):
    # Load the rotated image
    img = Image.open(rotated_image_path).convert("RGBA")

    # Create a transparent overlay image
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(overlay)

    # Unpack overlay data
    overlay_poly, _, _, overlay_color = overlay_data

    # Draw polygons with specified colors onto the overlay
    for i, poly in enumerate(overlay_poly):
        this_color = overlay_color[i]
        cls, this_color = next(iter(this_color.items()))
        color_BGR = this_color[0]  # Use the fill color
        color = (color_BGR[2], color_BGR[1], color_BGR[0], color_BGR[3])
        # Draw the filled polygon on the overlay
        draw.polygon(poly, fill=color)

    # Combine the original image with the overlay
    combined = Image.alpha_composite(img, overlay)

    # Convert combined image to RGB and save it
    combined.convert("RGB").save(output_path)

def save_masks_color(save_individual_masks_color, use_efds_for_masks, overlay_data, cropped_overlay_size, leaf_type, output_path):
    if len(overlay_data) > 0:
        # unpack
        overlay_poly, overlay_efd, overlay_rect, overlay_color = overlay_data

        if use_efds_for_masks:
            use_polys = overlay_efd
        else:
            use_polys = overlay_poly
        
        if save_individual_masks_color:
            # Create a black image
            img = Image.new('RGB', (cropped_overlay_size[1], cropped_overlay_size[0]), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            if use_polys != []:
                for i, poly in enumerate(use_polys):
                    this_color = overlay_color[i]
                    cls, this_color = next(iter(this_color.items()))
                    # Set the color for the polygon based on its class    
                    if leaf_type == 0:
                        if 'leaf' in cls:
                            color = [46, 255, 0]
                        elif 'petiole' in cls:
                            color = [0, 173, 255]
                        elif 'hole' in cls:
                            color = [209, 0, 255]
                        else:
                            color = [255, 255, 255]
                    elif leaf_type == 1:
                        if 'leaf' in cls:
                            color = [0, 200, 255]
                        elif 'petiole' in cls:
                            color = [255, 140, 0]
                        elif 'hole' in cls:
                            color = [200, 0, 255]
                        else:
                            color = [255, 255, 255]
                    # Draw the filled polygon on the image
                    draw.polygon(poly, fill=tuple(color))
            if leaf_type == 0:
                img.save(output_path)
            elif leaf_type == 1:
                img.save(output_path)

def segment_rotated_leaf(input_dir, output_dir, output_dir_overlay):
    logger = logging.basicConfig(filename='example.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
    # Initialize parameters
    THRESH = 0.5  # Example threshold
    LEAF_TYPE = 0  # Example leaf type, adjust as needed

    # Initialize the Detector_LM2 instance
    DIR_MODEL = os.path.join(parentdir, 'leafmachine2', 'segmentation', 'models', 'Group3_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR')#'GroupB_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR')
    detector = Detector_LM2(logger, DIR_MODEL, THRESH, LEAF_TYPE)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for image_name in os.listdir(input_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(input_dir, image_name)
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}__seg.png")
            output_path_overlay = os.path.join(output_dir_overlay, f"{os.path.splitext(image_name)[0]}__overlay.jpg")
            
            # Read the image
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            cropped_overlay_size = [height, width]
            
            # Segment the image
            out_polygons, out_bboxes, out_labels, out_color = detector.segment(img, generate_overlay=False, overlay_dpi=100, bg_color='black')
            
            
            if len(out_polygons) > 0: # Success
                out_polygons, out_bboxes, out_labels, out_color = keep_rows(out_polygons, out_bboxes, out_labels, out_color, get_string_indices(out_labels))
            
            overlay_color = []
            overlay_poly = []
            for i, polys in enumerate(out_polygons):
                color_rgb = tuple(map(lambda x: int(x*255), out_color[i]))
                fill_color = (color_rgb[0], color_rgb[1], color_rgb[2], 127)
                outline_color = (color_rgb[0], color_rgb[1], color_rgb[2])
                overlay_color.append({out_labels[i]: [fill_color, outline_color]})

                max_poly = get_largest_polygon(polys)#, value['height'], value['width'])
                overlay_poly.append(max_poly)

            overlay_data = [overlay_poly, None, None, overlay_color]
            save_masks_color(True, False, overlay_data, cropped_overlay_size, 0, output_path)
            save_masks_color_and_overlay(image_path, overlay_data, output_path_overlay)


class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model='yolov8n-pose.pt', source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, dir_oriented_images, dir_keypoint_overlay, save_oriented_images, save_keypoint_overlay, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.dir_oriented_images = dir_oriented_images
        self.dir_keypoint_overlay = dir_keypoint_overlay

        self.save_oriented_images = save_oriented_images
        self.save_keypoint_overlay = save_keypoint_overlay


        super().__init__(cfg, overrides, _callbacks)
        self.metadata = defaultdict(dict)
        self.args.task = 'pose'

        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')
            
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
            'lamina_tip': (0, 255, 0),  # Bright Green
            'apex_left': (255, 182, 193),  # Light Pink
            'apex_center': (255, 0, 255),  # Magenta
            'apex_right': (221, 160, 221),  # Light Purple
            'midvein_*': (128, 128, 128),  # Medium Gray
            'base_left': (173, 216, 230),  # Light Blue
            'base_center': (0, 0, 255),  # Blue
            'base_right': (0, 0, 139),  # Dark Blue
            'lamina_base': (255, 0, 0),  # Bright Red
            'petiole_*': (211, 211, 211),  # Light Gray
            'petiole_tip': (255, 165, 0),  # Orange
            'width_left': (0, 0, 0),  # Black
            'width_right': (0, 0, 0),  # Black
        }

    def get_color_for_keypoint(self, key):
        for pattern, color in self.color_map.items():
            if pattern.endswith('*') and key.startswith(pattern.split('_*')[0]):
                return color
            elif key == pattern:
                return color
        return (255, 255, 255)  # Default to white if no match
            
    def calc_angle(self, pred_kpts_np):
        # Extract the coordinates of points 0 and 22
        p0 = pred_kpts_np[0]
        p22 = pred_kpts_np[22]
        
        # Calculate the differences in coordinates
        dx = p22[0] - p0[0]  # Change in x
        dy = p22[1] - p0[1]  # Change in y
        
        # Calculate the angle in radians between the line connecting p0 to p22 and the vertical axis
        # Note: np.arctan2 returns the angle in radians, and it takes (y, x) instead of (x, y)
        angle_rad = np.arctan2(dx, dy)
        
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        
        # The angle returned by arctan2 might need adjustment depending on your coordinate system's orientation
        # This assumes the origin (0,0) is at the top-left corner of the image, and y values increase downwards.
        return angle_deg
    
    def visualize_keypoints(self, img, keypoints, img_path):
        for key, idx in self.mapping.items():
            point = keypoints[idx]
            color = self.get_color_for_keypoint(key)
            cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)

        # Save visualization
        visual_path = os.path.join(self.dir_keypoint_overlay, os.path.basename(img_path))
        if self.save_keypoint_overlay:
            cv2.imwrite(visual_path, img)
    
    def rotate_image(self, angle, orig_img, fname, output_dir):
        original_fname = os.path.basename(fname)

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Construct the full path for saving the rotated image
        save_path = os.path.join(output_dir, original_fname)

        # Store the metadata
        # self.metadata[fname]['original_image'] = orig_img
        # self.metadata[fname]['rotated_image'] = rotated_img
        self.metadata[fname]['original_keypoints'] = self.metadata[fname].get('keypoints_before_rotation')
        self.metadata[fname]['rotated_keypoints'] = np.dot(rot_mat, np.vstack([self.metadata[fname]['original_keypoints'], np.ones((1, self.metadata[fname]['original_keypoints'].shape[1]))]))
        self.metadata[fname]['rotation_angle'] = angle
        self.metadata[fname]['image_path'] = save_path

        # Save the rotated image
        if self.save_oriented_images:
            cv2.imwrite(save_path, rotated_img)
        
        # Display the rotated image
        # cv2.imshow('Rotated Image', rotated_img)
        # cv2.waitKey(0)  # Wait for a key press to close the image window
    
        # return rotated_img


    # def rotate_image(self, angle, orig_img, img_path):
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
    #     self.metadata[img_path].update({
    #         'rotated_image': save_path,
    #         'rotation_angle': angle,
    #         'rotated_keypoints': np.dot(rot_mat[:2, :2], self.metadata[img_path]['original_keypoints'].T + rot_mat[:2, 2]).T
    #     })
            
    # def postprocess(self, preds, img, orig_imgs):
    #     super().postprocess(preds, img, orig_imgs)
    #     # preds = ops.non_max_suppression(preds,
    #     #                                 self.args.conf,
    #     #                                 self.args.iou,
    #     #                                 agnostic=self.args.agnostic_nms,
    #     #                                 max_det=self.args.max_det,
    #     #                                 classes=self.args.classes,
    #     #                                 nc=len(self.model.names))
    #     preds = super().postprocess(preds, img, orig_imgs)

    #     if not isinstance(orig_imgs, list):
    #         orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    #     results = []
    #     for i, pred in enumerate(preds):
    #         orig_img = orig_imgs[i]
    #         pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
    #         pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else []
    #         pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

    #         img_path = self.batch[0][i]
    #         pred_kpts_np = pred_kpts.xy.cpu().numpy()[0]  # assuming this conversion is correct
    #         angle = self.calc_angle(pred_kpts_np)

    #         self.metadata[img_path] = {
    #             'original_keypoints': pred_kpts_np,
    #             'img_path': img_path,
    #         }

    #         self.rotate_image(-angle, orig_img, img_path)

    #         self.visualize_keypoints(orig_img, pred_kpts_np, img_path)

    #         results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts))

    #     return results

    def postprocess(self, preds, img, orig_imgs): # From the original playing around with it. Now use the super version
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        rotate_vertical_results = []
        for i, pred in enumerate(preds):
            rotate_vertical = {'pred_kpts_np':None, 
                               'angle':None,
                               'img_path':None,}
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            
            img_path = self.batch[0][i]

            rotate_vertical_results.append(rotate_vertical)

            res = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            pred_kpts_np = res.keypoints.xy.cpu().numpy()[0]

            angle = self.calc_angle(pred_kpts_np)

            # print(pred_kpts_np)
            # print(angle)

            rotate_vertical['pred_kpts_np'] = pred_kpts_np
            # rotate_vertical['angle'] = angle
            rotate_vertical['img_path'] = img_path

            self.rotate_image(-angle, orig_img, img_path, self.dir_oriented_images)

            results.append(
                res)
        return results
    



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
    save_dir_seg = "D:/Dropbox/PH/image_tests/LM2_viburnum/Key_Points_SEGTEST"
    # save_dir_overlay = "D:/Dropbox/LeafMachine2/KP_2024/OUTPUT8_uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2_OVERLAY"
    # save_dir_overlay = "D:/Dropbox/PH/image_tests/LM2_viburnum_2000_NY/Key_Points_OVERLAY"
    save_dir_overlay = "D:/Dropbox/PH/image_tests/LM2_viburnum/Key_Points_OVERLAYTEST"

    input_dir = dir_oriented_images


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

    if not os.path.exists(save_dir_seg):
        print(f"Creating {save_dir_seg}")
        os.makedirs(save_dir_seg, exist_ok=True)
    
    if not os.path.exists(save_dir_overlay):
        print(f"Creating {save_dir_overlay}")
        os.makedirs(save_dir_overlay, exist_ok=True)
    
    # Create dictionary for overrides
    overrides = {
        'model': model_path,
        'source': img_path,
        'name':'uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2',
        'boxes':False,
        'max_det':1,
        'visualize':True,
        # 'show':True
    }
    
    # Initialize PosePredictor
    pose_predictor = PosePredictor(dir_oriented_images, dir_keypoint_overlay, save_oriented_images=True, save_keypoint_overlay=True, overrides=overrides)
    
    # Run prediction
    results = pose_predictor.predict_cli()


    if do_segment:
        segment_rotated_leaf(input_dir, save_dir_seg, save_dir_overlay)
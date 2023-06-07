import os
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from scipy.special import softmax

from color_mapping import get_label_color
from utils_VV import set_index_for_annotation


def preprocess(image):
    # Convert the image to YOLOv5 (640, 640) input format, float16
    image = image.resize((640, 640))
    image = F.to_tensor(image).unsqueeze(0).numpy().astype(np.float16)  # add batch dimension and convert to float16
    return image

def nms(boxes, scores, threshold, max_boxes=50):
    # initialize list for selected box indices
    pick = []
    # calculate areas of boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # check for large values
    large_values_mask = (widths > 10000) | (heights > 10000)

    if np.any(large_values_mask):
        print("Warning: Some bounding boxes have large dimensions that could cause numeric overflow when calculating area.")
        widths[large_values_mask] = 10000
        heights[large_values_mask] = 10000

    area = widths * heights

    # sort box indices by scores
    indices = scores.argsort()[::-1]

    # limit the number of boxes
    indices = indices[:max_boxes]

    while len(indices) > 0:
        # select box with highest current score
        i = indices[0]
        pick.append(i)
        # calculate IoU of this box with all remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (area[i] + area[indices[1:]] - intersection)
        # discard indices of boxes where IoU is above threshold
        indices = indices[1:][iou < threshold]
    return pick


def postprocess(predictions, threshold_obj=0.15, threshold_nms=0.15):
    boxes = predictions[..., 0:4]  # box coordinates
    objectness = predictions[..., 4:5]  # objectness score
    classes = predictions[..., 5:]  # class probabilities
    scores = objectness * softmax(classes, axis=-1)  # overall confidence score
    max_scores = scores.max(axis=-1)
    
    # apply objectness score threshold
    keep = max_scores >= threshold_obj
    boxes, scores, classes = boxes[keep], max_scores[keep], classes[keep]

    # apply NMS
    keep = nms(boxes, scores, threshold_nms)
    boxes, scores = boxes[keep], scores[keep]
    
    # get class indices
    print("classes shape:", classes.shape)
    class_preds = classes.argmax(axis=-1)
    
    # format as list of dicts
    print(len(boxes), len(scores), len(class_preds)) # They should all have the same length
    results = [{'bbox': boxes[i], 'class_idx': class_preds[i], 'score': scores[i]} for i in range(len(boxes))]

    return results

def draw_boxes(image, results):
    # Draw bounding boxes and labels onto the image
    draw = ImageDraw.Draw(image)
    for result in results:
        box = result['bbox']
        class_idx = result['class_idx']
        score = result['score']
        label = f"{set_index_for_annotation(class_idx)}: {score:.2f}"
        c_outline, c_fill = get_label_color("ruler")
        draw.rectangle([(box[0], box[1]), (box[0]+box[2], box[1]+box[3])], outline=c_outline, fill=c_fill)
        draw.text((box[0], box[1]), label)
    return image

def create_session(path_model):
    try:
        # Try to use TensorRT
        ort_sess = ort.InferenceSession(path_model, providers=['TensorrtExecutionProvider'])
        print("Using TensorRT")
        return ort_sess
    except Exception as e:
        print("Failed to use TensorRT, Error: ", e)
        try:
            # Try to use CUDA
            ort_sess = ort.InferenceSession(path_model, providers=['CUDAExecutionProvider'])
            print("Using CUDA")
            return ort_sess
        except Exception as e:
            print("Failed to use CUDA, Error: ", e)
            try:
                # Try to use CPU
                ort_sess = ort.InferenceSession(path_model, providers=['CPUExecutionProvider'])
                print("Using CPU")
                return ort_sess
            except Exception as e:
                print("Failed to use CPU, Error: ", e)
                raise Exception("Could not find a suitable provider. Please check your environment.")

def setup(path_img):
    # load the model
    dir_home = os.path.dirname(__file__)
    path_model = os.path.join(dir_home, "models","LM2_archival_v-2-1.onnx")

    # check the model
    onnx_model = onnx.load(path_model)
    onnx.checker.check_model(onnx_model)

    # use the path string to create the InferenceSession, not the loaded onnx_model
    ort_sess = create_session(path_model)
    
    # Get input name
    input_name = ort_sess.get_inputs()[0].name
    print(f"Model Input Name: {input_name}")

def process_labels(ort_sess, input_name, path_img):
    # load and preprocess the image
    image = Image.open(path_img)
    image_pp = preprocess(image)
    
    # pass the preprocessed image to the model
    raw_predictions = ort_sess.run(None, {input_name: image_pp})

    # reshape the output
    raw_predictions = np.reshape(raw_predictions, (1, -1, 14))

    # postprocess the raw predictions
    results = postprocess(raw_predictions)
    
    # draw the results on the original image
    overlay = draw_boxes(image, results)
    
    # save or display the image with bounding boxes
    # overlay.show()  # uncomment to display the image
    overlay.save("output.jpg")  # save the output
    
    # draw boxes and labels on the image
    image_with_boxes = draw_boxes(image, results)
    
    # save or display the result
    os.path.splitext(os.path.basename(path_img))[0]
    image_with_boxes.save("output.jpg")
    image_with_boxes.show()
    
    print("Detection completed.")

def run_voucher_vision(cfg, logger, dir_home, Project, Dirs):
    if cfg['leafmachine']['use_RGB_label_images']:
        dir_labels = os.path.join(Dirs.save_per_annotation_class,'label')
    else:
        dir_labels = os.path.join(Dirs.save_per_annotation_class,'label_binary')

    

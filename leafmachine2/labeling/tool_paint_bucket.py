import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os
from streamlit_drawable_canvas import st_canvas
from scipy import ndimage


st.set_page_config(layout="wide", page_title='Paint', initial_sidebar_state="collapsed")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

if 'clicked_point' not in st.session_state:
    st.session_state.clicked_point = None

if 'img' not in st.session_state:
    st.session_state.img = None

if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

if 'should_rerun' not in st.session_state:
    st.session_state.should_rerun = False

if 'despeckled_img' not in st.session_state:
    st.session_state.despeckled_img = None

if 'inverted_img' not in st.session_state:
    st.session_state.inverted_img = None

def fill_bucket(image, start_point, fill_color):
    ImageDraw.floodfill(image, start_point, fill_color, thresh=150)
    return image

def invert_colors(image):
    # Convert the image to grayscale and invert all pixels
    inverted_image = image.convert('L').point(lambda x: 255 - x)
    return inverted_image.convert('RGB')


def remove_specks(image, min_size=10, aggressiveness=0.5):
    # Convert the image to binary (black and white)
    binary = image.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
    
    # Convert to numpy array
    np_image = np.array(binary)
    
    # Label connected components
    labeled_array, num_features = ndimage.label(np_image)
    
    # Calculate the size threshold based on aggressiveness
    max_size = np.max([np.sum(labeled_array == i) for i in range(1, num_features + 1)])
    size_threshold = int(max_size * aggressiveness)
    
    # Remove objects smaller than the threshold
    for label in range(1, num_features + 1):
        component = labeled_array == label
        if np.sum(component) < size_threshold or np.sum(component) < min_size:
            np_image[component] = 0
    
    # Convert back to PIL Image
    result = Image.fromarray(np_image.astype('uint8') * 255)
    return result



def process_image(image_path, fill_color, start_point):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = fill_bucket(img, start_point, fill_color)
    img.save(image_path)
    return img

def main():
    st.title("Paint Bucket Tool")

    image_dir = 'G:/train_ruler_binary/Custom_Unet/training_inprogress'
    fill_color_hex = st.sidebar.color_picker("Pick Fill Color", "#000000")
    fill_color = hex_to_rgb(fill_color_hex)

    if os.path.isdir(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) == 0:
            st.error("No images found in the directory.")
            return
    else:
        st.error("Invalid directory. Please input a valid image directory.")
        return
    
    image_index = st.sidebar.number_input("Image Index", min_value=0, max_value=len(image_files)-1, value=st.session_state.image_index, step=1)

    # Update session state if user changes index via number input
    if image_index != st.session_state.image_index:
        st.session_state.image_index = image_index
        st.session_state.clicked_point = None
        st.session_state.should_rerun = True

    image_index = st.session_state.image_index
    selected_image = image_files[image_index]
    image_path = os.path.join(image_dir, selected_image)
    
    # Load or reload the image
    st.session_state.img = Image.open(image_path)
    img_width, img_height = st.session_state.img.size

    st.subheader(f"Image {image_index + 1} of {len(image_files)}: {selected_image}")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Previous") and image_index > 0:
            st.session_state.image_index -= 1
            st.session_state.clicked_point = None
            st.session_state.should_rerun = True
    
    with col2:
        if st.button("Next") and image_index < len(image_files) - 1:
            st.session_state.image_index += 1
            st.session_state.clicked_point = None
            st.session_state.should_rerun = True
    
    # Display the canvas with the current image
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.5)",
        stroke_width=1,
        background_image=st.session_state.img,
        update_streamlit=True,
        height=img_height,
        width=img_width,
        drawing_mode="point",
        point_display_radius=5,
        key=f"canvas_{image_index}"
    )

    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        last_object = canvas_result.json_data["objects"][-1]
        new_point = (int(last_object["left"]), int(last_object["top"]))
        if new_point != st.session_state.clicked_point:
            st.session_state.clicked_point = new_point
            st.write(f"Clicked point: {st.session_state.clicked_point}")
            st.session_state.img = process_image(image_path, fill_color, st.session_state.clicked_point)
            st.session_state.should_rerun = True
    
    # Speck Removal
    if st.session_state.despeckled_img is not None:
        if st.button("Apply Speck Removal"):
            st.session_state.img = st.session_state.despeckled_img
            st.session_state.img.save(image_path)
            st.success("Speck removal applied and saved!")
            st.session_state.despeckled_img = None
            st.session_state.should_rerun = True
    
    min_speck_size = st.slider("Speck Size", 0, 100, 20, 1)
    speck_removal_aggressiveness = st.slider("Speck Removal Aggressiveness", 0.0, 0.2, 0.05, 0.005)
    st.session_state.despeckled_img = remove_specks(st.session_state.img, min_speck_size, speck_removal_aggressiveness)
    st.image(st.session_state.despeckled_img, caption="Despeckled Preview", use_column_width=True)
    
    # Inversion
    if st.session_state.inverted_img is not None:
        if st.button("Apply Inversion"):
            st.session_state.img = st.session_state.inverted_img
            st.session_state.img.save(image_path)
            st.success("Inversion applied and saved!")
            st.session_state.inverted_img = None
            st.session_state.should_rerun = True
    
    if st.button("Invert"):
        st.session_state.inverted_img = invert_colors(st.session_state.img)
        st.image(st.session_state.inverted_img, caption="Inverted Preview", use_column_width=True)
    
    if st.session_state.should_rerun:
        st.session_state.should_rerun = False
        st.rerun()

if __name__ == "__main__":
    main()
import os
import random
import shutil
# Streamlit integration
import streamlit as st
from streamlit_image_select import image_select
import json

def save_project_state(file_path, state):
    """Save the current project state to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(state, f)

def load_project_state(file_path):
    """Load a project state from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def list_files(directory):
    """List all files in the given directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def parse_filename(filename):
    """Parse filename and return a dictionary with herbarium, family, genus, species, and fullname."""
    parts = filename.split('_')
    return {
        'herbarium': parts[0],
        'family': parts[2],
        'genus': parts[3],
        'species': parts[4].split('.')[0],  # Assuming file extension follows
        'fullname': '_'.join(parts[2:5]).split('.')[0]
    }

def select_images(directory, n_imgs, category, download_dict=None, swap_out=None):
    """Select n_imgs for each unique key in the category, optionally swapping out one image."""
    files = list_files(directory)
    parsed_files = {f: parse_filename(f) for f in files}
    category_dict = {}

    for file, attrs in parsed_files.items():
        key = attrs[category]
        category_dict.setdefault(key, []).append(file)

    swap_category = None
    if swap_out is not None:
        swap_category = parsed_files[swap_out][category]

        # Ensure the swapped out image is not considered in the future
        category_dict[swap_category].remove(swap_out)

        available_files = [f for f in category_dict[swap_category] if f not in download_dict[swap_category]]
        
        if available_files:
            # Replace the swapped-out image with a new one
            new_file = random.choice(available_files)
            download_dict[swap_category].remove(swap_out)
            download_dict[swap_category].append(new_file)
        else:
            # No replacement available, just remove the swapped-out image
            download_dict[swap_category].remove(swap_out)

    elif swap_out is None:
        # Initial selection of images
        for key, files in category_dict.items():
            download_dict[key] = random.sample(files, min(len(files), n_imgs))

    if swap_category is None:
        for key, files in category_dict.items():
            download_dict[key] = random.sample(files, min(len(files), n_imgs))

    return download_dict


def copy_images(download_dict, dir_new):
    """Copy images from the download_dict to the new directory."""
    if not os.path.exists(dir_new):
        os.makedirs(dir_new)
    
    for category, files in download_dict.items():
        for file in files:
            shutil.copy(os.path.join(st.session_state.directory, file), dir_new)

def save_load_projects():
    # Dropdown to load existing projects
    existing_projects = [f for f in os.listdir(st.session_state.project_dir) if f.endswith('.json')]
    project_to_load = st.selectbox("Load Project", [""] + existing_projects)

    # Input field to name a new project
    new_project_name = st.text_input("Name for New Project")

    # Save button
    if st.button("Save Current State"):
        if new_project_name or project_to_load:
            if new_project_name:
                use_name = new_project_name
            elif project_to_load:
                use_name = project_to_load.split('.')[0]
            save_path = os.path.join(st.session_state.project_dir, use_name + ".json")
            state = {
                'directory': st.session_state.directory,
                'dir_new': st.session_state.dir_new,
                'n_imgs': st.session_state.n_imgs,
                'category': st.session_state.category,
                'download_dict': st.session_state.download_dict,
                'current_key_index': st.session_state.current_key_index
            }
            save_project_state(save_path, state)
            st.success(f"Project '{use_name}' saved!")

    # Load button
    if project_to_load and st.button("Load Selected Project"):
        load_path = os.path.join(st.session_state.project_dir, project_to_load)
        loaded_state = load_project_state(load_path)
        for key, value in loaded_state.items():
            st.session_state[key] = value
        st.success(f"Project '{project_to_load}' loaded!")


def main():
    col1, col2, col3 = st.columns([4, 4, 2])
    
    with col1:
        save_load_projects()

        # Input fields with default values from session_state
        st.session_state.directory = st.text_input("Enter the directory path:", value=st.session_state['directory'])
        st.session_state.dir_new = st.text_input("Enter the new directory path:", value=st.session_state['dir_new'])
        Lcol1, Lcol2 = st.columns([1,1])
        delcol, _, __, swapcol, nextcol, backcol = st.columns([1,1,1,1,1,1])

        with Lcol1:
            st.session_state.n_imgs = st.number_input("Number of images per category:", min_value=1, value=st.session_state['n_imgs'])
        with Lcol2:
            st.session_state.category = st.selectbox("Select category:", index=["herbarium", "family", "genus", "species", "fullname"].index(st.session_state['category']), options=["herbarium", "family", "genus", "species", "fullname"])

    download_dict = {}
    current_key_index = st.session_state.get('current_key_index', 0)

    

    if 'download_dict' in st.session_state:
        download_dict = st.session_state['download_dict']
        keys = list(download_dict.keys())

        if keys:
            current_key = keys[current_key_index]
            with col1:
                st.subheader(f"Item: {current_key}")
                images = [os.path.join(st.session_state.directory, f) for f in download_dict[current_key]]
                selected_image = image_select(label="Select an Image", images=images, use_container_width=False, return_value="original")

                # Button to delete the current key and its images
                with delcol:
                    if st.button("Delete Category"):
                        del download_dict[current_key]
                        st.session_state['download_dict'] = download_dict
                        if current_key_index > 0:
                            st.session_state['current_key_index'] -= 1
                        st.success(f"Category '{current_key}' deleted!")
                        st.rerun()

                with swapcol:
                    if selected_image and st.button("Swap Image"):
                        download_dict = select_images(st.session_state.directory, st.session_state.n_imgs, st.session_state.category, download_dict, swap_out=os.path.basename(selected_image))
                        st.session_state['download_dict'] = download_dict
                        st.rerun()

                with nextcol:
                    if st.button("NEXT"):
                        if current_key_index < len(keys) - 1:
                            st.session_state['current_key_index'] += 1
                        else:
                            st.warning("Reached the end of categories!")
                        st.rerun()

                with backcol:
                    if st.button("BACK"):
                        if current_key_index > 0:
                            st.session_state['current_key_index'] -= 1
                        else:
                            st.warning("At the start of categories!")
                        st.rerun()



        with col2:
            mcol1, mcol2 = st.columns([1,1])
            
            with mcol1:
                if 'download_dict' in st.session_state and st.session_state['download_dict']:
                    current_item = st.session_state['current_key_index'] + 1  # Adding 1 to make it human-readable
                    total_keys = len(st.session_state['download_dict'])
                    progress = (st.session_state['current_key_index'] / total_keys) if total_keys else 0
                    st.progress(progress,text=f"Working on item {current_item} of {total_keys}")
            
            with mcol2:
                W = st.slider(label="image width",value=600,max_value=2000,min_value=100,step=50)

            st.image(selected_image, width=W)

    with col3:
        rcol1, rcol2 = st.columns([1,1])
        with rcol1:
            if st.button("Generate Download List") and st.session_state.directory and st.session_state.dir_new and st.session_state.category:
                # Initialize download_dict as an empty dictionary
                download_dict = {}
                download_dict = select_images(st.session_state.directory, st.session_state.n_imgs, st.session_state.category, download_dict)
                st.session_state['download_dict'] = download_dict
                st.session_state['current_key_index'] = 0
                current_key_index = 0
        with rcol2:
            if st.button("Copy Files") and st.session_state.directory and st.session_state.dir_new and st.session_state.category:
                copy_images(download_dict, st.session_state.dir_new)
                st.success("Images have been copied!")
        st.json(download_dict)

    

st.set_page_config(layout="wide", page_title='SupervisedSelection')

# Set default values
if 'dir_home' not in st.session_state:
    st.session_state['dir_home'] = os.path.dirname(__file__)
if 'project_dir' not in st.session_state:
    st.session_state['project_dir'] = os.path.join(st.session_state.dir_home, 'saved_projects_supervised_selection')
    os.makedirs(st.session_state.project_dir, exist_ok=True)

if 'directory' not in st.session_state:
    st.session_state['directory'] = '/media/data/Shape_Datasets/Tropical/Eschweilera_Mart_ex_DC/img'
if 'dir_new' not in st.session_state:
    st.session_state['dir_new'] = '/media/data/Shape_Datasets/Tropical/Eschweilera_Mart_ex_DC/img_sub'
if 'n_imgs' not in st.session_state:
    st.session_state['n_imgs'] = 10
if 'category' not in st.session_state:
    st.session_state['category'] = "fullname"

if __name__ == "__main__":
    main()

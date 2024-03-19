import streamlit as st
import os, yaml, platform, re, queue, time, csv
import threading
import pandas as pd
from dwc_downloader import download_dwc_archive, get_taxon_key, month_name_to_number
from mappings import continent_mapping, country_mapping, type_status_mapping, iucn_status_mapping
from download_from_GBIF_all_images_in_file import download_all_images_in_images_csv
from leafmachine2.machine.utils_GBIF import create_subset_file

def get_default_download_folder():
    system_platform = platform.system()  # Gets the system platform, e.g., 'Linux', 'Windows', 'Darwin'

    if system_platform == "Windows":
        # Typically, the Downloads folder for Windows is in the user's profile folder
        st.session_state.output_folder = os.path.join(os.getenv('USERPROFILE'), 'Downloads')
    elif system_platform == "Darwin":
        # Typically, the Downloads folder for macOS is in the user's home directory
        st.session_state.output_folder = os.path.join(os.path.expanduser("~"), 'Downloads')
    elif system_platform == "Linux":
        # Typically, the Downloads folder for Linux is in the user's home directory
        st.session_state.output_folder = os.path.join(os.path.expanduser("~"), 'Downloads')
    else:
        st.session_state.output_folder = "set/path/to/downloads/folder"
        print("Please manually set the output folder")
    
def run_download(job, output_folder, do_extract_zip, has_coord):
    st.session_state.output_zip_file = download_dwc_archive(
        do_extract_zip, job['genus'], job['family'], job['species'], job['start_year'], job['end_year'], output_folder,
        month_name_to_number(job['month']), job['publisher'], country_mapping[job['country']],
        continent_mapping[job['continent']], job['institution_code'],
        type_status_mapping[job['type_status']], iucn_status_mapping[job['iucn_status']],
        has_coord
    )
    st.session_state.thread_status = "finished"

def queue_download_job():
    job = {
        "family": st.session_state.family,
        "genus": st.session_state.genus,
        "species": st.session_state.species,
        "start_year": st.session_state.start_year,
        "end_year": st.session_state.end_year,
        "month": st.session_state.month,
        "continent": st.session_state.continent,
        "country": st.session_state.country,
        "publisher": st.session_state.publisher,
        "institution_code": st.session_state.institution_code,
        "type_status": st.session_state.type_status,
        "iucn_status": st.session_state.iucn_status
    }
    st.session_state.download_jobs.append(job)

    st.session_state.family = None
    st.session_state.genus = None
    st.session_state.species = None
    st.session_state.start_year = None
    st.session_state.end_year = None
    st.session_state.month = None
    st.session_state.continent = None
    st.session_state.country = None
    st.session_state.publisher = None
    st.session_state.institution_code = None
    st.session_state.type_status = None
    st.session_state.iucn_status = None
    job = {}

    st.success("Job added to queue")

def process_queued_jobs(output_folder, do_extract_zip, has_coord):
    for i, job in enumerate(st.session_state.download_jobs, start=1):
        st.session_state.thread_status = "started"
        run_download(job, output_folder, do_extract_zip, has_coord)
        st.success(f'Job {i} completed')
        st.success(f'Zipped folder saved to: {st.session_state.output_zip_file}')

    st.session_state.download_jobs = [] 

def create_input_field(label, col_1, col_2, do_verify_query=False, input_type="text_input", options=None):
    with col_1:
        if input_type == "text_input":
            user_input = st.text_input(label)
        elif input_type == "selectbox":
            user_input = st.selectbox(label, options)
        else:
            user_input = None
    
    # Check if the user has entered something
    with col_2:
        if user_input:
            if do_verify_query:
                key = get_taxon_key(user_input, label.lower())  
                display_key_result(key, label)
        else:
            st.write("")  # Placeholder
    
    return user_input

def clean_folder_name(folder_name):
    # Replace any special characters with nothing
    folder_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", folder_name)
    # Replace spaces with underscores
    folder_name = folder_name.replace(" ", "_")
    return folder_name


def display_key_result(key, label):
    if key is not None:
        st.success(f"{label} Key: {key}", icon="✅")
    else:
        st.warning(f"{label} Key: Not found", icon="⚠️")

def get_non_zip_folders(output_folder):
    # Check if the output folder path is valid
    if not os.path.exists(output_folder):
        return []

    # List all items in the output folder
    items = os.listdir(output_folder)

    # Filter out non-folder items and zip folders
    folders = [item for item in items if os.path.isdir(os.path.join(output_folder, item)) and not item.endswith('.zip')]

    # Check for the special case where the only folder is named "dataset"
    if len(folders) == 1 and folders[0] == "dataset":
        return [output_folder]
    elif len(folders) == 0:
        return [output_folder]
    else:
        # Return a list of full paths for each non-zip folder
        return [os.path.join(output_folder, folder) for folder in folders]

def set_download_folder():
    st.header(f"Download Folder")

    # Layout for download folder
    col_1_download, col_2_download = st.columns([4, 2])

    # Text input and checkbox to enter new folder and decide on its creation
    with col_1_download:
        create_dir_if_not_exist = st.checkbox("Create directory if it does not exist", value=True)
        st.session_state.new_folder = st.text_input("Enter new download folder:", value=st.session_state.new_folder)

        st.session_state.do_extract_zip = st.checkbox("Automatically extract the contents of the downloaded ZIP file into a separate folder", value=True)

        # Button to activate folder change
        if st.button("Change"):
            st.session_state.set_folder = True
        
        # Logic to handle folder change
        if st.session_state.new_folder and st.session_state.set_folder:
            
            # Check if the path is a valid directory
            if os.path.isdir(st.session_state.new_folder) and not create_dir_if_not_exist:
                st.session_state.output_folder = st.session_state.new_folder
                st.success(f"Directory set to '{st.session_state.new_folder}'")
            elif not os.path.isdir(st.session_state.new_folder) and not create_dir_if_not_exist:
                get_default_download_folder()
                st.error("Invalid file path. Please enter a different file path")
            elif create_dir_if_not_exist:
                os.makedirs(st.session_state.new_folder,exist_ok=True)
                st.session_state.output_folder = st.session_state.new_folder
                st.success(f"Directory '{st.session_state.new_folder}' created")
            else:
                get_default_download_folder()
                st.error("Invalid file path. Please enter a different file path")
        
        # Reset the folder change flag
        if st.session_state.set_folder:
            st.session_state.set_folder = False

    st.subheader(f"Location: {st.session_state.output_folder}")

def read_spreadsheet_length(file_path):
    try:
        # Determine the file extension
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                row_count = sum(1 for row in reader)
                return row_count - 1  # Adjust for header
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                row_count = sum(1 for row in file)
                return row_count - 1  # Adjust for header if needed
        else:
            return "Unsupported file format"
    except Exception as e:
        # Log the exception if needed
        print(f"Error {e}")
        return "None"

def count_files(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])


def download_images():

    # Create input fields
    st.session_state.DWC_folder_containing_records = st.text_input("Folder Containing DWC Records",value=st.session_state.output_folder)

    st.session_state.DWC_occ_name = st.text_input("DWC Occurrence File Name",value=st.session_state.DWC_occ_name)

    st.session_state.DWC_img_name = st.text_input("DWC Multimedia File Name",value=st.session_state.DWC_img_name)

    st.session_state.DWC_min_res = st.number_input("Minimum Resolution",value=st.session_state.DWC_min_res,min_value=1)

    st.session_state.DWC_max_res = st.number_input("Maximum Resolution",value=st.session_state.DWC_max_res,min_value=1)

    st.session_state.do_resize = st.checkbox("Resize images larger than 'Maximum Resolution' to have long side of 5000 pixels", value=False, help="If the Maximum Resolution is set to be 30MP, any images larger that 30MP will be A) skipped/not downloaded if this is set to False B) will be downloaded and resized to ~24MP if this is set to True. If you want to download all images regardless of size, set very large an very low bounds like 1 and 200.")

    st.session_state.DWC_n_threads = st.number_input("Number of Threads",value=st.session_state.DWC_n_threads,min_value=1)

    st.session_state.project_download_list = get_non_zip_folders(st.session_state.DWC_folder_containing_records)

    # Create a DataFrame to store the information
    project_info = []

    # Iterate through each project path
    try:
        for path in st.session_state.project_download_list:
            occ_path = os.path.join(path, st.session_state.DWC_occ_name)
            img_path = os.path.join(path, st.session_state.DWC_img_name)

            occ_length = read_spreadsheet_length(occ_path)
            img_length = read_spreadsheet_length(img_path)

            project_info.append([path, occ_length, img_length])
            # Convert the list to a DataFrame
            project_df = pd.DataFrame(project_info, columns=["Project Path", "Occurrence Length", "Multimedia Length"])

            # Display the DataFrame as a table
        st.table(project_df)

    except Exception as e:
        # Log the exception if needed
        print(f"Error {e}")
        occ_length = "None"
        img_length = "None"
        st.info(f"The folder {st.session_state.DWC_folder_containing_records} does not contain DWC occ or image files.")

    total_projects = len(st.session_state.project_download_list)

    # /home/brlab/Downloads/Populus_test_auto
    if st.button("Download Images"):
        for index, path in enumerate(st.session_state.project_download_list):

            dir_destination_images = os.path.join(path, 'img')
            dir_destination_csv = os.path.join(path, 'occ')
            cfg = {
                'dir_home': path,
                'dir_destination_images': dir_destination_images,
                'dir_destination_csv': dir_destination_csv,
                'filename_occ': st.session_state.DWC_occ_name,
                'filename_img': st.session_state.DWC_img_name, 
                'filename_combined': 'combined_occ_img_downloaded.csv',
                'MP_low': st.session_state.DWC_min_res,
                'MP_high': st.session_state.DWC_max_res,
                'do_resize': st.session_state.do_resize,
                'n_threads': st.session_state.DWC_n_threads,
                'ignore_banned_herb': False,
                # 'banned_url_stems': [], #['mediaphoto.mnhn.fr'] # ['mediaphoto.mnhn.fr', 'stock.images.gov'] etc....
                'is_custom_file': False,
                # 'col_url': 'url',
                # 'col_name': 'lab_code',
            }
            ######################################################################################################
            new_filename = create_subset_file(cfg)
            cfg['filename_occ'] = new_filename
            cfg['dir_destination_images'] = os.path.join(path, 'img_subset')
            ######################################################################################################
            download_all_images_in_images_csv(cfg)
            st.success(f"Download images for {os.path.basename(os.path.normpath(path))}")
            percentage_completed = (index + 1) / total_projects
            st.session_state.progress_bar_total.progress(percentage_completed)

        st.success('Download completed')

    st.session_state.progress_bar_total = st.progress(0, text="Working on projects...")

    


def main():

    # Streamlit UI components
    st.title("Download DWC Archive")
    st.write("Species must include genus (e.g., 'Quercus alba') and must match the GBIF naming convention (e.g., 'Acer rubrum L.'). \nRun Check Query prior to download.")

    set_download_folder()

    st.header("Options")
    st.session_state.has_coord = st.checkbox("Specimen record must have coordinates", value=True)

    col_1, col_2 = st.columns([4, 2])

    st.session_state.family = create_input_field("Family", col_1, col_2, do_verify_query=True)
    st.session_state.genus = create_input_field("Genus", col_1, col_2, do_verify_query=True)
    st.session_state.species = create_input_field("Species", col_1, col_2, do_verify_query=True)

    st.session_state.start_year = create_input_field("Year Start", col_1, col_2)
    st.session_state.end_year = create_input_field("Year End", col_1, col_2)

    st.session_state.month = create_input_field("Month", col_1, col_2, input_type="selectbox", options=["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

    st.session_state.continent = create_input_field("Continent", col_1, col_2, input_type="selectbox", options=list(continent_mapping.keys()))
    st.session_state.country = create_input_field("Country", col_1, col_2, input_type="selectbox", options=list(country_mapping.keys()))

    st.session_state.publisher = create_input_field("Publisher", col_1, col_2)
    st.session_state.institution_code = create_input_field("Institution Code", col_1, col_2)

    st.session_state.type_status = create_input_field("Type Status", col_1, col_2, input_type="selectbox", options=list(type_status_mapping.keys()))
    st.session_state.iucn_status = create_input_field("IUCN Status", col_1, col_2, input_type="selectbox", options=list(iucn_status_mapping.keys()))


    if st.button("Add to Queue"):
        queue_download_job()

    st.write('---')
    st.subheader("Current Download Queue")
    if st.session_state.download_jobs:
        # Convert the list of dictionaries (jobs) into a DataFrame
        queue_df = pd.DataFrame(st.session_state.download_jobs)
        # Display the DataFrame as a table
        st.table(queue_df)
    else:
        st.write("No jobs in the queue.")

    st.write('---')

    # Add button to start processing queued jobs
    if st.button("Start Downloads"):
        with st.spinner('Starting queued downloads...'):
            process_queued_jobs(st.session_state.output_folder, st.session_state.do_extract_zip, st.session_state.has_coord)
            st.balloons()
            st.success('All queued downloads completed')
            st.success('Ready to download more records')


    if st.session_state.thread_status == "finished":
        st.session_state.thread_status = "not_started"

    # YAML load/save functionality
    yaml_file = st.file_uploader("Load YAML", type=['yaml'])

    if yaml_file is not None:
        yaml_content = yaml.safe_load(yaml_file)
        st.write("Loaded YAML:", yaml_content)

    if st.button("Save YAML"):
        data = {
            "family": st.session_state.family,
            "genus": st.session_state.genus,
            "species": st.session_state.species,
            "start_year": st.session_state.start_year,
            "end_year": st.session_state.end_year,
            "month": st.session_state.month,
            "continent": st.session_state.continent,
            "country": st.session_state.country,
            "publisher": st.session_state.publisher,
            "institution_code": st.session_state.institution_code,
            "type_status": st.session_state.type_status,
            "iucn_status": st.session_state.iucn_status
        }
        st.download_button(
            label="Download YAML",
            data=yaml.dump(data),
            file_name="config.yaml",
            mime="application/x-yaml"
        )

    st.write('---')
    download_images()

    



st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='Download DWC Archive')

if "output_folder" not in st.session_state:
    get_default_download_folder()
if "set_folder" not in st.session_state:
    st.session_state.set_folder = False
if "created_new_folder" not in st.session_state:
    st.session_state.created_new_folder = False
if 'new_folder' not in st.session_state:
    st.session_state.new_folder = ""
if 'thread_status' not in st.session_state:
    st.session_state.thread_status = "not_started"
if 'output_zip_file' not in st.session_state:
    st.session_state.output_zip_file = ""
if 'do_extract_zip' not in st.session_state:
    st.session_state.do_extract_zip = True
if 'has_coord' not in st.session_state:
    st.session_state.has_coord = True
if 'completed_downloads' not in st.session_state:
    st.session_state.completed_downloads = 0
if 'stop_update' not in st.session_state:
    st.session_state.stop_update = False

if 'progress_baprogress_bar_total_current' not in st.session_state:
    st.session_state.progress_bar_total = st.empty()
if 'progress_bar_current' not in st.session_state:
    st.session_state.progress_bar_current = st.empty()

if 'download_jobs' not in st.session_state:
    st.session_state.download_jobs = []

if 'family' not in st.session_state:
    st.session_state.family = None
if 'genus' not in st.session_state:
    st.session_state.genus = None
if 'species' not in st.session_state:
    st.session_state.species = None
if 'start_year' not in st.session_state:
    st.session_state.start_year = None
if 'end_year' not in st.session_state:
    st.session_state.end_year = None
if 'month' not in st.session_state:
    st.session_state.month = None
if 'continent' not in st.session_state:
    st.session_state.continent = None
if 'country' not in st.session_state:
    st.session_state.country = None
if 'publisher' not in st.session_state:
    st.session_state.publisher = None
if 'institution_code' not in st.session_state:
    st.session_state.institution_code = None
if 'type_status' not in st.session_state:
    st.session_state.type_status = None
if 'iucn_status' not in st.session_state:
    st.session_state.iucn_status = None

if "project_download_list" not in st.session_state:
    st.session_state.project_download_list = []
if "DWC_folder_containing_records" not in st.session_state:
    st.session_state.DWC_folder_containing_records = ""
if "DWC_occ_name" not in st.session_state:
    st.session_state.DWC_occ_name = 'occurrence.txt'
if "DWC_img_name" not in st.session_state:
    st.session_state.DWC_img_name = 'multimedia.txt'
if "DWC_min_res" not in st.session_state:
    st.session_state.DWC_min_res = 1
if "DWC_max_res" not in st.session_state:
    st.session_state.DWC_max_res = 200
if "DWC_n_threads" not in st.session_state:
    st.session_state.DWC_n_threads = 32
if "do_resize" not in st.session_state:
    st.session_state.do_resize = False

# TODO add logic for downloading from custom files
# is_custom_file: False
# col_url: 'url'
# col_name: 'lab_code'
# # Use for custom
# filename_img: 'P_serotina_79_urls.csv' 

main()

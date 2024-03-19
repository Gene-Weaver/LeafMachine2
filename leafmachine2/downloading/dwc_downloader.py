import requests
import os
import time
import yaml, re
import tkinter as tk
from tkinter import messagebox
import zipfile

def clean_folder_name(folder_name):
    # Replace any special characters with nothing
    folder_name = re.sub(r"[^a-zA-Z0-9_\- ]", "", folder_name)
    # Replace spaces with underscores
    folder_name = folder_name.replace(" ", "_")
    return folder_name

def download_dwc_archive(do_extract_zip=True, genus=None, family=None, species=None, start_year=None, end_year=None, 
                         output_path=None, month=None, publisher=None, country=None, continent=None, institution_code=None, type_status=None, iucn_status=None,
                         HAS_COORDINATE=True):

    # Load login information
    private_login_data = load_private_login('PRIVATE_LOGIN.yaml')
    username = private_login_data["username"]
    password = private_login_data["password"]
    notification_address = private_login_data["notification_address"]
    
    family_key = get_taxon_key(family, 'family')
    genus_key = get_taxon_key(genus, 'genus')
    species_key = get_taxon_key(species, 'species')

    if HAS_COORDINATE:
        HAS_COORDINATE_val = "true"
    else:    
        HAS_COORDINATE_val = "false"

    predicates = [
        {"type": "equals", "key": "BASIS_OF_RECORD", "value": "PRESERVED_SPECIMEN"},
        {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"},
        {"type": "equals", "key": "HAS_COORDINATE", "value": HAS_COORDINATE_val},
        {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
    ]

    if family_key is not None:
        predicates.append({"type": "equals", "key": "FAMILY_KEY", "value": family_key})

    if genus_key is not None:
        predicates.append({"type": "equals", "key": "GENUS_KEY", "value": genus_key})

    if species_key is not None:
        predicates.append({"type": "equals", "key": "SPECIES_KEY", "value": species_key})

    if start_year:
        predicates.append({"type": "greaterThanOrEquals", "key": "YEAR", "value": start_year})

    if end_year:
        predicates.append({"type": "lessThanOrEquals", "key": "YEAR", "value": end_year})

    if month:
        predicates.append({"type": "equals", "key": "MONTH", "value": month})

    if publisher:
        predicates.append({"type": "equals", "key": "PUBLISHING_ORG", "value": publisher})

    if country:
        predicates.append({"type": "equals", "key": "COUNTRY", "value": country})

    if continent:
        predicates.append({"type": "equals", "key": "CONTINENT", "value": continent})

    if institution_code:
        predicates.append({"type": "equals", "key": "INSTITUTION_CODE", "value": institution_code})

    if type_status:
        predicates.append({"type": "equals", "key": "TYPE_STATUS", "value": type_status})

    if iucn_status:
        predicates.append({"type": "equals", "key": "IUCN_RED_LIST_CATEGORY", "value": iucn_status})

    base_url = "https://api.gbif.org/v1/occurrence/download/request"
    headers = {"Content-Type": "application/json"}
    data = {
        "creator": username,
        "notification_address": notification_address,
        "sendNotification": True,
        "format": "DWCA",
        "predicate": {
            "type": "and",
            "predicates": predicates
        }
    }

    response = requests.post(base_url, json=data, headers=headers, auth=(username, password))
    print(f"Response content: {response}")

    if response.status_code == 201:
        download_key = response.text
        print(f"Download request successful. Download key: {download_key}")
        print("Waiting for the file to be ready...")


        status_url = f"https://api.gbif.org/v1/occurrence/download/{download_key}"
        while True:
            status_response = requests.get(status_url)
            status_data = status_response.json()
            status = status_data["status"]
            
            if status == "SUCCEEDED":
                download_url = status_data["downloadLink"]
                download_response = requests.get(download_url, stream=True)
                print(f"Download URL: {download_url}")
                if download_response.status_code == 200:
                    # print(f"Download response headers: {download_response.headers}")
                    file_name_parts = [part for part in [family, genus, species, start_year, end_year, month, publisher, country, continent, institution_code, type_status, iucn_status] if part]
                    file_name = '_'.join(file_name_parts)
                    file_name = clean_folder_name(file_name)
                    file_name = file_name + '.zip'
                    output_file = os.path.join(output_path, file_name)
                    with open(output_file, "wb") as file:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    print(f"File successfully saved to {output_file}")

                    # Extract the zip file
                    if do_extract_zip:
                        with zipfile.ZipFile(output_file, 'r') as zip_ref:
                            # Create a directory with the same name as the zip file (without the .zip extension)
                            extract_folder = os.path.splitext(output_file)[0]
                            os.makedirs(extract_folder, exist_ok=True)
                            zip_ref.extractall(extract_folder)
                            print(f"Contents extracted to {extract_folder}")

                    return output_file
                else:
                    print(f"Error downloading file: {download_response.status_code}, {download_response.text}")
                    return None
            elif status == "FAILED":
                print("Download request failed.")
                return None
            else:
                time.sleep(60)  # Wait for 1 minute before checking the status again

    else:
        print(f"Error: {response.status_code}, {response.text}")

def get_taxon_key(name, rank):
        url = f"https://api.gbif.org/v1/species/match?name={name}&rank={rank}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            key_name = f"{rank.lower()}Key"
            return data.get(key_name)
        else:
            print(f"Error fetching taxon key for {name} ({rank}): {response.status_code}, {response.text}")
            return None

def load_private_login(file_path):
    with open(file_path, 'r') as file:
        private_login_data = yaml.safe_load(file)
    return private_login_data

def month_name_to_number(month_name):
    month_names = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    return month_names.index(month_name)

def create_private_login_yaml():
    def save_data():
        data = {
            "notification_address": [notification_address_entry.get()],
            "username": username_entry.get(),
            "password": password_entry.get()
        }

        with open("PRIVATE_LOGIN.yaml", "w") as file:
            yaml.safe_dump(data, file)

        messagebox.showinfo("Info", "PRIVATE_LOGIN.yaml created successfully!")
        root.destroy()

    root = tk.Tk()
    root.title("Enter PRIVATE_LOGIN Information")

    tk.Label(root, text="Notification Address:").grid(row=1, column=0, sticky="e")
    notification_address_entry = tk.Entry(root)
    notification_address_entry.grid(row=1, column=1)

    tk.Label(root, text="Username:").grid(row=2, column=0, sticky="e")
    username_entry = tk.Entry(root)
    username_entry.grid(row=2, column=1)

    tk.Label(root, text="Password:").grid(row=3, column=0, sticky="e")
    password_entry = tk.Entry(root, show="*")
    password_entry.grid(row=3, column=1)

    tk.Button(root, text="Save", command=save_data).grid(row=4, column=1, sticky="e")

    root.mainloop()

def load_private_login(file_path):
    if not os.path.exists(file_path):
        messagebox.showwarning("Warning", "PRIVATE_LOGIN.yaml not found. Please enter the required information.")
        create_private_login_yaml()

    with open(file_path, 'r') as file:
        private_login_data = yaml.safe_load(file)

    return private_login_data

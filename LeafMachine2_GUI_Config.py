import requests, os
import time
import shutil
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog
from PIL import Image, ImageTk
import yaml
import webbrowser

import tkinter.filedialog as filedialog

def select_directory():
    selected_directory = filedialog.askdirectory()
    dir_output_var.set(selected_directory)



def save_input():
    # Your code to save the input data to a YAML file
    pass

def open_url():
    webbrowser.open("https://leafmachine.org")

def close_app():
    root.destroy()


# Create the tkinter window
font_size = 16
root = ThemedTk(theme="Adapta")
root.title("LeafMachine Configuration")
root.columnconfigure(1, weight=1)

style = ttk.Style()
style.configure("TLabel", font=("Roboto", font_size))
style.configure("TEntry", font=("Roboto", font_size))
style.configure("TButton", font=("Roboto", font_size))

# Add the logo image
image = Image.open("D:/Dropbox/LM2_Env/LeafMachine2_Manuscript/logo/LM2_Desktop_Narrow.jpg")
new_height = 300
aspect_ratio = image.width / image.height
new_width = int(new_height * aspect_ratio)
image = image.resize((new_width, new_height), Image.ANTIALIAS)
logo = ImageTk.PhotoImage(image)
logo_label = tk.Label(root, image=logo, cursor="hand2")
logo_label.grid(row=0, column=0, columnspan=5)
logo_label.bind("<Button-1>", lambda event: open_url())

# Create StringVars for form entries
dir_output_var = tk.StringVar()
run_name_var = tk.StringVar()
image_location_var = tk.StringVar()
GBIF_mode_var = tk.StringVar()
batch_size_var = tk.StringVar()
num_workers_var = tk.StringVar()
dir_images_local_var = tk.StringVar()
path_combined_csv_local_var = tk.StringVar()
path_occurrence_csv_local_var = tk.StringVar()
path_images_csv_local_var = tk.StringVar()
use_existing_plant_component_detections_var = tk.StringVar()
use_existing_archival_component_detections_var = tk.StringVar()
process_subset_of_images_var = tk.StringVar()
dir_images_subset_var = tk.StringVar()
n_images_per_species_var = tk.StringVar()
species_list_var = tk.StringVar()

# Create labels and entry widgets
dir_output_label = ttk.Label(root, text="Project Output Dir:")
dir_output_entry = ttk.Entry(root, textvariable=dir_output_var, width=100)
dir_output_button = ttk.Button(root, text="Select", command=select_directory)

run_name_label = ttk.Label(root, text="Run Name:")
run_name_entry = ttk.Entry(root, textvariable=run_name_var)
image_location_label = ttk.Label(root, text="Image Location:")
image_location_entry = ttk.Combobox(root, textvariable=image_location_var, values=['local', 'GBIF'])
GBIF_mode_label = ttk.Label(root, text="GBIF Mode:")
GBIF_mode_entry = ttk.Entry(root, textvariable=GBIF_mode_var)
batch_size_label = ttk.Label(root, text="Batch Size:")
batch_size_entry = ttk.Entry(root, textvariable=batch_size_var)
num_workers_label = ttk.Label(root, text="Num Workers:")
num_workers_entry = ttk.Combobox(root, textvariable=num_workers_var, values=[1, 2, 4, 8])

dir_images_local_label = ttk.Label(root, text="Dir Images Local:")
dir_images_local_entry = ttk.Entry(root, textvariable=dir_images_local_var)
dir_images_local_button = ttk.Button(root, text="Select", command=select_directory)

path_combined_csv_local_label = ttk.Label(root, text="Path Combined CSV Local:")
path_combined_csv_local_entry = ttk.Entry(root, textvariable=path_combined_csv_local_var)
path_occurrence_csv_local_label = ttk.Label(root, text="Path Occurrence CSV Local:")
path_occurrence_csv_local_entry = ttk.Entry(root, textvariable=path_occurrence_csv_local_var)
path_images_csv_local_label = ttk.Label(root, text="Path Images CSV Local:")
path_images_csv_local_entry = ttk.Entry(root, textvariable=path_images_csv_local_var)
use_existing_plant_component_detections_label = ttk.Label(root, text="Use Existing Plant Component Detections:")
use_existing_plant_component_detections_entry = ttk.Entry(root, textvariable=use_existing_plant_component_detections_var)
use_existing_archival_component_detections_label = ttk.Label(root, text="Use Existing Archival Component Detections:")
use_existing_archival_component_detections_entry = ttk.Entry(root, textvariable=use_existing_archival_component_detections_var)
process_subset_of_images_label = ttk.Label(root, text="Process Subset Of Images:")
process_subset_of_images_entry = ttk.Entry(root, textvariable=process_subset_of_images_var)
dir_images_subset_label = ttk.Label(root, text="Dir Images Subset:")
dir_images_subset_entry = ttk.Entry(root, textvariable=dir_images_subset_var)
n_images_per_species_label = ttk.Label(root, text="N Images Per Species:")
n_images_per_species_entry = ttk.Entry(root, textvariable=n_images_per_species_var)
species_list_label = ttk.Label(root, text="SpeciesList:")
species_list_entry = ttk.Entry(root, textvariable=species_list_var)

# Create buttons
save_button = ttk.Button(root, text="Save YAML", command=save_input)
close_button = ttk.Button(root, text="Close", command=close_app)

# Grid layout
dir_output_label.grid(row=2, column=0, sticky="e")
dir_output_entry.grid(row=2, column=1, sticky="ew")#, padx=10, pady=20)
dir_output_button.grid(row=2, column=2)

run_name_label.grid(row=3, column=0, sticky="e")
run_name_entry.grid(row=3, column=1, columnspan=2, sticky="ew")
image_location_label.grid(row=4, column=0, sticky="e")
image_location_entry.grid(row=4, column=1, columnspan=2, sticky="ew")
GBIF_mode_label.grid(row=5, column=0, sticky="e")
GBIF_mode_entry.grid(row=5, column=1, columnspan=2, sticky="ew")
batch_size_label.grid(row=6, column=0, sticky="e")
batch_size_entry.grid(row=6, column=1, columnspan=2, sticky="ew")
num_workers_label.grid(row=7, column=0, sticky="e")
num_workers_entry.grid(row=7, column=1, columnspan=2, sticky="ew")

dir_images_local_label.grid(row=8, column=0, sticky="e")
dir_images_local_entry.grid(row=8, column=1, sticky="ew")
dir_images_local_button.grid(row=8, column=2)

path_combined_csv_local_label.grid(row=9, column=0, sticky="e")
path_combined_csv_local_entry.grid(row=9, column=1, columnspan=2, sticky="ew")
path_occurrence_csv_local_label.grid(row=10, column=0, sticky="e")
path_occurrence_csv_local_entry.grid(row=10, column=1, columnspan=2, sticky="ew")
path_images_csv_local_label.grid(row=11, column=0, sticky="e")
path_images_csv_local_entry.grid(row=11, column=1, columnspan=2, sticky="ew")
use_existing_plant_component_detections_label.grid(row=12, column=0, sticky="e")
use_existing_plant_component_detections_entry.grid(row=12, column=1, columnspan=2, sticky="ew")
use_existing_archival_component_detections_label.grid(row=13, column=0, sticky="e")
use_existing_archival_component_detections_entry.grid(row=13, column=1, columnspan=2, sticky="ew")
process_subset_of_images_label.grid(row=14, column=0, sticky="e")
process_subset_of_images_entry.grid(row=14, column=1, columnspan=2, sticky="ew")
dir_images_subset_label.grid(row=15, column=0, sticky="e")
dir_images_subset_entry.grid(row=15, column=1, columnspan=2, sticky="ew")
n_images_per_species_label.grid(row=16, column=0, sticky="e")
n_images_per_species_entry.grid(row=16, column=1, columnspan=2, sticky="ew")
species_list_label.grid(row=17, column=0, sticky="e")
species_list_entry.grid(row=17, column=1, columnspan=2, sticky="ew")

# Create a spacer in column 3
spacer_label = ttk.Label(root, text="")
spacer_label.grid(row=0, column=3, rowspan=16)

# Place buttons in column 4
save_button.grid(row=2, column=4, sticky="ew")
close_button.grid(row=4, column=4, sticky="ew")

root.columnconfigure(1, weight=1)

root.mainloop()
import os, sys
import win32com.client
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageEnhance

def create_shortcut():
    # Request user's confirmation
    confirmation = input("Do you want to create a shortcut for the LeafMachine2? (y/n): ")

    if confirmation.lower() != "y":
        print("Okay, no shortcut will be created.")
        return

    # Get the script path
    script_path = os.path.abspath(__file__)
    #  Get the directory of the script
    script_dir = os.path.dirname(script_path)

    # Path to the icon file
    icon_path = os.path.join(script_dir, 'img', 'icon.jpg')
    img = Image.open(icon_path)  
    enhancer = ImageEnhance.Color(img)
    img_enhanced = enhancer.enhance(1.5) 
    img_enhanced.save(os.path.join(script_dir, 'img', 'icon.ico'), format='ICO', sizes=[(256,256)])
    icon_path_ico = os.path.join(script_dir, 'img', 'icon.ico')

    # Construct the path to the static folder
    static_dir = os.path.join(script_dir, "static")

    # Ask for the name of the shortcut
    shortcut_name = "LeafMachine2"

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    root.update()  # Ensures that the dialog appears on top
    folder_path = filedialog.askdirectory(title="Choose location to save the shortcut")
    print(f"Shortcut will be saved to {folder_path}")

    venv_path = filedialog.askdirectory(title="Choose the location of your Python virtual environment")
    print(f"Using virtual environment located at {venv_path}")

    # Path to the activate script in the venv
    activate_path = os.path.join(venv_path, "Scripts")

    shortcut_path = os.path.join(folder_path, f'{shortcut_name}.lnk')

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = "%windir%\System32\cmd.exe"

    streamlit_exe = os.path.join(venv_path, "Scripts","streamlit")
    print(script_dir)
    print(streamlit_exe)
    activate_path = os.path.join(script_dir,".venv_LM2","Scripts")
    print(activate_path)
    shortcut.Arguments = f'/K cd /D ""{activate_path}"" && activate && cd /D ""{script_dir}"" && python run_LeafMachine2.py'
    # Set the icon of the shortcut
    shortcut.IconLocation = icon_path_ico

    shortcut.save()

    print(f"Shortcut created with the name '{shortcut_name}' in the chosen directory.")

if __name__ == "__main__":
    create_shortcut()
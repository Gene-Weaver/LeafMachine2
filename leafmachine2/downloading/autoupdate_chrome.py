import os, sys, inspect, re
import subprocess
import requests
import yaml  # To parse the YAML configuration file

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.general_utils import get_cfg_from_full_path


def get_latest_chrome_version():
    # URL to the metadata file in Google's repository
    repo_url = "https://dl.google.com/linux/chrome/deb/dists/stable/main/binary-amd64/Packages"
    try:
        response = requests.get(repo_url)
        if response.status_code == 200:
            # Extract the version information from the Packages metadata
            match = re.search(r"Version: (\d+\.\d+\.\d+\.\d+)-1", response.text)
            if match:
                return match.group(1)
            else:
                print("Could not parse the latest Chrome version from the repository.")
                return None
        else:
            print(f"Failed to fetch metadata from {repo_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching latest Chrome version: {e}")
        return None

def get_installed_chrome_version():
    try:
        result = subprocess.run(
            ["google-chrome", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            version_output = result.stdout.strip()
            match = re.search(r"Google Chrome (\d+\.\d+\.\d+\.\d+)", version_output)
            if match:
                return match.group(1)
        print("Google Chrome is not installed or could not fetch the version.")
        return None
    except Exception as e:
        print(f"Error checking installed Chrome version: {e}")
        return None

def is_new_update_available():
    latest_version = get_latest_chrome_version()
    installed_version = get_installed_chrome_version()

    if not latest_version:
        print("Could not retrieve the latest Chrome version.")
        return False

    if not installed_version:
        print("No installed Chrome version found. Update is required.")
        return True

    print(f"Latest Chrome Version: {latest_version}")
    print(f"Installed Chrome Version: {installed_version}")

    # Compare the versions
    return latest_version != installed_version

def get_sudo_password_from_config():
    # Path to the configuration file
    path_cfg_private = os.path.join(parentdir, 'PRIVATE_DATA.yaml')
    
    # Load the YAML configuration
    try:
        with open(path_cfg_private, 'r') as file:
            cfg_private = yaml.safe_load(file)
        return cfg_private.get('sudo_password')
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {path_cfg_private}")
        return None
    except KeyError:
        print("Error: 'sudo_password' key not found in the configuration file")
        return None
    except Exception as e:
        print(f"Error reading configuration: {e}")
        return None

def download_and_install_chrome():
    sudo_password = get_sudo_password_from_config()
    # URL to get the latest Chrome `.deb` package
    chrome_url = "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
    deb_file = "google-chrome-stable_current_amd64.deb"
    
    # Step 1: Download the .deb package
    print(f"Downloading Google Chrome from {chrome_url}...")
    response = requests.get(chrome_url, stream=True)
    
    if response.status_code == 200:
        with open(deb_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {deb_file}")
    else:
        print(f"Failed to download Chrome. Status code: {response.status_code}")
        return

    # Step 2: Install the package
    print("Installing Google Chrome...")
    try:
        # Provide the sudo password for dpkg
        install_command = f"echo {sudo_password} | sudo -S dpkg -i {deb_file}"
        subprocess.run(install_command, shell=True, check=True)
        print("Google Chrome installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        
        # Fix missing dependencies and re-run dpkg
        print("Attempting to fix dependencies...")
        fix_command = f"echo {sudo_password} | sudo -S apt-get -f install -y"
        subprocess.run(fix_command, shell=True, check=True)
    
    # Cleanup: Remove the downloaded .deb file
    if os.path.exists(deb_file):
        os.remove(deb_file)
        print(f"Removed the downloaded file: {deb_file}")

if __name__ == "__main__":
    # Get sudo password from the configuration file
    
    if is_new_update_available():
        print("A new stable Chrome update is available!")
        download_and_install_chrome()
    else:
        print("You are up to date.")
    
    # download_and_install_chrome()

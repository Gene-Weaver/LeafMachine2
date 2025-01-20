import os
import re
import requests
import subprocess
import yaml
import winreg

def get_latest_chrome_version():
    # URL to fetch the latest stable Chrome version (official Google URL)
    metadata_url = "https://versionhistory.googleapis.com/v1/chrome/platforms/win/channels/stable/versions"
    try:
        response = requests.get(metadata_url)
        if response.status_code == 200:
            data = response.json()
            if data and "versions" in data and len(data["versions"]) > 0:
                return data["versions"][0]["version"]
            else:
                print("No version data found in the response.")
                return None
        else:
            print(f"Failed to fetch metadata from {metadata_url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching latest Chrome version: {e}")
        return None

def get_installed_chrome_version():
    try:
        # Access registry key to find the installed Chrome version
        reg_path = r"SOFTWARE\\Google\\Chrome\\BLBeacon"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path) as key:
            version, _ = winreg.QueryValueEx(key, "version")
            return version
    except FileNotFoundError:
        print("Google Chrome is not installed.")
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
    path_cfg_private = os.path.join(os.getcwd(), 'PRIVATE_DATA.yaml')
    
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
    chrome_url = "https://dl.google.com/chrome/install/latest/chrome_installer.exe"
    installer_file = "chrome_installer.exe"

    # Step 1: Download the installer
    print(f"Downloading Google Chrome from {chrome_url}...")
    response = requests.get(chrome_url, stream=True)

    if response.status_code == 200:
        with open(installer_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {installer_file}")
    else:
        print(f"Failed to download Chrome. Status code: {response.status_code}")
        return

    # Step 2: Install Chrome
    print("Installing Google Chrome...")
    try:
        subprocess.run([installer_file, '/silent', '/install'], check=True)
        print("Google Chrome installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
    finally:
        # Cleanup: Remove the installer file
        if os.path.exists(installer_file):
            os.remove(installer_file)
            print(f"Removed the installer file: {installer_file}")

if __name__ == "__main__":
    if is_new_update_available():
        print("A new stable Chrome update is available!")
        download_and_install_chrome()
    else:
        print("You are up to date.")

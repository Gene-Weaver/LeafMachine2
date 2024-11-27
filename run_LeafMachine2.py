import streamlit.web.cli as stcli
import os, sys, random, time, subprocess
import socket
from pathlib import Path
import git
from importlib.metadata import distributions
from packaging.requirements import Requirement
from packaging import version

def update_setuptools():
    """Update the setuptools package using pip."""
    print("Updating setuptools to avoid compatibility issues...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"], check=True)
        print("setuptools updated successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to update setuptools:", e)
        sys.exit(1)  # Exit if setuptools can't be updated
        

def normalize_package_name(name):
    """Normalize package names to match the naming convention used in distributions."""
    return name.lower().replace('-', '_').replace(' ', '')

def get_installed_distributions():
    """Retrieve installed distributions as a dict with normalized package names as keys."""
    return {normalize_package_name(dist.metadata['Name']): dist.version for dist in distributions()}

def check_and_fix_requirements(requirements_file):
    """
    Checks if installed packages in the virtual environment satisfy the requirements specified
    in the requirements.txt file and fixes them if they do not.
    """
    installed_distributions = get_installed_distributions()
    missing_or_incompatible = []
    
    with open(requirements_file, 'r') as req_file:
        requirements = [Requirement(line.strip()) for line in req_file if line.strip() and not line.startswith('#')]

    for req in requirements:
        pkg_name = normalize_package_name(req.name)
        if pkg_name not in installed_distributions:
            missing_or_incompatible.append(f"{req} is not installed")
        elif req.specifier:
            installed_ver = version.parse(installed_distributions[pkg_name])
            if installed_ver not in req.specifier:
                missing_or_incompatible.append(f"{pkg_name}=={installed_distributions[pkg_name]} does not satisfy {req}")

    if missing_or_incompatible:
        print("The following packages are missing or incompatible:")
        for issue in missing_or_incompatible:
            print(f"  - {issue}")
        
        print("Attempting to fix the package issues by running 'pip install -r requirements.txt'...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
            print("Packages have been successfully updated.")
        except subprocess.CalledProcessError as e:
            print("Failed to install packages:", e)
    else:
        print("All requirements are satisfied.")


def find_github_desktop_git():
    """Search for the most recent GitHub Desktop Git installation."""
    # Base path where GitHub Desktop versions are located
    base_path = Path(f"C:/Users/{os.getlogin()}/AppData/Local/GitHubDesktop/")
    print(f"base_path: {base_path}")

    # Searching recursively for git.exe within any directories under the base path
    versions = sorted(base_path.rglob('git.exe'), key=lambda x: x.parent, reverse=True)
    for git_path in versions:
        print(f"git_path: {git_path}")
        if "app-" in str(git_path.parent):  # Ensuring it's in an 'app-' directory if that's still relevant
            print(f"git_path_exists: TRUE")
            return str(git_path)

    print(f"git_path_exists: FALSE")
    return None

def update_repository(repo_path):
    print(f"changing path to: {repo_path}")
    os.chdir(repo_path)
    print(f"changed to: {repo_path}")

    try:
        # Open the existing repository at the specified path
        repo = git.Repo(repo_path)
        # Check for the current working branch
        current_branch = repo.active_branch
        print(f"Updating repository on branch: {current_branch.name}")

        # Pulls updates for the current branch
        origin = repo.remotes.origin
        result = origin.pull()

        # Check if the pull was successful
        if result[0].flags > 0:
            print("Repository updated successfully.")
        else:
            print("No updates were available.")
                
    except Exception as e:
        print(f"Error while updating repository: {e}")


def find_available_port(start_port, end_port):
    ports = list(range(start_port, end_port + 1))
    random.shuffle(ports)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except socket.error:
                print(f"Port {port} is in use, trying another port...")
    raise ValueError(f"Could not find an available port in the range {start_port}-{end_port}.")


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path

if __name__ == "__main__":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    start_port = 8501
    end_port = 8599
    retry_count = 0
    repo_path = resolve_path(os.path.dirname(__file__))
    print(f"repo_path: {repo_path}")

    update_setuptools()
    check_and_fix_requirements(resolve_path(os.path.join(os.path.dirname(__file__),'requirements.txt')))

    try:
        update_repository(repo_path)
    except:
        print(f"Could not update VVE using git pull.")
        print(f"Make sure that 'Git' is installed and can be accessed by this user account.")

    # Update again in case the pull introduced a new package
    check_and_fix_requirements(resolve_path(os.path.join(os.path.dirname(__file__),'requirements.txt')))

    try:
        free_port = find_available_port(start_port, end_port)
        sys.argv = [
            "streamlit",
            "run",
            resolve_path(os.path.join(os.path.dirname(__file__),"leafmachine2","machine", "LeafMachine2_GUI.py")),
            "--global.developmentMode=false",
            f'--server.maxUploadSize=51200',
            f'--server.runOnSave=true',
            f'--server.port={free_port}',
            f'--theme.primaryColor=#16a616',
            f'--theme.backgroundColor=#1a1a1a',
            f'--theme.secondaryBackgroundColor=#303030',
            f'--theme.textColor=cccccc',
        ]
        sys.exit(stcli.main())

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    else:
        print("Failed to start the application after multiple attempts.")
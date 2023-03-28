

Table of Contents
=================


## Installing LeafMachine2

### Prerequisites
- Python 3.8.10
- PyTorch 1.11 
- CUDA version 11.3 (if utilizing a GPU)
- Git

### Installation - Cloning the LeafMachine2 Repository
1. First, install Python 3.8.10 on your machine of choice.
2. Open a terminal window and `cd` into the directory where you want to install LeafMachine2.
3. Clone the LeafMachine2 repository from GitHub by running `git clone https://github.com/Gene-Weaver/LeafMachine2.git` in the terminal.
4. Move into the LeafMachine2 directory by running `cd LeafMachine2` in the terminal.
5. To run LeafMachine2 we need to install its dependencies inside of a python virtual environmnet. Follow the instructions below for your operating system. 

### About Python Virtual Environments
A virtual environment is a tool to keep the dependencies required by different projects in separate places, by creating isolated python virtual environments for them. This avoids any conflicts between the packages that you have installed for different projects. It makes it easier to maintain different versions of packages for different projects.

### Installation - Ubuntu 20.04

#### Virtual Environment

1. Still inside the LeafMachine2 directory, show that a venv is currently not active: 
    - `which python`
2. Then create the virtual environment (venv_fp is the name of our new virtual environment):  
    - `python3 -m venv venv_fp`
3. Activate the virtual environment:  
    - `source ./venv_fp/bin/activate`
4. Confirm that the venv is active (should be different from step 1):  
    - `which python`
5. If you want to exit the venv, deactivate the venv using:  
    - `deactivate`

#### Installing Packages

1. Install the required dependencies to use LeafMachine2: 
    - `pip install opencv-python pandas pybboxes scipy scikit-image numpy tqdm pyyaml IPython matplotlib seaborn tensorboard fpdf`
2. Upgrade numpy: 
    - `pip install numpy -U`
3. Install qrcode[pil]:  
    - `pip install qrcode[pil]`
4. The LeafMachine2 machine learning algorithm requires PyTorch version 1.11 for CUDA version 11.3. If your computer does not have a GPU, then use the CPU version and the CUDA version is not applicable. PyTorch is large and will take a bit to install.
    - WITH GPU: `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    - WITH CPU ONLY: `pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu`
5. Test the installation:  
    - `python3 test.py`
    - If you see large red messages, then the installation was not successful. The rror message will be below the large red boxes, providing information on how to correct the installation error. If you need more help, please submit an inquiry in the form at [LeafMachine2.org](https://LeafMachine2.org/)
6. You can view the test output in `LeafMachine2/demo/demo_output/`

### Installation - Windows 10+

#### Virtual Environment

1. Still inside the LeafMachine2 directory, show that a venv is currently not active: 
    - `python --version`
2. Then create the virtual environment (venv_fp is the name of our new virtual environment):  
    - `python3 -m venv venv_fp`
3. Activate the virtual environment:  
    - `.\venv_fp\Scripts\activate`
4. Confirm that the venv is active (should be different from step 1):  
    - `python --version`
5. If you want to exit the venv, deactivate the venv using:  
    - `deactivate`

#### Installing Packages

1. Install the required dependencies to use LeafMachine2:  
    - `pip install opencv-python pandas pybboxes scipy scikit-image numpy tqdm pyyaml IPython matplotlib seaborn tensorboard fpdf`
2. Upgrade numpy:  
    - `pip install numpy -U`
3. Install qrcode[pil]:  
    - `pip install qrcode[pil]`
4. The LeafMachine2 machine learning algorithm requires PyTorch version 1.11 for CUDA version 11.3. If your computer does not have a GPU, then use the CPU version and the CUDA version is not applicable. PyTorch is large and will take a bit to install.
    - WITH GPU: `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    - WITH CPU ONLY: `pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu`
5. Test the installation. test.py will test the QR code builder and image processing portions of LeafMachine2:  
    - `python test.py`
    - If you see large red messages, then the installation was not successful. The rror message will be below the large red boxes, providing information on how to correct the installation error. If you need more help, please submit an inquiry in the form at [LeafMachine2.org](https://LeafMachine2.org/)
6. You can view the test output in at [this Zenodo repo]https://zenodo.org/record/7764379

---

### Troubleshooting CUDA

- If your system already has another version of CUDA (e.g., CUDA 11.7) then it can be complicated to switch to CUDA 11.3. 
- The simplest solution is to install pytorch with CPU only, avoiding the CUDA problem entirely.
- Alternatively, you can install the [latest pytorch release]https://pytorch.org/get-started/locally/ for your specific system, either using the cpu only version `pip3 install torch`, `pip3 install torchvision`, `pip3 install torchaudio` or by matching the pythorch version to your CUDA version.
- We have not validated CUDA 11.6 or CUDA 11.7, but our code is likely to work with them too. If you have success with other versions of CUDA/pytorch, let us know and we will update our instructions. 

## Using LeafMachine2

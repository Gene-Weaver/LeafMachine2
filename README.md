[![LeafMachine2](https://LeafMachine.org/img/LM2_Desktop_Narrow2.jpg "LeafMachine2")](https://LeafMachine.org/)

Table of Contents
=================

* [Table of Contents](#table-of-contents)
   * [Installing LeafMachine2](#installing-leafmachine2)
      * [Prerequisites](#prerequisites)
      * [Hardware](#hardware)
      * [Installation - Cloning the LeafMachine2 Repository](#installation---cloning-the-leafmachine2-repository)
      * [About Python Virtual Environments](#about-python-virtual-environments)
      * [Installation - Ubuntu 20.04](#installation---ubuntu-2004)
         * [Virtual Environment](#virtual-environment)
         * [Installing Packages](#installing-packages)
      * [Installation - Windows 10+](#installation---windows-10)
         * [Virtual Environment](#virtual-environment-1)
         * [Installing Packages](#installing-packages-1)
      * [Troubleshooting CUDA](#troubleshooting-cuda)
   * [Testing the LeafMachine2 Installation](#testing-the-leafmachine2-installation)
   * [Using LeafMachine2](#using-leafmachine2)

---

## Installing LeafMachine2

### Prerequisites
- Python 3.8.10 (Later versions should work. We have validated Python 3.10.4 too)
- PyTorch 1.11 
- Git
- CUDA version 11.3 (if utilizing a GPU)
    * see [Troubleshooting CUDA](#Troubleshooting-CUDA)



### Hardware
- A GPU with at least 8 GB of VRAM is required
- LeafMachine2 v.2.1 is RAM limited. A batch size of 50 images could potentially utilize 48 GB+ of system memory. Setting batch sizes to 20 will only increase the number of summary and data files, but performance speed differences are minimal.
- The PCD confidence threshold dictates RAM usage. More leaves detected -> more RAM to store more leaves and derived measurements until they are saved to batch files and summary images. 
- The number of leaves per image dictates RAM usage. Taxa with hundred of leaves per image (e.g. _Diospyros buxifolia_) will require much more RAM than taxa with few leaves per image (e.g. _Quercus alba_)
- For most PCs, set the number of workers to 2 or 4. If you have a high performance PC with 128 GB+ of RAM and a powerful CPU, then 8 workers and batch sizes of 100+ are possible. 
> **Note:** An average PC with 32 GB of RAM and a consumer-grade GPU is more than capable of running LeafMachine2, just dial back the batch size. With the right configs, a PC with 16 GB of RAM can run LeafMachine2 if the batch size is set to 10 or 20. 


### Installation - Cloning the LeafMachine2 Repository
1. First, install Python 3.8.10, or greater, on your machine of choice.
    - Make sure that you can use `pip` to install packages on your machine, or at least inside of a virtual environment.
    - Simply type `pip` into your terminal or PowerShell. If you see a list of options, you are all set. Otherwise, see
    either this [PIP Documentation](https://pip.pypa.io/en/stable/installation/) or [this help page](https://www.geeksforgeeks.org/how-to-install-pip-on-windows/)
2. Open a terminal window and `cd` into the directory where you want to install LeafMachine2.
3. Clone the LeafMachine2 repository from GitHub by running `git clone https://github.com/Gene-Weaver/LeafMachine2.git` in the terminal.
4. Move into the LeafMachine2 directory by running `cd LeafMachine2` in the terminal.
5. To run LeafMachine2 we need to install its dependencies inside of a python virtual environmnet. Follow the instructions below for your operating system. 

### About Python Virtual Environments
A virtual environment is a tool to keep the dependencies required by different projects in separate places, by creating isolated python virtual environments for them. This avoids any conflicts between the packages that you have installed for different projects. It makes it easier to maintain different versions of packages for different projects.

For more information about virtual environments, please see [Creation of virtual environments](https://docs.python.org/3/library/venv.html)

> We include `requirements.txt` files in the `LeafMachine2/requirements/` folder. If you experience version incompatability following the instructions below, please refer to `LeafMachine2/requirements/requirements_all.txt` for an exhaustive list of packages and versions that are officially supported. 

---

### Installation - Ubuntu 20.04

#### Virtual Environment

1. Still inside the LeafMachine2 directory, show that a venv is currently not active 
    <pre><code class="language-python">which python</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
2. Then create the virtual environment (venv_LM2 is the name of our new virtual environment)  
    <pre><code class="language-python">python3 -m venv venv_LM2</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. Activate the virtual environment  
    <pre><code class="language-python">source ./venv_LM2/bin/activate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Confirm that the venv is active (should be different from step 1)  
    <pre><code class="language-python">which python</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
5. If you want to exit the venv, deactivate the venv using  
    <pre><code class="language-python">deactivate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

#### Installing Packages

1. Install the required dependencies to use LeafMachine2 
    - With the venv active, install wheel 
        <pre><code class="language-python">python3 -m pip install wheel</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet"></button>
    - Update pip
        <pre><code class="language-python">python3 -m pip install --upgrade pip setuptools</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet"></button>
    - Allow bash file to run
        <pre><code class="language-python">chmod +x install_dependencies.sh</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet"></button>
    - Then install dependencies
        <pre><code class="language-python">bash install_dependencies.sh</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet"></button>
    - If you encounter an error, you can try running the following install command instead
        <pre><code class="language-python">pip install astropy asttokens beautifulsoup4 cachetools certifi cloudpickle colorama contourpy cycler Cython dask dataclasses debugpy decorator einops entrypoints executing fairscale filelock fonttools fsspec future fuzzywuzzy fvcore geojson gitdb GitPython grpcio huggingface-hub hydra-core idna imageio imagesize imutils iopath ipykernel ipython jedi joblib jsonpatch jsonpointer jupyter_client jupyter_core kiwisolver labelbox Levenshtein locket Markdown MarkupSafe matplotlib matplotlib-inline mypy-extensions ndjson nest-asyncio networkx numpy oauthlib omegaconf packaging pandas parso partd pathspec pathtools pickleshare Pillow platformdirs pooch portalocker promise prompt-toolkit protobuf psutil pure-eval py-cpuinfo pyamg pyasn1 pyasn1-modules pydantic pydot pyefd pyerfa pyGeoTile Pygments pyparsing pyproj python-dateutil python-Levenshtein pytz PyWavelets PyYAML pyzenodo3 pyzmq QtPy rapidfuzz reportlab requests requests-oauthlib rsa scikit-image scikit-learn scipy seaborn sentry-sdk setproctitle Shapely shortuuid SimpleITK six scikit-learn smmap soupsieve stack-data tabulate termcolor threadpoolctl tifffile timm tomli toolz tornado tqdm traitlets typing_extensions urllib3 wandb wcwidth websocket-client Werkzeug wget yacs zenodo-get</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet"></button>
2. Upgrade numpy 
    <pre><code class="language-python">pip install numpy -U</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. Install COCO annotation tools
    <pre><code class="language-python">pip install git+https://github.com/waspinator/pycococreator.git@fba8f4098f3c7aaa05fe119dc93bbe4063afdab8#egg=pycococreatortools</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Install COCO annotation tools
    <pre><code class="language-python">pip install pycocotools==2.0.5</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
5. We need a special version of Open CV
    <pre><code class="language-python">pip install opencv-contrib-python==4.7.0.68</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
6. The LeafMachine2 machine learning algorithm requires PyTorch version 1.11 for CUDA version 11.3. If your computer does not have a GPU, then use the CPU version and the CUDA version is not applicable. PyTorch is large and will take a bit to install.
    - WITH GPU 
    <pre><code class="language-python">pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
7. Install ViT for PyTorch. ViT is used for segmenting labels and rulers.
    <pre><code class="language-python">pip install vit-pytorch==0.37.1</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

> If you need help, please submit an inquiry in the form at [LeafMachine.org](https://LeafMachine.org/)

---

### Installation - Windows 10+

#### Virtual Environment

1. Still inside the LeafMachine2 directory, show that a venv is currently not active 
    <pre><code class="language-python">python --version</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
2. Then create the virtual environment (venv_LM2 is the name of our new virtual environment)  
    <pre><code class="language-python">python3 -m venv venv_LM2</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. Activate the virtual environment  
    <pre><code class="language-python">.\venv_LM2\Scripts\activate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Confirm that the venv is active (should be different from step 1)  
    <pre><code class="language-python">python --version</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
5. If you want to exit the venv, deactivate the venv using  
    <pre><code class="language-python">deactivate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

#### Installing Packages

1. Install the required dependencies to use LeafMachine2  
    <pre><code class="language-python">pip install astropy asttokens beautifulsoup4 cachetools certifi cloudpickle colorama contourpy cycler Cython dask dataclasses debugpy decorator einops entrypoints executing fairscale filelock fonttools fsspec future fuzzywuzzy fvcore geojson gitdb GitPython grpcio huggingface-hub hydra-core idna imageio imagesize imutils iopath ipykernel ipython jedi joblib jsonpatch jsonpointer jupyter_client jupyter_core kiwisolver labelbox Levenshtein locket Markdown MarkupSafe matplotlib matplotlib-inline mypy-extensions ndjson nest-asyncio networkx numpy oauthlib omegaconf packaging pandas parso partd pathspec pathtools pickleshare Pillow platformdirs pooch portalocker promise prompt-toolkit protobuf psutil pure-eval py-cpuinfo pyamg pyasn1 pyasn1-modules pydantic pydot pyefd pyerfa pyGeoTile Pygments pyparsing pyproj python-dateutil python-Levenshtein pytz PyWavelets pywin32 PyYAML pyzenodo3 pyzmq QtPy rapidfuzz reportlab requests requests-oauthlib rsa scikit-image scikit-learn scipy seaborn sentry-sdk setproctitle Shapely shortuuid SimpleITK six sklearn smmap soupsieve stack-data tabulate tensorboard tensorboard-data-server tensorboard-plugin-wit termcolor threadpoolctl tifffile timm tomli toolz tornado tqdm traitlets typing_extensions urllib3 wandb wcwidth websocket-client Werkzeug wget yacs zenodo-get</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
2. Upgrade numpy  
    <pre><code class="language-python">pip install numpy -U</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
3. Install COCO annotation tools
    <pre><code class="language-python">pip install git+https://github.com/waspinator/pycococreator.git@fba8f4098f3c7aaa05fe119dc93bbe4063afdab8#egg=pycococreatortools</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
4. Install COCO annotation tools
    <pre><code class="language-python">pip install pycocotools==2.0.5</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
5. We need a special version of Open CV
    <pre><code class="language-python">pip install opencv-contrib-python==4.7.0.68</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
6. The LeafMachine2 machine learning algorithm requires PyTorch version 1.11 for CUDA version 11.3. If your computer does not have a GPU, then use the CPU version and the CUDA version is not applicable. PyTorch is large and will take a bit to install.
    - WITH GPU 
    <pre><code class="language-python">pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
7. Install ViT for PyTorch. ViT is used for segmenting labels and rulers.
    <pre><code class="language-python">pip install vit-pytorch==0.37.1</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>

> If you need help, please submit an inquiry in the form at [LeafMachine.org](https://LeafMachine.org/)

---

### Troubleshooting CUDA

- If your system already has another version of CUDA (e.g., CUDA 11.7) then it can be complicated to switch to CUDA 11.3. 
- The simplest solution is to install pytorch with CPU only, avoiding the CUDA problem entirely, but that is not recommended given that 
LeafMachine2 is designed to use GPUs. We have not tested LeafMachine2 on systems that lack GPUs.
- Alternatively, you can install the [latest pytorch release](https://pytorch.org/get-started/locally/) for your specific system, either using the cpu only version `pip3 install torch`, `pip3 install torchvision`, `pip3 install torchaudio` or by matching the pythorch version to your CUDA version.
- We have not validated CUDA 11.6 or CUDA 11.7, but our code is likely to work with them too. If you have success with other versions of CUDA/pytorch, let us know and we will update our instructions. 

---

## Testing the LeafMachine2 Installation
Assuming no errors with the previous instructions, we can now test LeafMachine2 and make sure that everything is funtional. There are two options for working with LeafMachine2. 

If you plan to mostly use it with default setting, then working from the terminal (or PowerShell) will be fine. 

We recommend using [Microsoft Visual Studio Code](https://code.visualstudio.com/download) (or another IDE e.g. Sublime, PyCharm...) since LeafMachine2 relies on configuration files. Editing config files in an IDE helps reduce syntax mistakes, plus it's just easier to work with. 

If you plan on changing lots of settings, we recommend running LeafMachine2 in debug mode (in the IDE, LM2 does not have a debug mode). It won't hurt anything, and it will show you more information about any errors. We log and handle errors, but not all errors fail gracefully and inevitably there will be some novel errors as we continue to refine LeafMachine2. 

1. When you first run LeafMachine2, it must download all of the machine learning files that are not stored on GitHub. You will see white text in the console indicating progress or failure. 
    - If LM2 displays an error while downloading the networks, first try to run `test.py` described below.
        * Note: If you attempt to download the ML networks multiple times, you may see error messages, but often the networks were successfully downloaded anyway. 
    - If you still have trouble, submit your log file (or all files) to [LeafMachine2 Logs](https://docs.google.com/forms/d/e/1FAIpQLSdTOBBt4LNQBy9NPzxGFXGywwJoc52YGcVTY9dO6VKM1iz0Fw/viewform?usp=sf_link) for assistance. 
2. In the future, we will release updated ML networks. This system will also handle updates. 
3. We have included demo images and a test file to make sure that all components are functional. The demo images are in `../LeafMachine2/demo/demo_images/` and all output from the test will go to `../LeafMachine2/demo/demo_output/`. After a successful test, you can delete the contents of `demo_output/` if you want, or keep it as a reference. 
    - The demo will run all components of LeafMachine2, including some features that you may not want. For example, the demo will test skeletonization of the labels and rulers, so in the folder `../LeafMachine2/demo/demo_output/test_run/Cropped_Images/By_Class/label_binary` you will see strange looking labels, but that is intentional. 
4. Running the installation test (from the terminal)
    - make sure that the virtual environment is active and that the virtual environment is located inside the LM2 home directory: `LeafMachine2/venv_LM2`
    - cd into the LeafMachine2 directory
    <pre><code class="language-python">python test.py</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet"></button>
    - You should see some blue text, and then a lot of information in the console. 
    - If the run completes (usually after ~5 minutes) and you see a :grinning: then you should be all set!
    - Otherwise, double check that you followed each step and reach out by submitting an inquiry in the form at [LeafMachine.org](https://LeafMachine.org/)

## Using LeafMachine2

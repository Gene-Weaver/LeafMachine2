<!-- Include Clipboard.js -->
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.8/clipboard.min.js"></script>
<script>
  var clipboard = new ClipboardJS('.btn');
</script>
```
[![LeafMachine2](https://LeafMachine.org/img/LM2_Desktop_Narrow2.jpg "LeafMachine2")](https://LeafMachine.org/)

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

> We include `requirements.txt` files in the `LeafMachine2/requirements/` folder. If you experience version incompatability following the
instructions below, please refer to `LeafMachine2/requirements/requirements_all.txt` for an exhaustive list of packages and versions that
are officially supported. 

---

### Installation - Ubuntu 20.04

#### Virtual Environment

1. Still inside the LeafMachine2 directory, show that a venv is currently not active: 
    <pre><code class="language-python">which python</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
2. Then create the virtual environment (venv_fp is the name of our new virtual environment):  
    <pre><code class="language-python">python3 -m venv venv_fp</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
3. Activate the virtual environment:  
    <pre><code class="language-python">source ./venv_fp/bin/activate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
4. Confirm that the venv is active (should be different from step 1):  
    <pre><code class="language-python">which python</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
5. If you want to exit the venv, deactivate the venv using:  
    <pre><code class="language-python">deactivate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>

#### Installing Packages

1. Install the required dependencies to use LeafMachine2: 
    - With the venv active, run: 
        <pre><code class="language-python">chmod +x install_dependencies</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
    - Then:
        <pre><code class="language-python">bash install_dependencies.sh</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
    - If you encounter an error, you can try running the following install command instead
        <pre><code class="language-python">pip install astropy asttokens beautifulsoup4 cachetools certifi cloudpickle colorama contourpy cycler Cython dask dataclasses debugpy decorator einops entrypoints executing fairscale filelock fonttools fsspec future fuzzywuzzy fvcore geojson gitdb GitPython grpcio huggingface-hub hydra-core idna imageio imagesize imutils iopath ipykernel ipython jedi joblib jsonpatch jsonpointer jupyter_client jupyter_core kiwisolver labelbox Levenshtein locket Markdown MarkupSafe matplotlib matplotlib-inline mypy-extensions ndjson nest-asyncio networkx numpy oauthlib omegaconf packaging pandas parso partd pathspec pathtools pickleshare Pillow platformdirs pooch portalocker promise prompt-toolkit protobuf psutil pure-eval py-cpuinfo pyamg pyasn1 pyasn1-modules pydantic pydot pyefd pyerfa pyGeoTile Pygments pyparsing pyproj python-dateutil python-Levenshtein pytz PyWavelets pywin32 PyYAML pyzenodo3 pyzmq QtPy rapidfuzz reportlab requests requests-oauthlib rsa scikit-image scikit-learn scipy seaborn sentry-sdk setproctitle Shapely shortuuid SimpleITK six sklearn smmap soupsieve stack-data tabulate tensorboard tensorboard-data-server tensorboard-plugin-wit termcolor threadpoolctl tifffile timm tomli toolz tornado tqdm traitlets typing_extensions urllib3 wandb wcwidth websocket-client Werkzeug wget yacs zenodo-get</code></pre>
        <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
2. Upgrade numpy: 
    <pre><code class="language-python">pip install numpy -U</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
3. Install ViT for PyTorch. ViT is used for segmenting labels and rulers.
    <pre><code class="language-python">pip install vit-pytorch==0.37.1</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
4. Install COCO annotation tools
    <pre><code class="language-python">pip install git+https://github.com/waspinator/pycococreator.git@fba8f4098f3c7aaa05fe119dc93bbe4063afdab8#egg=pycococreatortools</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
5. Install COCO annotation tools
    <pre><code class="language-python">pip install pycocotools==2.0.5</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
6. We need a special version of Open CV:
    <pre><code class="language-python">pip install opencv-contrib-python==4.7.0.68</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
7. The LeafMachine2 machine learning algorithm requires PyTorch version 1.11 for CUDA version 11.3. If your computer does not have a GPU, then use the CPU version and the CUDA version is not applicable. PyTorch is large and will take a bit to install.
    - WITH GPU: 
    <pre><code class="language-python">pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
8. Test the installation:  
    <pre><code class="language-python">python3 test.py</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
    - If you see large red messages, then the installation was not successful. The error message will be below the large red boxes, providing information on how to correct the installation error. If you need more help, please submit an inquiry in the form at [LeafMachine2.org](https://LeafMachine2.org/)
9. You can view the test output in `LeafMachine2/demo/demo_output/`

---

### Installation - Windows 10+

#### Virtual Environment

1. Still inside the LeafMachine2 directory, show that a venv is currently not active: 
    <pre><code class="language-python">python --version</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
2. Then create the virtual environment (venv_fp is the name of our new virtual environment):  
    <pre><code class="language-python">python3 -m venv venv_fp</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
3. Activate the virtual environment:  
    <pre><code class="language-python">.\venv_fp\Scripts\activate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
4. Confirm that the venv is active (should be different from step 1):  
    <pre><code class="language-python">python --version</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
5. If you want to exit the venv, deactivate the venv using:  
    <pre><code class="language-python">deactivate</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>

#### Installing Packages

1. Install the required dependencies to use LeafMachine2:  
    <pre><code class="language-python">pip install astropy asttokens beautifulsoup4 cachetools certifi cloudpickle colorama contourpy cycler Cython dask dataclasses debugpy decorator einops entrypoints executing fairscale filelock fonttools fsspec future fuzzywuzzy fvcore geojson gitdb GitPython grpcio huggingface-hub hydra-core idna imageio imagesize imutils iopath ipykernel ipython jedi joblib jsonpatch jsonpointer jupyter_client jupyter_core kiwisolver labelbox Levenshtein locket Markdown MarkupSafe matplotlib matplotlib-inline mypy-extensions ndjson nest-asyncio networkx numpy oauthlib omegaconf packaging pandas parso partd pathspec pathtools pickleshare Pillow platformdirs pooch portalocker promise prompt-toolkit protobuf psutil pure-eval py-cpuinfo pyamg pyasn1 pyasn1-modules pydantic pydot pyefd pyerfa pyGeoTile Pygments pyparsing pyproj python-dateutil python-Levenshtein pytz PyWavelets pywin32 PyYAML pyzenodo3 pyzmq QtPy rapidfuzz reportlab requests requests-oauthlib rsa scikit-image scikit-learn scipy seaborn sentry-sdk setproctitle Shapely shortuuid SimpleITK six sklearn smmap soupsieve stack-data tabulate tensorboard tensorboard-data-server tensorboard-plugin-wit termcolor threadpoolctl tifffile timm tomli toolz tornado tqdm traitlets typing_extensions urllib3 wandb wcwidth websocket-client Werkzeug wget yacs zenodo-get</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
2. Upgrade numpy:  
    <pre><code class="language-python">pip install numpy -U</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
3. Install ViT for PyTorch. ViT is used for segmenting labels and rulers.
    <pre><code class="language-python">pip install vit-pytorch==0.37.1</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
4. Install COCO annotation tools
    <pre><code class="language-python">pip install git+https://github.com/waspinator/pycococreator.git@fba8f4098f3c7aaa05fe119dc93bbe4063afdab8#egg=pycococreatortools</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
5. Install COCO annotation tools
    <pre><code class="language-python">pip install pycocotools==2.0.5</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
6. We need a special version of Open CV:
    <pre><code class="language-python">pip install opencv-contrib-python==4.7.0.68</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
7. The LeafMachine2 machine learning algorithm requires PyTorch version 1.11 for CUDA version 11.3. If your computer does not have a GPU, then use the CPU version and the CUDA version is not applicable. PyTorch is large and will take a bit to install.
    - WITH GPU: 
    <pre><code class="language-python">pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
8. Test the installation. test.py will test the QR code builder and image processing portions of LeafMachine2:  
    <pre><code class="language-python">python test.py</code></pre>
    <button class="btn" data-clipboard-target="#code-snippet">Copy to Clipboard</button>
    - If you see large red messages, then the installation was not successful. The rror message will be below the large red boxes, providing information on how to correct the installation error. If you need more help, please submit an inquiry in the form at [LeafMachine2.org](https://LeafMachine2.org/)
9. You can view the test output in at [this Zenodo repo]https://zenodo.org/record/7764379

---

### Troubleshooting CUDA

- If your system already has another version of CUDA (e.g., CUDA 11.7) then it can be complicated to switch to CUDA 11.3. 
- The simplest solution is to install pytorch with CPU only, avoiding the CUDA problem entirely, but that is not recommended given that 
LeafMachine2 is designed to use GPUs. We have not tested LeafMachine2 on systems that lack GPUs.
- Alternatively, you can install the [latest pytorch release]https://pytorch.org/get-started/locally/ for your specific system, either using the cpu only version `pip3 install torch`, `pip3 install torchvision`, `pip3 install torchaudio` or by matching the pythorch version to your CUDA version.
- We have not validated CUDA 11.6 or CUDA 11.7, but our code is likely to work with them too. If you have success with other versions of CUDA/pytorch, let us know and we will update our instructions. 

## Using LeafMachine2

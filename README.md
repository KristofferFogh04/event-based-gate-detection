# event-based-gate-detection

This is the code for the master thesis titled xxx

## Installation
First set up an [Anaconda](https://www.anaconda.com/) environment:

    conda create -n event-gate python=3.7  
    conda activate event-gate

Then clone the repository and install the dependencies with pip

    git clone https://github.com/KristofferFogh04/event-based-gate-detection.git
    cd event-based-gate-detection/
    pip install -r requirements.txt
    
Install VCS tool and clone external dependencies with:

    sudo apt-get install python-vcstool
    vcs-import < event-based-gate-detection/dependencies.yaml

To setup the external library SparseConvNet

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch # See https://pytorch.org/get-started/locally/
    bash SparseConvNet/develop.sh

### CPP Bindings
To build the cpp bindings for rpg_asynet, use the following commands:

    pip install rpg_asynet/event_representation_tool/
    conda install -c conda-forge pybind11
    pip install rpg_asynet/async_sparse_py/


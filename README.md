# MattPort


# Setup the environment

```
# Clone the repo
git clone --recurse-submodules git@github.com:ethanweber/mattport.git

# Create the python environment
conda create --name mattport python=3.8.13
conda activate mattport
pip install -r environment/requirements.txt

# Install helper library
cd external/goat
python setup.py develop
# then go back, i.e., `cd /path/to/mattport`

# Install mattport as a library
python setup.py develop

# Running the test cases
pytest tests
```

# Getting the data

Download the original NeRF dataset and put it in the following format.

```
- data/
    - fern/
    - lego/
```

# Training a model

```
python scripts/run_train_nerf.py
```

# Setting up Jupyter

```
python -m jupyter lab build
bash environments/run_jupyter.sh
```
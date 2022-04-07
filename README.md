# MattPort


# Setup the environment

```
# Clone the repo
git clone --recurse-submodules git@github.com:ethanweber/mattport.git

# Create the python environment
conda create --name mattport python=3.8
conda activate mattport
pip install -r environment/requirements.txt

# Install helper library
cd external/goat
python setup.py develop
# then go back, i.e., `cd /path/to/mattport`

# For using with Jupyter
python -m ipykernel install --user --name mattport --display-name "mattport"

# Install mattport as a library
python setup.py develop

# Running the test cases
pytest tests
```

# Training a model

```
python scripts/run_train_nerf.py
```
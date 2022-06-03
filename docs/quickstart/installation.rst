.. _installation:

Installation
=================

Clone the repo::

   git clone --recurse-submodules git@github.com:ethanweber/pyrad.git

Create the python environment::

   conda create --name pyrad python=3.8.13
   conda activate pyrad
   pip install -r environment/requirements.txt

Install pyrad as a library::

   pip install -e .

Install library with CUDA support. Change setup.py to ``USE_CUDA = True`` and then::

   python setup.py develop

Running the test cases::

   pytest tests
   Field Modules

Getting the data
------------------

Download the original NeRF dataset and put it in the following format.::


   - data/
      - blender/
         - fern/
         - lego/

Training a model
------------------

Run with default config::

   python scripts/run_train.py

Run with config changes::

   python scripts/run_train.py machine_config.num_gpus=1
   python scripts/run_train.py data.dataset.downscale_factor=1

Run with different datasets::

   python scripts/run_train.py data/dataset=blender_lego
   python scripts/run_train.py data/dataset=friends_TBBT-big_living_room

Run with different datasets and config changes::

   python scripts/run_train.py data/dataset=friends_TBBT-big_living_room graph.network.far_plane=14

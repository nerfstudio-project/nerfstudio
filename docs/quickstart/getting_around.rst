.. _getting_around:

Getting around the codebase
=====================================

The entry point for training starts at ``scripts/run_train.py``, which spawns instances of our ``Trainer()`` class (in ``nerf/trainer.py``). The ``Trainer()`` is responsible for setting up the datasets and NeRF graph depending on the config specified. If you are planning on just using our codebase to build a new NeRF method or use an existing implementation, we've abstracted away the training routine in these two files and chances are you will not need to touch them.

The NeRF graph definitions can be found in ``nerf/graph/``. Each implementation of NeRF is definined in its own file. For instance, ``nerf/graph/instant_ngp.py`` contains populates the ``NGPGraph()`` class with all of the appropriate fields, colliders, and misc. modules.

* Fields (``nerf/fields/``): composed of field modules (``nerf/field_modules/``) and represents the radiance field of the NeRF.
* Misc. Modules (``nerf/misc_modules``- TODO(maybe move to misc_modules? better organization)): any remaining module in the NeRF (e.g. renderers, samplers, losses, and metrics).

To implement any pre-existing NeRF that we have not yet implemented under `nerf/graph/`, create a new graph structure by using provided modules or any new module you define. Then create an associated config making sure ``__target__`` points to your NeRF class (see [here](./configs/README.md) for more info on how to create the config). Then run training as described above.
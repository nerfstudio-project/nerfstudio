
|

.. image:: _static/imgs/logo.png
  :width: 400
  :align: center
  :alt: nerfactory

Neural Volumetric Rendering
"""""""""""""""""""""""""""""

All-in-one repository for state-of-the-art NeRFs.

nerfactory provides a simple API that allows for a seamless and simplified end-to-end process of creating, training, and testing NeRFs.
The library supports a more interpretable implementation of NeRFs by modularizing each component.
With more modular NeRFs, not only does your code become far more user-friendly, but using this framework also makes it easier for the community to build upon your implementation. 

It's as simple as plug and play with nerfactory!

If you have further questions or want a feature that is not yet supported, please do not hesitate to reach out to the Plenoptix team.

|


Contents
"""""""""""""""""""""""""""""

.. toctree::
   :hidden:
   :caption: Getting Started

   Quickstart<tutorials/quickstart_index>
   tutorials/data_index
   tutorials/creating_graphs.md
   tutorials/system_overview.md

.. toctree::
   :hidden:
   :caption: Guides

   
   models/index.rst
   notebooks/index

.. toctree::
   :hidden:
   :caption: Tooling

   tooling/index
   
.. toctree::
   :hidden:
   :caption: Reference

   Contributing<reference/contributing.md>
   reference/api/index



This documentation is organized into 3 parts:

* **Tutorials**: a great place to start if you are new to nerfactory. Contains a quick tour, installation, and an overview of the core structures that will allow you to get up and running with nerfactory |:metal:|
* **Tooling**: describes all of the additional support we provide to help you debug and improve your NeRFs (e.g logging, performance profiling)
* **Reference**: describes each class and function. Develop a better understanding of the core of our technology and terminology. This section includes descriptions of each module and component in the codebase.


|


Supported Models
"""""""""""""""""""""""""""""


+------------------------+------------------------------+----------------------------+
|                        |  PSNR                        | Rays / sec.                |
+========================+==============================+============================+
| NeRF                   |  |:hourglass_flowing_sand:|  | |:hourglass_flowing_sand:| |
+------------------------+------------------------------+----------------------------+
| Instant-NGP            |  |:hourglass_flowing_sand:|  | |:hourglass_flowing_sand:| |
+------------------------+------------------------------+----------------------------+


|


Indices and Tables
"""""""""""""""""""""""""""""

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

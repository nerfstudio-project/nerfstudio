Using the viewer
================

The nerfstudio web-based viewer makes it easy to view training in real-time, and to create content videos from your trained models 🌟!

In the tutorial below, we walk you through how you can turn a simple capture into a 3D video 📸 🎬

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/nSFsugarWzk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

For specific sections, click the links below to navigate to the associated portion of the video.

Getting started
"""""""""""""""

* `Hello from nerfstudio <https://youtu.be/nSFsugarWzk?t=0>`_
* `Preprocess your video <https://youtu.be/nSFsugarWzk?t=13>`_
* `Launching training and viewer <https://youtu.be/nSFsugarWzk?t=27>`_

Viewer basics
"""""""""""""""

* `Viewer scene introduction <https://youtu.be/nSFsugarWzk?t=63>`_
* `Moving around in viewer <https://youtu.be/nSFsugarWzk?t=80>`_
* `Overview of Controls Panel - train speed/output options <https://youtu.be/nSFsugarWzk?t=98>`_
* `Overview of Scene Panel - toggle visibility <https://youtu.be/nSFsugarWzk?t=115>`_

Creating camera trajectories
""""""""""""""""""""""""""""

* `Creating a custom camera path <https://youtu.be/nSFsugarWzk?t=136>`_
* `Camera spline options - cycle, speed, smoothness <https://youtu.be/nSFsugarWzk?t=158>`_
* `Camera options - move, add, view <https://youtu.be/nSFsugarWzk?t=177>`_

Rendering a video
"""""""""""""""""

* `Preview camera trajectory <https://youtu.be/nSFsugarWzk?t=206>`_
* `How to render final video <https://youtu.be/nSFsugarWzk?t=227>`_

|

..  seealso:: 

  For a more in-depth developer overview on how to hack with the viewer, see our `developer guide </docs/_build/html/developer_guides/viewer/index.html>`_


Local vs. remote compute
""""""""""""""""""""""""""

Training on a local machine
---------------------------

You should be able to click the link obtained while running a script to open the viewer in your browser.

Training on a remote machine
----------------------------

If you are training on a remote maching, the viewer will still let you view your trainings. 
You will need to forward the port that the viewer is running on.

For example, if you are running the viewer on port 7007, you will need to forward that port to your local machine. 
You can (without needing to modify router port settings) simply do this by opening a new terminal window and sshing into the remote machine with the following command:

..  code-block:: bash

    ssh -L 7007:localhost:7007 <username>@<remote-machine-ip>


..  admonition:: Note

    So long as you don't close this terminal window with this specific active ssh connection, the port will remain open for all your other ssh sessions. 
    
    For example, if you do this in a new terminal window, any existing ssh sessions (terminal windows, VSCode remote connection, etc) will still be able to access the port, but if you close this terminal window, the port will be closed.


You can now simply open the link (same ones shown in image above) in your browser on your local machine and it should connect!

..  warning::
    If the port is being used, you will need to switch the port using the `--viewer.websocket-port` flag tied to the model subcommand.
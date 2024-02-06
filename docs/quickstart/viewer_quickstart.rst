Using the viewer
================

The nerfstudio web-based viewer makes it easy to view training in real-time, and to create content videos from your trained models 🌟!

Connecting to the viewer
^^^^^^^^^^^^^^^^^^^^^^^^

The nerfstudio viewer is launched automatically with each :code:`ns-train` training run! It can also be run separately with :code:`ns-viewer`.
To access a viewer, we need to enter the server's address and port in a web browser.

Accessing on a local machine
""""""""""""""""""""""""""""

You should be able to click the link obtained while running a script to open the viewer in your browser. This should typically look something like :code:`http://localhost:7007`.

Accessing over an SSH connection
""""""""""""""""""""""""""""""""

If you are training on a remote machine, the viewer will still let you view your trainings. You will need to forward traffic from your viewing host to the listening port on the host running ns-train (7007 by default). You can achieve this by securely tunneling traffic on the specified port through an ssh session. In a terminal window on the viewing host, issue the following command:

..  code-block:: bash

   ssh -L 7007:127.0.0.1:7007 <username>@<training-host-ip>


..  admonition:: Note

    You can now simply open the link (same one shown in image above) in your browser on your local machine and it should connect!. So long as you don't close this terminal window with this specific active ssh connection, the port will remain open for all your other ssh sessions.

    For example, if you do this in a new terminal window, any existing ssh sessions (terminal windows, VSCode remote connection, etc) will still be able to access the port, but if you close this terminal window, the port will be closed.


..  warning::
    If the port is being used, you will need to switch the port using the `--viewer.websocket-port` flag tied to the model subcommand.


Accessing via a share link
""""""""""""""""""""""""""

To connect to remote machines without requiring an SSH connection, we also support creating share URLs.
This can be generated from the "Share:" icon in the GUI, or from the CLI by specifying the :code:`--viewer.make-share-url True` argument.
This will create a publically accessible link that you can share with others to view your training progress.
This is useful for sharing training progress with collaborators, or for viewing training progress on a remote machine without requiring an SSH connection.
However, it will introduce some latency in the viewer.


..  seealso::

  For a more in-depth developer overview on how to hack with the viewer, see our `developer guide </docs/_build/html/developer_guides/viewer/index.html>`_


Legacy viewer tutorial video
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the tutorial video below, we walk you through how you can turn a simple capture into a 3D video 📸 🎬

This video is for the legacy viewer, which was the default in :code:`nerfstudio<1.0`. It has some visual differences from the current viewer. However, the core functionality is the same.
The legacy viewer can still be enabled with the :code:`--vis viewer_legacy` option, for example, :code:`ns-train nerfacto --vis viewer_legacy`.

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

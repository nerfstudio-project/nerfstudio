# Python Kernel and Client Viewer App communication

> The purpose of this document is to explain how to communicate from Python with the Client Viewer app. We will eventually move this into the read the docs.

- Python Kernel (nerfstudio code)
- Bridge Server
- Client Viewer App

We have two types of components that we want to keep state updated in both locations.

- Widgets
  - The widgets are used to keep track of the
- SceneNode
  - The scene nodes are used to represent the three.js objects. The properties relevant to these objects are the following: `"object", "transform", "properties"`.

# Checklist

- [ ] Currently using request-reply (REQ, REP with zmq). I.e., Python Kernel -> Bridge Server <-> Client Viewer App. We want a way to update the Python Kernel when the Bridge Server is updated. This requries some form of binding with callbacks. When the Bridge Server is updated, we want to update the binded Python variable. We can take inspiration from [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Basics.html).

- [ ]

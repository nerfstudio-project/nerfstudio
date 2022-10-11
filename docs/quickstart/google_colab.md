# Google Colab

Here we briefly explain how to get our repo working in a Google Colab.

## Viewer suport

To get the viewer working, you have to do some steps outside the notebook.

#### Installing on Mac M1

```
git clone
cd nerfstudio
pip install torch==1.12.1
pip install --upgrade git+https://github.com/pytorch/functorch@v0.2.1
pip install -e .
```

#### Forwarding port

```
pip install -e ".[viewer]"
ns-bridge-server --use-ngrok
```

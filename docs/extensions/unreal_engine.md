# Exporting to Unreal Engine

 ```{image} imgs/desolation_unreal.png
 :width: 800 
 :align: center 
 :alt: NeRF in Unreal Engine 
 ``` 

## Overview

NeRFStudio models can be used in Unreal Engine if they are converted to an NVOL file. NVOL is a new standard file format to store NeRFs in a fast and efficient way. NVOL files can be obtained from NeRFStudio checkpoints files (.ckpt) using the [Volinga Suite](https://volinga.ai/).

:::{admonition} Note
:class: warning
Volinga Suite is not yet publicly available. If you are interested in converting your models to NVOL, sign up for the beta at [volinga.ai](https://volinga.ai).
:::


## Exporting your model to NVOL
Currently NVOL file only supports Volinga model (which is based on nerfacto). You can train your model using the following command:

```bash
ns-train volinga --data /path/to/your/data --vis viewer
```

Once the training is done, you can find your checkpoint file in the `outputs/path-to-your-data/volinga` folder. Then, you can drag it to Volinga Suite to export it to NVOL.

 ```{image} imgs/export_nvol.png 
 :width: 400 
 :align: center 
 :alt: Nvol export in Voliga Suite 
 ``` 

Once the NVOL is ready, you can download it and use it in Unreal Engine.

 ```{image} imgs/nvol_ready.png 
 :width: 800 
 :align: center 
 :alt: NVOL ready to use 
 ``` 
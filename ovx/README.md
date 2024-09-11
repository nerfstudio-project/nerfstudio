# OVX training

## Introduction
The docuement will demostrate how to train 3DGS on OVX. 
Most of the procedures are adapt from https://github.com/j3soon/omni-farm-isaac. 
And you can find all the supplement resource at https://drive.google.com/drive/folders/1iLafjBqnT-NvElK4Ijn1TRH34G7agAQf?usp=drive_link.

## Setup
Setup the VPN describe in user manual. And place the `secrets` into `ovx` directory. 
```bash
source ovx/secrets/env.sh
```
Upload training data to Nucleus. In the example, I upload data to `Projects/3DGS/datasets`.

## Omnicli API
Save a job definition
```bash
bash ovx/save_job.sh nerfstudio-bash
```

Check all job definition
```bash
bash ovx/load_job.sh
```

Remove a job definition
```bash
bash ovx/remove_job.sh nerfstudio-bash
```

Submit a task
```bash
bash ovx/scripts/submit_task.sh nerfstudio-bash "/run.sh --download-src 'omniverse://$NUCLEUS_HOSTNAME/Projects/3DGS/datasets/20240822_174147.zip' --download-dest '/app/20240822_174147.zip' --upload-src '/app/20240822_174147_processed.zip' --upload-dest 'omniverse://$NUCLEUS_HOSTNAME/Projects/3DGS/nerfstudio/20240822_174147_processed.zip' 'bash ovx/run_nerfstudio.sh /app/20240822_174147.zip /app/20240822_174147_processed.zip colmap'" "nerfstudio training"
```

## Submit a Training Task
I also write a python script for easier task submition.  
Submit one task
```bash
python -m ovx.submit_task --id 20240822_174147
```


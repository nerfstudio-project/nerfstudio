# Nerfstudio paper

> Below we outline the steps to run the experiments from the paper.

1. Download the MIP-NeRF 360 data and the Nerfstudio Dataset.

```bash
ns-download-data mipnerf360
```

```bash
ns-download-data nerfstudio --capture-name nerfstudio-dataset
```

2. Run Nerfacto on the MIP-NeRF 360 data.

```bash
python projects/nerfstudio_paper/benchmark_nerfstudio_paper.py nerfacto-on-mipnerf360 --dry-run
```

3. Run Nerfacto ablation experiments.

```bash
python projects/nerfstudio_paper/benchmark_nerfstudio_paper.py nerfacto-ablations --dry-run
```

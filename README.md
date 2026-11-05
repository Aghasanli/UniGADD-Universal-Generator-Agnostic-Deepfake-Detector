# UniGADD: Universal Generator-Agnostic Deepfake Detector

Research code for **UniGADD: Universal Generator-Agnostic Deepfake Detector**. UniGADD detects AI-generated images from both known and previously unseen generators using foundation-model features, contrastive learning, and feature-space clustering.

**Paper:** arXiv link forthcoming.

## Overview

- Frozen DINOv2 feature extraction with cached image embeddings.
- An MLP projection trained with supervised contrastive learning and K-means pseudo-labels for real images.
- Statistical outlier detection and nearest-centroid classification.
- Leave-one-generator-out and leave-N-generators-out experiments across GAN- and diffusion-based generators.

## Setup

Use Python with PyTorch and Jupyter Notebook or JupyterLab. A CUDA-capable GPU is recommended for feature extraction and training. Install the required packages:

```bash
pip install torch torchvision numpy pillow scikit-learn tqdm matplotlib jupyterlab
```

For the optional CLIP backbones, also install `open-clip-torch`. Pretrained backbone weights are downloaded on first use.

The experiments were run in Google Colab. The supplied notebook includes Colab-specific Google Drive loading; for local or server execution, remove the `google.colab` imports, replace the Drive mounting/extraction helper with local dataset loading, and update the data, cache, and output paths.

## Data and usage

1. Obtain real and generated images, arranged in a `real/` folder and one folder per generator matching the notebook's `fake_names` list (for example, `biggan_fake/`). Place image files directly inside each folder.
2. Open [`Copy_of_UniGADD.ipynb`](Copy_of_UniGADD.ipynb) in Jupyter or Colab. Configure dataset loading for your environment, then adjust the backbone selection, image limits, and training settings.
3. Run the backbone-comparison cell for the main UniGADD experiment. It contains both the required definitions and a `main()` call. Run the later leave-N-out experiment separately if needed; the final cell implements a linear-classifier baseline and redefines shared classes.

**Using Colab:** select a GPU runtime and place `real.zip` and the generator ZIP archives in `MyDrive/datasets/`. Each archive should contain its corresponding image folder. The included helper mounts Google Drive and extracts the archives.

The main experiment reports detection accuracy for seen and unseen generators and saves `backbone_results.json` in the configured output directory (`local_ex`, default: `/content/dataset_extracted/`). The leave-N-out experiment saves `leave_n_out_results.csv`. Feature embeddings are cached as `.npz` files; clear the relevant cache when changing the data or preprocessing.

Datasets must be obtained separately. Experiment settings in the notebook are configurable; consult the paper for the reported evaluation protocol.

## Citation

If you use this code in your research, please cite **UniGADD: Universal Generator-Agnostic Deepfake Detector**. Full citation details will be added after the arXiv preprint is available.

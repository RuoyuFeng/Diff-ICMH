# Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior

<div align="center">

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.22549-b31b1b.svg)](https://arxiv.org/abs/2511.22549)

</div>

## 📢 News

- **[2024-11]** 🎉 Our paper has been accepted by **NeurIPS 2025**!
- **[2024-11]** Code and pre-trained models released.

## 📝 Abstract

This repository contains the official implementation of **Diff-ICMH**, a novel image compression framework that harmonizes machine and human vision using generative priors. Our method achieves state-of-the-art performance in both perceptual quality and machine task compatibility.

## 👥 Authors

**Ruoyu Feng**<sup>1*</sup>, **Yunpeng Qi**<sup>1*</sup>, **Jinming Liu**<sup>2</sup>, **Yixin Gao**<sup>1</sup>, **Xin Li**<sup>1†</sup>, **Xin Jin**<sup>2</sup>, **Zhibo Chen**<sup>1†</sup>

<sup>1</sup>University of Science and Technology of China  
<sup>2</sup>Eastern Institute of Technology, Ningbo

<sup>*</sup>Equal contribution  
<sup>†</sup>Corresponding authors

## 🛠️ Installation

### Prerequisites
- Python 3.8
- CUDA 12.1
- PyTorch 2.4.1

### Setup Environment

```bash
# Create conda environment
conda create -n diff-icmh python=3.8
conda activate diff-icmh

# Install PyTorch and related packages
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install tb-nightly --index-url https://pypi.org/simple
pip install huggingface_hub

# Install other requirements
pip install -r requirements.txt

# Install RAM (Recognize Anything Model)
# Download from https://github.com/xinyu1205/recognize-anything
cd src/recognize-anything
pip install -e .
cd ../..
```

### Download Pre-trained Models

```bash
# Prepare SD2.1 Model Weights
mkdir -p checkpoints/sd2p1
wget https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt \
    --no-check-certificate -O checkpoints/sd2p1/v2-1_512-ema-pruned.ckpt

# Prepare RAM Model Weights
mkdir -p checkpoints/ram
wget https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth \
    --no-check-certificate -O checkpoints/ram/ram_plus_swin_large_14m.pth

# Prepare Diff-ICMH Model Weights
huggingface-cli download RuoyuFeng/Diff-ICMH --include "difficmh_models/*" --local-dir checkpoints
```

## 🚀 Quick Start

### Inference

Run image compression on the Kodak dataset:

```bash
export GPU_INFERENCE=0
# BPP_WEIGHT options: [2, 4, 8, 16, 32]
export BPP_WEIGHT=2
export FOLDER_NAME=CNscale1.0_1_1_${BPP_WEIGHT}_2_WTagGCM_bs16x1_lr0.00005_cfg7.0
export CKPT_LC=checkpoints/difficmh_models/${FOLDER_NAME}/model.ckpt
export CONTROL_MODULE_SCALE=1.0
export CFG_SCALE=5.0
export INPUT_DIR=data/kodak_subset
export OUTPUT_DIR=outputs/kodak/${FOLDER_NAME}

CUDA_VISIBLE_DEVICES=${GPU_INFERENCE} python3 inference_partition.py \
    --ckpt_lc $CKPT_LC \
    --config configs/model/diffeic.yaml \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR \
    --steps 50 \
    --device cuda \
    params.control_stage_config.params.control_model_ratio=${CONTROL_MODULE_SCALE} \
    params.preprocess_tag_config.params.enabled=True \
    params.c_cfg_scale=${CFG_SCALE}
```

### Configuration Options

- **BPP_WEIGHT**: Controls the bits-per-pixel trade-off. Available options: `[2, 4, 8, 16, 32]`
- **CONTROL_MODULE_SCALE**: Scale factor for the control module (default: `1.0`)
- **CFG_SCALE**: Classifier-free guidance scale (default: `5.0`)
- **--steps**: Number of diffusion sampling steps (default: `50`)

## 📊 Results

Our method achieves strong performance on multiple benchmarks. Please refer to our paper for detailed experimental results.

## 📄 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{fengdiff,
  title={Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior},
  author={Feng, Ruoyu and Qi, Yunpeng and Liu, Jinming and Gao, Yixin and Li, Xin and Jin, Xin and Chen, Zhibo},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

## 📧 Contact

For questions and discussions, please contact:
- Ruoyu Feng: [ustcfry@mail.ustc.edu.cn]

## 🙏 Acknowledgments

This work is mainly based on [ControlNet](https://github.com/lllyasviel/ControlNet), [RAM](https://github.com/xinyu1205/recognize-anything) and [DiffEIC](https://github.com/huai-chang/DiffEIC), thanks to their invaluable contributions.

## 📜 License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ by the Diff-ICMH Team
</div>
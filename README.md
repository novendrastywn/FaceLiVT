# FaceLiVT Series: Face Recognition using Linear Vision Transformer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-lfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-lfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-calfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-calfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-cplfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-cplfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-cfp-fp)](https://paperswithcode.com/sota/lightweight-face-recognition-on-cfp-fp?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-agedb-30)](https://paperswithcode.com/sota/lightweight-face-recognition-on-agedb-30?p=edgeface-efficient-face-recognition-model-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-ijb-b)](https://paperswithcode.com/sota/lightweight-face-recognition-on-ijb-b?p=edgeface-efficient-face-recognition-model-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-ijb-c)](https://paperswithcode.com/sota/lightweight-face-recognition-on-ijb-c?p=edgeface-efficient-face-recognition-model-for)	

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2307.01838-009d81v2.svg)](https://arxiv.org/abs/2506.10361)

Official gitlab repository for FaceLiVT Series: Face Recognition using Linear Vision Transformer.

---

## Overview

The FaceLiVT series introduces hybrid CNN–Transformer architectures with lightweight linear attention mechanisms for efficient mobile face recognition. The series currently includes two generations:

| Version | Venue | Key Contribution |
|:---|:---|:---|
| **FaceLiVTv1** | IEEE ICIP 2025 / IEEE APCCAS 2025 | Multi-Head Linear Attention (MHLA) with structural reparameterization |
| **FaceLiVTv2** | IEEE TBIOM (Under Review) | Lite MHLA with affine rescale transformation, GDConv head, unified RepMix–Lite MHLA block |

---

## FaceLiVTv1

> **FaceLiVT: Face Recognition Using Linear Vision Transformer with Structural Reparameterization for Mobile Device**
> 
> Published in *IEEE International Conference on Image Processing (ICIP) 2025* and *IEEE Asia Pacific Conference on Circuits and Systems (APCCAS) 2025*.

**Abstract**: This paper introduces FaceLiVT, a lightweight yet powerful face recognition model that integrates a hybrid Convolution Neural Network (CNN)-Transformer architecture with an innovative and lightweight Multi-Head Linear Attention (MHLA) mechanism. By combining MHLA alongside a reparameterized token mixer, FaceLiVT effectively reduces computational complexity while preserving competitive accuracy. Extensive evaluations on challenging benchmarks; including LFW, CFP-FP, AgeDB-30, IJB-B, and IJB-C; highlight its superior performance compared to state-of-the-art lightweight models. MHLA notably improves inference speed, allowing FaceLiVT to deliver high accuracy with lower latency on mobile devices. Specifically, FaceLiVT is 8.6× faster than EdgeFace, a recent hybrid CNN-Transformer model optimized for edge devices, and 21.2× faster than a pure ViT-Based model. With its balanced design, FaceLiVT offers an efficient and practical solution for real-time face recognition on resource-constrained platforms.

<img src="assets/FLiVTv1.png"/>

```bibtex
@INPROCEEDINGS{setyawan2025facelivt,
  author={Setyawan, Novendra and Sun, Chi-Chia and Hsu, Mao-Hsiu and Kuo, Wen-Kai and Hsieh, Jun-Wei},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)}, 
  title={FaceLiVT: Face Recognition Using Linear Vision Transformer with Structural Reparameterization for Mobile Device}, 
  year={2025},
  volume={},
  number={},
  pages={1720-1725},
  keywords={Performance evaluation;Computer vision;Accuracy;Face recognition;Computational modeling;Computer architecture;Benchmark testing;Transformers;Mobile;Real-time systems;Face Recognition;Vision Transformer;Multi-Head Linear Attention (MHLA);Structural Reparameterization;Lightweight Model},
  doi={10.1109/ICIP55913.2025.11084611}}
```
```bibtex
@inproceedings{setyawan2025facelivt,
  title={FaceLiVT: Energy Efficient Face Recognition with Linear Vision Transformer for Limited Resource Device},
  author={Setyawan, Novendra and Gu, Jun-Xian and Sun, Chi-Chia and Hsu, Mao-Hsiu and Kuo, Wen-Kai and Shen, Chung-An and Hsieh, Jun-Wei},
  booktitle={2025 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

### FaceLiVTv1 Model Variants

| Model | MAdds | Params |
|:---|:---:|:---:|
| `facelivt_s` | 160 | 5.05 M |
| `facelivt_m` | 386 | 9.75 M |

---

## FaceLiVTv2

> **FaceLiVTv2: An Improved Baseline Hybrid Architecture for Efficient Mobile Face Recognition**
>
> Submitted to *IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM)*.

**Abstract**: Lightweight face recognition is increasingly important for deployment on edge and mobile devices, where strict constraints on latency, memory, and energy consumption must be met alongside reliable accuracy. Although recent hybrid CNN-Transformer architectures have advanced global context modeling, striking an effective balance between recognition performance and computational efficiency remains an open challenge. In this work, we present FaceLiVTv2, an improved version of our FaceLiVT hybrid architecture designed for efficient global–local feature interaction in mobile face recognition. At its core is Lite MHLA, a lightweight global token interaction module that replaces the original multi-layer attention design with multi-head linear token projections and affine rescale transformations, reducing redundancy while preserving representational diversity across heads. We further integrate Lite MHLA into a unified RepMix block that coordinates local and global feature interactions and adopts global depthwise convolution for adaptive spatial aggregation in the embedding stage. Extensive experiments on LFW, CA-LFW, CP-LFW, CFP-FP, AgeDB-30, and IJB show that FaceLiVTv2 consistently improves the accuracy-efficiency trade-off over existing lightweight methods. Notably, FaceLiVTv2 reduces mobile inference latency by 22% relative to FaceLiVTv1, achieves speedups of up to 30.8% over GhostFaceNets on mobile devices, and delivers 20-41% latency improvements over EdgeFace and KANFace across platforms while maintaining higher recognition accuracy.

<!-- TODO: Add FaceLiVTv2 architecture figure -->
<!-- <img src="assets/FLiVTv2.png"/> -->

### Key Improvements over FaceLiVTv1

| Component | FaceLiVTv1 | FaceLiVTv2 |
|:---|:---|:---|
| **Global Token Interaction** | MHLA (2-layer MLP-style: Linear–GELU–Linear) | Lite MHLA (single linear projection per head, activation-free) |
| **Normalization** | LayerNorm | Affine Rescale Transformation (`Aff(X) = α ⊙ X + β`) |
| **Block Design** | Separate RepMix + MHLA branches | Unified RepMix–Lite MHLA block |
| **Stage Strategy** | Same mixing across all stages | Stage-specific: RepMix+FFN (stages 1-2), RepMix+LiteMHLA+FFN (stages 3-4) |
| **Embedding Head** | Global Average Pooling (GAP) | Global Depthwise Convolution (GDConv) for adaptive spatial aggregation |
| **Complexity** | `2(NrN)C` per MHLA block | `≈ N²C + ε` per Lite MHLA block |

### FaceLiVTv2 Model Variants

| Model | Channels (S1/S2/S3/S4) | MAdds (M) | Params (M) |
|:---|:---:|:---:|:---:|
| `facelivtv2_xs` | 32 / 64 / 128 / 256 | 90 | 2.9 |
| `facelivtv2_s` | 48 / 96 / 192 / 384 | 179 | 4.62 |
| `facelivtv2_m` | 56 / 112 / 224 / 448 | 258 | 7.04 |
| `facelivtv2_l` | 64 / 128 / 256 / 512 | 309 | 8.52 |



## Installation Instructions

### Step 1: Install Necessary Components

Install dependencies of Insight face repo. You can find them [here](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). Install [DALI](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/install_dali.md) as well.

#### Substep: Install PyTorch

Install PyTorch to 2.0.0 with CUDA.

### Step 2: Install TIMM

Run the following commands:

```bash
pip install timm==0.6.12
pip install pandas tabulate mxnet
```

### Step 3: Understand Configurations

There are two generations with multiple configurations in this source code:

**FaceLiVTv1:**
- `facelivt_s` : FaceLiVTv1-Small
- `facelivt_m` : FaceLiVTv1-Medium

**FaceLiVTv2:**
- `facelivtv2_xs` : FaceLiVTv2-XSmall (2.9M params, 90M MAdds)
- `facelivtv2_s` : FaceLiVTv2-Small (4.62M params, 179M MAdds)
- `facelivtv2_m` : FaceLiVTv2-Medium (7.04M params, 258M MAdds)
- `facelivtv2_l` : FaceLiVTv2-Large (8.52M params, 309M MAdds)

To see the model parameters, flops, and size on disk, run the following commands:

```bash
# FaceLiVTv1
python speed_gpu.py facelivt_s
python speed_gpu.py facelivt_m

# FaceLiVTv2
python speed_gpu.py facelivtv2_xs
python speed_gpu.py facelivtv2_s
python speed_gpu.py facelivtv2_m
python speed_gpu.py facelivtv2_l
```

---

## Inference

The following code shows how to use the model for inference:

```python
import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model

# Choose architecture: "facelivt_s", "facelivt_m" for v1
#                      "facelivtv2_xs", "facelivtv2_s", "facelivtv2_m", "facelivtv2_l" for v2
arch="facelivtv2_s"
model=get_model(arch)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

checkpoint_path=f'checkpoints/{arch}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()
path = 'checkpoints/synthface.jpeg'
aligned = align.get_aligned_face(path)
transformed_input = transform(aligned).unsqueeze(0)
embedding = model(transformed_input)
print(embedding.shape)

```

---

## Data Preparation

### Glint360K (Main Training)

Download and prepare Glint360K: place the `.rec` files in `data/glint360k`. You can find more instructions [here](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/prepare_webface42m.md).

### WebFace4M / WebFace12M (Optional)

Download and prepare WebFace4M and WebFace12M: place the `.rec` files in `data/webface4m` and `data/webface12m`.

### TinyFace (Low-Resolution Fine-tuning for FaceLiVTv2)

Download and prepare the [TinyFace](https://qmul-tinyface.github.io/) dataset for low-resolution face recognition evaluation.


---

## Training

### FaceLiVTv1

#### FaceLiVT-S

Launch the following command after setting the root path and output path in the config files:

```bash
torchrun --nproc_per_node=8 train_v2.py configs/distil_glint360k_facelivt_s_li.py
```

#### FaceLiVT-M

Launch the following command after setting the root path and output path in the config files:

```bash
torchrun --nproc_per_node=4 train_v2.py configs/distil_glint360k_facelivt_m_li.py
```

### FaceLiVTv2

Training was performed over 50 epochs with a total batch size of 1026 on three NVIDIA RTX A6000 GPUs. AdamW optimizer with polynomial decay learning rate scheduler, initial learning rate of 6e-3, and l2 regularization of 1e-2.

#### FaceLiVTv2-XS

```bash
torchrun --nproc_per_node=3 train_v2.py configs/glint360k_facelivtv2_xs.py
```

#### FaceLiVTv2-S

```bash
torchrun --nproc_per_node=3 train_v2.py configs/glint360k_facelivtv2_s.py
```

#### FaceLiVTv2-M

```bash
torchrun --nproc_per_node=3 train_v2.py configs/glint360k_facelivtv2_m.py
```

#### FaceLiVTv2-L

```bash
torchrun --nproc_per_node=3 train_v2.py configs/glint360k_facelivtv2_l.py
```

#### TinyFace Fine-tuning (FaceLiVTv2)

For low-resolution face recognition evaluation on TinyFace, fine-tune the pretrained model for 40 epochs with batch size 8, AdamW optimizer, and initial learning rate of 2e-4. No distillation or restoration is applied.

```bash
python train_tinyface.py configs/tinyface_facelivtv2_s.py --pretrained checkpoints/facelivtv2_s.pt
```

---

## Evaluation Benchmarks

| Benchmark | Type | Protocol |
|:---|:---|:---|
| LFW | Face Verification | 6,000 pairs |
| CA-LFW | Cross-Age Verification | Age variation |
| CP-LFW | Cross-Pose Verification | Pose variation |
| CFP-FP | Frontal-Profile Verification | Profile variation |
| AgeDB-30 | Age-gap Verification | 30-year age gap |
| IJB-B | Mixed Verification/Identification | 1,845 subjects |
| IJB-C | Mixed Verification/Identification | 3,531 subjects |
| TinyFace | Low-Resolution Retrieval | Rank-1/5/10 (FaceLiVTv2) |

### Cross-Platform Inference (FaceLiVTv2)

FaceLiVTv2 was evaluated across multiple platforms:

| Platform | Device | Runtime |
|:---|:---|:---|
| Mobile | iPhone 15 Pro | CoreML |
| Edge | Jetson AGX Orin | ONNX Runtime |
| CPU | Intel i5-12500 (64GB) | ONNX Runtime |
| GPU | NVIDIA RTX 5090 | ONNX Runtime |

---

## Pretrained Models

<!-- TODO: Add download links for pretrained models -->

| Model | Training Data | Loss | Checkpoint |
|:---|:---|:---|:---|
| `facelivt_s` | Glint360K | CosFace | [Download](#) |
| `facelivt_m` | Glint360K | CosFace | [Download](#) |
| `facelivtv2_xs` | Glint360K | CosFace / ArcFace | [Download](#) |
| `facelivtv2_s` | Glint360K | CosFace / ArcFace | [Download](#) |
| `facelivtv2_m` | Glint360K | CosFace / ArcFace | [Download](#) |
| `facelivtv2_l` | Glint360K | CosFace / ArcFace | [Download](#) |

---

## Acknowledgements

This work was conducted in collaboration between the Department of Electro-Optics at National Formosa University, the Department of Electrical Engineering at National Taipei University, and the College of Artificial Intelligence and Green Energy at National Yang Ming Chiao Tung University.

---

> :warning: **Note About the License:** Please refer to the `LICENSE` file in the parent directory for information about the license terms and conditions.

# EdgeFace: Efficient Face Recognition Model for Edge Devices

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-lfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-lfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-calfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-calfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-cplfw)](https://paperswithcode.com/sota/lightweight-face-recognition-on-cplfw?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-cfp-fp)](https://paperswithcode.com/sota/lightweight-face-recognition-on-cfp-fp?p=edgeface-efficient-face-recognition-model-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-agedb-30)](https://paperswithcode.com/sota/lightweight-face-recognition-on-agedb-30?p=edgeface-efficient-face-recognition-model-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-ijb-b)](https://paperswithcode.com/sota/lightweight-face-recognition-on-ijb-b?p=edgeface-efficient-face-recognition-model-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/edgeface-efficient-face-recognition-model-for/lightweight-face-recognition-on-ijb-c)](https://paperswithcode.com/sota/lightweight-face-recognition-on-ijb-c?p=edgeface-efficient-face-recognition-model-for)	

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2307.01838-009d81v2.svg)](https://arxiv.org/abs/2307.01838v2)

Official gitlab repository for EdgeFace: Efficient Face Recognition Model for Edge Devices 
published in IEEE Transactions on Biometrics, Behavior, and Identity Science.


> Abstract: We present EdgeFace - a lightweight and efficient face recognition network inspired by the hybrid architecture of EdgeNeXt. By effectively combining the strengths of both CNN and Transformer models, and a low rank linear layer, EdgeFace achieves excellent face recognition performance optimized for edge devices. The proposed EdgeFace network not only maintains low computational costs and compact storage, but also achieves high face recognition accuracy, making it suitable for deployment on edge devices. The proposed EdgeFace model achieved the top ranking among models with fewer than 2M parameters in the IJCB 2023 Efficient Face Recognition Competition. Extensive experiments on challenging benchmark face datasets demonstrate the effectiveness and efficiency of EdgeFace in comparison to state-of-the-art lightweight models and deep face recognition models.
```angular2html
@article{george2023edgeface,
  title={Edgeface: Efficient face recognition model for edge devices},
  author={George, Anjith and Ecabert, Christophe and Shahreza, Hatef Otroshi and Kotwal, Ketan and Marcel, Sebastien},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science.},
  year={2024}
}
```

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


<img src="assets/edgeface.png"/>

The following code shows how to use the model for inference:
```python
import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model
arch="edgeface_base"# or "edgeface_s_gamma_05" # or edgeface_xs_gamma_06
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




### Step 3: Understand Configurations

There are two configurations in this source code. The mappings of names and method names in the results are:

- Method Name : `{name}`
- Idiap EdgeFace-S(𝛾=0.5) : `edgeface_s_gamma_05`
- Idiap EdgeFace-XS(𝛾=0.6) : `edgeface_xs_gamma_06`

To see the model parameters, flops, and size on disk, run the following commands:

```bash
python eval_edgeface.py edgeface_s_gamma_05
python eval_edgeface.py edgeface_xs_gamma_06
```


### Step 4: Data Preparation

Download and prepare WebFace4M and WebFace12M: place the `.rec` files in `data/webface4m` and `data/webface12m`. You can find more instructions [here](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/prepare_webface42m.md).

### Step 5: Training

#### EdgeFace-S(𝛾=0.5)

Launch the following command after setting the root path and output path in the config files:

```bash
torchrun --nproc_per_node=8 train_v2.py configs/edgeface_s_gamma_05.py
```
After finishing this step, launch:

```bash
torchrun --nproc_per_node=8 train_v2_restart.py configs/edgeface_s_gamma_05_restart.py
```

#### EdgeFace-XS(𝛾=0.6)

Launch the following command after setting the root path and output path in the config files:

```bash
torchrun --nproc_per_node=4 train_v2.py configs/edgeface_xs_gamma_06.py
```
After finishing this step, launch:

```bash
torchrun --nproc_per_node=4 train_v2_restart.py configs/edgeface_xs_gamma_06_restart.py
```


# EdgeFace Models via `torch.hub`

## Available Models

- `edgeface_base`
- `edgeface_xs_gamma_06`
- `edgeface_xs_q`
- `edgeface_xxs`
- `edgeface_xxs_q`
- `edgeface_s_gamma_05`

## Usage

You can load the models using `torch.hub` as follows:

```python
import torch
variant='edgeface_xs_gamma_06'
model = torch.hub.load('otroshi/edgeface', variant, source='github', pretrained=True)
model.eval()


> :warning: **Note About the License:** Please refer to the `LICENSE` file in the parent directory for information about the license terms and conditions.

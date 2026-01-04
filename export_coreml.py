import torch
import os
from timm import create_model
import timm
# import models.efficientnext.EfficientNeXt
from backbones import get_model, reparameterize
# import models.emo_model.emo

import utils

import torch
import torchvision
from argparse import ArgumentParser
from ptflops import get_model_complexity_info

parser = ArgumentParser()

parser.add_argument('--model', default='efficientnext_a', type=str)
parser.add_argument('--resolution', default=112, type=int)
parser.add_argument('--ckpt', default=None, type=str)


if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()
    model = get_model(args.model)
    if args.ckpt:
        print("Load Checkpoint")
        checkpoint = torch.load(os.path.join(args.ckpt, "current_best_model.pt"))
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     name = k[28:33] # remove 'module.' of dataparallel
        #     if name == "lin_1":
        #         k = list(k)
        #         k[28:33] = "lin.0"
        #         k = ''.join(k)
        #     elif name == "lin_2":
        #         k = list(k)
        #         k[28:33] = "lin.1"
        #         k = ''.join(k)
        #     elif name == "lin_3":
        #         k = list(k)
        #         k[28:33] = "lin.2"
        #         k = ''.join(k)

            # new_state_dict[k]=v
        
        model.load_state_dict(checkpoint)
    
        
    # utils.replace_batchnorm(model)
    # model=timm.utils.reparameterize_model(model)
    # print("Reparameterize") 
    # model = reparameterize(model) 
    model.eval()
    print(model)

    # Trace the model with random data.
    resolution = args.resolution
    example_input = torch.rand(1, 3, resolution, resolution) 
    traced_model = torch.jit.trace(model, torch.Tensor(example_input))
    out = traced_model(example_input)
    # print("======export_onnx======")
    # inputs = torch.randn(64, 3, resolution, resolution, device='cpu')
    # torch.onnx.export(model, inputs, './onnx/'+args.model+".onnx")

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model.eval(), (3, resolution, resolution), as_strings=False,
        print_per_layer_stat=False, verbose=False)
    gmacs = macs / (1000**3) #+ model.extra_gflops
    print(f'GFLOPs: {gmacs:.3f}, Mparams: {(params/(1000**2)):.3f}')

    import coremltools as ct

    # Using image_input in the inputs parameter:
    # Convert to Core ML neural network using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",
        inputs=[ct.ImageType(shape=example_input.shape)]
    )

    # Save the converted model.
    model.save(f"./coreml/{args.model}_{resolution}.mlmodel")
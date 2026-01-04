import timm.utils
import torch
# import onnxruntime as ort
import time
import timm
from timm import create_model

import utils
# from fvcore.nn import FlopCountAnalysis, parameter_count
torch.autograd.set_grad_enabled(False)

T0 = 10
T1 = 60

from backbones import get_model, reparameterize

def throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    model.to('cpu')
    model = model.eval()
    print(f"Running Throughput Test On {device}")
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)


def throughput_cpu_onnx(name, model, device, batch_size, resolution=224):
    model.to('cpu')
    model = model.eval()
    print("Convert To ONNX...")
    inputs = torch.randn(1, 3, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx/{name}_{1}.onnx", verbose = False, opset_version=16)
    inputs = torch.randn(batch_size, 3, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx/{name}_{batch_size}.onnx", verbose = False, opset_version=16)
    # torch.onnx.export(model, inputs, f"{name}.onnx", verbose = False)
    # ort_session = ort.InferenceSession(f"./onnx/{name}_{batch_size}.onnx")
    # ort_inputs = {ort_session.get_inputs()[0].name: inputs.numpy()}
    print("Finish...")
    # start = time.time()
    # while time.time() - start < T0:
    #     _ = ort_session.run(None, ort_inputs)
    
    # timing = []
    # while sum(timing) < T1:
    #     start = time.time()
    #     _ = ort_session.run(None, ort_inputs)
    #     timing.append(time.time() - start)
    # timing = torch.as_tensor(timing, dtype=torch.float32)
    # print(name, device+"onnx", batch_size / timing.mean().item(), 'images/s @ batch size', batch_size)
    # print(name, device+"onnx", 1000 / (batch_size / timing.mean().item()), 'ms latency', batch_size)

def throughput_gpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    model.to(device)
    model = model.eval()
    print(f"Running Throughput Test On {device}")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    while time.perf_counter() - start < T0:
        model(inputs)
    timing = []
    torch.cuda.synchronize()
    while sum(timing) < T1:
        start = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.perf_counter() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

def latency(name, model, device, repetition, resolution=224):
    input = torch.randn(1, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()
    while time.perf_counter() - start < T0:
        model(input)
    timing = []
    torch.cuda.synchronize()
    for i in range(repetition):
        start = time.perf_counter()
        model(input)
        torch.cuda.synchronize()
        timing.append((time.perf_counter() - start)*100.0)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, (timing.mean().item()), 'ms')

device = "cuda:0"

from argparse import ArgumentParser
import torchvision

parser = ArgumentParser()

parser.add_argument('--model', default='repinc_m2_3', type=str) #repinc_m1
parser.add_argument('--resolution', default=112, type=int)
parser.add_argument('--batch-size', default=256, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size
    resolution = args.resolution
    torch.cuda.empty_cache()
    # if args.model == 'shufflenet_v2_x1_0':
    #     model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    # elif args.model == 'shufflenet_v2_x1_5':
    #     model = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
    # elif args.model == 'shufflenet_v2_x2_0':
    #     model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
    #     # model = torchvision.models.mobilenet
    # else:
    #     model = create_model(model_name, num_classes=1000).eval() #inference_mode=True,

    model = get_model(
        model_name, dropout=0.0, fp16=False, num_features=512).cuda()
    # model=timm.utils.reparameterize_model(model) 
    # model = fuse_conv_bn(model)
    model = reparameterize(model)
    print(model)


    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model.eval(), (3, 112, 112), as_strings=False,
        print_per_layer_stat=False, verbose=False)
    gmacs = macs / (1000**3)
    print(f'GFLOPs: {gmacs:.3f}, Mparams: {(params/(1000**2)):.3f}')
    # inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # throughput_cpu_onnx(model_name, model, device='cpu', 
    #                     batch_size=64, resolution=resolution)
    # throughput_cpu(model_name, model, device='cpu', batch_size=batch_size, resolution=resolution)
    throughput_gpu(model_name, model, device, batch_size, resolution=resolution)
    # latency(model_name, model, device, 100, resolution=resolution)
    # print(model)

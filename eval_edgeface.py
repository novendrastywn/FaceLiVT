"""
===============================================================================
Author: Anjith George
Institution: Idiap Research Institute, Martigny, Switzerland.

Copyright (C) 2023 Anjith George

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at anjith.george@idiap.ch
===============================================================================
"""

import torch
import os
import pandas as pd
import sys
from tabulate import tabulate
from backbones import get_model
import argparse

def count_parameters_in_millions(model):
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    return total_params / 1e6


def code_to_profile(model):
    _ = model(torch.rand((1,3,112,112)))

def get_mflops(model):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], with_flops=True
    ) as p:
        code_to_profile(model)

    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # Get the key averages from the profiler
    key_averages = p.key_averages()

    # Initialize total FLOPs counter
    total_flops = 0

    # Iterate over key averages and sum the FLOPs
    for record in key_averages:
        if hasattr(record, 'flops'):
            total_flops += record.flops

    # Print the total FLOPs
    MFLOPS=total_flops/ 1e6
    print(f"Total FLOPs: {MFLOPS} M")
    
    return MFLOPS


def main(args):

    all_dict={}

    os.makedirs('dummy', exist_ok=True)

    name = args.name
        
    model=get_model(name)
    
    model.eval()
    
    MODEL_PATH=os.path.join('dummy', name+'.pth')
    
    # saves a dummy state dict to see model size
    torch.save(model.state_dict(), MODEL_PATH)

    MPARAMS=count_parameters_in_millions(model)

    ret_dict={}

    ret_dict['MFLOPS']=get_mflops(model)

    ret_dict['MPARAMS']=MPARAMS

    ret_dict['MODEL SIZE-MB']=(os.stat(MODEL_PATH).st_size/1e6)
    
    
    all_dict[name]=ret_dict

    df= pd.DataFrame.from_dict(all_dict).T

    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the MFLOPS , parameters and model size on disk")
    parser.add_argument("name", 
                        type=str, 
                        help="Name of the model to evaluate",
                        default="edgeface_s_gamma_05",
                        choices=['edgeface_xs_gamma_06', 'edgeface_s_gamma_05'])

    main(parser.parse_args())
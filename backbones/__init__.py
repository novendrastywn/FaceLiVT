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
from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2
from .pocketnet.augment_cnn import AugmentCNN 
from .pocketnet import genotypes as gt
from .vargfacenet import VarGFaceNet
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .facelivt import *
# from .facelivtv2 import *
from .mobilefacenet import MobileFaceNet
from .kanface.KANFace import KANFace
import timm
import torch

def get_model(name, **kwargs):

    if name=='edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False), rank_ratio=0.6)
    elif name=='edgeface_xs_q':
        model= get_timmfrv2('edgenext_x_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif  name=='edgeface_xxs':
        return get_timmfrv2('edgenext_xx_small', batchnorm=False)
    elif  name=='edgeface_base':
        return get_timmfrv2('edgenext_base', batchnorm=False)
    elif name=='edgeface_xxs_q':
        model=get_timmfrv2('edgenext_xx_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model   
    elif name=='edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=True), rank_ratio=0.5)
    # elif name=='r50':
    #     return get_timmfrv2('resnet50')
    elif name == 'mobilefacenet':
        return MobileFaceNet()
    elif name == 'vargfacenet':
        return VarGFaceNet()
    elif name == 'KANFace06':
        return KANFace(rank_ratio=0.6, grid_size=25, neuron_fun="mean", num_features=512)
    elif name == 'KANFace05':
        return KANFace(rank_ratio=0.5, grid_size=25, neuron_fun="mean", num_features=512)
    elif name == "PocketNetS128":
        genotypes = dict({
        "softmax_cifar10": "Genotype(normal=[[('dw_conv_7x7', 0), ('dw_conv_3x3', 1)], [('dw_conv_1x1', 1), ('dw_conv_1x1', 2)], [('max_pool_3x3', 2), ('dw_conv_7x7', 3)], [('dw_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dw_conv_7x7', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))",
        "softmax_casia": "Genotype(normal=[[('dw_conv_3x3', 0), ('dw_conv_1x1', 1)], [('dw_conv_3x3', 2), ('dw_conv_5x5', 0)], [('dw_conv_3x3', 3), ('dw_conv_3x3', 0)], [('dw_conv_3x3', 4), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dw_conv_3x3', 1), ('dw_conv_7x7', 0)], [('skip_connect', 2), ('dw_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))"    })
        genotype = gt.from_str(genotypes["softmax_casia"])
        return AugmentCNN(C=16, n_layers=18, genotype=genotype, stem_multiplier=4,
                       emb=128)
    elif name == "PocketNetS256":
        genotypes = dict({
        "softmax_cifar10": "Genotype(normal=[[('dw_conv_7x7', 0), ('dw_conv_3x3', 1)], [('dw_conv_1x1', 1), ('dw_conv_1x1', 2)], [('max_pool_3x3', 2), ('dw_conv_7x7', 3)], [('dw_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dw_conv_7x7', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))",
        "softmax_casia": "Genotype(normal=[[('dw_conv_3x3', 0), ('dw_conv_1x1', 1)], [('dw_conv_3x3', 2), ('dw_conv_5x5', 0)], [('dw_conv_3x3', 3), ('dw_conv_3x3', 0)], [('dw_conv_3x3', 4), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dw_conv_3x3', 1), ('dw_conv_7x7', 0)], [('skip_connect', 2), ('dw_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))"    })
        genotype = gt.from_str(genotypes["softmax_casia"])
        return AugmentCNN(C=16, n_layers=18, genotype=genotype, stem_multiplier=4,
                       emb=256)
    elif name == "PocketNetM128":
        genotypes = dict({
        "softmax_cifar10": "Genotype(normal=[[('dw_conv_7x7', 0), ('dw_conv_3x3', 1)], [('dw_conv_1x1', 1), ('dw_conv_1x1', 2)], [('max_pool_3x3', 2), ('dw_conv_7x7', 3)], [('dw_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dw_conv_7x7', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))",
        "softmax_casia": "Genotype(normal=[[('dw_conv_3x3', 0), ('dw_conv_1x1', 1)], [('dw_conv_3x3', 2), ('dw_conv_5x5', 0)], [('dw_conv_3x3', 3), ('dw_conv_3x3', 0)], [('dw_conv_3x3', 4), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dw_conv_3x3', 1), ('dw_conv_7x7', 0)], [('skip_connect', 2), ('dw_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))"    })
        genotype = gt.from_str(genotypes["softmax_casia"])
        return AugmentCNN(C=32, n_layers=9, genotype=genotype, stem_multiplier=4,
                       emb=128)
    elif name == "PocketNetM256":
        genotypes = dict({
        "softmax_cifar10": "Genotype(normal=[[('dw_conv_7x7', 0), ('dw_conv_3x3', 1)], [('dw_conv_1x1', 1), ('dw_conv_1x1', 2)], [('max_pool_3x3', 2), ('dw_conv_7x7', 3)], [('dw_conv_5x5', 4), ('max_pool_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('dw_conv_7x7', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('max_pool_3x3', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))",
        "softmax_casia": "Genotype(normal=[[('dw_conv_3x3', 0), ('dw_conv_1x1', 1)], [('dw_conv_3x3', 2), ('dw_conv_5x5', 0)], [('dw_conv_3x3', 3), ('dw_conv_3x3', 0)], [('dw_conv_3x3', 4), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dw_conv_3x3', 1), ('dw_conv_7x7', 0)], [('skip_connect', 2), ('dw_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 0), ('max_pool_3x3', 1)]], reduce_concat=range(2, 6))"    })
        genotype = gt.from_str(genotypes["softmax_casia"])
        return AugmentCNN(C=32, n_layers=9, genotype=genotype, stem_multiplier=4,
                       emb=128)
    elif name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(  
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    else:
        return timm.create_model(name)

def reparameterize(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'deploy'):
            reparametrized = child.deploy()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        else:
            reparameterize(child)
    
    return net
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import trunc_normal_
from timm.models import register_model
from torch.nn.modules.batchnorm import _BatchNorm
from timm.models import register_model

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def reparam(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], 
                            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, 
                            groups=self.c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def reparam(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class LoRaConv(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio, bias=True):
        super(LoRaConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(2,int(min(in_features, out_features) * rank_ratio))
        self.block = nn.Sequential(
                        Conv2d_BN(in_features, self.rank, 1, 1),
                        nn.Conv2d(self.rank, out_features, 1, 1, bias=bias))

    def forward(self, x):
        return self.block(x)

class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop
        if self.training and self.drop > 0:
            self.forward = self.forward_train
        else:
            self.forward = self.forward_deploy

    def forward_train(self, x):
        return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                         device=x.device).ge_(self.drop).div(1 - self.drop).detach()

    def forward_deploy(self, x):
        return x + self.m(x)
        
    @torch.no_grad()
    def reparam(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.reparam()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, RepConv):
            m = self.m.reparam()
            assert(m.conv.groups == m.conv.in_channels)
            ("Identity RepMSDWConv Reparam")
            kw, kh = (m.conv.weight.shape[2]-1)//2, \
                     (m.conv.weight.shape[3]-1)//2
            identity = torch.ones(m.conv.weight.shape[0], m.conv.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [kh,kh,kw,kw])
            m.conv.weight += identity.to(m.conv.weight.device)
            return m
        else:
            return self        

class FFN(torch.nn.Module):
    def __init__(self, ed, h, act_layer=nn.GELU):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h) 
        self.act = act_layer()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0) 

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def reparam(self):
        classifier = self.classifier.reparam()
        if self.distillation:
            classifier_dist = self.classifier_dist.reparam()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier

class RepConv(nn.Module):
    def __init__(self, inc, ouc, ks=1, stride=1, pad=0, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(inc, ouc, ks, stride, pad, groups=groups)
        self.repconv = nn.Conv2d(inc, ouc, ks//2, stride, pad//2, groups=groups)
        self.bn = nn.BatchNorm2d(ouc)
    
    def forward(self, x):
        xr = self.conv(x) + self.repconv(x) 
        return self.bn(xr)
    
    def forward_deploy(self, x):
        return self.conv(x)
    
    @torch.no_grad()
    def reparam(self):
        conv = self.conv

        repconv=self.repconv; self.__delattr__('repconv')
        kw, kh = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                     (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv_w = nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv_b = repconv.bias         

        final_conv_w = conv.weight + repconv_w 
        final_conv_b = conv.bias + repconv_b 

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
                    (bn.running_var + bn.eps)**0.5
        self.__delattr__('bn')

        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)

        self.forward = self.forward_deploy
        self.conv = conv
        return self

class StemLayer(nn.Module):
    def __init__(self, inc, ouc, ks=3, ps=16, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2

        blocks = math.ceil(ps**0.5)
        dims = [inc] + [x.item() for x in ouc//2**torch.arange(blocks-1, -1, -1)]
        stem = [nn.Sequential(
                RepConv(dims[i], dims[i+1], ks=ks, stride=2, pad=pad),
                act_layer() if i < (blocks-1) else nn.Identity())
                for i in range (blocks)]
        self.stem = nn.Sequential(*stem)
        
    def forward(self, x):
        return self.stem(x)
    
class PatchMerging(nn.Module):
    def __init__(self, inc, ouc, ks=7, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2 

        self.spatial = nn.Sequential(
                        RepConv(inc, inc, ks=ks, stride=2, pad=pad, groups=inc),
                        # act_layer(),
                        Conv2d_BN(inc, ouc, ks=1, stride=1)
                        )
        self.channel = Residual(FFN(ouc, ouc*2, act_layer))
        
    def forward(self, x):
        return self.channel(self.spatial(x))
        

class MHSA(torch.nn.Module):
    """Structural Reparameterization Single-Head Self-Attention"""
    def __init__(self, dim, resolution, ratio=0.5, act_layer=nn.ReLU):
        super().__init__()

        self.n_head = 6
        self.ratio  = ratio
        self.qk_dim = 32
        self.v_dim  = int( dim * self.ratio) 
        self.att_dim = self.n_head * self.v_dim
        self.split_idx = (self.qk_dim, self.qk_dim, self.v_dim)
        self.head_dim = self.qk_dim + self.qk_dim + self.v_dim

        self.scale     = self.qk_dim ** -0.5
        proj_dim       = 2 * self.qk_dim * self.n_head + self.v_dim  * self.n_head
        self.qkv_proj  = Conv2d_BN(dim, proj_dim, 1, 1) 
        self.out_proj  = Conv2d_BN(self.att_dim, dim, 1, 1) 


    def forward(self, x):
        B, _, H, W = x.shape

        qkv = self.qkv_proj(x).reshape(B, self.n_head, self.head_dim, -1)
        q, k, v = qkv.permute(0, 1, 3, 2).split(self.split_idx, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = torch.matmul(attn, v).transpose(-2, -1).reshape(B, -1, H, W)
        x = self.out_proj(attn)
        # print(x.shape)
        return x
    
class LinearAttention(torch.nn.Module):
    """Reparameterized Linear-Attention"""
    def __init__(self, dim, resolution, ratio=4, act_layer=nn.ReLU):
        super().__init__()

        self.n_head = 16
        self.res=resolution**2
        self.dim=dim
        linear1=[]
        linear2=[]
        self.norm=nn.GroupNorm(1, dim)
        self.act_layer = act_layer()
        for i in range(self.n_head):
            linear1.append(nn.Linear(self.res, self.res*ratio))
            linear2.append(nn.Linear(self.res*ratio, self.res))
        self.linear1 = torch.nn.ModuleList(linear1)
        self.linear2 = torch.nn.ModuleList(linear2)

    def forward(self, x):
        B,C,H,W= x.shape
        # print(x.shape, H*W, self.res)
        x = self.norm(x).reshape(-1, self.dim, self.res)
        x = x.chunk(self.n_head, dim=1)
        x_out = []
        for i in range(self.n_head):
            feat = self.linear1[i](x[i])
            feat = self.act_layer(feat)
            feat = self.linear2[i](feat)
            x_out.append(feat)
        x = torch.cat(x_out, dim=-1)
        x = x.reshape(B,C,H,W)
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio,  resolution, type, act_layer=nn.ReLU):
        super().__init__()

        if type=='repmix':
            token_mixer = RepConv(dim, dim, ks=3, stride=1, pad=1, groups=dim)
            # token_mixer = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, groups=dim)
            self.block  = nn.Sequential(
                Residual(token_mixer),
                Residual(FFN(dim, dim*mlp_ratio, act_layer)))
            
        elif type=='mhsa':
            token_mixer = MHSA(dim,  resolution, act_layer=act_layer)
            self.block  = nn.Sequential(
                Residual(token_mixer),
                Residual(FFN(dim, dim*mlp_ratio, act_layer)))
            
        elif type=='mhla':
            token_mixer = LinearAttention(dim, resolution, act_layer=act_layer)
            self.block  = nn.Sequential(
                Residual(token_mixer),
                Residual(FFN(dim, dim*mlp_ratio, act_layer)))
                
        else:
            print("type is not listed")
        
    def forward(self, x):
        return self.block(x)
    
class Stage(nn.Module):
    def __init__(self, dim, depth,  mlp_ratio,  resolution, type, act_layer=nn.ReLU):
        super().__init__()
        block = [Block(dim=dim, 
                       mlp_ratio=mlp_ratio,
                       resolution=resolution,
                       type=type, 
                       act_layer=act_layer,
                      )for i in range (depth)]
        self.blocks = nn.Sequential(*block)
    def forward(self, x):
        return self.blocks(x)

class FaceLiVT(nn.Module):  
    def __init__(self, in_chans=3, img_size=112,
                 num_classes=512,
                 dims=[ 48, 96, 192, 384],
                 depths=[ 2, 2, 8, 2],
                 type =["repmix", "repmix", "mhsa", "mhsa"],
                 ks_pe= 3,
                 patch_size=4, 
                 mlp_ratio=3, 
                 act_layer=nn.GELU, 
                 distillation=False,
                 final_feature_dim=None, 
                 drop_rate=0.0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.final_feature_dim = final_feature_dim

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        
        num_stage = len(depths)
        self.num_stage = num_stage

        stages = []
        img_res = img_size//patch_size
        patch_embedds=[]
        patch_embedds.append(StemLayer(in_chans, dims[0], ps=patch_size, act_layer=act_layer))

        for i_stage in range(num_stage):
            # print(img_res)
            stage = Stage(
                    dim=dims[i_stage],
                    depth=depths[i_stage], 
                    type=type[i_stage],
                    resolution=img_res, 
                    mlp_ratio=mlp_ratio, 
                    act_layer=act_layer,
            )
            stages.append(stage)
            if i_stage < (num_stage-1):
                patch_embedd=PatchMerging(dims[i_stage], dims[i_stage+1], ks=ks_pe, act_layer=act_layer)
                patch_embedds.append(patch_embedd)
                img_res = math.ceil(img_res/2)
            

        self.patch_embedds = nn.Sequential(*patch_embedds)
        self.stages = nn.Sequential(*stages)
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        # Classifier head
        if self.final_feature_dim is not None:
            if isinstance(self.final_feature_dim, (list, tuple)):
                self.pre_head = nn.Sequential(
                            Conv2d_BN(dims[-1], self.final_feature_dim[0]),
                            nn.AdaptiveAvgPool2d(1)
                            )
            else:
                self.pre_head = nn.AdaptiveAvgPool2d(1)
                self.final_feature_dim=[dims[-1], self.final_feature_dim]

            self.head = nn.Sequential(
                BN_Linear(self.final_feature_dim[0], self.final_feature_dim[1]),
                act_layer(),
                self.head_drop,
                Classfier(self.final_feature_dim[1], num_classes, distillation)
                )
        else:
            self.pre_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), self.head_drop)
            self.head = Classfier(dims[-1], num_classes, distillation)

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feature(self, x):
        for i in range(self.num_stage):
            x = self.patch_embedds[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        x = self.pre_head(x).flatten(1)
        x = self.head(x)
        return x

def reparameterize(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'reparam'):
            reparametrized = child.reparam()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        else:
            reparameterize(child)
    
    return net

@register_model
def facelivt_s_sa(num_classes = 512, distillation=False, pretrained=True, **kwargs):
    model=FaceLiVT(
            num_classes=num_classes,
            dims=[ 40, 80, 160, 320],
            depths=[ 2, 4, 6, 2],
            type =["repmix", "repmix", "mhsa", "mhsa"],
            final_feature_dim=None, 
            distillation=False,
            **kwargs)
    # reparameterize(model)
    return model


@register_model
def facelivt_m_sa(num_classes = 512, distillation=False, pretrained=True, **kwargs):
    model=FaceLiVT(
            num_classes=num_classes,
            dims=[ 64, 128, 256, 512],
            depths=[ 2, 4, 6, 2],
            type =["repmix", "repmix", "mhsa", "mhsa"],
            final_feature_dim=None, 
            distillation=False,
            **kwargs)
    # reparameterize(model)
    return model

@register_model
def facelivt_l_sa(num_classes = 512, distillation=False, pretrained=True, **kwargs):
    model=FaceLiVT(
            num_classes=num_classes,
            dims=[ 96, 192, 384, 768],
            depths=[ 2, 4, 6, 2],
            type =["repmix", "repmix", "mhsa", "mhsa"],
            final_feature_dim=None, 
            distillation=False,
            **kwargs)
    # reparameterize(model)
    return model

@register_model
def facelivt_s(num_classes = 512, distillation=False, pretrained=True, **kwargs):
    model=FaceLiVT(
            num_classes=num_classes,
            dims=[ 40, 80, 160, 320],
            depths=[ 2, 4, 6, 2],
            type =["repmix", "repmix", "mhla", "mhla"],
            final_feature_dim=None, 
            distillation=False,
            **kwargs)
    # reparameterize(model)
    return model

@register_model
def facelivt_m(num_classes = 512, distillation=False, pretrained=True, **kwargs):
    model=FaceLiVT(
            num_classes=num_classes,
            dims=[ 64, 128, 256, 512],
            depths=[ 2, 4, 6, 2],
            type =["repmix", "repmix", "mhla", "mhla"],
            final_feature_dim=None, 
            distillation=False,
            **kwargs)
    # reparameterize(model)
    return model

# @register_model
# def facelivt_s_repmix(num_classes = 512, distillation=False, pretrained=True, **kwargs):
#     model=FaceLiVT(
#             num_classes=num_classes,
#             dims=[ 48, 96, 192, 384],
#             depths=[ 2, 2, 6, 2],
#             type =["repmix", "repmix", "repmix", "repmix"],
#             final_feature_dim=None, 
#             distillation=False,
#             **kwargs)
#     # reparameterize(model)
#     return model

# @register_model
# def facelivt_s(num_classes = 512, distillation=False, pretrained=True, **kwargs):
#     model=FaceLiVT(
#             num_classes=num_classes,
#             dims=[ 48, 96, 192, 384],
#             depths=[ 2, 2, 6, 2],
#             type =["repmix", "repmix", "repmix", "mhla"],
#             final_feature_dim=None, 
#             distillation=False,
#             **kwargs)
#     # reparameterize(model)
#     return model
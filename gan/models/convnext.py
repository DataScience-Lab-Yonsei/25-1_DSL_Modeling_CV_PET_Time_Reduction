from base64 import encode
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), # patchify 4x4
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first") # LN
        )
        self.downsample_layers.append(stem)
        for i in range(3): # stem이 첫번째 down sample, 얘네는 block 사이 downsample
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# class ConvNeXtGenerator(nn.Module):
#     r""" ConvNeXt
#         A PyTorch impl of : `A ConvNet for the 2020s`  -
#           https://arxiv.org/pdf/2201.03545.pdf
#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """
#     def __init__(self, in_chans=3, out_chans=3, 
#                  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
#                  drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]
#                  ):
#         super().__init__()
        
#         self.model = nn.Sequential(
#             ConvNeXtEncoder(in_chans=in_chans, depths=depths, dims=dims, out_indices=out_indices),
#             ConvNeXtDecoder(out_chans=out_chans, dims=dims, out_indices=out_indices)
#         )

#     def forward(self, x, nce_layers=None, encode_only=False):
#         x = self.Encoder(x)
#         if encode_only:
#             return x
#         x = self.Decoder(x)
#         return x


class ConvNeXtGenerator(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, out_chans=3, depths=[3, 3, 9], dims=[96, 192, 384], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2],
                 unet_structure=True,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths)-1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        self.depths = depths
        self.out_indices = out_indices
        self.unet_structure = unet_structure

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(depths)):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.upsampling_layers = nn.ModuleList()

        mul = 2 if unet_structure else 1

        self.upsampling_layers.append(nn.Sequential(
            LayerNorm(dims[-1] , eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose2d(dims[-1], dims[-2],
                               kernel_size=3, stride=2, padding=1 , output_padding=1),
            # *[Block(dim=dims[-2],
            #     layer_scale_init_value=layer_scale_init_value) for j in range(3)]
            nn.GELU()
        ))
        for i in range(len(depths)-2, 0, -1):
            upsampling_layer = nn.Sequential(
                LayerNorm(dims[i] * mul, eps=1e-6, data_format='channels_first'),
                nn.ConvTranspose2d(dims[i] * mul, dims[i-1],
                                   kernel_size=3, stride=2, padding=1 , output_padding=1),
                # *[Block(dim=dims[i-1],
                # layer_scale_init_value=layer_scale_init_value) for j in range(3)]
                nn.GELU()
            )
            self.upsampling_layers.append(upsampling_layer)
        self.upsampling_layers.append(nn.Sequential(
            LayerNorm(dims[0] * mul, eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose2d(dims[0] * mul, dims[0] * mul // 2,
                               kernel_size=3, stride=2, padding=1 , output_padding=1),
            # *[Block(dim=3,
            #   layer_scale_init_value=layer_scale_init_value) for j in range(3)],
            nn.GELU(),
            LayerNorm(dims[0] * mul // 2, eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose2d(dims[0] * mul // 2, out_chans,
                               kernel_size=3, stride=2, padding=1 , output_padding=1),
            
            nn.Tanh()
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if isinstance(pretrained, str):
            ckpt = torch.load(pretrained, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            self.load_state_dict(state_dict, strict=False)
            print(f'Loaded checkpoint from "{pretrained}"')
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x, nce_layers=None, encode_only=False):
        features = self.forward_features(x)
        
        if encode_only:
            return features
        
        x = self.upsampling_layers[0](features[self.out_indices[-1]])

        if self.unet_structure:
            for i in self.out_indices[-2::-1]:
                x = self.upsampling_layers[-1-i](torch.cat((x, features[i]), dim=1))
        else:
            for i in self.out_indices[-2::-1]:
                x = self.upsampling_layers[-1-i](x)
        
        return x
    

class ConvNeXtDecoder(nn.Module):
    """ConvNeXt-based decoder that consists of a few Resnet blocks + a few upsampling operations.
    """

    def __init__(self, out_chans=3, dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2 ,3]):
        """Construct a Resnet-based decoder
        [downsample] = [LayerNorm] -> [ConvTranspose]
        indice[3]: 768 -> [downsample] -> [Block] * 3-> 384
        indice[2]: 384 + 384(<-768) -> [downsample] -> [Block] * 3 -> 192
        indice[1]: 192 + 192(<-384) -> [downsample] -> [Block] * 3 -> 96
        indice[0]: 96 + 96(<-192) -> [downsample] -> [Block] * 3 -> [256x256x3]
        """
        super().__init__()
        
        self.upsampling_layers = []
        self.upsampling_layers.append(nn.Sequential(
            LayerNorm(dims[0] , eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose2d(dims[0], out_chans,
                               kernel_size=4, stride=4),
            nn.GELU()
        ))
        for i in range(2, 0, -1):
            upsampling_layer = nn.Sequential(
                LayerNorm(dims[i] * 2, eps=1e-6, data_format='channels_first'),
                nn.ConvTranspose2d(dims[i] * 2, dims[i-1],
                                   kernel_size=2, stride=2),
                nn.GELU()
            )
            self.upsampling_layers.append(upsampling_layer)
        self.upsampling_layers.append(nn.Sequential(
            LayerNorm(dims[0] * 2, eps=1e-6, data_format='channels_first'),
            nn.ConvTranspose2d(dims[0] * 2, out_chans,
                               kernel_size=4, stride=4),
            nn.Tanh()
        ))
        
        self.out_indices = out_indices

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.upsampling_layers[0](input[self.out_indices[0]])
        for i in range(self.out_indices[1:]):
            x = self.upsampling_layers[i](torch.cat(x, input[i]))
        return x


class ConvNeXtIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, out_chans=3, patch_size = 16,
                 depth=18, dim=384, drop_path_rate=0., 
                 layer_scale_init_value=0,
                 ):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i], 
                                    layer_scale_init_value=layer_scale_init_value)
                                    for i in range(depth)])
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first') # final norm layer

        self.mets = nn.ConvTranspose2d(dim, out_chans, kernel_size=patch_size, stride=patch_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.mets(x)
        return x


if __name__ == '__main__':
    cv = ConvNeXtGenerator()
    for i, x in enumerate(cv.stages[1]):
        print(i,',', x)
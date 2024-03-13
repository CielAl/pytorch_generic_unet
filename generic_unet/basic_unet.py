from torch import nn
import torch
from fastai.callback.hook import model_sizes, hook_outputs, dummy_eval, Hooks, Hook
import numpy as np
from fastai.torch_basics import ConvLayer, BatchNorm, SigmoidRange, ToTensorBase, \
    PixelShuffle_ICNR, apply_init, SequentialEx
from fastai.vision.models.unet import ResizeToOrig
from .skip_connect import SkipBlock, SKIP_TYPE
from .ppm import SpatialPyramidPooling
from typing import List, Tuple, Callable, Optional
from torchvision.models.resnet import resnet50
from fastai.layers import NormType


def _get_sz_change_idxs(sizes):
    """Get the indexes of the layers where the size of the activation changes."""

    feature_szs = [size[-1] for size in sizes]
    sz_chg_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    return sz_chg_idxs


class BasicUNet(SequentialEx):  # nn.Module
    """Adapted from DynamicUnet.

    """
    encoder: List[nn.Module]
    bridge: List[nn.Module]
    decoder: List[nn.Module]
    output: List[nn.Module]
    encoder_hook: Hook

    @staticmethod
    def decoder_with_skips(dummy_tensor: torch.Tensor, *,
                           layer_hooks: Hooks | List[Hook],
                           size_change_ind_rev: List[int],
                           sizes: List,
                           blur: bool,
                           blur_final: bool,
                           norm_type: NormType,
                           skip_type: SKIP_TYPE,
                           skip_bottleneck: bool,
                           act_cls: Callable,
                           init: Callable,
                           out_layers: Optional[List],
                           **kwargs) -> Tuple[List[nn.Module], torch.Tensor]:
        """Helper function to create decoder with skip connection

        Args:
            dummy_tensor: dummy tensor output of previous layers
            layer_hooks:
            size_change_ind_rev:
            sizes:
            blur:
            blur_final:
            norm_type:
            skip_type:
            skip_bottleneck:
            act_cls:
            init:
            out_layers:
            **kwargs:

        Returns:

        """
        # if not given then create a new list.
        if out_layers is None:
            out_layers = []

        # start from the deepest level block
        for i, idx in enumerate(size_change_ind_rev):

            not_final = i != len(size_change_ind_rev) - 1
            # input of upsampling / encoding path target size
            up_in_c, x_in_c = int(dummy_tensor.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            # would be too memory intensive to add attention here
            # sa = self_attention and (i==len(size_change_indices)-3)

            skip_block = SkipBlock(up_in_c, x_in_c, layer_hooks[i], final_div=not_final, blur=do_blur,
                                   act_cls=act_cls, init=init, norm_type=norm_type, skip_type=skip_type,
                                   bottleneck=skip_bottleneck,
                                   **kwargs).eval()
            out_layers.append(skip_block)
            dummy_tensor = skip_block(dummy_tensor)
        # out_modules = nn.Sequential(*out_layers)
        return out_layers, dummy_tensor

    @staticmethod
    def get_mid_bridge(dummy_tensor: torch.Tensor, *,
                       num_input: int,
                       feature_map_size: int,
                       pool_sizes,
                       ppm_flatten: bool,
                       act_cls: Callable, norm_type: NormType,
                       init: Callable,
                       **kwargs) -> Tuple[List[nn.Module], torch.Tensor]:
        """add midconv and poolings

        Returns:

        """
        # ni = sizes[-1][1]
        # map_size = sizes[-1][2]
        # ref_size = x.shape[-2:]
        ppm = SpatialPyramidPooling(grid_size_list=pool_sizes, flatten=False, feature_map_size=feature_map_size,
                                    in_channels=num_input, out_channels=num_input)
        ppm_out_size = SpatialPyramidPooling.output_size(pool_sizes, num_input, ppm_flatten)
        reduction_size = num_input

        reduction = ConvLayer(ppm_out_size, reduction_size, act_cls=act_cls, norm_type=norm_type, **kwargs)

        middle_conv = nn.Sequential(ConvLayer(num_input, num_input*2, act_cls=act_cls, norm_type=norm_type, **kwargs),
                                    ConvLayer(num_input*2, num_input, act_cls=act_cls,
                                    norm_type=norm_type, **kwargs)).eval()
        # dummy_tensor = middle_conv(dummy_tensor)
        apply_init(nn.Sequential(middle_conv), init)
        bridge_list = [BatchNorm(num_input), nn.ReLU(), ppm, reduction, middle_conv]
        out_modules = bridge_list
        dummy_tensor = SequentialEx(*out_modules)(dummy_tensor)  # middle_conv(dummy_tensor)
        # down path + middle
        # encoder, BatchNorm(ni), nn.ReLU()
        return out_modules, dummy_tensor

    @staticmethod
    def get_output_layer(dummy_tensor: torch.Tensor,
                         imsize: Tuple[int, ...],
                         ref_size: Tuple[int, ...],
                         num_input: int,
                         num_output: int,
                         act_cls: Callable,
                         norm_type: NormType,
                         y_range: Optional = None,
                         **convlayer_kwargs) -> Tuple[List[nn.Module], torch.Tensor]:
        # ni = x.shape[1]
        layers = []
        if imsize != ref_size:
            layers.append(PixelShuffle_ICNR(num_input, act_cls=act_cls, norm_type=norm_type))

        layers.append(ResizeToOrig())
        layers += [ConvLayer(num_input, num_output, ks=1, act_cls=None, norm_type=norm_type, **convlayer_kwargs)]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        layers.append(ToTensorBase())
        layer_module = SequentialEx(*layers)
        dummy_tensor: torch.Tensor = layer_module(dummy_tensor)
        return layers, dummy_tensor

    @staticmethod
    def parse_encoder(encoder: nn.Module, imsize: Tuple[int, ...])\
            -> Tuple[List[Tuple[int, ...]], List[int], Hooks | List[Hook], torch.Tensor]:
        sizes = model_sizes(encoder, size=imsize)
        size_change_ind_rev = list(reversed(_get_sz_change_idxs(sizes)))
        hooks_bottom_to_top = hook_outputs([encoder[i] for i in size_change_ind_rev], detach=False)
        dummy_tensor = dummy_eval(encoder, imsize).detach()
        return sizes, size_change_ind_rev, hooks_bottom_to_top, dummy_tensor

    @staticmethod
    def basic_unet_modules(encoder: nn.Module,
                           *,
                           n_out: int,
                           img_size: Tuple[int, ...],
                           sizes: List[Tuple[int, ...]],
                           size_change_ind_rev: List[int],
                           hooks: Hooks | List[Hooks] | List[List[Hook]],
                           dummy_tensor: torch.Tensor,
                           pool_sizes: List[int],
                           ppm_flatten: bool = False,
                           blur: bool = False,
                           blur_final: bool = True,
                           y_range=None,
                           act_cls: Callable = nn.ReLU,
                           norm_type: NormType = NormType.Batch,
                           init: Callable = nn.init.kaiming_normal_,
                           skip_type: SKIP_TYPE = 'sum',
                           skip_bottleneck: bool = False,
                           **convlayer_kwargs) -> Tuple[Tuple[List[nn.Module], ...], torch.Tensor]:
        # channel size
        ni = sizes[-1][1]
        # feature map height/width
        map_size = sizes[-1][2]

        down_path = [encoder]   # BatchNorm(ni), nn.ReLU() moved to middle layers.

        bridge_layers, dummy_tensor = BasicUNet.get_mid_bridge(dummy_tensor, num_input=ni, feature_map_size=map_size,
                                                               pool_sizes=pool_sizes, ppm_flatten=ppm_flatten,
                                                               act_cls=act_cls, norm_type=norm_type,
                                                               init=init, **convlayer_kwargs)
        # ref_size = x.shape[-2:]

        # down path + middle

        # up path
        decoders, dummy_tensor = BasicUNet.decoder_with_skips(dummy_tensor,
                                                              layer_hooks=hooks,
                                                              size_change_ind_rev=size_change_ind_rev,
                                                              sizes=sizes, blur=blur, blur_final=blur_final,
                                                              norm_type=norm_type,
                                                              skip_type=skip_type,
                                                              skip_bottleneck=skip_bottleneck, act_cls=act_cls,
                                                              init=init, out_layers=None)
        # add decoders to the list
        output, dummy_tensor = BasicUNet.get_output_layer(dummy_tensor,
                                                          imsize=img_size,
                                                          ref_size=sizes[0][-2:], num_input=dummy_tensor.shape[1],
                                                          num_output=n_out,
                                                          act_cls=act_cls, norm_type=norm_type,
                                                          y_range=y_range, **convlayer_kwargs)

        # add output layers

        return (down_path, bridge_layers, decoders, output), dummy_tensor

    def __init__(self,
                 encoder: nn.Module,
                 *,
                 n_out: int,
                 img_size: Tuple[int, ...],
                 sizes: List[Tuple[int, ...]],
                 size_change_ind_rev: List[int],
                 hooks_bottom_to_top: Hooks | List[Hooks] | List[List[Hook]],
                 dummy_tensor: torch.Tensor,
                 pool_sizes: List[int],
                 ppm_flatten: bool = False,
                 blur: bool = False,
                 blur_final: bool = True,
                 y_range=None,
                 act_cls: Callable = nn.ReLU,
                 norm_type: NormType = NormType.Batch,
                 init: Callable = nn.init.kaiming_normal_,
                 skip_type: SKIP_TYPE = 'sum',
                 skip_bottleneck: bool = False,
                 **convlayer_kwargs):
        layers_tuple, dummy_tensor = BasicUNet.basic_unet_modules(encoder=encoder,
                                                                  n_out=n_out, img_size=img_size,
                                                                  sizes=sizes, size_change_ind_rev=size_change_ind_rev,
                                                                  dummy_tensor=dummy_tensor, hooks=hooks_bottom_to_top,
                                                                  pool_sizes=pool_sizes,
                                                                  ppm_flatten=ppm_flatten,
                                                                  act_cls=act_cls,
                                                                  init=init,
                                                                  norm_type=norm_type,
                                                                  blur=blur, blur_final=blur_final,
                                                                  skip_type=skip_type,
                                                                  y_range=y_range,
                                                                  skip_bottleneck=skip_bottleneck,
                                                                  **convlayer_kwargs)

        self.encoder_hook: Hook = hook_outputs(encoder, detach=False)[-1]
        self.encoder, self.bridge, self.decoder, self.output = layers_tuple
        layer_list_flattened = sum(layers_tuple, [])
        super().__init__(*layer_list_flattened)

    @classmethod
    def build(cls,
              encoder: nn.Module,
              *,
              n_out: int,
              img_size: Tuple[int, ...],
              pool_sizes: List[int],
              ppm_flatten: bool = False,
              blur: bool = False,
              blur_final: bool = True,
              y_range=None,
              act_cls: Callable = nn.ReLU,
              norm_type: NormType = NormType.Batch,
              init: Callable = nn.init.kaiming_normal_,
              skip_type: SKIP_TYPE = 'sum',
              skip_bottleneck: bool = False,
              **convlayer_kwargs):

        sizes, size_change_ind_rev, hooks_bottom_to_top, dummy_tensor = BasicUNet.parse_encoder(encoder, img_size)

        return cls(encoder=encoder,
                   n_out=n_out, img_size=img_size,
                   sizes=sizes, size_change_ind_rev=size_change_ind_rev,
                   dummy_tensor=dummy_tensor, hooks_bottom_to_top=hooks_bottom_to_top,
                   pool_sizes=pool_sizes,
                   ppm_flatten=ppm_flatten,
                   act_cls=act_cls,
                   init=init,
                   norm_type=norm_type,
                   blur=blur, blur_final=blur_final,
                   skip_type=skip_type,
                   y_range=y_range,
                   skip_bottleneck=skip_bottleneck,
                   ** convlayer_kwargs)

    @classmethod
    def build_resnet(cls, n_out: int,
                     img_size: Tuple[int, ...],
                     pool_sizes: List[int],
                     ppm_flatten: bool = False,
                     blur: bool = False,
                     blur_final: bool = True,
                     y_range=None,
                     act_cls: Callable = nn.ReLU,
                     norm_type: NormType = NormType.Batch,
                     init: Callable = nn.init.kaiming_normal_,
                     skip_type: SKIP_TYPE = 'sum',
                     skip_bottleneck: bool = False,
                     **convlayer_kwargs):

        encoder = nn.Sequential(*list(resnet50().children())[:-2])
        return cls.build(encoder=encoder,
                         n_out=n_out, img_size=img_size,
                         pool_sizes=pool_sizes,
                         ppm_flatten=ppm_flatten,
                         act_cls=act_cls,
                         init=init,
                         norm_type=norm_type,
                         blur=blur, blur_final=blur_final,
                         skip_type=skip_type,
                         y_range=y_range,
                         skip_bottleneck=skip_bottleneck,
                         ** convlayer_kwargs)

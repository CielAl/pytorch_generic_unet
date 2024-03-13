from typing import List, Tuple, Callable, Optional, Literal

import numpy as np
import torch
from fastai.callback.hook import Hook, hook_output, Hooks
from fastai.layers import NormType, SequentialEx
from torch import nn

from .basic_unet import BasicUNet
from .skip_connect import SKIP_TYPE

REDUCE_MEAN = Literal['mean']
REDUCE_CAT = Literal['cat']
REDUCE_SUM = Literal['sum']
REDUCE_NONE = Literal['none']
TYPE_REDUCE = Literal[REDUCE_MEAN, REDUCE_CAT, REDUCE_SUM, REDUCE_NONE]


class SeqExSpecifyOrig(SequentialEx):

    def forward_helper(self, x, orig):
        res = x
        for l in self.layers:
            res.orig = orig
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig, nres.orig = None, None
            res = nres
        return res

    def forward(self, x: torch.Tensor, orig: Optional[torch.Tensor] = None):
        orig = orig if orig is not None else x
        return self.forward_helper(x, orig)


class NestedUNet(nn.Module):

    model_list: nn.ModuleList
    encoder_hook_list: List[Hook]
    num_levels: int

    @staticmethod
    def inspect_slice_sizes(img_size: Tuple[int, int], backbone_slices: List[nn.Module],
                            sizes: List[torch.Size | Tuple[int, ...]],
                            size_change_ind_forward: List[int]) -> None:
        with torch.no_grad():
            x = torch.ones(1, 3, *img_size)
            for idx, segment in enumerate(backbone_slices):
                x = segment(x)
                target_size_ind = size_change_ind_forward[idx] + 1
                target_size = sizes[target_size_ind]
                assert x.shape == target_size

    @staticmethod
    def indices_to_slices(*, size_change_ind_forward: List[int],
                          max_length: int) -> List[Tuple[int, int]]:
        """Convert the indices to segment of slices.

        Args:
            size_change_ind_forward: index of layers that induce tensor size change (up/downsample) in forward order.
            max_length: max length of array that is indexed
        Returns:

        """

        size_change_ind_forward_arr = np.asarray(size_change_ind_forward, dtype=int)
        offset_size_change = 1  # 1 as offset as the ind is the layer right before size change happens
        offset_slice_end = 1  # 1 as the offset of slice end --> the end "j" of slice i:j is not inclusive.
        # end indices
        end_indices = size_change_ind_forward_arr + (offset_slice_end + offset_size_change)
        end_indices[-1] = max_length
        # start indices
        start_indices = np.roll(end_indices, 1)
        start_indices[0] = 0

        return [(int(start), int(end))
                for start, end in zip(start_indices, end_indices)]

    @staticmethod
    def slice_backbone_helper(backbone: nn.Sequential, slice_idx: List[Tuple[int, int]])\
            -> Tuple[List[nn.Module], List[Tuple[int, int]]]:
        """Helper function
        Args:
            backbone: encoder object
            slice_idx: list of pairs of int as a list of slice indices.

        Returns:

        """
        out_modules = []
        # out_heads = nn.ModuleList()
        for slice_pair in slice_idx:
            start, end = slice_pair
            out_modules.append(backbone[start: end])
        return out_modules, slice_idx

    @staticmethod
    def slice_backbone(backbone: nn.Sequential,
                       size_change_ind_forward: List[int])\
            -> Tuple[List[nn.Module], List[Tuple[int, int]]]:
        slice_idx = NestedUNet.indices_to_slices(size_change_ind_forward=size_change_ind_forward,
                                                 max_length=len(backbone))
        return NestedUNet.slice_backbone_helper(backbone, slice_idx)

    @staticmethod
    def hook_extend_helper(decoders: nn.Sequential | List[nn.Module],
                           hooks: List[List[Hook]]):
        """Add new hooks to decoder and add to the existing list of hooks for dense connection.
        Return a new copied list.

        Args:
            decoders: decoders in forward order
            hooks: hooks (reverse encoder order and forward decoder order)

        Returns:
            new nested list with hooks of decoders added to the hook list of the same level.
        """
        new_out_list = []
        assert len(decoders) == len(hooks)
        for d, hk in zip(decoders, hooks):
            new_hook = hook_output(d, detach=False)
            hk_copy = hk.copy()
            hk_copy.append(new_hook)
            new_out_list.append(hk_copy)
        return new_out_list

    @staticmethod
    def new_extended_hooks(decoders: nn.Sequential | List[nn.Module] | nn.ModuleList,
                           hooks_stack_rev, hook_start_ind):
        hooks_stack_copy = hooks_stack_rev.copy()
        curr_hooks = hooks_stack_copy[hook_start_ind::]
        hooks_stack_copy[hook_start_ind::] = NestedUNet.hook_extend_helper(decoders, curr_hooks)
        return hooks_stack_copy

    @staticmethod
    def forward_hook_store_shape(model: BasicUNet, num_channel: int, img_size: Tuple[int, int]):
        with torch.no_grad():
            dummy = next(model.parameters()).new(1, num_channel, *img_size)
            model(dummy)

    @staticmethod
    def get_inner_net(curr_change_ind_rev: List[int],
                      *,
                      curr_sizes: List[Tuple[int, ...]],
                      curr_hooks: List[List[Hook]],
                      curr_encoder: nn.Sequential,
                      n_out: int,
                      img_size: Tuple[int, int],
                      dummy_tensor_in: torch.Tensor,
                      pool_sizes: List[int],
                      ppm_flatten: bool,
                      blur: bool,
                      blur_final: bool,
                      y_range,
                      act_cls: Callable,
                      norm_type: NormType,
                      skip_type: SKIP_TYPE,
                      init: Callable,
                      skip_bottleneck: bool,
                      **convlayer_kwargs) -> BasicUNet:

        # get the unet layers
        basic_unet = BasicUNet(curr_encoder, n_out=n_out, img_size=img_size,
                               sizes=curr_sizes, size_change_ind_rev=curr_change_ind_rev,
                               hooks_bottom_to_top=curr_hooks, dummy_tensor=dummy_tensor_in,
                               pool_sizes=pool_sizes, ppm_flatten=ppm_flatten, blur=blur,
                               blur_final=blur_final, y_range=y_range, act_cls=act_cls,
                               norm_type=norm_type, skip_type=skip_type, init=init,
                               skip_bottleneck=skip_bottleneck, **convlayer_kwargs)
        return basic_unet

    @staticmethod
    def nested_unet_modules(encoder: nn.Sequential,
                            num_levels: int = 0,
                            *,
                            n_out: int,
                            img_size: Tuple[int, int],
                            sizes: List[Tuple[int, ...]],
                            size_change_ind_rev: List[int],
                            hooks_bottom_to_top: Hooks | List[Hook],
                            dummy_tensor_in: torch.Tensor,
                            pool_sizes: List[int],
                            ppm_flatten: bool = False,
                            blur: bool = False,
                            blur_final: bool = True,
                            y_range=None,
                            act_cls: Callable,
                            norm_type: NormType,
                            init: Callable,
                            skip_type: SKIP_TYPE = 'sum',
                            skip_bottleneck: bool = False,
                            **convlayer_kwargs) -> Tuple[List[nn.ModuleList | List[nn.Module]],
                                                         List[nn.ModuleList | List[nn.Module]],
                                                         List[Hook]]:
        # if nest_level = 0 ---> only create the out-most Unet ---> essentially a basic unet
        # after appending each level of inner unet, append the output/decoder nodes to inner
        # hook list for the next level
        hooks_stack_rev: List[List[Hook]] = [[hook] for hook in hooks_bottom_to_top]

        (backbone_slices,
         paired_slice) = NestedUNet.slice_backbone(encoder, size_change_ind_forward=size_change_ind_rev[::-1])

        NestedUNet.inspect_slice_sizes(img_size, backbone_slices, sizes, size_change_ind_rev[::-1])

        # final level is appended manually in the end - excluded
        num_levels = min(len(size_change_ind_rev) - 1, num_levels)

        head_list = []
        tail_list = []
        encoder_hook_list = []

        assert num_levels <= len(size_change_ind_rev), f"{num_levels} vs. {len(size_change_ind_rev) - 1}"
        for idx, level in enumerate(range(num_levels)):
            # start index for hooks and current size_change_ind_rev
            hook_start_ind = len(size_change_ind_rev) - level - 1
            # get the unet layers
            encoder_head = backbone_slices.pop(0)
            start_end_pair = paired_slice.pop(0)
            # sizes aligned with encoder
            # curr_sizes = sizes[:start_end_pair[-1]]
            # curr_change_ind_rev = size_change_ind_rev[hook_start_ind::]
            # [torch.Size([1, 64, 256, 256])]
            curr_hooks = hooks_stack_rev[hook_start_ind::]

            curr_encoder = encoder[:start_end_pair[-1]]
            # _ -> curr_hooks_basic_unet
            curr_sizes, curr_change_ind_rev, _, dummy_tensor_nested = BasicUNet.parse_encoder(curr_encoder, img_size)

            assert size_change_ind_rev[hook_start_ind::] == curr_change_ind_rev
            assert sizes[:start_end_pair[-1]] == curr_sizes
            head_list.append(encoder_head)
            # each unet should have its own hook

            inner_net: BasicUNet = NestedUNet.get_inner_net(curr_change_ind_rev,
                                                            curr_sizes=curr_sizes,
                                                            curr_hooks=curr_hooks,
                                                            curr_encoder=curr_encoder,
                                                            n_out=n_out,
                                                            img_size=img_size,
                                                            dummy_tensor_in=dummy_tensor_nested,
                                                            pool_sizes=pool_sizes,
                                                            ppm_flatten=ppm_flatten,
                                                            blur=blur, blur_final=blur_final,
                                                            y_range=y_range, act_cls=act_cls,
                                                            norm_type=norm_type,
                                                            skip_type=skip_type, init=init,
                                                            skip_bottleneck=skip_bottleneck,
                                                            **convlayer_kwargs)

            hooks_stack_rev = NestedUNet.new_extended_hooks(inner_net.decoder, hooks_stack_rev, hook_start_ind)
            NestedUNet.forward_hook_store_shape(inner_net, num_channel=3, img_size=img_size)
            decoder_head = inner_net.bridge + inner_net.decoder + inner_net.output
            tail_list.append(decoder_head)
            encoder_hook_list.append(inner_net.encoder_hook)

        # add the outer

        head_list.append(backbone_slices)
        out_unet: BasicUNet = NestedUNet.get_inner_net(size_change_ind_rev,
                                                       curr_sizes=sizes,
                                                       curr_hooks=hooks_stack_rev,
                                                       curr_encoder=encoder,
                                                       n_out=n_out,
                                                       img_size=img_size,
                                                       dummy_tensor_in=dummy_tensor_in,
                                                       pool_sizes=pool_sizes,
                                                       ppm_flatten=ppm_flatten,
                                                       blur=blur,
                                                       blur_final=blur_final,
                                                       y_range=y_range,
                                                       init=init,
                                                       act_cls=act_cls,
                                                       norm_type=norm_type,
                                                       skip_type=skip_type,
                                                       skip_bottleneck=skip_bottleneck,
                                                       **convlayer_kwargs)
        # rest of the path as encoder
        decoder_head = out_unet.bridge + out_unet.decoder + out_unet.output
        tail_list.append(decoder_head)
        encoder_hook_list.append(out_unet.encoder_hook)
        return head_list, tail_list, encoder_hook_list

    def __init__(self,
                 encoder: nn.Sequential,
                 *,
                 n_out: int,
                 num_levels: int = 0,
                 img_size: Tuple[int, int],
                 pool_sizes: List[int],
                 ppm_flatten: bool = False,
                 blur: bool = False,
                 blur_final: bool = True,
                 y_range=None,
                 act_cls=nn.ReLU,
                 init=nn.init.kaiming_normal_, norm_type=None,
                 skip_type: SKIP_TYPE = 'sum',
                 skip_bottleneck: bool = False,
                 **convlayer_kwargs):
        """For simplification, last_cross was removed for now as it is not in the scope of our study.

        Args:
            encoder:
            n_out:
            img_size:
            pool_sizes:
            ppm_flatten:
            blur:
            blur_final:
            y_range:
            act_cls:
            init: Initialization function. Note it assumes that the encoder is already initialized by itself.
                Therefore, only midconv and upsampling routes are initialized
            norm_type:
            skip_type:
            skip_bottleneck:
            **convlayer_kwargs:
        """
        super().__init__()
        self.skip_type = skip_type

        sizes, size_change_ind_rev, self.hooks_bottom_to_top, x = BasicUNet.parse_encoder(encoder, img_size)

        (head_list, tail_list,
         encoder_hook_list) = NestedUNet.nested_unet_modules(encoder, num_levels=num_levels,
                                                             n_out=n_out, img_size=img_size,
                                                             sizes=sizes,
                                                             size_change_ind_rev=size_change_ind_rev,
                                                             hooks_bottom_to_top=self.hooks_bottom_to_top,
                                                             dummy_tensor_in=x, pool_sizes=pool_sizes,
                                                             ppm_flatten=ppm_flatten, blur=blur, blur_final=blur_final,
                                                             y_range=y_range, act_cls=act_cls, norm_type=norm_type,
                                                             skip_type=skip_type, init=init,
                                                             skip_bottleneck=skip_bottleneck, **convlayer_kwargs)

        self.num_levels = num_levels

        # debug
        encoder_hook_list, model_list = NestedUNet.assemble(encoder_hook_list, head_list, tail_list)

        self.encoder_hook_list = encoder_hook_list
        self.model_list = nn.ModuleList(model_list)

    @staticmethod
    def assemble(encoder_hook_list: List[Hook],
                 head_list: List[nn.ModuleList | List[nn.Module]],
                 tail_list: List[nn.ModuleList | List[nn.Module]]) -> Tuple[List[Hook], List[nn.Module]]:
        head_hooks = []
        model_list = []

        for hk, head, tail in zip(encoder_hook_list, head_list, tail_list):
            head_hooks.append(hk)
            model_list.append(SeqExSpecifyOrig(*head, *tail))
        return head_hooks, model_list

    @staticmethod
    def collate_output(tensor: torch.Tensor | List[torch.Tensor],
                       reduce: Optional[TYPE_REDUCE] = 'sum',
                       dim: int = 0,
                       **reduce_kwargs) -> torch.Tensor | List[torch.Tensor]:
        if reduce is None or isinstance(tensor, torch.Tensor):
            return tensor
        if isinstance(tensor, List) and len(tensor) == 1:
            return tensor[0]
        # tensor is a list here
        match reduce:
            case 'cat':
                return torch.cat(tensor, dim=dim, **reduce_kwargs)
            case 'sum':
                return torch.stack(tensor).sum(dim=dim, **reduce_kwargs)
            case 'mean':
                return torch.stack(tensor).mean(dim=dim, **reduce_kwargs)
        return tensor

    def forward(self, x: torch.Tensor,
                reduce: Optional[TYPE_REDUCE] = 'sum',
                **reduce_kwargs) -> torch.Tensor | List[torch.Tensor]:
        out_list: List[torch.Tensor] = []
        prev_hook: Optional[Hook] = None

        for idx, (hk, model) in enumerate(zip(self.encoder_hook_list, self.model_list)):
            is_first = idx == 0
            input_tensor = x if is_first else prev_hook.stored
            out = model(input_tensor, orig=x)
            prev_hook = hk
            out_list.append(out)
        collated_out = NestedUNet.collate_output(out_list, reduce=reduce, **reduce_kwargs)
        return collated_out

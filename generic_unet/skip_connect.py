from typing import List, Callable, Optional, Literal
import torch
from fastai.layers import PixelShuffle_ICNR, BatchNorm, ConvLayer, NormType
from fastai.torch_core import Module, apply_init
from torch import nn
from torch.nn import functional as F
from fastai.callback.hook import Hook, Hooks


SKIP_CAT = Literal['cat']
SKIP_SUM = Literal['sum']
SKIP_TYPE = Literal[SKIP_CAT, SKIP_SUM]


def valid_add(tensor1, tensor2):
    """Check if two  tensors can be added (e.g., same shape or broadcast)
    """
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    # different dim and neither is one
    for dim1, dim2 in zip(shape1[::-1], shape2[::-1]):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False

    # different number of dimensions
    if len(shape1) != len(shape2):
        shorter, longer = sorted([shape1, shape2], key=len)
        if any(dim != 1 for dim in longer[:len(longer) - len(shorter)]):
            return False

    return True


class SkipBlock(Module):
    final_div: bool
    _x_in_c: int
    _up_in_c: int
    _x_in_c_original: int

    @staticmethod
    def channel_size(up_out_c: int, x_in_c: int, final_div: bool, skip_type: SKIP_TYPE = 'sum'):
        match skip_type:
            case 'cat':
                ni = up_out_c + x_in_c  # up_in_c // 2
            case 'sum':
                ni = up_out_c  # up_in_c // 2
            case _:
                raise ValueError(f"Invalid skip_type: {skip_type}")

        nf = ni if final_div else ni // 2
        return ni, nf

    @staticmethod
    def _skip_channel_collate(hook: Hook | Hooks | List[Hook], x_in_c: int, channel_dim: int = 1) -> int:
        if hook is None:
            return x_in_c
        match hook:
            case Hook():
                hook_size = x_in_c if hook.stored is None else hook.stored.shape[channel_dim]
                assert hook_size == x_in_c
            case Hooks() | list():
                hook_size_list = [h.stored.shape[channel_dim] for h in hook if h is not None and h.stored is not None]
                hook_size = sum(hook_size_list) if len(hook_size_list) > 0 else x_in_c

                assert (len(hook) <= 0 or hook[0] is None
                        or hook[0].stored is None
                        or hook[0].stored.shape[channel_dim] == x_in_c), f"{x_in_c} vs. {hook[0].stored.shape}"

            case _:
                raise TypeError(f"Unsupported Hook type: {type(hook)}")

        return hook_size

    def __init__(self,
                 up_in_c: int, x_in_c: int,
                 hook: Hook | Hooks | List[Hook],
                 final_div: bool = True, blur: bool = False,
                 act_cls: Callable = nn.ReLU,
                 init: Callable = nn.init.kaiming_normal_, norm_type: Optional[NormType] = None,
                 bottleneck: bool = False,
                 skip_type: SKIP_TYPE = 'sum',
                 **kwargs):
        super().__init__()
        self.skip_type = skip_type
        self.all_hook = hook
        self.final_div = final_div

        self._x_in_c_original = x_in_c
        self._x_in_c = SkipBlock._skip_channel_collate(self.all_hook, x_in_c, channel_dim=1)
        self._up_in_c = up_in_c

        up_out_c = self._up_in_c // 2 if skip_type == SKIP_CAT or final_div else up_in_c // 4
        # assert shuffle_nf == x_in_c, f"{up_in_c} != {x_in_c}"

        self.pixel_shuffle = PixelShuffle_ICNR(up_in_c, up_out_c,  # if final_div else up_in_c // 4
                                               blur=blur, act_cls=act_cls, norm_type=norm_type)
        self.bn = BatchNorm(self._x_in_c)

        ni, nf = SkipBlock.channel_size(up_out_c=up_out_c, x_in_c=self._x_in_c, final_div=final_div,
                                        skip_type=skip_type)
        conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        conv2 = ConvLayer(nf, nf, act_cls=act_cls, norm_type=norm_type, xtra=None, **kwargs)

        self.relu = act_cls()
        self.bottle_neck = bottleneck

        self.btn_conv = nn.Sequential(conv1, conv2) if self.bottle_neck else nn.Identity()
        apply_init(self.btn_conv, init)

    @staticmethod
    def _skip_connect(tensor1: torch.Tensor, tensor2, skip_type: SKIP_TYPE = 'sum'):
        match skip_type:
            case 'sum':
                assert valid_add(tensor1, tensor2), (f"{tensor1.shape} vs. {tensor2.shape}."
                                                     f" Since PixelShuffle is used for upsampling, the "
                                                     f"corresponding downsample layer in the encoder should reduce"
                                                     f"the size of channels accordingly."
                                                     f"Try using concatenate connection or add extra conv blocks"
                                                     f"to skip path.")
                return tensor1 + tensor2
            case 'cat':
                return torch.cat([tensor1, tensor2], dim=1)
            case _:
                raise ValueError(f"Invalid skip_type: {skip_type}")

    @staticmethod
    def hook_out_helper(hook: Hook):
        assert isinstance(hook, Hook)
        # assert hook.stored is not None
        return hook.stored

    @staticmethod
    def hook_size_collation_inplace(tensor_list: List[torch.Tensor], flag: bool):
        """
        """
        if (not flag) or tensor_list is None or len(tensor_list) == 0:
            return tensor_list

        # HW
        ref_shape = tensor_list[0].shape[-2:]

        for idx, tensor in enumerate(tensor_list):
            if tensor.shape[-2:] != ref_shape:
                # Resize the tensor to match the reference shape
                tensor_list[idx] = F.interpolate(tensor, size=ref_shape, mode='bilinear', align_corners=False)

        return tensor_list

    @staticmethod
    def get_hook_out(hook: Hook | List[Hook] | Hooks, skip_type: Optional[SKIP_TYPE] = None) -> torch.Tensor:
        if isinstance(hook, Hook):
            return SkipBlock.hook_out_helper(hook)
        assert skip_type is not None
        # if a list is encountered, then reduce the outcome first
        out_list = [SkipBlock.hook_out_helper(x) for x in hook]
        # disabled for now - for debugging: check if the activation in hooks on the same level have the same HW

        out_list = SkipBlock.hook_size_collation_inplace(out_list, flag=False)

        match skip_type:
            case 'sum':
                # noinspection PyTypeChecker
                return sum(out_list)
            case 'cat':
                return torch.cat(out_list, dim=1)
            case _:
                raise NotImplementedError(f"{skip_type}")

    def forward(self, up_in):

        hook_out = SkipBlock.get_hook_out(self.all_hook, self.skip_type)  # self.hook.stored
        up_out = self.pixel_shuffle(up_in)

        skip_shape = hook_out.shape[-2:]
        if skip_shape != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, hook_out.shape[-2:], mode='nearest')

        cat_x = SkipBlock._skip_connect(up_out, self.bn(hook_out), self.skip_type)
        cat_x = self.relu(cat_x)
        return self.btn_conv(cat_x)

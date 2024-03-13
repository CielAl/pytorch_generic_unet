from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from .resized_conv import ResizeConv


class SpatialPyramidPooling(nn.Module):
    grid_size_list: List[int]
    flatten: bool
    upconv_dict: nn.ModuleDict
    _do_upsample: bool

    def __init__(self, grid_size_list: List[int], flatten: bool = True,
                 feature_map_size: Optional[int] = None,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None):
        super().__init__()
        assert len(grid_size_list) > 0
        self.grid_size_list = grid_size_list
        self.flatten = flatten
        self.upconv_dict = nn.ModuleDict()
        # placeholder
        self._identity = nn.Identity()
        self.create_upconv_blocks(self.grid_size_list, feature_map_size, in_channels, out_channels)

    @staticmethod
    def output_size(grid_size_list: List[int], num_channels: int, flatten: bool):
        if flatten:
            sum([x ** 2 for x in grid_size_list]) * num_channels
        return len(grid_size_list) * num_channels

    def get_upsample_block(self, grid_size) -> Optional[nn.Module]:
        key = str(grid_size)
        if key not in self.upconv_dict:
            return self._identity
        return self.upconv_dict[key]

    def create_upconv_blocks(self, grid_size_list, new_size, in_channels, out_channels):
        self._do_upsample = new_size is not None

        optional_arg_check = [new_size is None, in_channels is None, out_channels is None]
        assert all(optional_arg_check) or not any(optional_arg_check), \
            f"new_size, in_channels, out_channels must all be none or not-none"
        if not self._do_upsample:
            return
        for grid_size in grid_size_list:
            upconv = ResizeConv(new_size=new_size, kernel_size=1, in_channels=in_channels, out_channels=out_channels)
            self.upconv_dict[str(grid_size)] = upconv

    @staticmethod
    def _pool(x: torch.Tensor, grid_size: int, flatten: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        pooled_output = F.adaptive_avg_pool2d(x, output_size=grid_size)
        if flatten:
            pooled_output = pooled_output.view(batch_size, -1, 1, 1)
        return pooled_output

    @staticmethod
    def _upsample(x: torch.Tensor, block: Optional[nn.Module], do_upsample: bool):
        if do_upsample:
            assert block is not None
            return block(x)
        return x

    def forward(self, x) -> torch.Tensor | List[torch.Tensor]:
        pooled_outputs = []
        for grid_size in self.grid_size_list:
            # pool given the target grid size
            pooled_output = SpatialPyramidPooling._pool(x, grid_size, self.flatten)
            # upsample if needed
            upconv = self.get_upsample_block(grid_size)
            pooled_output = SpatialPyramidPooling._upsample(pooled_output, upconv, self._do_upsample)
            pooled_outputs.append(pooled_output)

        spp_output = torch.cat(pooled_outputs, dim=1)
        return spp_output

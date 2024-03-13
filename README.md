# pytorch_generic_unet
Generic implementation of pytorch basic and nested unet.
Certain internal implementation (e.g., hooks) relies on ```fastai```.

# Basic Usage
Basic UNet with customizable encoder path:

```python
generic_unet.models.BasicUNet
```

Nested UNet with customizable encoder and number of nested levels.

```python
generic_unet.models.NestedUNet
```

# TODO
Detailed showcase and full documentation.

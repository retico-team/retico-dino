# retico-dino
A ReTiCo module for extracting visual features of detected objects with Meta's [DINOv2](https://github.com/facebookresearch/dinov2) models.

## Installation and Requirements

To use the DINO module you first need to install the retico-dino package:
* Install retico-dino:
```pip install retico-dino```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-dino.git```

It is advised to install PyTorch for your hardware first, following PyTorch's [install guide](https://pytorch.org/get-started/locally/), e.g.:
- For CUDA 12.6 (NVIDIA): `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
- For ROCm 6.4 (AMD): `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4`

## Modules

### `Dinov2ObjectFeatures`
Subscribes to `ExtractedObjectsIU` (produced by `ExtractObjectsModule` from `retico_vision`) and encodes each cropped object image with DINOv2, producing an `ObjectFeaturesIU` containing a dictionary of feature vectors, one per object.

#### Arguments:
* `model_name` (str): the DINOv2 model to load from `torch.hub`; defaults to `"dinov2_vits14"`
* `top_objects` (int): the maximum number of objects to encode per incoming frame; defaults to `1`
* `show` (bool): if `True`, opens each object crop in an image viewer as it's processed; defaults to `False`

### Example
**Note:** The example uses `WebcamModule` and `ExtractObjectsModule` from `retico_vision` and `SAMModule` from [retico-sam](https://github.com/retico-team/retico-sam) (`pip install retico-sam`). `ExtractObjectsModule` lets you choose how many objects are extracted from each frame.

```python
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from retico_core import *
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule, ExtractObjectsModule
from retico_sam.sam import SAMModule
from retico_dino.dino import Dinov2ObjectFeatures

path_var = '<path-to-sam-checkpoint>'  # e.g. sam_vit_h_4b8939.pth

webcam = WebcamModule()
sam = SAMModule(model='h', path_to_chkpnt=path_var, use_bbox=True)
extractor = ExtractObjectsModule(num_obj_to_display=1)
feats = Dinov2ObjectFeatures(show=False, top_objects=1)
debug = DebugModule()

webcam.subscribe(sam)
sam.subscribe(extractor)
extractor.subscribe(feats)
feats.subscribe(debug)

network.run(webcam)

print("Network is running")
input()

network.stop(webcam)
```

Citation
```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

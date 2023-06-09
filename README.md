# retico-dino
A ReTiCo module for DINO. See below for more information.

### Installation and requirements

### Example
```python
import sys
from retico import *

prefix = '/path/to/modules/'
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-yolov8')
sys.path.append(prefix+'retico-dino')

from retico_vision.vision import WebcamModule 
from retico_yolov8.yolov8 import Yolov8
from retico_dino.dino import Dinov2ObjectFeatures



webcam = WebcamModule()
yolo = Yolov8()
feats = Dinov2ObjectFeatures(show=True)
debug = modules.DebugModule()


webcam.subscribe(yolo)
yolo.subscribe(feats)
feats.subscribe(debug)

run(webcam)

print("Network is running")
input()

stop(webcam)
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
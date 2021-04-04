<img src="./images/arch.png"></img>

## ViViT: A Video Vision Transformer - Pytorch

Implementation of <a href="https://arxiv.org/pdf/2103.15691v1.pdf">ViViT</a>.
We present pure-transformer based models for video
classification, drawing upon the recent success of such models in image classification. Our model extracts spatiotemporal tokens from the input video, which are then encoded by a series of transformer layers. In order to handle the long sequences of tokens encountered in video, we
propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input. Although transformer-based models are known to only be effective when large training datasets are available, we show
how we can effectively regularise the model during training
and leverage pretrained image models to be able to train on
comparatively small datasets. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple
video classification benchmarks including Kinetics 400 and
600, Epic Kitchens, Something-Something v2 and Moments
in Time, outperforming prior methods based on deep 3D
convolutional networks. To facilitate further research, we
will release code and models.


## Notes:
* Currently the implementation only includes Model-3.
* For Model-2, refer to the repo: https://github.com/rishikksh20/ViViT-pytorch by [@rishikksh20](https://github.com/rishikksh20): 


## Usage

```python
import torch
from models import ViViTBackbone

v = ViViTBackbone(
    t=32,
    h=64,
    w=64,
    patch_t=8,
    patch_h=4,
    patch_w=4,
    num_classes=10,
    dim=512,
    depth=6,
    heads=10,
    mlp_dim=8,
    model=3
)

device = torch.device('cpu')
vid = torch.rand(32, 3, 32, 64, 64).to(device)

pred = v(vid)  # (32, 10)
```

## Citation:
```
@misc{arnab2021vivit,
      title={ViViT: A Video Vision Transformer}, 
      author={Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lučić and Cordelia Schmid},
      year={2021},
      eprint={2103.15691},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement:
* Code implementation is inspired from [@lucidrains](https://github.com/lucidrains) repo : https://github.com/lucidrains/vit-pytorch

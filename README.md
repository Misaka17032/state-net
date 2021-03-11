# State Net

State net is an deep learning framework which aims to improve the stability of training process.

## Requirements

Training data is required. Downloads ILSVRC2012 from [ImageNet](http://image-net.org/) and then put it in /data/.

It should be:
```
- data
-- train
--- 0
---- 1.jpg, 2.jpg ...
--- 1
...
```

[pytorch](https://github.com/pytorch/pytorch) and [numpy](https://github.com/numpy/numpy) are also required. Follow the instructions in the links to install.

## Runtime

Training: `python train.py`

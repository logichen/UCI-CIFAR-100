### MicroNet-CIFAR100: *EfficiennNet-B0 with Angleloss and Label Refinery*
By [Biao Yang] biaoy1@uci.edu,
[Fanghui Xue] fanghuix@uci.edu,
[Jiancheng Lyu] jianlyu@qti.qualcomm.com,
[Shuai Zhang] shuazhan@qti.qualcomm.com,
[Yingyong Qi] yingyong@qti.qualcomm.com,
and [Jack xin] jack.xin@uci.edu


### Introduction
This is a pytorch training script for light weight network on
CIFAR-100. We aim to attend the MicroNet Chanllenge hosted at NeurIPS 2019. Our codes are modified from label refinery:
(https://arxiv.org/abs/1805.02641), EfficientNet: (https://arxiv.org/pdf/1905.11946.pdf) and additive margin softmax loss: (https://arxiv.org/pdf/1801.05599.pdf).

Our model is based on EfficientNet. We changed the structure to meet the input size of 32x32. And we also changed the cross-entropy to additive margin softmax loss with s=5.0 and m=0.0.
Also we enlarged our dataset with different transformations of the original CIFAR-100, without usage of any extra data.


### Usage
#### Prerequisite
You need Python3.5+ to implement this code. Other requirements can be found in [requirements.txt](requirements.txt).
You may install the packages by:
```
pip3 install -r requirements.txt
```

#### Train models
Before you run the code, you should make sure you have downloaded the data. 
Fisrt, you need to train EfficientNet-B3 with default parameters. Usually the best result can achieve an accuracy of over 79%. Choose the best model.
Then you can retrain the EfficientNet-B3 with the refined labels of the previous best model. After the second training of EfficientNet-B3, it can achieve 
an accuracy of over 80%. Now you can train EfficientNet-ex with the refined labels of EfficientNet-B3. 

1. Train EfficientNet-B3:
```
python train.py --model efficientnet_b3 --data_dir (path to you data) --s (default 5.0) --coslinear True
```
2. Train EfficientNet-B3 with refined labels:
```
python train.py --model efficientnet_b3 --label-refinery-model efficientnet_b3 --label-refinery-state-file (path to best model_state.pytar) --s (default 5.0) --coslinear True
```
3. Train EfficientNet-B0 with refined labels of EfficientNet-B3:
```
python train.py --model efficientnet_b0 --label-refinery-model efficientnet_b3 --label-refinery-state-file (path to best model_state.pytar) --s (default 5.0) --coslinear True
```
4. Train EfficientNet-ex with refined labels of EfficientNet-B0:
```
python train.py --model efficientnet_ex --label-refinery-model efficientnet_b0 --label-refinery-state-file (path to best model_state.pytar) --s (default 5.0) --coslinear True
```

#### Test models
To test a trained EfficientNet-B0 model:
```
python test.py --model efficientnet_ex --model-state-file (path to best model.pytar) --data_dir (path to you data)
```


#### Print number of operations and parameters
```
python ./torchscope-master/count.py
```

#### Running result:
Counting multiplication operations as 1/2 operation

Total params: 2,793,064
Trainable params: 2,773,064
Non-trainable params: 20,000
Total FLOPs: 251,891,224
Total Madds: 382,885,052.0

Input size (MB): 0.01
Forward/backward pass size (MB): 42.61
Params size (MB): 2.66
Estimated Total Size (MB): 45.28
FLOPs size (GB): 0.25
Madds size (GB): 0.38


Counting multiplication as 1 operation

Total params: 2,793,064
Trainable params: 2,773,064
Non-trainable params: 20,000
Total FLOPs: 251,891,224
Total Madds: 512,835,116

Input size (MB): 0.01
Forward/backward pass size (MB): 42.61
Params size (MB): 2.66
Estimated Total Size (MB): 45.28
FLOPs size (GB): 0.25
Madds size (GB): 0.51


*Change the "mul_factor" in the line 83 of "scope.py" to decide how to count multiplications:
madds = compute_madd(module, input[0], output, mul_factor = 0.5)


#### Scoring: 0.074762
EfficiennNet-ex with Angleloss and Label Refinery:
Accuracy: 80.12% (efficientnet_b0.pytar)
Parameter number: 2,793,064
Total operations: 382,885,052.0

WideResNet-28-10
Parameter number: 36.5M
Total operations: 10.49B

Scoring:
Since we do not use quantization method.
Count all parameters as 16-bit and 1 multiplication operation as 1/2 operation:
(0.5 x 2,793,064)/36.5M + 382,885,052.0/10.49B = 0.074762

Count all parameters as 16-bit and 1 multiplication operation as 1 operation:
2,793,064/36.5M + 512,835,116/10.49B = 0.12541

#### License
By downloading this software you acknowledge that you read and agreed all the
terms in the `LICENSE` file.

Sep 29th, 2019

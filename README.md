# HADS: Hardware-Aware Deep Subnetworks 

In this work we introduce Resource-Efficient Deep Subnetworks (REDS) to tackle model adaptation to variable resources. In contrast to the state-of-the-art, REDS use structured sparsity constructively by exploiting permutation invariance of neurons, which allows for hardware-specific optimizations. Specifically, REDS achieve computational efficiency by (1) skipping sequential computational blocks identified by a novel iterative knapsack optimizer, and (2) leveraging simple math to re-arrange the order of operations in REDS computational graph to take advantage of the data cache. REDS support conventional deep networks frequently deployed on the edge and provide computational benefits even for small and simple networks. We evaluate REDS on six benchmark architectures trained on the [Google Speech Commands](https://arxiv.org/pdf/1804.03209.pdf), [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets, and test on four off-the-shelf mobile and embedded hardware platforms. We provide a theoretical result and empirical evidence for REDS outstanding performance in terms of submodels' test set accuracy, and demonstrate an adaptation time in response to dynamic resource constraints of under 40 microseconds of models deployed on [Arduino Nano 33 BLE Sense](https://docs.arduino.cc/hardware/nano-33-ble-sense) through [Tensorflow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro). 

## Software prerequisites

Install the software packages required for reproducing the experiment by running the
command: `pip3 install -r requirements.txt` inside the project folder.

Run the `setup.sh` script file to create the hierarchy of folders used to store the results of the experiments.

[Install](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-)
the [GUROBI](https://www.gurobi.com/) solver and obtain
a license (see [free academic license](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license-)).
To link the license and the solver to the programs you have to pass the arguments: **--gurobi_home** and
**--gurobi_license_file** to each program. The former points to the absolute path of the installation of Gurobi and the
latter to its license.

```
python kws_ds_convolution.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```

Change linux64 with your operating system version/type.

## Training REDS models

For each program, you can specify the usage of the GPU by passing an id number from the  `--cuda_device` argument. In the default configuration, all the experiments results are stored inside the /logs directory and printed to the screen.
For each program, you can specify the solver's maximum running time per iteration by passing the value in seconds to the
`--solver_time_limit` argument. For the [DS-CNN size L](https://arxiv.org/pdf/1711.07128.pdf), the suggested time is at least 3 hours (10800
seconds).

All the individual subnetwork architectures can be trained in isolation by running the `_full_training.py` files. 

**Train DS-CNN models**

```
python kws_ds_convolution.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```

To train the REDS DS-CNN S models on *CIFAR10* or *Fashion-MNIST* run for the former the *vision_ds_convolution_fashion_mnist.py* file and for the latter *vision_ds_convolution_cifar10.py* file. The pre-trained models are stored in the models/ folder.

**Train DNN models**

```
python kws_dnn.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```

**Train CNN models**

```
python kws_convolution_cnn.py --gurobi_license_file path/to/license/gurobi.lic --gurobi_home path/to/installation//gurobi/gurobi1002/linux64 
```


## Analysis results on Pixel 6 and Xiaomi Redmi Note 9 Pro 
The results obtained from the subnetworks configuration are obtained from the official Google
Tensorflow Lite benchmarking [tool](https://www.tensorflow.org/lite/performance/measurement). From left to right: number of model parameters, model accuracy and model inference
as a function of MAC percentage in each REDS subnetwork. 

**(1) Models size S**

<img src="result/plots/plotly_mobile_parameters_sizeS.png" width="270"/> <img src="result/plots/plotly_mobile_accuracy_sizeS.png" width="270"/> <img src="result/plots/plotly_mobile_inference_sizeS.png" width="270"/> 

**(2) Models size L**

<img src="result/plots/plotly_mobile_parameters_sizeL.png" width="270"/> <img src="result/plots/plotly_mobile_accuracy_sizeL.png" width="270"/> <img src="result/plots/plotly_mobile_inference_sizeL.png" width="270"/>


## TensorFlow Lite for Microcontrollers Analysis 
REDS zero-overhead was assessed on Tensorflow Lite for Microcontrollers by [implementing](https://github.com/aschesklave/TFlite-Micro-Cutting/blob/main/src/model_modifier.cpp) the runtime dynamic adaptation of the deployed model and by [modifying](https://github.com/chint95/tflite-micro/blob/b6878ecd464fcf442ee03a4e279257b20bcda731/src/tensorflow/lite/kernels/internal/reference/fully_connected.h#L31) the fully connected floating point kernel. 

## Knapsack for depth-wise convolutions
REDS's iterative knapsack for depth-wise convolutions is modelled with [OR-Tools](https://developers.google.com/optimization) and its implementation can be found [here](https://github.com/FraCorti/REDS/blob/b47f9d5e370e5a7b5a726e832d00030ea401ee00/utils/knapsack.py#L475).  

## BibTeX

If you found this repository useful, please consider citing our work.
```
@article{corti2023reds,
  title={REDS: Resource-Efficient Deep Subnetworks for Dynamic Resource Constraints},
  author={Corti, Francesco and Maag, Balz and Schauer, Joachim and Pferschy, Ulrich and Saukh, Olga},
  journal={arXiv preprint arXiv:2311.13349},
  year={2023}
}
```

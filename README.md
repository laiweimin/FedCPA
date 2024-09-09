# FedCPA

This repository hosts the implementation of FedCPA: Federated Camouflaged Poisoning Attack. 

FedCPA is crafted to perform covert and targeted poisoning attacks within Federated Unlearning (FU) environments. 

The codebase builds upon PFLlib, a widely-used library for federated learning experiments on GitHub. 

In addition to implementing FedCPA, this project introduces various Federated Unlearning algorithms ("Retrain", "FedEraser", "FedRecover", "FedRecovery", "FedForgotten") and extends Byzantine-robust aggregation rules("Krum", "TrimmedMean", "Bulyan", "LIE", "Median") to counteract the FedCPA threat effectively.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following packages:
- numpy
- pandas
- pytorch

### Preparing the Dataset
Before running the training script, it is necessary to partition the CIFAR-10 dataset for an IID and balanced scenario. Use the following commands to prepare the dataset:

```bash
cd ./dataset
python generate_cifar10.py iid balance -
```
This will create the necessary data partitions to ensure that each client in the federated learning setup receives a balanced and independently identically distributed subset of the CIFAR-10 dataset.

### Training Configuration
Once the dataset is prepared, you can proceed to configure and run the training process.

The main parameters for configuring the camouflage poisoning attacks are set through the command line as follows:

```bash
cd ./system
python main.py -lbs 32 -nc 20 -lr 0.01 -ls 10 -jr 1 -nb 10 -data Cifar10 -m cnn -algo Camouflaged_FedAvg -gr 100 -did 0 -go cnn -cstart 5 -crestart 3 -cimages 5 -def NoDefense -unl Retrain -camoutype WithinClient -unlt UnlearningData
```

## Detailed Parameter Descriptions
### Device Configuration
- `-did` or `--device_id`: Specifies the GPU device ID for training if using CUDA.
### Training Configuration
- `-gr` or `--global_rounds`: Total number of training rounds to be executed, default is `100`.
- `-lr` or `--local_learning_rate`: Specifies the learning rate for local model updates, default is `0.01`.
- `-lbs` or `--batch_size`: Sets the batch size for training, default is `32`.
- `-ls` or `--local_epochs`: Number of epochs each client will train locally per round, default is `10`.
- `-t- or `--times-: Number of running times.
### Attack Strategies
* ```-camoutype:``` or ```--camouflage_type```: Type of camouflage ```choices: "WithinClient", "CrossClient"```. Default is ```"WithinClient"```.
* ```-cstart``` or ```--camouflage_start_epoch```: Epoch at which to start camouflage. Default is `5`.
* ```-ceps``` or ```--camouflage_eps```: Epsilon value for perturbations. Default is `16`.
* ```-crestart``` or ```--camouflage_restarts```: Number of restarts for camouflage attempts. Default is `3`.
* ```-cattackiter``` or ```--camouflage_attackiter```: Number of iterations for the camouflage attack. Default is `100`.
* ```-cimages``` or ```--camouflage_images_count```: Number of images to camouflage. Default is `5`.
### Unlearning Options
- `-unl` or `--unlearning`: Specifies the type of federated unlearning method to use, options include ```"NoUnlearning"```, ```"Retrain"```, ```"FedEraser"```, ```"FedRecover"```, ```"FedRecovery"```, ```"FedForgotten"```.
- `-unlt` or `--unlearning_type`: Determines the scope of data unlearning, options are ```"UnlearningData"``` for specific data points or ```"UnlearningClient"``` for entire client data. Default is ```"UnlearningData"```.
### Defense Mechanisms
- `-def` or `--defense`: Type of defense mechanism to be applied, choices are ```"NoDefense"```, ```"Krum"```, ```"TrimmedMean"```, ```"Bulyan"```, ```"LIE"```, ```"Median"```. Default is ```"NoDefense"```.

## Acknowledgments
Credit goes to the developers of [PFLlib](https://github.com/TsingZ0/PFLlib), whose tools were crucial in developing the FedCPA. 

Modifications and extensions were made to suit the specific needs of federated unlearning attack scenarios.

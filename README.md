## About this project

## Features

## Getting Started
#### Create and activate new conda environment
```
conda create -n crowdmagail python=3.8
conda activate crowdmagail
```

#### Install pip requirements
```
pip install requirements.txt
```

#### Install Python bindings for ORCA
```
cd ./model/Python-RVO2
python setup.py build
python setup.py install
```

#### Install CARLA Simulator
Please refer to: https://carla.readthedocs.io/en/latest/build_linux/

## Usage
#### Training
Execute the following, the model will be saved in `./checkpoint/testproj/model_final.bin` by default. \
See the function `get_args()` in `utils/utils.py` for default parameters.
```
python train.py
```
Training using grid search, the default configuration path is located at `./configs/exp_configs/test.yaml`.
```
python run_experiments.py
```
To use tensorboard:
```
tensorboard --logdir <LOG_DIR>
```

#### Evaluation
Run this script to evaluate to calculate the metrics of the model.
```
python evaluate.py --LOAD_MODEL <MODEL_PATH>
```

#### Visualization @ PyQt5
Run this script to visualize the performance of the model.
```
python visualize_qt.py --LOAD <MODEL_PATH>
```

#### Visualization @ CARLA
First, navigate to the folder where CARLA is installed. Then run the CARLA simulator.
```
cd CARLA_0.9.12
sh CarlaUE4.sh
```

Then, run the script to visualize the model in CARLA.
```
cd carla
python visualize_carla.py
```

## Demonstrations

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
```
python train.py
```

#### Evaluation
```
python evaluate.py --LOAD_MODEL <MODEL>
```

#### Visualization
```
python visualize.py --LOAD <MODEL>
```

`<MODEL> = ./checkpoint/demonstration/model_final.bin`

tensorboard --logdir <LOG_DIR>

## Demonstrations
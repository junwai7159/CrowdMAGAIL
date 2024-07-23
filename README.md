## About this project
Crowd simulation holds significant practical implications in fields such as urban planning and entertainment. Simulating realistic virtual crowds is a challenging yet crucial task for various applications. Microscopic crowd simulation, particularly modeling the obstacle avoidance behavior of pedestrians is one of the most complex aspects in crowd simulation. The process which pedestrians to avoid obstacles, both static and dynamic is pivotal for achieving realistic crowd simulation. Inspired by the recent advances in deep multi-agent reinforcement learning, in this paper we propose Crowd-MAGAIL, a data-driven microscopic crowd simulation model based on the MAPPO backbone algorithm. 

## Features
- **Feature extraction** module, which extracts key intrinsic and extrinsic features of pedestrians from their local observations.
- **Decision** module, responsible for sampling the deciding the actions of all pedestrians using GMM.
- **Attention module** models pedestrian interactions with other pedestrians or obstacles.
- Data-driven **GAIL reward** based on the GC and UCY pedestrian trajectory dataset where MAPPO-Clip and W-GAN serves as the generator and discriminator respectively.

The overall model architecture is CTDE-based, i.e. Centrailized Training Decentralized Execution:
<img src="https://github.com/user-attachments/assets/ab117d51-ffc8-4f57-a8db-5a7d9dab6875" width="700">

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
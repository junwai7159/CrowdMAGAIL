import argparse
import random 
import torch 
import numpy as np

########## CONFIG ########## 
def get_args(enable_carla=False, **kwargs):
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument("--UUID", type=str, default="testproj", help='custom model name')
    parser.add_argument("--SAVE_DIRECTORY", type=str, default="checkpoint", help='save directory')
    parser.add_argument("--LOAD_MODEL", type=str, default=None, help='pretrained model to load')
    parser.add_argument("--SEED", type=int, default=42, help='random seed')
    parser.add_argument("--DEVICE", type=str, default="cuda", help='cpu or cuda')
    parser.add_argument("--MODEL", type=str, default="MAGAIL", help="SFM or ORCA or TECRL or BC or MAGAIL")

    # parameters in environments
    parser.add_argument("--DT", type=float, default=0.08, help='simulation time step')
    parser.add_argument("--DYNAMIC", action="store_true", help='include dynamic agents')
    parser.add_argument("--NO_COLLISION_DETECTION", action="store_true", help='disable the collision detection mechanism')
    parser.add_argument("--NUM_PED", type=int, default=20, help='the number of pedestrians in the environment')
    parser.add_argument("--NUM_OBS", type=int, default=10, help='the number of obstacles in the environment')
    parser.add_argument("--SIZE_ENV", type=float, default=10.0, help='half the length of the side of the environment')
    parser.add_argument("--RW_ARRIVE", type=float, default=1.0, help='reward for arrived pedestrian')
    parser.add_argument("--RW_WORK", type=float, default=-0.01, help='reward: process work')
    parser.add_argument("--RW_ENERGY", type=float, default=-0.0015, help='reward: energy consumption')
    parser.add_argument("--RW_MENTAL", type=float, default=6.3, help='reward: mental effort')
    parser.add_argument("--RW_SMOOTH_V", type=float, default=-4.0, help='reward: smooth speed')
    parser.add_argument("--RW_SMOOTH_W", type=float, default=-1.0, help='reward: smooth direction')
    parser.add_argument("--RW_URGENT", type=float, default=-0.01, help='reward: GAIL')
    parser.add_argument("--RW_COLL", type=float, default=-0.01, help='reward: collision')
    parser.add_argument("--RW_GAIL", type=float, default=2.0, help='reward: GAIL')
    parser.add_argument("--SCENARIO", type=str, default="RANDOM", help='scenario of the environment')
    parser.add_argument("--DATASET", type=str, default="UCY", help='dataset used')

    # parameters in agents
    parser.add_argument("--MEMORY_CAPACITY", type=int, default=600)    # batch_size for generator
    parser.add_argument("--MAX_EPISODES", type=int, default=50000)
    parser.add_argument("--MAX_EP_STEPS", type=int, default=200)
    parser.add_argument("--G_EPOCH", type=int, default=128)
    parser.add_argument("--D_EPOCH", type=int, default=10)
    parser.add_argument("--BATCH_SIZE", type=int, default=600)                   # batch_size for discriminator 
    parser.add_argument("--GAMMA", type=float, default=0.95)
    parser.add_argument('--LAMBDA', type=float, default=0.9)
    parser.add_argument("--EPSILON", type=float, default=0.2)
    parser.add_argument('--ENTROPY', type=float, default=0.01)    
    parser.add_argument("--LR_0", type=float, default=1e-4, help='learning rate: the feature extractor')
    parser.add_argument("--LR_A", type=float, default=1e-4, help='learning rate: actor')
    parser.add_argument("--LR_C", type=float, default=3e-4, help='learning rate: critic')
    parser.add_argument("--LR_D", type=float, default=1e-5, help='learning rate: discriminator')
    parser.add_argument("--H_FEATURE", type=str, default="(128, )")
    parser.add_argument("--H_ATTENTION", type=str, default="(128, )")
    parser.add_argument("--H_SEQ", type=str, default="(128, 128)")
    parser.add_argument("--H_DISCRIMINATOR", type=str, default="(128, )")

    # CARLA
    if enable_carla:
        parser.add_argument('--ego-walker', type=int, default=1, help="set ego walker to attach camera")
        parser.add_argument('--offset', type=str, default="(-41, 238)", help='offset of (x, y) coordinates from the origin')
        parser.add_argument( '--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
        parser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
        parser.add_argument('-n', '--number-of-vehicles', metavar='N', default=0, type=int, help='Number of vehicles (default: 30)')
        parser.add_argument('--safe', action='store_true', help='Avoid spawning vehicles prone to accidents')
        parser.add_argument('--filterv', metavar='PATTERN',default='vehicle.*', help='Filter vehicle model (default: "vehicle.*")')
        parser.add_argument('--generationv', metavar='G', default='All', help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
        parser.add_argument('--filterw', metavar='PATTERN', default='walker.pedestrian.*', help='Filter pedestrian type (default: "walker.pedestrian.*")')
        parser.add_argument('--generationw', metavar='G', default='2', help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
        parser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port to communicate with TM (default: 8000)')
        parser.add_argument('--asynch', action='store_true', help='Activate asynchronous mode execution')
        parser.add_argument('--hybrid', action='store_true', help='Activate hybrid mode for Traffic Manager')
        parser.add_argument('--car-lights-on', action='store_true', default=False, help='Enable automatic car light management')
        parser.add_argument('--hero', action='store_true', default=False, help='Set one of the vehicles as hero')
        parser.add_argument('--respawn', action='store_true', default=False, help='Automatically respawn dormant vehicles (only in large maps)')
        parser.add_argument('--no-rendering', action='store_true', default=False, help='Activate no rendering mode')

    # custom parameters
    for key, value in kwargs.items():
        if type(value) is dict:  # add a new parameter, such as `get_args(NEW_ARG=dict(type=str, default='new_arg_value'))`
            parser.add_argument(f"--{key}", **value)
        else:  # reset default value of a existing parameter, `such as get_args(DEVICE='cuda')`
            parser.set_defaults(**{key: value})
    args, unknown = parser.parse_known_args()
    if len(unknown):
        print(f'[Warning] Unknown Arguments: {unknown}')
    return args



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


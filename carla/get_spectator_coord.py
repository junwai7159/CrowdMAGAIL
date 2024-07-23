import glob
import os
import sys
import time
import random
import logging
import shutil
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import carla
from carla import VehicleLightState as vls
import numpy as np
import torch

from configs.config import get_args, set_seed

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def main():
    ARGS = get_args(enable_carla=True)       
    ARGS.DEVICE = 'cpu'       
    set_seed(ARGS.SEED)

    ########## CARLA SIMULATOR ##########


    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(ARGS.host, ARGS.port)
    client.set_timeout(10.0)
    synchronous_master = False

    try:
        ###### load world #####
        world = client.get_world()

        ###### load spectator #####
        spectator = world.get_spectator()
        # spectator.set_transform(carla.Transform(carla.Location(), carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)))

        ##### traffic manager #####
        traffic_manager = client.get_trafficmanager(ARGS.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if ARGS.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if ARGS.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if ARGS.SEED is not None:
            traffic_manager.set_random_device_seed(ARGS.SEED)

        ###### define settings #####
        settings = world.get_settings()
        if not ARGS.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = ARGS.DT
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if ARGS.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, ARGS.filterv, ARGS.generationv)
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = get_actor_blueprints(world, ARGS.filterw, ARGS.generationw)
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        if ARGS.safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if ARGS.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif ARGS.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, ARGS.number_of_vehicles, number_of_spawn_points)
            ARGS.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if ARGS.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)
        
        ##### simulate carla #####
        while True:
            if not ARGS.asynch and synchronous_master:
                print(spectator.get_location())
                world.tick()

            else:
                world.wait_for_tick()

    finally:
        if not ARGS.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
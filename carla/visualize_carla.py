import glob
import os
import sys
import time
import random
import logging
import shutil
from queue import Queue, Empty
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import carla
from carla import VehicleLightState as vls
import numpy as np
import torch

from configs.config import get_args, set_seed
from envs.pedsim import Pedsim
from model.ppo import PPO
from model.sfm import SFM
from model.orca import ORCA
from utils.envs import init_env
from utils.state import pack_state
from utils.visualization_cv import *
from img_to_vid import img_to_vid

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_data.frame))

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

    # initialize env & model
    env = Pedsim(ARGS)
    env_carla = Pedsim(ARGS)
    init_env(env, ARGS)
    init_env(env_carla, ARGS)
    
    if ARGS.MODEL in ['TECRL', 'MAGAIL']:
        model = PPO(ARGS).to(ARGS.DEVICE)
        if ARGS.LOAD_MODEL is not None:
            model.load_state_dict(torch.load(ARGS.LOAD_MODEL, map_location=torch.device(ARGS.DEVICE)))
        env.ppo = model
    else:
        if ARGS.MODEL == 'SFM':
            model = SFM(env, ARGS)
        elif ARGS.MODEL == 'ORCA':
            model = ORCA(env, ARGS)

    ##### simulate model ######
    print(f'Simulating {ARGS.MODEL} ...', end='\n')
    while True:
        with torch.no_grad():
            mask = env.mask[:, -1] & ~env.arrive_flag[:, -1]
            if not mask.any():
                break
            if ARGS.MODEL in ['TECRL', 'MAGAIL']:
                action = torch.full((env.num_pedestrians, 2), float('nan'), device=env.device)
                state = pack_state(*env.get_state())
                action[mask, :], _ = model(state[mask], explore=True)
                env.action(action[:, 0], action[:, 1], enable_nan_action=True)
            elif ARGS.MODEL in ['SFM', 'ORCA']:
                model()
    

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
        for layer in [carla.MapLayer.Buildings, carla.MapLayer.Decals, carla.MapLayer.Foliage, carla.MapLayer.ParkedVehicles, 
                                carla.MapLayer.Particles, carla.MapLayer.Props, carla.MapLayer.StreetLights, carla.MapLayer.Walls]:
            world.unload_map_layer(layer)

        ###### load spectator #####
        x_offset, y_offset = eval(ARGS.offset)
        spectator = world.get_spectator()
        loc = carla.Location(x=x_offset, y=y_offset, z=30.0)
        rot = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        spectator.set_transform(carla.Transform(loc, rot))

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
        record_file_dir = '/home/violeteyes/.config/Epic/CarlaUE4/Saved/test1.log'
        print("Recording on file: %s" % client.start_recorder(record_file_dir))

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

        # create sensor queue
        sensor_queue = Queue()

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = ARGS.hero
        for n, transform in enumerate(spawn_points):
            if n >= ARGS.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if ARGS.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 1.0     # how many pedestrians will walk through the road

        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(ARGS.NUM_PED):
            spawn_point = carla.Transform()
            loc = carla.Location(x=float(env.position[i, 0, 0]), y=float(env.position[i, 0, 1]), z=1)
            if (loc != None):
                spawn_point.location = loc
                # world.debug.draw_string(loc, 'O', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=50.0, persistent_lines=True)
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        batch = []
        for i in range(len(walkers_list)):
            control = carla.WalkerControl(carla.Vector3D(x=0, y=0),speed=float(walker_speed[i]))
            all_actors[i].apply_control(control)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if ARGS.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        # 6. set camera on ego walker
        ego_walker = all_actors[ARGS.ego_walker]
        print(ego_walker)
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(640))
        cam_bp.set_attribute("image_size_y",str(480))
        cam_bp.set_attribute("fov",str(105))
        cam_location = ego_walker.get_transform().location
        cam_rotation = ego_walker.get_transform().rotation
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_walker, attachment_type=carla.AttachmentType.Rigid)
        ego_cam.listen(lambda image: sensor_callback(image, sensor_queue, 'cam_rgb'))

        
        ##### simulate carla #####
        t = 1
        while True:
            if not ARGS.asynch and synchronous_master:
                world.tick()

                # sensor (rgb camera)
                try:
                    image, frame = sensor_queue.get(True, 1.0)
                    image.save_to_disk(f'./result/Carla/image_frame/{image.frame}.jpg')
                except Empty:
                    print('Some of the sensor information is missing')

                
                # walker control
                mask_carla = [float('nan')] * len(walkers_list)
                position_carla = []
                velocity_carla = []
                for i in range(len(walkers_list)):
                    loc, rot = all_actors[i].get_transform().location, all_actors[i].get_transform().rotation
                    delta_x = (env.position[i, t, 0].item() - loc.x) 
                    delta_y = (env.position[i, t, 1].item() - loc.y) 
                    speed = torch.norm(env.velocity[i, t, :]).item()

                    # if i == ARGS.ego_walker:
                        # spectator.set_transform(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=2.3), rot))

                    if np.isnan([delta_x, delta_y, speed]).any():
                        mask_carla[i] = False
                        control = carla.WalkerControl(carla.Vector3D(x=0.0, y=0.0),speed=0.0)
                    else:
                        mask_carla[i] = True
                        control = carla.WalkerControl(carla.Vector3D(x=delta_x, y=delta_y),speed=speed)
                    all_actors[i].apply_control(control)

                    if mask_carla[i]:
                        position_carla.append([loc.x, loc.y])
                        velocity_carla.append([(loc.x - env_carla.position[i, -1, 0]) / ARGS.DT, (loc.y - env_carla.position[i, -1, 1]) / ARGS.DT])
                    else:
                        position_carla.append([float('nan'), float('nan')])
                        velocity_carla.append([float('nan'), float('nan')])

                # add step to env_carala
                # env_carla.add_step(torch.tensor(position_carla), torch.tensor(velocity_carla))

            else:
                world.wait_for_tick()
            
            if t + 1 < env.num_steps:
                t += 1
            else:
                break

    finally:
        if not ARGS.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        print("Stop recording")
        client.stop_recorder()
        shutil.move(record_file_dir, './result/Carla/test1.log')
        ego_cam.destroy()
        img_to_vid(input_path='./result/Carla/image_frame/', output_path='./result/Carla/test.mp4')
        
        time.sleep(0.5)

        ##### plot results #####
        # plot_traj(env_carla, save_path=f'./result/Carla/traj.png', xrange=(-ARGS.SIZE_ENV + x_offset, ARGS.SIZE_ENV + x_offset), yrange=(-ARGS.SIZE_ENV + y_offset, ARGS.SIZE_ENV + y_offset))


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
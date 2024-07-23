import math
import torch
import numpy as np
from utils.metrics import calc_collision
from envs.pedsim import Pedsim

########## ENVS ##########
# 定义行人、障碍物、地图
def init_ped_obs_random(ARGS):
    n1, n2, size = ARGS.NUM_PED, ARGS.NUM_OBS, ARGS.SIZE_ENV
    positions = torch.distributions.Uniform(-size, size).sample([n1, 2])
    destinations = torch.distributions.Uniform(-size, size).sample([n1, 2])
    obstacles = torch.distributions.Uniform(-1.5 * size, 1.5 * size).sample([n2, 2])
    return positions, destinations, obstacles

def init_ped_circle(ARGS, circle_radius=5.0):
    angles = np.linspace(0, 2*np.pi, ARGS.NUM_PED, endpoint=False)
    positions = [[circle_radius * np.cos(theta), circle_radius * np.sin(theta)] for theta in angles]
    destinations = [[-pos[0], -pos[1]] for pos in positions]
    return torch.tensor(positions, dtype=torch.float32), torch.tensor(destinations, dtype=torch.float32)

def init_ped_corridor(ARGS, length=20.0, width=7.0, vertical=False, horizontal=True):
    positions, destinations = list(), list()
    rng = np.random.RandomState()
    r2 = 0.3 ** 2
    i, placeable = 0, None
    
    # Positions
    while len(positions) < ARGS.NUM_PED:
        if vertical:
            x = rng.rand() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5
            if y < 0: 
                y -= 0.25
            else: 
                y += 0.25
        elif horizontal:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            y = rng.random() - 0.5
            if vertical: y *= 0.5
        
        if vertical:
            ped_pos0 = [x * (width / 2), y * (length / 2) * 2.0]
        elif horizontal:
            ped_pos0 = [x * (length / 2) * 2.0, y * (width / 2)]

        for ped_pos1 in positions:
            dist2 = (ped_pos0[0] - ped_pos1[0])**2 + (ped_pos0[1] - ped_pos1[1])**2
            if dist2 <= 3.0: # Collision
                placeable = False
            placeable = True
        if placeable or not positions:
            positions.append(ped_pos0)
    
    # Destinations
    while len(destinations) < ARGS.NUM_PED:
        if vertical:
            x = rng.random() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5 
            if y < 0:
                y -= 0.25
            else:
                y += 0.25
            if ((positions[i][1] > 0 and y > 0) or (positions[i][1] < 0 and y < 0)):
                y = -y
        elif horizontal:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            if (positions[i][0] > 0 and x > 0) or (positions[i][0] < 0 and x < 0):
                x = -x
            y = rng.random()-0.5
            if vertical: y *= 0.5
        
        if vertical:
            x *= width / 2
            y *= (length / 2) * 2.0
        elif horizontal: 
            x *= (length / 2) * 2.0
            y *= width / 2

        if (positions[i][0] - x)**2 + (positions[i][1] - y)**2 <= r2:
            continue
        placeable = True
        for gx, gy in destinations:
            if (gx-x)**2 + (gy-y)**2 <= r2:
                placeable = False
        if placeable:
            destinations.append([x, y])
            i+=1

    return torch.tensor(positions, dtype=torch.float32), torch.tensor(destinations, dtype=torch.float32)

def init_ped_crossing(ARGS, length=20.0, width=7.0):
    positions, destinations = list(), list()
    rng = np.random.RandomState()
    r2 = 0.3 ** 2
    i, placeable = 0, None
    flag = list()
    
    # Positions
    while len(positions) < ARGS.NUM_PED:
        if rng.random() > 0.5:
            vertical, horizontal = True, False
        else:
            vertical, horizontal = False, True
        if vertical:
            x = rng.rand() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5
            if y < 0: 
                y -= 0.25
            else: 
                y += 0.25
            y = abs(y)
        elif horizontal:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            x = -abs(x)
            y = rng.random() - 0.5
            if vertical: y *= 0.5
        
        if vertical:
            ped_pos0 = [x * (width / 2), y * (length / 2) * 2.0]
        elif horizontal:
            ped_pos0 = [x * (length / 2) * 2.0, y * width / 2]

        for ped_pos1 in positions:
            dist2 = (ped_pos0[0] - ped_pos1[0])**2 + (ped_pos0[1] - ped_pos1[1])**2
            if dist2 <= 3.0: # Collision
                placeable = False
            placeable = True
        if placeable or not positions:
            positions.append(ped_pos0)
            flag.append([vertical, horizontal])
    
    # Destinations
    while len(destinations) < ARGS.NUM_PED:
        vertical, horizontal = flag[i][0], flag[i][1]

        if vertical:
            x = rng.random() - 0.5
            if horizontal: x *= 0.5
            y = (rng.random() - 0.5) * 0.5 
            if y < 0:
                y -= 0.25
            else:
                y += 0.25
            if ((positions[i][1] > 0 and y > 0) or (positions[i][1] < 0 and y < 0)):
                y = -y
                y = -abs(y)

        else:
            x = (rng.random() - 0.5) * 0.5
            if x < 0:
                x -= 0.25
            else:
                x += 0.25
            if (positions[i][0] > 0 and x > 0) or (positions[i][0] < 0 and x < 0):
                x = -x
                x = abs(x)
            y = rng.random()-0.5
            if vertical: y *= 0.5

        if vertical:
            x *= width / 2
            y *= (length / 2 ) * 2.0
        elif horizontal: 
            x *= (length / 2) * 2.0
            y *= width / 2

        if (positions[i][0] - x)**2 + (positions[i][1] - y)**2 <= r2:
            continue
        placeable = True
        for gx, gy in destinations:
            if (gx-x)**2 + (gy-y)**2 <= r2:
                placeable = False
        if placeable:
            destinations.append([x, y])
            i+=1

    return torch.tensor(positions, dtype=torch.float32), torch.tensor(destinations, dtype=torch.float32)

def init_env(env, ARGS, pos=None, des=None, obs=None):
    n1, n2, size = ARGS.NUM_PED, ARGS.NUM_OBS, ARGS.SIZE_ENV
    velocity = 0.0 * torch.rand((n1, 2))

    if 'offset' in ARGS:
        x_offset, y_offset = eval(ARGS.offset)
    else:
        x_offset, y_offset = 0, 0

    ########## Pedestrians ##########
    if ARGS.SCENARIO == 'CIRCLE':
        n2 = 0
        positions, destinations = init_ped_circle(ARGS, circle_radius=10.0)
        obstacles = torch.distributions.Uniform(-10.0, 10.0 * size).sample([n2, 2])
    
    elif ARGS.SCENARIO == 'CORRIDOR':
        length, width = 20.0, 7.0
        vertical, horizontal = False, True
        positions, destinations = init_ped_corridor(ARGS, length, width, vertical, horizontal)

        if horizontal:
            start_top, end_top = np.array([-length / 2, width / 2]), np.array([length / 2, width / 2])
            start_bottom, end_bottom = np.array([-length / 2, -width / 2]), np.array([length / 2, -width / 2])
            n_top = int(np.linalg.norm(end_top - start_top) / (env.obstacle_radius * 2))
            n_bottom = int(np.linalg.norm(end_bottom - start_bottom) / (env.obstacle_radius * 2))
            obstacles_top = np.linspace(start_top, end_top, n_top)
            obstacles_bottom = np.linspace(start_bottom, end_bottom, n_bottom)
            obstacles = torch.tensor(np.concatenate((obstacles_top, obstacles_bottom)), dtype=torch.float32)
        elif vertical:
            start_top, end_top = np.array([-length / 2, -width / 2]), np.array([length / 2, -width / 2])
            start_bottom, end_bottom = np.array([-length / 2, width / 2]), np.array([length / 2, width / 2])
            n_top = int(np.linalg.norm(end_top-start_top) / (env.obstacle_radius * 2))
            n_bottom = int(np.linalg.norm(end_bottom-start_bottom) / (env.obstacle_radius * 2))
            obstacles_top = np.linspace(start_top, end_top, n_top)
            obstacles_bottom = np.linspace(start_bottom, end_bottom, n_bottom)
            obstacles = torch.tensor(np.concatenate((obstacles_top, obstacles_bottom)), dtype=torch.float32)
    
    elif ARGS.SCENARIO == 'CROSSING':        
        length, width = 20.0, 7.0
        positions, destinations = init_ped_crossing(ARGS, length, width)

        quadrant1 = np.array([[width / 2, length / 2], [width / 2, width / 2], [length / 2, width / 2]])
        quadrant2 = np.array([[-width / 2, length / 2], [-width / 2, width / 2], [-length / 2, width / 2]])
        quadrant3 = np.array([[-width / 2, -length / 2], [-width / 2, -width / 2], [-length / 2, -width / 2]])
        quadrant4 = np.array([[width / 2, -length / 2], [width / 2, -width / 2], [length / 2, -width / 2]])

        def init_quad_crossing(quadrant):
            for vertex in quadrant:
                n_line1 = int(np.linalg.norm(quadrant[0] - quadrant[1]) / (env.obstacle_radius * 2))
                n_line2 = int(np.linalg.norm(quadrant[1] - quadrant[2]) / (env.obstacle_radius * 2)) - 1
                obstacles_line1 = np.linspace(quadrant[0], quadrant[1], n_line1)
                obstacles_line2 = np.linspace(quadrant[1], quadrant[2], n_line2)[1:]
                obstacles = torch.tensor(np.concatenate((obstacles_line1, obstacles_line2)), dtype=torch.float32)
            return obstacles
        obstacles = torch.cat((init_quad_crossing(quadrant1), init_quad_crossing(quadrant2), init_quad_crossing(quadrant3), init_quad_crossing(quadrant4)))

    elif ARGS.SCENARIO == 'RANDOM':
        positions = torch.distributions.Uniform(-size, size).sample([n1, 2])
        destinations = torch.distributions.Uniform(-size, size).sample([n1, 2])
        obstacles = torch.distributions.Uniform(-1.5 * size, 1.5 * size).sample([n2, 2])
    
    elif ARGS.SCENARIO == 'PREDEFINED':
        positions = pos
        destinations = des
        obstacles = obs

    # load
    # positions = torch.tensor(np.load(f'./configs/scenario_configs/{ARGS.SCENARIO}/pos10.npy'))
    # destinations = torch.tensor(np.load(f'./configs/scenario_configs/{ARGS.SCENARIO}/des10.npy'))
    # obstacles = torch.tensor(np.load(f'./configs/scenario_configs/{ARGS.SCENARIO}/obs10.npy'))
    
    # offsets = torch.tensor([[x_offset, y_offset]] * n1)
    # positions += offsets
    # destinations += offsets
    
    env.add_pedestrian(positions, velocity, destinations, init=True)
    env.add_obstacle(obstacles, init=True)
    spawn_collision = calc_collision(env)
    print('Spawn Collisions:', spawn_collision)
    
    # if ARGS.SCENARIO != 'PREDEFINED':
    #     # save positions, destinations, obstacles:
    #     pos_np = positions.numpy(); np.save(f'./configs/scenario_configs/{ARGS.SCENARIO}/pos10.npy', pos_np)
    #     des_np = destinations.numpy(); np.save(f'./configs/scenario_configs/{ARGS.SCENARIO}/des10.npy', des_np)
    #     obs_np = obstacles.numpy(); np.save(f'./configs/scenario_configs/{ARGS.SCENARIO}/obs10.npy', obs_np)
    
    return positions, destinations, obstacles, spawn_collision




import itertools
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from PIL import Image
from scipy.stats.kde import gaussian_kde

from configs.config import get_args
from utils.action import  mod2pi
from tqdm import tqdm

def fig2array(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image[:, :, :3]

def plot_traj(env, save_path='plot.png', xrange=(-20, 20), yrange=(-20, 20), show_tqdm_flag=True):
    xmin, xmax = xrange
    ymin, ymax = yrange    
    
    plt.figure()
    fig, ax = plt.subplots()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    if show_tqdm_flag: 
        tqdm.write(f"Saving whole trajectory to '{save_path}'...")

    for p in tqdm(range(env.num_pedestrians)) if show_tqdm_flag else range(env.num_pedestrians):
        x = env.position[p, :, 0]; x = x[~torch.isnan(x)]
        y = env.position[p, :, 1]; y = y[~torch.isnan(y)]
        plt.plot(x, y, linewidth=0.8, color='grey')

    ax.set_aspect('equal')
    fig.savefig(save_path)

def plot_heatmap(env, save_path='plot.png', xrange=(-20, 20), yrange=(-20, 20), return_array=False):
    # pos_x_list, pos_y_list = pos_list
    x_min, x_max = xrange
    y_min, y_max = yrange   
    
    x = env.position[:, :, 0].view(-1); x = x[~torch.isnan(x)]
    y = env.position[:, :, 1].view(-1); y = y[~torch.isnan(y)]
    
    # KDE on the data
    xy = torch.vstack([x, y])
    kde = gaussian_kde(xy, bw_method='silverman')

    # KDE on the meshgrid
    x_grid, y_grid = torch.meshgrid(torch.linspace(x_min, x_max, 100), torch.linspace(y_min, y_max, 100))
    xy_grid  = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
    z = kde(xy_grid.T).reshape(x_grid.shape)

    plt.figure()
    plt.axis('off')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.imshow(z, origin='lower', aspect='auto', extent=[x_min, x_max, y_min, y_max],  cmap='jet')
    # plt.colorbar(label='Density')
    # plt.title('Trajectory Heatmap')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    
    # Plot the smoothed heatmap
    if not return_array:
        print(f'Saving heatmap to {save_path}...')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        fig = plt.gcf()
        fig.canvas.draw()
        array_rgb = np.array(fig.canvas.renderer.buffer_rgba())
        array_bgr = cv2.cvtColor(array_rgb, cv2.COLOR_RGBA2BGR)
        plt.close()
        return array_rgb

def plot_distribution():
    pass

def generate_gif(env, save_path, focus_id=None, start_time=None, final_time=None, 
                 speed_up=1, downsample=1, xrange=(-20, 20), yrange=(-20, 20), mpp=0.05, 
                 show_tqdm_flag=True, imit_agent_mask=None, rl_agent_mask=None):
    """
    将 env 生成 gif 图像, 存储为 save_path. 若指定 focus_id, 还可实现视野跟随 focus_id 移动
    也实现了旋转视角的功能, 但是很晕所以算了
    :param env: 需要可视化的环境
    :param save_path: 保存为 gif 文件
    :param focus_id: 行人 id, 设为 [0, N) 之间的正整数时视角跟随 focus_id 移动, 视野被限制到 focus_id 附近 5m
    :param start_time: 开始时刻, 默认为 0
    :param final_time: 结束时刻(不含), 默认为 env.num_steps
    :param speed_up: 通过增加 fps 实现倍速播放
    :param downsample: 通过抽帧实现倍速播放
    :param xrange: 场景横坐标范围
    :param yrange: 场景纵坐标范围
    :param mpp: 每像素代表多少米, 建议不小于 0.05。当指定 focus 时视野缩小, 此时可以将 mpp 设为 0.01
    :param show_tqdm_flag: 是否显示进度条
    """
    xmin, xmax = xrange
    ymin, ymax = yrange
    start_time = 0 if start_time is None else start_time
    final_time = env.num_steps if final_time is None else final_time
    dpi = 200

    # 生成背景图
    fig = plt.figure(figsize=((xmax - xmin) / mpp / dpi, (ymax - ymin) / mpp / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # 生成占满 figure 的 ax
    ax.grid(linestyle='dotted')
    ax.set_aspect(1.0, 'datalim')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    background_image = fig2array(fig)
    plt.close()
    H, W = background_image.shape[0], background_image.shape[1]
    background_image2 = deepcopy(background_image)
    for obs in env.obstacle if env.obstacle is not None else []:
        x = int((obs[0] - xmin) / mpp)
        y = H - int((obs[1] - ymin) / mpp)
        cv2.circle(background_image2, (x, y), int(env.obstacle_radius / mpp), (128, 128, 128), thickness=-1)
    background_image = background_image2
    
    fps = 1. / env.meta_data['time_unit']
    if show_tqdm_flag: 
        tqdm.write(f"Saving animation to '{save_path}'...")
    seq = []
    for t in tqdm(range(start_time, final_time, downsample)) if show_tqdm_flag else range(start_time, final_time, downsample):
        if focus_id and not env.mask[focus_id, t]:
            continue
        im = deepcopy(background_image)
        # im = np.zeros((W, H, 3), dtype=np.uint8)

        for p in range(env.num_pedestrians):
            if env.mask[p, t] and ~(torch.isnan(env.position[p, t, :]).any()):
                # and ~(torch.isnan(env.position[p, t, :]).any())
                # TO-DO: if nan then continue
                # if torch.isnan(env.position[p, t, :]).any():
                #     print(rl_agent_mask, p)
                #     print(env.position[p, :, :], env.mask[:, :], env.destination[p, :])
                x = int((env.position[p, t, 0] - xmin) / mpp)
                y = H - int((env.position[p, t, 1] - ymin) / mpp)
                # speed = env.velocity[p, t, :].norm()
                # color = tuple((np.array([0, 1.34 / (1.34 + speed), speed / (1.34 + speed)]) * 255))
                if rl_agent_mask is not None:
                    color = (0, 0, 240) if rl_agent_mask[p] else (0, 240, 0)
                else:
                    color = (0, 0, 240)
                if p == focus_id:
                    color = (240, 0, 0)
                cv2.circle(im, (x, y), int(env.ped_radius / mpp), color, thickness=-1)
                cv2.circle(im, (x, y), int(env.ped_radius / mpp), (0, 0, 0), thickness=1)
                if imit_agent_mask is not None:
                    p_real = imit_agent_mask[p].item()
                else:
                    p_real = p
                cv2.putText(im, str(p_real), (x+int(env.ped_radius/mpp), y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                traj = env.position[p, max(0, t - int(2. * fps)):t+1, :]
                traj = traj[env.mask[p, max(0, t - int(2. * fps)):t+1]]
                traj[:, 0] -= xmin
                traj[:, 1] -= ymin
                traj_ = (traj / mpp).cpu().numpy().astype(np.int32)
                traj_[:, 1] = H - traj_[:, 1]
                cv2.polylines(im, [traj_], False, (200, 200, 200), 1)
            elif env.arrive_flag[p, t]:
                x = int((env.destination[p, 0] - xmin) / mpp)
                y = H - int((env.destination[p, 1] - ymin) / mpp)
                cv2.drawMarker(im, position=(x, y), color=(120, 120, 240), markerSize=int(env.ped_radius / mpp), markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
                    
        cv2.putText(im, f'Frame {t} / {t * env.meta_data["time_unit"]:.2f}s', (0+5, H-5), cv2.FONT_ITALIC, 1, (255, 0, 0), 1)

        if focus_id is not None:
            x = int((env.position[focus_id, t, 0] - xmin) / mpp)
            y = H - int((env.position[focus_id, t, 1] - ymin) / mpp)
            # 将 focus_id 平移到画面中心
            im = cv2.warpAffine(im, np.float32([
                [1, 0, W / 2 - x], 
                [0, 1, H / 2 - y]
            ]), (H, W), borderValue=(0, 0, 0))
            # 旋转画面
            # if 'ang' not in dir(): 
            #     ang = env.direction[focus_id, t, 0] / np.pi * 180
            # # vec = env.position[focus_id, max(0, t - int(2. * fps)):t + 1, :]
            # # vec = vec[~vec.isnan().any(dim=-1), :]
            # # vec = vec[-1] - vec[0]
            # # ang = .9 * ang + .1 * torch.atan2(vec[1], vec[0]) / np.pi * 180
            # tmp = env.direction[focus_id, max(0, t - int(2. * fps)):t + 1, 0]
            # alpha = 0.1 / tmp[~tmp.isnan()].std().clamp(1.)
            # if ~alpha.isnan():
            #     ang = (1 - alpha) * ang + (alpha) * env.direction[focus_id, t, 0] / np.pi * 180
            # im = cv2.warpAffine(im, cv2.getRotationMatrix2D(
            #     (W / 2, H / 2), angle=90 - int(ang), scale=1.
            # ), (H, W), borderValue=(128, 128, 128))
            # 剪裁画面
            im = im[int(H / 2 - 5 / mpp):int(H / 2 + 5 / mpp) + 1, int(W / 2 - 5 / mpp):int(W / 2 + 5 / mpp) + 1]
        seq.append(Image.fromarray(im))
    seq[0].save(save_path, save_all=True, append_images=seq[1:], duration=1000. / fps / speed_up, loop=0)

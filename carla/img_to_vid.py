import os
import sys
import skvideo.io
import numpy as np
from PIL import Image
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

def img_to_vid(input_path='./result/Carla/image_frame/', output_path='./result/Carla/test.mp4'):
    image_frame_path = input_path
    video_path = output_path
    writer = skvideo.io.FFmpegWriter(video_path, inputdict = {'-r': str(1 / 0.08),'-s': f'{640}x{480}'},
                                    outputdict={'-r': str(1 / 0.08), '-vcodec': 'libx264', '-preset': 'ultrafast', '-pix_fmt': 'yuv420p'})

    print(f'Saving video to {video_path} ...')
    for img in os.listdir(image_frame_path):
        file_path = os.path.join(image_frame_path, img)
        image = Image.open(file_path)
        image = np.array(image, dtype=np.uint8)
        writer.writeFrame(image)
        os.remove(file_path)

    writer.close()


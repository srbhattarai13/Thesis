from moviepy.editor import *
import os


path = '../datasets/avenue/testing/optical_flow_png/'
base_dir = os.path.realpath("../datasets/avenue/testing/optical_flow_png/")

print(base_dir)

img = sorted(os.listdir(path))

img = img[3:]
# print(img)

clips = [ImageClip(m).set_duration(2) for m in img]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("prediction/actual_optflow.mp4", fps=24)
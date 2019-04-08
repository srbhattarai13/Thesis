from PIL import Image
import os


size = (256, 256)


avenue = '../datasets/ped1/training/frames/'

frame_resize = '../datasets/ped1//frame_resize/'
video_names = sorted(os.listdir(avenue))

# print(video_names)


for i, vid in enumerate(video_names):
    frame_list = sorted(os.listdir(avenue + vid + '/'))

    if not os.path.exists(frame_resize + vid):
        os.makedirs(frame_resize + vid)

    for frame in frame_list:
        img = Image.open(avenue + vid + '/' + frame)
        img = img.resize(size, Image.ANTIALIAS)
        img.save(frame_resize + vid + '/' + frame)

        print('processing video:', vid , 'Frame: ', frame)



#         path = avenue_optflow + '/' + vid + ofs + ',' + avenue_optflow + '/' + vid + next1opt + ',' + avenue_optflow   + '/' + vid + next2opt  + ',' + avenue_optflow +'/' + vid + next3opt  + '\n'
#
#         # if i ==6:
#         #     print(path)
#
#         # print(path)
#         with open('data/traingListAVENUE.csv', 'a') as f:
#             f.write(path)
#

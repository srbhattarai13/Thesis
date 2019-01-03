import csv, os

avenue_training = '../datasets/avenue/training/frames'
avenue_optflow = '../datasets/avenue/training/optical_flowmap'

video_names = sorted(os.listdir(avenue_training))

totalFrames = []
for i, vid in enumerate(video_names):
    frame_list = sorted(os.listdir(avenue_training + '/' + vid + '/'))

    for frame in range(len(frame_list)- 1):
        path = avenue_training + '/' + vid +  avenue_training + '/' + vid + '/' + str(frame) + ',' + avenue_optflow + '/' + vid + '/' + str(frame).rstrip('.jpg') + '.png'

        # print(path)
        with open('data/traingListAVENUE.csv', 'a') as f:
            f.write(path)



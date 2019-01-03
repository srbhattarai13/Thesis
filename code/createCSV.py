import csv, os

avenue_training = '../datasets/avenue/training/frames'
avenue_optflow = '../datasets/avenue/training/optical_flow'

video_names = sorted(os.listdir(avenue_training))

totalFrames = []
for i, vid in enumerate(video_names):
    frame_list = sorted(os.listdir(avenue_training + '/' + vid + '/'))

    for frame in range(len(frame_list)- 1):
        fs = '/%04d.jpg'%frame
        ofs = '/%04d.flo' % frame
        # print(s)
        path = avenue_training + '/' + vid +',' +  avenue_training + '/' + vid + '/' + fs + ',' + avenue_optflow + '/' + vid + ofs + '\n'
        print(path)

        # print(path)
        with open('data/traingListAVENUE.csv', 'a') as f:
            f.write(path)
        # #


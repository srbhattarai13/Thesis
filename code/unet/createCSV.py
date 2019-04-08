import csv, os

# avenue_training = '../datasets/avenue/training/frames'
dataset_dir = '../datasets/avenue/testing/'

video_names = sorted(os.listdir(dataset_dir +'frames'))

totalFrames = []
for i, vid in enumerate(video_names):
    frame_list = sorted(os.listdir(dataset_dir + 'frames' + '/' + vid + '/'))
    print(len(frame_list))

    for frame in range(len(frame_list)-5):
        # fs = '/%04d.jpg'%frame
        ofs = '/%04d.jpg' % frame
        # print(s)
        next1 = frame + 1
        # print(nextframe)
        next1opt= '/%04d.jpg' % next1

        next2 = next1 + 1
        next2opt = '/%04d.jpg' % next2

        next3 = next2 + 1
        next3opt = '/%04d.jpg' % next3

        next4 = frame + 4
        next4opt = '/%04d.flo' % next4

        path = dataset_dir + 'frames/'  + vid +'/' + frame_list[frame]  + ',' + dataset_dir + 'frames/' + vid +'/' + frame_list[frame+1] + ',' + dataset_dir + 'frames/' + vid +'/' + frame_list[frame+2]  + ',' + dataset_dir +'frames/' + vid +'/' + frame_list[frame+3] + ',' + dataset_dir + 'optical_flow_resize/' + vid + next4opt + ',' + str(len(frame_list)) + ',' + vid + '\n'
        print(path)
        # break
    # break
        # if i ==6:
        #     print(path)

        # # print(path)
        # with open('data/training_avenue.csv', 'a') as f:
        #     f.write(path)

    # break


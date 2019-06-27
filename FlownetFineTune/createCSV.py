import csv, os

dataset_dir = '../../datasets/shanghaitech/training/'
# dataset_dir = '../../datasets/shanghaitech/testing/'

video_names = sorted(os.listdir(dataset_dir +'frames/'))

totalFrames = []
for i, vid in enumerate(video_names):
    frame_list = sorted(os.listdir(dataset_dir + 'frames/' + vid + '/'))
    print(len(frame_list))

    for frame in range(len(frame_list)-2):
        # fs = '/%04d.jpg'%frame
        ofs = '/%04d.jpg' % frame
        # print(s)
        next1 = frame + 1
        # print(nextframe)
        next1opt= '/%04d.flo' % next1

        next2 = next1 + 1
        next2opt = '/%04d.jpg' % next2

        next3 = next2 + 1
        next3opt = '/%04d.jpg' % next3

        next4 = frame +1
        next4opt = '/%04d.flo' % next4

        # print(frame_list[frame])

        path = dataset_dir + 'frames/'  + vid +'/' + frame_list[frame] + ',' + dataset_dir + 'frames/' + vid +'/' + frame_list[frame+1]  + ',' + dataset_dir + 'optflow_norm/' + vid + next4opt  + '\n'
        print(path)
        # break
    # break
        # path = dataset_dir + 'optflow_norm/'  + vid +'/' + next4opt+ '\n'
        # print(path)
        # break
    # break

        # if i ==6:
        #     print(path)

        # # print(path)
        with open('data/training_shanghaitech.csv', 'a') as f:
            f.write(path)


    # break


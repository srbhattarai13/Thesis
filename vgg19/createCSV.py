import csv, os

dataset_dir = '../../datasets/shanghaitech/training/'
# dataset_dir = '../../datasets/shanghaitech/testing/'

video_names = sorted(os.listdir(dataset_dir +'vggfeatures/'))

totalFrames = []
for i, vid in enumerate(video_names):
    frame_list = sorted(os.listdir(dataset_dir + 'frames/' + vid + '/'))
    print(len(frame_list))

    for frame in range(len(frame_list)-1):
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

        next4 = frame + 1
        next4opt = '/%04d.flo' % next4

        # print(frame_list[frame])

        # path = dataset_dir + 'vggfeatures/'  + vid +'/' + frame_list[frame] + ',' + dataset_dir + 'frames/' + vid +'/' + frame_list[frame+1]  + ',' + dataset_dir + 'optical_flow_resize/' + vid + next4opt  + '\n'
        # print(path)
        # break
        path = dataset_dir + 'frames/'  + vid +'/' + frame_list[frame]+ ',' + dataset_dir + 'vggfeatures/'  + vid +'/' + str(frame_list[frame]).strip('.jpg') +'.npy'+ '\n'
        print(path)
    #     if frame == 10:
    #         break
    #     # print(path)
    # break
    # break

        # if i ==6:
        #     print(path)

        # # print(path)
        with open('data/vgg_shanghaitech_train.csv', 'a') as f:
            f.write(path)


    # break


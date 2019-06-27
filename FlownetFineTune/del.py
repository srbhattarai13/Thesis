import os

data_dir = '../../datasets/ChairsSDHom/data/test/'


tolist = sorted(os.listdir(data_dir+'t1/'))

for i in reversed(range(51, len(tolist))):

        os.remove(data_dir+'t1/'+tolist[i])
        print('removed',i)


print(len(tolist))
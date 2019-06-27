import numpy as np



def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:, :, 0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    # print('Width', width)

    height = np.fromfile(f, np.int32, 1).squeeze()
    # print('Height',height)

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)



flow = readFlow('../../datasets/avenue/training/optical_flow/01/0000.flo')

print(flow.shape)
import numpy as np
import calc_feature as cf

ldata_1, ldata_2  = np.empty((20, 100, 64), dtype='float32'), np.empty((20, 100, 64), dtype='float32')
tdata_1, tdata_2 = np.empty((20, 100, 64), dtype='float32'), np.empty((20, 100, 64), dtype='float32')
ldata, tdata = np.empty((20, 200, 64), dtype='float32'), np.empty((20, 200, 64), dtype='float32')


def load(rpath='./Data'):
    (ldata_1[:], tdata_1[:]) = read_writer(rpath + '/hira', 0)
    (ldata_2[:], tdata_2[:]) = read_writer(rpath + '/hira', 1)
    for i in range(20):
        ldata[i][0:100], tdata[i][0:100] = ldata_1[i], tdata_1[i]
        ldata[i][100:200], tdata[i][100:200] = ldata_2[i], tdata_2[i]

def read_file(path):
    l = np.empty((100,64), dtype='float32')
    with open(path) as f:
        chara = np.array(f.readlines())
    for i in range(100):
        l[i][:] = cf.in_feature(chara[64*i:64*i+64])
    return l

def read_writer(path, writer_num):
    ldata, tdata = np.empty((20, 100, 64), dtype='float32'), np.empty((20, 100, 64), dtype='float32')
    path = path + str(writer_num) + '_'
    for i in range(20):
        ldata[i][:] = read_file(path + str(i).zfill(2) + 'L.dat')
        tdata[i][:]= read_file(path + str(i).zfill(2) + 'T.dat')
    return (ldata, tdata)


import numpy as np

def mesh_feature(p, ii, jj):
    cnt = 0
    for i in range(8):
        for j in range(8):
            if(p[8*ii + i][8*jj + j] == '1'):
                cnt += 1
    cnt = cnt/64
    return cnt

def in_feature(p):
    l = np.empty(64, dtype='float32')
    for i in range(8):
        for j in range(8):
            l[8*i + j] = mesh_feature(p, i, j)
    return l
'''
.h5 File Data Loader for Stardist
'''


import h5py

def load_h5(f):
    with h5py.File(f,'r') as f:
        dataset = f['data']
        data = dataset[...]
    return data[0]

def load_h5_instance(f):
    with h5py.File(f,'r') as f:
        dataset = f['instance']
        data = dataset[...]
    return data[0]
import numpy as np




def parse_off(path):
    with open(path) as f:
        lines=f.readlines()
    lines = [x.strip() for x in lines if x[0]!='#']
    
    if lines[0]=='OFF':
        lines = lines[1:]
    info = [int(x.replace('OFF','')) for x in lines[0].split()]
    lines = lines[1:]
    n_verts = info[0]
    n_faces=info[1]

    vertices = [[float(y) for y in x.split()] for x in lines[:n_verts]]
    faces = [[int(y) for y in x.split()[1:]] for x in lines[n_verts:]]

    return np.array(vertices,np.float32) , np.array(faces,int)

     





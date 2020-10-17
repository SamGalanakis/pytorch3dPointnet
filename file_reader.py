import numpy as np
from pathlib import Path


class FileReader:
    def __init__(self):
        pass
    
    def convert_ply_to_off(self, path):
        off_file = ["OFF\n"]

        with path.open() as f:
            ply = f.readlines()

        ply = [x for x in ply if not x.startswith("comment")]

        vertex = ply[2].split()[2]
        indeces = ply[6].split()[2]
        off_file.append(f"{vertex} {indeces} 0\n")

        ply = ply[9:]

        off_file.extend(ply[:int(vertex)])
        off_file.extend(ply[int(vertex):])

        return off_file

    def read(self, path,verbose=False):
        if type(path)==str:
            path = Path(path)
        lines=False
        if path.suffix == ".ply":
            lines = self.convert_ply_to_off(path)
        elif path.suffix != ".off":
            raise Exception("Invalid file type, can only process .off and .ply")

        if not lines:
            with path.open() as f:
                lines = f.readlines()
            lines = [x for x in lines if x[0] != "#"]
        if "OFF" in lines[0]:
            lines = lines[1:]
            

        lines = [x.rstrip() for x in lines]

        info = [int(x) for x in lines[0].split()]
        lines = lines[1:]

        if len(info) == 4:
            n_vertices = info[0]
            n_faces = info[1]
            n_edges = info[2]
            n_attributes = info[3]
        else:
            n_vertices = info[0]
            n_faces = info[1]
            n_attributes = info[2]

        if n_attributes > 0:
            raise Exception("Extra properties")

        vertices = lines[:n_vertices]
        vertices = np.array([list(map(lambda y: float(y), x.split()))
                                for x in vertices], dtype=np.float32).flatten()
        elements = lines[n_vertices:]
        elements = [list(map(lambda y: int(y), x.split())) for x in elements]

        triangles = np.array([x[1:] for x in elements if x[0]==3],dtype = np.uint32)
        quads = np.array([x[1:] for x in elements if x[0]==4],dtype = np.uint32)
        assert triangles.size/3 +quads.size/4 == len(elements), "Non quad/triangle elements!"
        element_dict = {"triangles":triangles, "quads":quads}
        if verbose:
            print(f" File type: {path.suffix} Triangles: {triangles.size}, Quads: {quads.size}.")
        return vertices, element_dict, info

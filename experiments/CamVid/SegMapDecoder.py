import numpy as np


class SegMapDecoder:
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colours = np.uint8([Sky, Building, Pole, Road_marking, Road,
                              Pavement, Tree, SignSymbol, Fence, Car,
                              Pedestrian, Bicyclist])

    @classmethod
    def decode(cls, segMap):
        r = np.empty_like(segMap, dtype=np.uint8)
        g = np.empty_like(segMap, dtype=np.uint8)
        b = np.empty_like(segMap, dtype=np.uint8)
        for l in range(0, 12):
            map = segMap == l
            r[map] = cls.label_colours[l, 0]
            g[map] = cls.label_colours[l, 1]
            b[map] = cls.label_colours[l, 2]

        rgb = np.zeros([segMap.shape[0], segMap.shape[1], 3], np.uint8)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    @classmethod
    def decode2(cls, segMap):
        decodedShape = (segMap.shape[0], segMap.shape[1], 3)
        rgb = np.empty(decodedShape, dtype=np.uint8)
        map = np.empty_like(segMap, dtype=np.bool)
        for l in range(0, 12):
            map = np.equal(segMap, l, out=map)
            rgb[..., 0][map] = cls.label_colours[l, 0]
            rgb[..., 1][map] = cls.label_colours[l, 1]
            rgb[..., 2][map] = cls.label_colours[l, 2]

        return rgb

    def __call__(cls, segMap):
        return cls.decode(segMap)


decoder = SegMapDecoder()

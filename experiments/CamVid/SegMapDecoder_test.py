import time
import numpy as np
import cv2
from experiments.CamVid.SegMapDecoder import SegMapDecoder

def main():
    testannotDir = './content/trainannot/'
    imageFileName = '0006R0_f01260.png'
    trueLabels = cv2.imread(testannotDir + imageFileName, cv2.IMREAD_GRAYSCALE)

    np.testing.assert_array_equal(SegMapDecoder.decode(trueLabels), SegMapDecoder.decode2(trueLabels))

    iters = 1000

    t0 = time.time()
    for _ in range(iters):
        decoded = SegMapDecoder.decode(trueLabels)
    t1 = time.time()
    print(t1-t0)

    t0 = time.time()
    for _ in range(iters):
        decoded = SegMapDecoder.decode2(trueLabels)
    t1 = time.time()
    print(t1 - t0)


    return

    cv2.imshow('decoded', decoded)
    cv2.imshow('decoded2', decoded2)
    cv2.waitKey()

def main():
    im = np.zeros([2, 2, 1], np.uint8)
    mask = np.array([[True]], np.bool)
    print(im[mask])

main()

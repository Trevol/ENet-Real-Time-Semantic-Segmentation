import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def main():
    training_path = './content/test/'
    imageFileName = 'Seq05VD_f05100.png'

    img = cv2.imread(training_path + imageFileName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
    img = np.divide(img, 255.0, dtype=np.float32)

    print(img.dtype, img.shape, img.min(), img.max())

    img2 = plt.imread(training_path + imageFileName)
    # img2 = cv2.resize(img2, (512, 512), cv2.INTER_NEAREST)
    print(img2.dtype, img2.shape, img2.min(), img2.max())

    np.testing.assert_array_equal(img, img2)


main()

import torch
import matplotlib.pyplot as plt
import cv2

from experiments.CamVid.SegMapDecoder import decoder
from models.ENet import ENet


def rgb2bgr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def imshow(winname, im):
    cv2.imshow(winname, im)


def main():
    enet = ENet(12)
    device = torch.device('cuda:0')
    enet = enet.to(device)

    # enet.load_state_dict(torch.load('../content/ckpt-enet-CamVid-downloaded.pth')['state_dict'])
    # enet.load_state_dict(torch.load('./content/ckpt-enet-100-21.788276344537735.pth')['state_dict'])
    enet.load_state_dict(torch.load('./checkpoints/ckpt-enet-50-314.6487820148468.pth')['state_dict'])

    # testDir = '../content/test/'
    # testannotDir = '../content/testannot/'
    # imageFileName = 'Seq05VD_f05100.png'

    testDir = './content/train/'
    testannotDir = './content/trainannot/'
    imageFileName = '0006R0_f01260.png'

    inputImg = plt.imread(testDir + imageFileName)  # read in RGB
    inputImg = cv2.resize(inputImg, (512, 512), cv2.INTER_NEAREST)

    inputTensor = torch.tensor(inputImg).unsqueeze(0).float()
    inputTensor = inputTensor.transpose(2, 3).transpose(1, 2).to(device)

    with torch.no_grad():
        outTensor = enet(inputTensor.float()).squeeze(0)

    # Индекс слоя с максимальным значением в соотв. пикселе и является индексом класса для этого пикселя
    # Альтернатива: outTensor.data.max(0)[1].cpu().numpy()
    computedLabels = outTensor.data.argmax(0).cpu().numpy()
    decodedComputedLabels = decoder(computedLabels)

    trueLabels = cv2.imread(testannotDir + imageFileName, cv2.IMREAD_GRAYSCALE)
    trueLabels = cv2.resize(trueLabels, (512, 512), cv2.INTER_NEAREST)
    decodedTrueLabels = decoder(trueLabels)

    figure = plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)  # rows, cols, index
    plt.title('Input Image')
    plt.axis('off')
    plt.imshow(inputImg)

    plt.subplot(1, 3, 2)
    plt.title('Output Labels')
    plt.axis('off')
    plt.imshow(decodedComputedLabels)

    plt.subplot(1, 3, 3)
    plt.title('True Labels')
    plt.axis('off')
    plt.imshow(decodedTrueLabels)

    plt.show()


if __name__ == '__main__':
    main()

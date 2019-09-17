import numpy as np
import torch
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from models.ENet import ENet


def imreadFloatRgb(fileName):
    img = cv2.imread(fileName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
    return np.divide(img, 255.0, dtype=np.float32)


def loader(training_path, segmented_path, batch_size, h=512, w=512):
    filenames_t = os.listdir(training_path)
    total_files_t = len(filenames_t)

    filenames_s = os.listdir(segmented_path)
    total_files_s = len(filenames_s)

    assert (total_files_t == total_files_s)

    if str(batch_size).lower() == 'all':
        batch_size = total_files_s

    assert batch_size <= total_files_s

    while (1):
        # Choosing random indexes of images and labels
        batch_idxs = np.random.permutation(total_files_s)[:batch_size]

        inputs = []
        labels = []

        for jj in batch_idxs:
            img = imreadFloatRgb(training_path + filenames_t[jj])
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            inputs.append(img)

            # Reading semantic image
            img = cv2.imread(segmented_path + filenames_s[jj], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            labels.append(img)

        inputs = np.stack(inputs, axis=2)
        # Changing image format to C x H x W
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)

        labels = torch.tensor(labels)

        yield inputs, labels


def get_class_weights(num_classes, c=1.02):
    pipe = loader('./content/train/', './content/trainannot/', batch_size='all')
    _, labels = next(pipe)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    return class_weights


def main():
    device = torch.device('cuda:0')
    enet = ENet(12).to(device)

    lr = 5e-4
    batch_size = 20
    class_weights = get_class_weights(12)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = torch.optim.Adam(enet.parameters(),
                                 lr=lr,
                                 weight_decay=2e-4)

    print_every = 5
    eval_every = 5

    # ## Training loop

    # In[17]:

    train_losses = []
    eval_losses = []

    bc_train = 367 // batch_size  # mini_batch train
    bc_eval = 101 // batch_size  # mini_batch validation

    iterationsPerEpochs = 1000

    # Define pipeline objects
    pipe = loader('./content/train/', './content/trainannot/', batch_size)
    eval_pipe = loader('./content/val/', './content/valannot/', batch_size)

    epochs = 100
    print()
    for e in range(1, epochs + 1):

        train_loss = 0
        print('-' * 15, 'Epoch %d' % e, '-' * 15)

        enet.train()

        for _ in tqdm(range(iterationsPerEpochs)):
            X_batch, mask_batch = next(pipe)

            # assign data to cpu/gpu
            X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

            optimizer.zero_grad()

            out = enet(X_batch.float())

            # loss calculation
            loss = criterion(out, mask_batch.long())
            # update weights
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print()
        train_losses.append(train_loss)

        if (e + 1) % print_every == 0:
            print(f'Epoch {e}/{epochs}...',
                  f'Loss {train_loss:6f}')

        if e % eval_every == 0:
            with torch.no_grad():
                enet.eval()

                eval_loss = 0

                # Validation loop
                for _ in tqdm(range(bc_eval)):
                    inputs, labels = next(eval_pipe)

                    inputs, labels = inputs.to(device), labels.to(device)

                    out = enet(inputs)

                    out = out.data.max(1)[1]

                    eval_loss += (labels.long() - out.long()).sum()

                print()
                print(f'Loss {eval_loss:6f}')

                eval_losses.append(eval_loss)

        if e % print_every == 0:
            checkpoint = {
                'epochs': e,
                'state_dict': enet.state_dict()
            }
            torch.save(checkpoint, f'./checkpoints/ckpt-enet-{e}-{train_loss}.pth')
            print('Model saved!')

    print(f'Epoch {e}/{epochs}...',
          f'Total Mean Loss: {sum(train_losses) / epochs:6f}')


if __name__ == '__main__':
    main()

import torch
import torch.version
import time


def main():
    devices = dict(
        cuda0=torch.device('cuda:0'),
        cpu=torch.device('cuda:0')
    )

    for devName, dev in devices.items():
        tensor = torch.randn([212, 2512, 2512], dtype=torch.float32, device=dev)
        for i in range(2):
            t0 = time.time()
            for i in range(1):
                c = torch.argmax(tensor).cpu().numpy()
            t1 = time.time()
            print(devName, f'{t1 - t0:.6f}')


def main__():
    # TODO: test argmax on cuda:0
    devices = dict(
        cuda0=torch.device('cuda:0'),
        cpu=torch.device('cpu')
    )

    output = torch.randn([3, 4, 4], device=devices['cuda0'])
    # output[0, :, :] = 3
    # output[1, :, :] = 5
    # output[2, :, :] = 2
    print(output.argmax(0))
    print(output.max(0))
    print('#################')
    for v in output.max(0):
        print('-------------------------')
        print(v)


if __name__ == '__main__':
    main()

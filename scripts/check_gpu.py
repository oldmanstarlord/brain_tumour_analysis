import torch


def main():
    print('PyTorch version:', torch.__version__)
    cuda_avail = torch.cuda.is_available()
    print('CUDA available:', cuda_avail)
    print('CUDA device count:', torch.cuda.device_count())
    if cuda_avail and torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            try:
                print('Device', i, ':', torch.cuda.get_device_name(i))
            except Exception:
                pass


if __name__ == '__main__':
    main()

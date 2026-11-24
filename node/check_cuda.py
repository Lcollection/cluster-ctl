"""Run a small matrix multiplication on each GPU visible to PyTorch."""

import torch


def main():
    count = torch.cuda.device_count()
    assert count, "No CUDA devices visible"
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPUs {count}")
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        a = torch.eye(128, device=device)
        b = a @ a
        torch.cuda.synchronize(device)
        assert torch.equal(a, b), f"Incorrect result on {device}"
        print(f"GPU {index}: {torch.cuda.get_device_name(index)} — matmul passed")


if __name__ == "__main__":
    main()

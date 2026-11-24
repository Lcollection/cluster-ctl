# PART 计算节点管理

宿主机用 uv 管理 Python 和管理工具；模型项目分别维护自己的
`pyproject.toml`、`uv.lock`。训练和推理通过 rootless Docker 运行。

每张卡的显存独立；使用全部 GPU 需要模型代码支持分布式或任务并行。

## 已部署目录

```text
~/ai-training/
├── .venv/                     # uv 管理的宿主机环境，Python 3.12
├── pyproject.toml / uv.lock
├── infrastructure/part/       # 本目录的部署副本
├── projects/PART/             # 初始化的 PART 工作区，尚未安装模型
├── datasets/                  # 容器中 /data，只读
├── models/                    # 容器中 /models
├── runs/                      # 容器中 /runs，保存训练结果
└── cache/                     # uv、Hugging Face、Torch、Triton 缓存
```

Docker 镜像和容器写入层位于节点本地
`/var/tmp/ai-training-<username>/docker`，由 Docker 管理；数据、权重和结果保存在
`~/ai-training` 的 BeeGFS 目录中。容器内临时文件不作为持久化结果保存。

## 宿主机

```bash
cd ~/ai-training
uv run --locked python --version
uv run --locked ruff --version
docker context show                    # rootless
systemctl --user status docker
```

uv 安装在 `~/.local/bin`，shell 已配置 `UV_CACHE_DIR` 和
`UV_LINK_MODE=copy`。系统 Python 保留用于操作系统工具。
Docker 用户服务已启用，`loginctl show-user <ssh_username> -p Linger` 应为 `Linger=yes`，
从而支持退出 SSH 后继续运行。节点重启后本地镜像若被清理，需要重新导入或拉取。

## 模型项目

`projects/PART` 当前是空依赖工作区。后续按目标模型官方环境要求添加依赖；
RFD4、AtomWorks、Tmol 等尚未安装，也未下载权重。
需要不同 Python 或 CUDA 版本的模型使用独立项目和镜像。

例如，在确认模型兼容 PyTorch 2.7.1 / CUDA 12.8 后，可在**模型容器内**执行：

```bash
partctl shell
uv add torch==2.7.1 --index pytorch=https://download.pytorch.org/whl/cu128
uv lock
exit
```

`uv add` 会更新绑定挂载的项目文件；将生成的 `pyproject.toml` 和 `uv.lock`
保存到版本控制中。容器内 `uv run --locked` 按锁文件安装并运行。
每个容器的 Python 虚拟环境位于 `/opt/venv`，不会复用宿主机 `.venv`。
当前基础镜像含 Python 与 uv；需要编译器和系统库的模型应提供自己的 Dockerfile，
并通过 `PART_IMAGE` 选择构建后的镜像。

## 训练、推理和 GPU 选择

部署后的 `~/.local/bin/partctl` 指向管理脚本，其配置位于
`~/ai-training/infrastructure/part/.env`。另一个模型可通过环境变量覆盖项目目录：

```bash
partctl status
partctl gpu-check
partctl shell

# 单卡训练，前台输出日志
PART_GPU_DEVICE=nvidia.com/gpu=0 partctl train python train.py --output /runs/example

# 多卡任务需要项目自己提供分布式训练入口
partctl train torchrun --standalone --nproc_per_node=8 train.py --output /runs/example

# 后台任务返回容器 ID；用它查看日志、停止和删除容器
partctl submit python train.py --output /runs/example
partctl logs CONTAINER_ID
partctl stop CONTAINER_ID
docker rm CONTAINER_ID

# 另一个模型项目
PART_PROJECT_DIR="$HOME/ai-training/projects/my-model" partctl train python train.py

# 服务器必须在容器内监听 0.0.0.0:8000
PART_GPU_DEVICE=nvidia.com/gpu=1 partctl infer python serve.py
```

推理端口默认只发布到节点的 `127.0.0.1:8000`。在本地通过 SSH 转发访问：

```bash
ssh -p <ssh_port> -L 8000:127.0.0.1:8000 <username>@<cluster_ip>
```

`PART_GPU_DEVICE=nvidia.com/gpu=all` 是默认值；可改成单卡编号或 GPU UUID。
这里没有自动分配 GPU 的调度器，提交任务前用 `partctl status` 检查占用。

## 环境验证

`smoke/check_driver.py` 用 CUDA Driver API 在每张可见 GPU 上执行 `41 + 1` 内核，
断言结果为 42；只依赖 Python 标准库，避免为了验证透传下载整套训练框架。

```bash
PART_PROJECT_DIR="$HOME/ai-training/infrastructure/part/smoke" \
  partctl train python check_driver.py
```

安装 PyTorch 后，还可运行 `check_cuda.py` 验证 PyTorch 矩阵乘法。
这些检查验证环境和 GPU 计算，不代表某个蛋白质模型已经部署。

## 镜像与 NVIDIA CDI

基础镜像是从 GHCR 的 `astral-sh/uv:python3.12-bookworm-slim` 导入的
Linux amd64 镜像，本地标签为 `part-runtime:uv-python3.12`。
网络正常时可以拉取并重新标记；部署记录另存于节点的 `image-source.json`。
节点直连 Docker Hub / GHCR 下载层曾超时，因此本次通过本地下载后传入。
导入存档保留在 `/var/tmp/ai-training-<username>/part-python-image.tar`，
可用 `docker load -i` 重新导入，或将存档另存到持久化存储。

```bash
docker pull ghcr.io/astral-sh/uv:python3.12-bookworm-slim
docker tag ghcr.io/astral-sh/uv:python3.12-bookworm-slim part-runtime:uv-python3.12
```

节点无需系统级 NVIDIA runtime：用户级 NVIDIA Toolkit 1.20.2 提供 CDI，
工具位于 `~/.local/share/ai-training/nvidia/usr/bin`，规范文件位于
`~/.config/cdi/nvidia.yaml`，Docker 的 `cdi-spec-dirs` 指向该目录。
Compose 使用 CDI `devices`，不使用依赖传统 NVIDIA runtime 的 `gpus: all`。
驱动升级后重新生成 CDI：

```bash
toolkit="$HOME/.local/share/ai-training/nvidia/usr/bin"
"$toolkit/nvidia-ctk" cdi generate \
  --nvidia-cdi-hook-path="$toolkit/nvidia-cdi-hook" \
  --output="$HOME/.config/cdi/nvidia.yaml"
partctl gpu-check
```

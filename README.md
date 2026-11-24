# Cluster Manager

A small local CLI for managing uv projects and Docker workloads on GPU compute
nodes over SSH. Node setup files are in `node/`; `clusterctl` invokes the
`partctl` installed on each node.

## Install locally

```bash
cp cluster.example.toml cluster.toml
# Edit the node name, host, port and user. Keep credentials in SSH keys/agent.
uv sync
uv run clusterctl --config cluster.toml status <cluster_name>
uv run clusterctl --config cluster.toml gpu-check <cluster_name>
```

`cluster.toml` is git-ignored. The example contains the node address and account
name, but no password. Configure SSH key access before running commands; the
CLI uses `BatchMode=yes` and never prompts for or stores a password.

## Commands

```bash
uv run clusterctl --config cluster.toml status <cluster_name>
uv run clusterctl --config cluster.toml shell <cluster_name>
uv run clusterctl --config cluster.toml train <cluster_name> python train.py --epochs 5
uv run clusterctl --config cluster.toml submit <cluster_name> python train.py
uv run clusterctl --config cluster.toml logs <cluster_name> CONTAINER_ID
uv run clusterctl --config cluster.toml stop <cluster_name> CONTAINER_ID
uv run clusterctl --config cluster.toml infer <cluster_name> python serve.py
```

Training runs inside the node's uv-enabled GPU container. Projects keep their
own `pyproject.toml` and `uv.lock`; data, weights and run outputs use persistent
mounts. `submit` returns a container ID for logs and stop operations. Set the
node's `PART_GPU_DEVICE` in `~/ai-training/infrastructure/part/.env` to select
one GPU. Multi-GPU work requires a distributed project entry point such as
`torchrun`.

## Deploy node runtime

Copy `node/` to `~/ai-training/infrastructure/part/` on a node and follow
`node/README.md` to configure Docker CDI, the local project and `partctl`.
This runtime targets Linux amd64, rootless Docker, NVIDIA CDI and Python 3.12.

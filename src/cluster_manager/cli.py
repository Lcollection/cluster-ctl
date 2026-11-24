"""SSH command-line interface for deployed GPU compute nodes."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path


def load_config(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except OSError as error:
        raise SystemExit(f"Cannot read config {path}: {error}") from error


def ssh_command(node: dict, remote_args: list[str], *, interactive: bool = False) -> list[str]:
    host, user = node.get("host"), node.get("user")
    if not host or not user:
        raise SystemExit("Each node needs non-empty 'host' and 'user' fields")
    ssh = ["ssh", "-p", str(int(node.get("port", 22))), "-o", "BatchMode=yes"]
    if interactive:
        ssh.append("-tt")
    if identity := node.get("identity_file"):
        ssh.extend(["-i", str(Path(identity).expanduser())])

    home = node.get("work_root", "~/ai-training").removeprefix("~/")
    project = node.get("project_dir", "projects/PART").removeprefix("~/")
    remote = (
        'export PATH="$HOME/.local/bin:$PATH"; '
        f'export PART_ROOT="$HOME"/{shlex.quote(home)}; '
        f'export PART_PROJECT_DIR="$PART_ROOT"/{shlex.quote(project)}; '
        f'exec partctl {" ".join(shlex.quote(arg) for arg in remote_args)}'
    )
    return [*ssh, f"{user}@{host}", remote]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="clusterctl",
        description="Control uv/Docker training and inference jobs on GPU nodes.",
    )
    result.add_argument("--config", type=Path, default=Path("cluster.toml"))
    subparsers = result.add_subparsers(dest="action", required=True)
    for action in ("status", "gpu-check", "shell"):
        command = subparsers.add_parser(action)
        command.add_argument("node", help="Name from [[nodes]] in the config")
    for action in ("train", "submit", "infer"):
        command = subparsers.add_parser(action)
        command.add_argument("node", help="Name from [[nodes]] in the config")
        command.add_argument("command", nargs=argparse.REMAINDER,
                             help="Command to run in the node container")
    for action in ("logs", "stop"):
        command = subparsers.add_parser(action)
        command.add_argument("node", help="Name from [[nodes]] in the config")
        command.add_argument("container", help="Container ID returned by submit")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    nodes = {entry.get("name"): entry for entry in config.get("nodes", [])}
    if args.node not in nodes:
        available = ", ".join(sorted(name for name in nodes if name)) or "none"
        raise SystemExit(f"Unknown node {args.node!r}; configured nodes: {available}")
    if args.action in {"status", "gpu-check", "shell"}:
        remote_args = [args.action]
    elif args.action in {"train", "submit", "infer"}:
        if not args.command:
            raise SystemExit(f"{args.action} requires a command, e.g. python train.py")
        remote_args = [args.action, *args.command]
    else:
        remote_args = [args.action, args.container]
    try:
        result = subprocess.run(
            ssh_command(nodes[args.node], remote_args, interactive=args.action == "shell"),
            check=False,
        )
    except FileNotFoundError as error:
        raise SystemExit("ssh was not found on PATH") from error
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

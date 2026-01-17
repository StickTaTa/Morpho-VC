from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _map_key(name: str) -> str:
    if name.endswith(".gamma"):
        return name[:-6] + ".weight"
    if name.endswith(".beta"):
        return name[:-5] + ".bias"
    return name


def convert_ckpt(ckpt_path: Path, output_path: Path, list_only: bool = False) -> None:
    try:
        import mindspore as ms
    except Exception as exc:  # pragma: no cover - depends on mindspore install
        raise ImportError("mindspore is required to convert .ckpt files") from exc

    params = ms.load_checkpoint(str(ckpt_path))
    if list_only:
        for key in sorted(params.keys()):
            print(key)
        return

    state = {}
    for name, param in params.items():
        key = _map_key(name)
        tensor = param.value() if hasattr(param, "value") else param
        array = tensor.asnumpy() if hasattr(tensor, "asnumpy") else tensor
        state[key] = torch.from_numpy(array)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
    print(f"Saved PyTorch checkpoint to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CellFM MindSpore .ckpt to PyTorch .pt")
    parser.add_argument("--ckpt", required=True, help="Path to MindSpore .ckpt")
    parser.add_argument("--out", required=True, help="Output .pt path")
    parser.add_argument("--list_keys", action="store_true", help="Only list keys in ckpt")
    args = parser.parse_args()

    convert_ckpt(Path(args.ckpt), Path(args.out), list_only=args.list_keys)


if __name__ == "__main__":
    main()

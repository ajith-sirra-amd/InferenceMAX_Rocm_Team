"""Read only the DSpark context projection/norm from a local checkpoint.

No model construction, downloads, checkpoint writes or other tensors are needed.
The caller supplies Torch so command help can run without importing GPU packages.
"""

import hashlib
import json
from pathlib import Path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_selected_tensors(folder, safe_open):
    aliases = {
        "projection": {"context_proj.weight", "model.context_proj.weight"},
        "norm": {"context_norm.weight", "model.context_norm.weight"},
    }
    matches = {name: [] for name in aliases}
    index = folder / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        for name, keys in aliases.items():
            matches[name] = [(folder / mapping[key], key) for key in sorted(keys) if key in mapping]
    else:
        # safe_open.keys reads headers; get_tensor is used only after selection.
        for path in sorted(folder.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
            for name, accepted in aliases.items():
                matches[name].extend((path, key) for key in sorted(keys & accepted))
    for name, found in matches.items():
        if len(found) != 1:
            raise ValueError(f"Expected exactly one {name} tensor; found {found}")
        path = found[0][0]
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError(f"Checkpoint tensor file is missing or outside checkpoint directory: {path}")
    return {name: found[0] for name, found in matches.items()}


def load_context_weights(folder, torch):
    """Return CPU projection weight, CPU norm weight, and selected-byte evidence."""
    from safetensors import safe_open

    folder = Path(folder)
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text())
    actual = (
        config.get("target_hidden_size"), config.get("num_target_layers"),
        config.get("hidden_size"), config.get("rms_norm_eps"),
    )
    if actual != (7168, 5, 7168, 1e-5):
        raise ValueError(f"Checkpoint geometry/epsilon differs from this bounded probe: {actual}")
    if config.get("quantization_config") is not None:
        raise ValueError("This probe requires an unquantized BF16 draft checkpoint")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise ValueError("This bounded probe requires declared BF16 checkpoint precision")
    tensors, records = {}, {}
    expected = {"projection": (7168, 35840), "norm": (7168,)}
    for name, (path, key) in _find_selected_tensors(folder, safe_open).items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if tuple(handle.get_slice(key).get_shape()) != expected[name]:
                raise ValueError(f"Unexpected shape for {key}; reject before materializing weights")
            tensor = handle.get_tensor(key)
        if tuple(tensor.shape) != expected[name] or tensor.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        tensor = tensor.contiguous()
        # Hash only the selected raw tensor bytes; avoid scanning multi-GB
        # checkpoint shards or making a second full projection-sized copy.
        digest = hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy()).cast("B")).hexdigest()
        stat = path.stat()
        records[name] = {
            "file": str(path.resolve()), "key": key, "shape": list(tensor.shape),
            "dtype": str(tensor.dtype), "tensor_sha256": digest,
            "file_size_bytes": stat.st_size, "file_mtime_ns": stat.st_mtime_ns,
        }
        tensors[name] = tensor
    evidence = {
        "directory": str(folder.resolve()), "config_sha256": _digest(config_path),
        "selected_tensors": records,
        "scope": "Only real context projection/norm weights; probe activations remain synthetic. No full-model/quality validation.",
        "loader_source_sha256": _digest(Path(__file__)),
    }
    index = folder / "model.safetensors.index.json"
    if index.exists():
        evidence["weight_index_sha256"] = _digest(index)
    return tensors["projection"], tensors["norm"], evidence

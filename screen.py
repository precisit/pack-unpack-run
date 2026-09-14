"""Pack, reopen the written package, decode to bf16, evaluate with stock MLX LM."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.perplexity import eval_ppl
from mlx_lm.utils import save_config

from codec import decode, pack_trits, quantize, unpack_trits

MODEL = "mlx-community/Qwen3.5-0.8B-bf16"
MODEL_REV = "3067585164dbcc505eec73d554349de6a27571a4"
CORPUS = "Salesforce/wikitext"
CORPUS_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
CORPUS_FILE = "wikitext-2-raw-v1/validation-00000-of-00001.parquet"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(work, samples, length, seed):
    source = snapshot_download(MODEL, revision=MODEL_REV, cache_dir=work.parent / "hub-cache")
    corpus = hf_hub_download(CORPUS, CORPUS_FILE, repo_type="dataset",
                             revision=CORPUS_REV, cache_dir=work.parent / "hub-cache")
    model, tokenizer = load(source)
    # English validation split: no calibration, training or fitting on this text.
    text = "\n\n".join(pq.read_table(corpus)["text"].to_pylist())
    tokens = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
    blocks = tokens[:len(tokens) // length * length].reshape(-1, length)
    if len(blocks) < samples:
        raise ValueError(f"Only {len(blocks)} disjoint blocks are available")
    indices = np.random.default_rng(seed).permutation(len(blocks))[:samples]
    selected = blocks[indices]
    np.save(work / "tokens.npy", selected)
    manifest = {
        "model": MODEL, "model_revision": MODEL_REV,
        "corpus": CORPUS, "corpus_revision": CORPUS_REV, "corpus_file": CORPUS_FILE,
        "corpus_sha256": digest(corpus), "tokens_sha256": digest(work / "tokens.npy"),
        "seed": seed, "sequence_length": length, "num_samples": samples,
        "scored_tokens": samples * (length - 1), "block_indices": indices.tolist(),
        "tokenization": "join rows with two newlines; no special tokens or chat template",
        "model_files": {p.name: digest(p) for p in sorted(Path(source).iterdir()) if p.is_file()},
    }
    write_json(work / "inputs.json", manifest)
    return source, model, tokenizer, mx.array(selected), manifest


def selected_names(model, scope):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".layers." in name:
            if scope == "all-linear" or ".mlp." in name:
                names.append(name + ".weight")
    if not names:
        raise ValueError("Tensor selection is empty")
    return sorted(names)


def pack_model(model, tokenizer, config, directory, group_size, scope):
    directory.mkdir()
    weights = dict(tree_flatten(model.parameters()))
    names = selected_names(model, scope)
    tensors = []
    for i, name in enumerate(names):
        weight = weights[name]
        if weight.dtype != mx.bfloat16:
            raise ValueError(f"Selected tensor is not bf16: {name}")
        trits, scales = quantize(np.asarray(weight.astype(mx.float32)), group_size)
        payload = pack_trits(trits)
        filename = f"{i:03d}.npz"
        np.savez(directory / filename, codes=payload, scales=scales)
        with np.load(directory / filename, allow_pickle=False) as written:
            np.testing.assert_array_equal(unpack_trits(written["codes"], group_size), trits)
            np.testing.assert_array_equal(written["scales"].view(np.uint16), scales.view(np.uint16))
        tensors.append({"name": name, "shape": list(weight.shape), "file": filename,
                        "sha256": digest(directory / filename), "weights": weight.size,
                        "payload_bytes": payload.nbytes + scales.nbytes})
    passthrough = {k: v for k, v in weights.items() if k not in names}
    mx.save_safetensors(str(directory / "passthrough.safetensors"), passthrough)
    save_config(config, directory / "config.json")
    tokenizer.save_pretrained(directory)
    manifest = {"schema": "ternary-demo-v1", "group_size": group_size, "scope": scope,
                "recipe": "absmean-rne-ternary-fp16-scale", "tensors": tensors,
                "passthrough_sha256": digest(directory / "passthrough.safetensors"),
                "selected_weights": sum(t["weights"] for t in tensors),
                "total_text_parameters": sum(w.size for w in weights.values())}
    write_json(directory / "manifest.json", manifest)
    return manifest


def unpack_model(package, destination):
    """The only input is the package, including its unchanged tensors and assets."""
    manifest = json.loads((package / "manifest.json").read_text())
    if digest(package / "passthrough.safetensors") != manifest["passthrough_sha256"]:
        raise ValueError("Passthrough checksum mismatch")
    weights = mx.load(str(package / "passthrough.safetensors"))
    for entry in manifest["tensors"]:
        path = package / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Packed tensor checksum mismatch")
        with np.load(path, allow_pickle=False) as stored:
            value = decode(stored["codes"], stored["scales"], entry["shape"], manifest["group_size"])
        weights[entry["name"]] = mx.array(value).astype(mx.bfloat16)
        mx.eval(weights[entry["name"]])
    destination.mkdir()
    mx.save_safetensors(str(destination / "model.safetensors"), weights)
    reloaded = mx.load(str(destination / "model.safetensors"))
    if reloaded.keys() != weights.keys():
        raise ValueError("Decoded checkpoint inventory changed during serialization")
    for name, expected in weights.items():
        actual = reloaded[name]
        if actual.dtype != expected.dtype or not mx.array_equal(actual, expected).item():
            raise ValueError(f"Decoded checkpoint serialization mismatch: {name}")
    save_config(json.loads((package / "config.json").read_text()), destination / "config.json")
    # Copy only tokenizer assets; never any original or packed weight file.
    import shutil
    for filename in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        if (package / filename).exists():
            shutil.copyfile(package / filename, destination / filename)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--scope", choices=["mlp", "all-linear"], default="mlp")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.num_samples < 1 or args.sequence_length < 2 or any(g < 1 for g in args.group_sizes):
        parser.error("Invalid sample count, sequence length or group size")
    args.work.mkdir(parents=True, exist_ok=False)
    mx.random.seed(args.seed)
    start = time.perf_counter()
    source, model, tokenizer, data, inputs = prepare(args.work, args.num_samples, args.sequence_length, args.seed)
    config = json.loads((Path(source) / "config.json").read_text())
    receipt = {"inputs": inputs, "scope": args.scope, "batch_size": 1,
               "platform": platform.platform(), "device": mx.device_info(),
               "versions": {n: importlib.metadata.version(n) for n in
                            ["mlx", "mlx-lm", "numpy", "huggingface-hub", "pyarrow", "transformers"]},
               "source_sha256": {n: digest(Path(__file__).with_name(n)) for n in ["screen.py", "codec.py"]},
               "results": []}
    for group_size in [None] + args.group_sizes:
        lane_start = time.perf_counter()
        if group_size is None:
            lane_model = model
            row = {"lane": "bf16-reference"}
        else:
            package = args.work / f"ternary-g{group_size}-packed"
            dense = args.work / f"ternary-g{group_size}-bf16"
            manifest = pack_model(model, tokenizer, config, package, group_size, args.scope)
            unpack_model(package, dense)
            lane_model, _ = load(str(dense))
            row = {"lane": f"ternary-g{group_size}", "group_size": group_size,
                   "selected_weights": manifest["selected_weights"],
                   "total_text_parameters": manifest["total_text_parameters"],
                   "selected_payload_bpw": 8 * sum(t["payload_bytes"] for t in manifest["tensors"]) / manifest["selected_weights"],
                   "selected_tensor_count": len(manifest["tensors"]),
                   "package_directory_bytes": sum(p.stat().st_size for p in package.iterdir() if p.is_file()),
                   "decoded_safetensors_bytes": sum(p.stat().st_size for p in dense.glob("*.safetensors"))}
        print(f"Evaluating {row['lane']}", flush=True)
        ppl, _ = eval_ppl(lane_model, data, batch_size=1)
        row.update(perplexity=ppl, lane_seconds=time.perf_counter() - lane_start)
        receipt["results"].append(row)
        receipt["total_seconds"] = time.perf_counter() - start
        write_json(args.work / "results.json", receipt)
        print(json.dumps(row), flush=True)
        if group_size is not None:
            del lane_model
            mx.clear_cache()


if __name__ == "__main__":
    main()

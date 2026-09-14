# Pack, unpack, run

Test a weight-format idea before implementing its runtime. This small example
quantizes selected Qwen3.5-0.8B text weights to ternary values, writes a package,
decodes **only that package** into a bf16 checkpoint, and measures perplexity
with stock MLX LM. It has no custom kernel and makes no packed-inference claim.

## Run

Use an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Python 3.12 and all dependencies are recorded in `uv.lock`. Allow about 10 GB
of disk space for the environment, model download, packages and decoded models.
The recorded run used an M1 Max with 32 GiB memory; smaller-memory machines
have not been tested.

```sh
git clone --branch demo/quality-screen https://github.com/precisit/pack-unpack-run.git
cd pack-unpack-run
uv sync --locked
uv run python -m unittest -v
uv run python screen.py --group-sizes 64 128
```

The first run downloads the public model (about 1.71 GB) and a small corpus
file. Downloads are retained in `hub-cache/`. Model code is supplied by the
installed MLX LM package, with no private fork or remote custom code.
The runner creates `work/` and refuses to overwrite an existing directory.
For another run, choose a new directory with `--work work-second`.

The default changes only feed-forward gate/up/down projections: 72 matrices,
264,241,152 of the text model's 752,393,024 parameters. Attention, embeddings,
norms and other parameters retain their values and dtypes. Vision is outside
this text-only experiment. `--scope all-linear` instead selects every `nn.Linear`
weight under the text transformer layers, still excluding embeddings and norms;
that optional scope has not been evaluated in the published receipt.

## What the example does

- `codec.py`: mean-absolute scale rounded to fp16, nearest-even ternary codes,
  five base-three digits per byte, and decoding. Each group is separately
  byte-aligned, with zero-trit padding. Values 243 through 255 are invalid.
- `test_codec.py`: all 243 codewords checked against a scalar definition,
  tail lengths, group boundaries, invalid bytes, zero groups and file replay.
  A changed valid code byte must change the reconstruction.
- `screen.py`: writes per-tensor NPZ files containing codes and fp16 scales,
  plus unchanged tensors and tokenizer/configuration assets. The decoder
  accepts only the package, verifies hashes, restores selected weights as
  bf16, and checks every tensor after checkpoint serialization. MLX LM then
  reloads each candidate for evaluation.

The scale rule uses no activation calibration or training. A zero stored scale
produces a zero group. Matrix columns must be divisible by the group size.
The default 64 and 128 work for the selected matrices. This is an educational
format, not a production loader for untrusted third-party packages.

For group size G, selected payload rate is `8 * (ceil(G / 5) + 2) / G`:
1.875 bpw at G=64 and 1.750 bpw at G=128. These rates include padding and scales,
but exclude NPZ headers, manifests, tokenizer assets and unchanged tensors.
Complete directory bytes are recorded separately. Decoded checkpoints retain
ordinary floating-point storage, approximately 1.50 GB of safetensors here.

## Recorded experiment

[The full receipt](results/qwen35-08b-mlp.json) records source-file hashes,
dependencies, input hashes, block indices, tensor counts and results. It was
produced on 2026-09-14 with stock MLX 0.32.0 and MLX LM 0.31.3 on an Apple
M1 Max, 32 GiB, macOS 26.5.1.

| Configuration | Perplexity |
| --- | ---: |
| Untouched bf16 reference | 24.92093 |
| Ternary feed-forward projections, G=64 | 8695.60368 |
| Ternary feed-forward projections, G=128 | 11110.01578 |

The simple recipe fails this quality screen. Correct packing preserves the
chosen trits; it cannot recover information discarded by the quantizer.
These results do not establish the quality of other ternary methods or models.

The complete recorded run took 108.82 seconds with downloads cached. Its timer
starts before input preparation and includes loading, hashing, tokenization,
packing, decoding, serialization checks and evaluation, through the final
evaluation. Dependency installation and the final result-file write are outside
that timer. This is an observed screen duration, not native inference performance.

### Fixed inputs and metric

- Model: [`mlx-community/Qwen3.5-0.8B-bf16`](https://huggingface.co/mlx-community/Qwen3.5-0.8B-bf16/tree/3067585164dbcc505eec73d554349de6a27571a4),
  revision `3067585164dbcc505eec73d554349de6a27571a4`.
- Corpus: [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1),
  revision `b08601e04326c79dfdd32d625aee71d232d685c3`, file
  `wikitext-2-raw-v1/validation-00000-of-00001.parquet`.
- Join corpus rows with two newlines and tokenize without added special tokens
  or a chat template. Divide into disjoint 512-token blocks, dropping the final
  incomplete block. Select 64 without replacement using `default_rng(123)`.
  Save these tokens once; reuse them in all three evaluations.
- Call MLX LM's [`eval_ppl`](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/perplexity.py#L60-L105)
  with batch size one. Each block has fresh model state and 511 next-token
  predictions: 32,704 scored tokens. Report the exponential of the mean
  negative log probability. Lower is better. No token-independence confidence
  interval is claimed.

No training or calibration uses this validation text. If you choose recipes by
repeatedly evaluating this pool, reserve separate held-out text for the finalists.
These values are not directly comparable to benchmark tables using different
tokenization, context lengths or document boundaries.

The run writes `inputs.json`, `tokens.npy`, `results.json`, and the packed and
decoded artifacts under the work directory. No model weights or corpus text are
committed to Git. The result receipt is a measurement of the source hashes it
records; later source changes do not retroactively become the measured version.

## License

Example code and documentation: Apache-2.0, copyright 2026 Precisit.
Dependencies retain their own licenses. The Qwen checkpoint is Apache-2.0;
WikiText is distributed under CC BY-SA 3.0. Downloads remain subject to their
respective licenses; this repository does not redistribute them.

# DeepSeek-V4.1-Flash on 8× V100

The most capable model this engine serves, for reasoning and world knowledge. As far as we know, no other engine runs it on V100s for local agentic coding. It is slow; it runs.

Official `deepseek-ai/DeepSeek-V4.1-Flash` on 8× V100-SXM2-32GB. This build has carried a multi-hour Claude Code session on a single conversation. Continuations that match a recorded chunk or request stop are not re-prefilled (a few dozen new tokens is a few seconds). A suffix of several thousand tokens that was never computed is still about 60 tok/s: each rank holds 30 experts on the GPU and a chunk often touches more, so the spill cache thrashes inside the chunk. Each rank's weight load reads only the experts it owns, and the draft reads only `mtp.*`. On this 8-card shape that was 1347 s for the target and 17 s for the draft, about 24 minutes until the server was ready. Set up host memory before the first launch ([below](#host-memory)).

## Get the model

Use the official DeepSeek mixed-quant checkpoint. Dense weights are block FP8 (`quant_method: fp8`, 32×32 `ue8m0`); routed experts are native FP4 (`expert_dtype: fp4`, MXFP4). The DSpark draft lives in the same repo. Runtime on this port is FP16 activations and an FP8-E4M3 KV cache. Leave the dense MXFP8 packed; unpacking it to FP16 does not fit in 32 GB. Skip third-party NVFP4 / GPTQ / AWQ re-quants, and skip DeepSeek-V4-Flash: that is a different architecture.

Weights: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

```bash
pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --local-dir ~/models/DeepSeek-V4.1-Flash

export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
```

~476 GB (48 shards). The export is multimodal. The vision tower is one fp32 copy on rank 0; each GEMM streams through that GPU as fp16 and is dropped. The model is MIT-licensed.

| | |
|---|---|
| checkpoint | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash |
| quantisation | native mixed: MXFP8 dense (e4m3 + UE8M0, 32×32) + MXFP4 routed experts |
| do not use | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash , https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 , https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark (V4-Flash NVFP4, not V4.1) |

## Host memory

Two large host-side pieces, both on the NUMA node(s) next to the GPUs. Find that node in the `NUMA Affinity` column of `nvidia-smi topo -m`.

- **Engram tables, ~189 GiB**, mapped from a boot-reserved 1 GiB hugetlb pool on the GPU-local node. The launch fails loudly if the pool is short instead of falling back to 4 KiB pages. On the measured box all eight GPUs sit on node 1, reserved on the kernel command line:

  ```
  default_hugepagesz=1G hugepagesz=1G hugepages=1:190
  ```

  `hugepages=<node>:<count>` reserves on one node. Then set `SGLANG_DSV41_ENGRAM_NUMA_NODE=<node>` (script default 1).
- **Pinned expert spill, 13 GiB per GPU** (~104 GiB), on transparent huge pages, not from the hugetlb pool. It is striped layer by layer over `SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES` (default `1,0`, both sockets). On a single-socket host set both variables to `0`.

If another model was loaded just before, the second node can be full of page cache and the spill pin fails with `mbind(MPOL_BIND, node=0, ...) failed errno=5`. Drop the cache first: `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`.

## Measured performance

2026-09-24, `llm-decode-bench` 0.6.2, the 8-card recipe (DSpark on, sticky last-seq, radix cache off, advertised 256k, `np=1`). Temperature 0. The server was not started with `--enable-metrics`, so this run has no speculative-accept gauge. The ship script pins `--max-running-requests 1`.

| check | result |
|---|---|
| Coding peak, `merge_sorted`, natural stop at 104 tokens, 3 runs | **9.3** tok/s median (8.8–9.3) |
| Sustained padding decode, 20 s, `ignore_eos` | **2.8** tok/s (ITL 322 ms) |
| Cold prefill scout, server counted 5,286 prompt tokens | TTFT **45.3 s**, **117** tok/s |
| Warmer one-token prefill, 8,004 tokens, spill already touched | TTFT **14.3 s**, **558** tok/s |

The 2.8 tok/s cell is greedy padding, the case this model loops on. It is a different number from the 9.3 tok/s coding rate and from the ~60 tok/s uncached suffix above. The 117 tok/s scout is the cold first touch; 558 tok/s is the same box after that spill was already warm. Temperature 0 is right for short code and wrong for long prose. Use `temperature=1`, `top_p=0.95` for chat. The ship script leaves `/health` as a liveness probe (no generation), so a load balancer GET does not drop the sticky pin.

## Reference recipe

The wrapper is the supported entry. It exports the env knobs that are not CLI flags, then launches the server.

```bash
export MODEL_PATH=~/models/DeepSeek-V4.1-Flash
export SGLANG_DSV41_DSPARK=1
export SGLANG_DSV41_STICKY_LAST_SEQ=1
export SGLANG_DSV41_CONTEXT_LEN=262144
bash scripts/serve_dsv41_v100.sh
```

Expanded (what the script actually runs when DSpark is on). Spill, Engram, and sticky are set by the env block, not implied by the CLI flags.

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_DSV41_EXPERT_SPILL_APPLY=1
export SGLANG_DSV41_EXPERT_SPILL_GB=13
export SGLANG_DSV41_SPILL_LANDING=36
export SGLANG_DSV41_STICKY_LAST_SEQ=1
export NCCL_ALGO=allreduce:tree
export NCCL_BUFFSIZE=2097152
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=4

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp 8 --ep-size 8 \
  --dtype float16 \
  --moe-runner-backend marlin \
  --attention-backend dsv4 \
  --context-length 262144 \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.87 \
  --max-running-requests 1 \
  --max-total-tokens 262144 \
  --max-prefill-tokens 262144 \
  --pre-warm-nccl \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --disable-radix-cache \
  --reasoning-parser deepseek-v41 \
  --tool-call-parser deepseekv41 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| DSpark | on (`γ=5` from the checkpoint) | Best measured TG on this box. `SGLANG_DSV41_DSPARK=0` is greedy Tree |
| sticky last-seq | on | One resident conversation. Exact continuation, or a shorter prefix saved at a chunk or request stop. Anything else, including a second conversation, drops the pin and prefills from 0. Radix stays off (a radix hit desyncs CSA2 pending state) |
| context / max tokens | 262144 | Advertised window. 8k prefill is what has been smoked; 512k has not left ~300 MiB for the Engram MXFP8 unpack |
| `--mem-fraction-static` | 0.87 | 0.99 OOMs the Engram unpack on T=6 verify capture. 0.88 OOMed the same 300 MiB unpack on a 461-token sticky prefill (TP7 had 284 MiB). 0.86 raises: no KV pool after draft weights |
| expert spill | 13 GiB/rank (landing 36) | Spill 12 left 8k ~8 MiB short of that unpack |
| `--max-running-requests` | 1 | DSpark would otherwise inflate this |
| `--chunked-prefill-size` | 2048 | Vestigial SWA floor is sized for this chunk |

Leave `--speculative-dspark-block-size` at the checkpoint default. Checkpoint weights stay mixed MXFP4 experts + packed MXFP8 dense.

## Limitations

- One conversation. A second session prefills from zero, and the resident image does not survive a restart.
- A long suffix that was never computed runs at about 60 tok/s (the spill cache thrashes inside a prefill chunk).
- A few GiB of HBM are left after load. Open-ended greedy decoding (temperature 0) can loop; use `temperature=1`, `top_p=0.95` for chat.
- Image requests stream the rank-0 vision tower through GPU GEMMs: fine for casual use, not fast.

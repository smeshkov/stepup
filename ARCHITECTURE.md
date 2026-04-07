# Architecture

- **`internal/tokenizer/`** — Byte-Pair Encoding (BPE) tokenizer with GPT-4-style regex pre-tokenization
  - `tokenizer.go` — `Tokenizer` struct with `TrainProduction()` (learns merge rules from text) and `Encode()` (converts text to token IDs)
  - `tokenizer_util.go` — Regex chunking (`chunkText`), pair statistics (`getChunkedStats`), and merge operations
  - Uses `github.com/dlclark/regexp2` instead of Go's stdlib `regexp` because BPE pre-tokenization requires PCRE lookaround assertions (Go's RE2 engine doesn't support them)

- **`internal/model/`** — Transformer model components
  - `tensor.go` — `Tensor` struct backed by a contiguous 1D `[]float32` slice with stride-based indexing (`row*Cols + col`), zero-copy `Row()` slicing, and `AppendRows()` for cache growth
  - `model.go` — `Embedding` struct with `Forward()` (token ID → dense vector lookup via `Tensor`)
  - `model_util.go` — `PositionalEncoding` struct with `Forward()` (sinusoidal position signals via `Tensor`)
  - `attention.go` — `MultiHeadAttention` and `Head` structs with `Forward()` and `ForwardCached()` (KV caching for autoregressive generation), plus `ResetCache()` to clear cached state
  - `layers.go` — `LayerNorm`, `FeedForward`, and `TransformerBlock` structs with `Forward()` and `ForwardCached()` methods, plus `ResetCache()`
  - `math_utils.go` — Linear algebra helpers: `matMul`, `transpose`, `scaleAndSoftmax`, `addTensors`, `addBias`, `relu` — all operating on `*Tensor`
  - `sampling.go` — Token sampling strategies: `SampleGreedy` (argmax), `SampleTemperature` (softmax with temperature scaling), `SampleTopP` (nucleus sampling with cumulative probability cutoff)
  - `inference.go` — `InferenceEngine` struct with `Infer()` (single request) and `RunBatch()` (concurrent batched inference using goroutine worker pool)

## Key Design Decisions

- **Chunked BPE**: Training and encoding operate on `[][]int` (slice of chunks) rather than a flat `[]int` to prevent merges across regex-defined boundaries
- **Pre-allocated slices**: The `merge()` function pre-allocates output capacity to avoid GC pressure during training on large text
- **Merge ordering**: During encoding, merges are applied oldest-first (lowest ID) to match training order
- **Contiguous Tensor**: All matrix data uses a 1D `[]float32` backing array instead of `[][]float32` — a single heap allocation per tensor instead of R+1, eliminating GC pressure from large matrices
- **Zero-copy row access**: `Tensor.Row(i)` returns a slice view into the backing array, enabling in-place mutation without allocation
- **KV Caching**: `ForwardCached()` stores past Key and Value projections per attention head, so autoregressive generation only computes Q/K/V for new tokens and reuses cached history via `AppendRows()`
- **Pre-norm architecture**: `TransformerBlock` applies `LayerNorm` *before* attention and FFN (Pre-LN), not after — this is more stable during training than the original Post-LN design
- **Residual connections**: Each sub-block uses `x + SubLayer(LayerNorm(x))` to preserve gradient flow through deep stacks
- **FFN expansion factor**: `FeedForward` expands to 4× `dModel` (standard transformer ratio) with ReLU activation
- **Sampling strategies**: Three decoding methods — greedy (deterministic argmax), temperature (controls distribution sharpness), and top-p/nucleus (dynamic vocabulary cutoff based on cumulative probability)
- **Concurrent Batched Inference**: `RunBatch()` processes multiple requests in parallel using a goroutine worker pool with semaphore-based concurrency limiting, maximizing GPU utilization while preserving result order
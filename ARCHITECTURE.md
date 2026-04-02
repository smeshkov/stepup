# Architecture

- **`internal/tokenizer/`** — Byte-Pair Encoding (BPE) tokenizer with GPT-4-style regex pre-tokenization
  - `tokenizer.go` — `Tokenizer` struct with `TrainProduction()` (learns merge rules from text) and `Encode()` (converts text to token IDs)
  - `tokenizer_util.go` — Regex chunking (`chunkText`), pair statistics (`getChunkedStats`), and merge operations
  - Uses `github.com/dlclark/regexp2` instead of Go's stdlib `regexp` because BPE pre-tokenization requires PCRE lookaround assertions (Go's RE2 engine doesn't support them)

- **`internal/model/`** — Transformer model components
  - `model.go` — `Embedding` struct with `Forward()` (token ID → dense vector lookup)
  - `model_util.go` — `PositionalEncoding` struct with `Forward()` (sinusoidal position signals)
  - `attention.go` — `MultiHeadAttention` struct with `Forward()` (parallel Q/K/V attention heads)
  - `layers.go` — `LayerNorm`, `FeedForward`, and `TransformerBlock` structs with `Forward()` methods
  - `math_utils.go` — Linear algebra helpers: `matMul`, `transpose`, `scaleAndSoftmax`, `addMatrices`, `addBias`, `relu`

## Key Design Decisions

- **Chunked BPE**: Training and encoding operate on `[][]int` (slice of chunks) rather than a flat `[]int` to prevent merges across regex-defined boundaries
- **Pre-allocated slices**: The `merge()` function pre-allocates output capacity to avoid GC pressure during training on large text
- **Merge ordering**: During encoding, merges are applied oldest-first (lowest ID) to match training order
- **Pre-norm architecture**: `TransformerBlock` applies `LayerNorm` *before* attention and FFN (Pre-LN), not after — this is more stable during training than the original Post-LN design
- **Residual connections**: Each sub-block uses `x + SubLayer(LayerNorm(x))` to preserve gradient flow through deep stacks
- **FFN expansion factor**: `FeedForward` expands to 4× `dModel` (standard transformer ratio) with ReLU activation
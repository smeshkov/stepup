# Architecture

- **`internal/tokenizer/`** — Byte-Pair Encoding (BPE) tokenizer with GPT-4-style regex pre-tokenization
  - `tokenizer.go` — `Tokenizer` struct with `TrainProduction()` (learns merge rules from text) and `Encode()` (converts text to token IDs)
  - `tokenizer_util.go` — Regex chunking (`chunkText`), pair statistics (`getChunkedStats`), and merge operations
  - Uses `github.com/dlclark/regexp2` instead of Go's stdlib `regexp` because BPE pre-tokenization requires PCRE lookaround assertions (Go's RE2 engine doesn't support them)

- **`internal/model/`** — Transformer model components
  - `model.go` — `Embedding`, `PositionalEncoding`, `SelfAttention`, `FeedForward`, `TransformerBlock`, `Transformer` structs
  - `attention.go` — `SelfAttention` implementation with `Forward()` method
  - `positional_encoding.go` — `PositionalEncoding` implementation with `Forward()` method
  - `feed_forward.go` — `FeedForward` implementation with `Forward()` method
  - `transformer_block.go` — `TransformerBlock` implementation with `Forward()` method
  - `transformer.go` — `Transformer` implementation with `Forward()` method

## Key Design Decisions

- **Chunked BPE**: Training and encoding operate on `[][]int` (slice of chunks) rather than a flat `[]int` to prevent merges across regex-defined boundaries
- **Pre-allocated slices**: The `merge()` function pre-allocates output capacity to avoid GC pressure during training on large text
- **Merge ordering**: During encoding, merges are applied oldest-first (lowest ID) to match training order
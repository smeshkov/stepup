# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**stepup** is a from-scratch transformer/LLM implementation in Go. The goal is to build every component of an LLM pipeline manually — tokenization, embeddings, attention, training, and inference. See `_docs/00_initial.md` for the full roadmap.

Current state: Phase 1 — BPE tokenizer is implemented in `internal/tokenizer/`.

## Build & Test Commands

```bash
# Run all tests
go test ./...

# Run a single package's tests
go test ./internal/tokenizer/

# Run a specific test
go test ./internal/tokenizer/ -run TestTokenizer_EndToEnd

# Run tests with verbose output
go test -v ./internal/tokenizer/
```

## Architecture

- **`internal/tokenizer/`** — Byte-Pair Encoding (BPE) tokenizer with GPT-4-style regex pre-tokenization
  - `tokenizer.go` — `Tokenizer` struct with `TrainProduction()` (learns merge rules from text) and `Encode()` (converts text to token IDs)
  - `tokenizer_util.go` — Regex chunking (`chunkText`), pair statistics (`getChunkedStats`), and merge operations
  - Uses `github.com/dlclark/regexp2` instead of Go's stdlib `regexp` because BPE pre-tokenization requires PCRE lookaround assertions (Go's RE2 engine doesn't support them)

## Key Design Decisions

- **Chunked BPE**: Training and encoding operate on `[][]int` (slice of chunks) rather than a flat `[]int` to prevent merges across regex-defined boundaries
- **Pre-allocated slices**: The `merge()` function pre-allocates output capacity to avoid GC pressure during training on large text
- **Merge ordering**: During encoding, merges are applied oldest-first (lowest ID) to match training order

## Development Docs

Design documents and milestone specs live in `_docs/`. See `_docs/_dev.md` for the index.

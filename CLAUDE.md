# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**stepup** is a from-scratch transformer/LLM implementation in Go. The goal is to build every component of an LLM pipeline manually — tokenization, embeddings, attention, training, and inference. See `_docs/00_initial.md` for the full roadmap.

**Current state:**
- Phase 1 — Data & Tokenization is implemented in `internal/tokenizer/`.
- Phase 1 — Embeddings & Positional Encoding is implemented in `internal/model/`.
- Phase 1 — Self-attention is implemented in `internal/model/`.
- Phase 1 — Transformer Block is implemented in `internal/model/`.

## Development process

- `make lint` and `make test` must pass.
- always make sure to add tests for any new code you write.
- tests should be placed next to the code they test.
- never modify schema without migrations.

## Architecture

See `ARCHITECTURE.md`

## Development Docs

Design documents and milestone specs live in `_docs/`. See `_docs/_dev.md` for the index.

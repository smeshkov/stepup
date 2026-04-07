package model

import (
	"math"
	"testing"
)

func TestHead_ForwardCached_MatchesUncached(t *testing.T) {
	dModel := 8
	dHead := 4

	// Create two identical heads (same weights)
	headCached := NewHead(dModel, dHead)
	headUncached := NewHead(dModel, dHead)
	copy(headUncached.Wq.Data, headCached.Wq.Data)
	copy(headUncached.Wk.Data, headCached.Wk.Data)
	copy(headUncached.Wv.Data, headCached.Wv.Data)

	// Full 3-token input
	input := NewTensor(3, dModel)
	for i := range input.Data {
		input.Data[i] = float32(i) * 0.1
	}

	// Uncached: process all 3 tokens at once
	uncachedOut := headUncached.Forward(input)

	// Cached: process token-by-token
	for i := range 3 {
		tokenInput := NewTensor(1, dModel)
		copy(tokenInput.Data, input.Row(i))
		cachedOut := headCached.ForwardCached(tokenInput)

		// Only the last step's output should match the uncached result for that token
		if i == 2 {
			for j := range dHead {
				if !almostEqual(cachedOut.Get(0, j), uncachedOut.Get(2, j), 1e-4) {
					t.Errorf("Token 2, dim %d: cached=%f, uncached=%f",
						j, cachedOut.Get(0, j), uncachedOut.Get(2, j))
				}
			}
		}
	}
}

func TestHead_ForwardCached_CacheGrows(t *testing.T) {
	dModel := 8
	dHead := 4
	head := NewHead(dModel, dHead)

	// Initially no cache
	if head.KCache != nil || head.VCache != nil {
		t.Fatal("Expected nil cache before any forward pass")
	}

	token := NewTensor(1, dModel)
	for i := range token.Data {
		token.Data[i] = 0.5
	}

	// After first token
	head.ForwardCached(token)
	if head.KCache.Rows != 1 {
		t.Errorf("Expected KCache with 1 row after 1 token, got %d", head.KCache.Rows)
	}

	// After second token
	head.ForwardCached(token)
	if head.KCache.Rows != 2 {
		t.Errorf("Expected KCache with 2 rows after 2 tokens, got %d", head.KCache.Rows)
	}

	// After third token
	head.ForwardCached(token)
	if head.KCache.Rows != 3 {
		t.Errorf("Expected KCache with 3 rows after 3 tokens, got %d", head.KCache.Rows)
	}
	if head.VCache.Rows != 3 {
		t.Errorf("Expected VCache with 3 rows after 3 tokens, got %d", head.VCache.Rows)
	}
}

func TestHead_ResetCache(t *testing.T) {
	dModel := 8
	dHead := 4
	head := NewHead(dModel, dHead)

	token := NewTensor(1, dModel)
	head.ForwardCached(token)
	head.ForwardCached(token)

	if head.KCache == nil {
		t.Fatal("Cache should not be nil after forward passes")
	}

	head.ResetCache()

	if head.KCache != nil || head.VCache != nil {
		t.Error("Cache should be nil after ResetCache")
	}
}

func TestMultiHeadAttention_ForwardCached_MatchesUncached(t *testing.T) {
	dModel := 16
	numHeads := 4

	// Create two identical MHAs
	mhaCached := NewMultiHeadAttention(numHeads, dModel)
	mhaUncached := NewMultiHeadAttention(numHeads, dModel)

	// Copy all weights
	copy(mhaUncached.Wo.Data, mhaCached.Wo.Data)
	for h := range numHeads {
		copy(mhaUncached.Heads[h].Wq.Data, mhaCached.Heads[h].Wq.Data)
		copy(mhaUncached.Heads[h].Wk.Data, mhaCached.Heads[h].Wk.Data)
		copy(mhaUncached.Heads[h].Wv.Data, mhaCached.Heads[h].Wv.Data)
	}

	// 4-token input
	input := NewTensor(4, dModel)
	for i := range input.Data {
		input.Data[i] = float32(i) * 0.01
	}

	// Uncached: all at once
	uncachedOut, err := mhaUncached.Forward(input)
	if err != nil {
		t.Fatalf("Uncached forward failed: %v", err)
	}

	// Cached: token-by-token, check the last token's output
	var lastCachedOut *Tensor
	for i := range 4 {
		tokenInput := NewTensor(1, dModel)
		copy(tokenInput.Data, input.Row(i))
		lastCachedOut, err = mhaCached.ForwardCached(tokenInput)
		if err != nil {
			t.Fatalf("Cached forward failed at token %d: %v", i, err)
		}
	}

	// The last token's output should match
	for j := range dModel {
		if !almostEqual(lastCachedOut.Get(0, j), uncachedOut.Get(3, j), 1e-3) {
			t.Errorf("Dim %d: cached=%f, uncached=%f",
				j, lastCachedOut.Get(0, j), uncachedOut.Get(3, j))
		}
	}
}

func TestMultiHeadAttention_ResetCache(t *testing.T) {
	dModel := 16
	numHeads := 4
	mha := NewMultiHeadAttention(numHeads, dModel)

	token := NewTensor(1, dModel)
	_, _ = mha.ForwardCached(token)

	mha.ResetCache()

	for i, head := range mha.Heads {
		if head.KCache != nil || head.VCache != nil {
			t.Errorf("Head %d cache should be nil after ResetCache", i)
		}
	}
}

func TestTransformerBlock_ForwardCached_Dimensions(t *testing.T) {
	dModel := 16
	numHeads := 4
	block := NewTransformerBlock(numHeads, dModel)

	// Feed 3 tokens one at a time
	for i := range 3 {
		token := NewTensor(1, dModel)
		for j := range dModel {
			token.Set(0, j, float32(i)*0.1)
		}

		out, err := block.ForwardCached(token)
		if err != nil {
			t.Fatalf("ForwardCached failed at step %d: %v", i, err)
		}

		// Output should always be [1, dModel] (only the new token)
		if out.Rows != 1 || out.Cols != dModel {
			t.Errorf("Step %d: expected 1x%d output, got %dx%d", i, dModel, out.Rows, out.Cols)
		}

		// No NaN/Inf
		for k := range out.Data {
			if math.IsNaN(float64(out.Data[k])) || math.IsInf(float64(out.Data[k]), 0) {
				t.Fatalf("Step %d: NaN or Inf in output at index %d", i, k)
			}
		}
	}
}

func TestTransformerBlock_ResetCache(t *testing.T) {
	dModel := 16
	numHeads := 4
	block := NewTransformerBlock(numHeads, dModel)

	token := NewTensor(1, dModel)
	_, _ = block.ForwardCached(token)
	_, _ = block.ForwardCached(token)

	block.ResetCache()

	// All heads should have nil caches
	for i, head := range block.Attn.Heads {
		if head.KCache != nil || head.VCache != nil {
			t.Errorf("Head %d cache should be nil after block ResetCache", i)
		}
	}
}

func TestTransformerBlock_ForwardCached_MatchesUncached(t *testing.T) {
	dModel := 16
	numHeads := 4

	blockCached := NewTransformerBlock(numHeads, dModel)
	blockUncached := NewTransformerBlock(numHeads, dModel)

	// Copy all weights to make them identical
	copy(blockUncached.Attn.Wo.Data, blockCached.Attn.Wo.Data)
	for h := range numHeads {
		copy(blockUncached.Attn.Heads[h].Wq.Data, blockCached.Attn.Heads[h].Wq.Data)
		copy(blockUncached.Attn.Heads[h].Wk.Data, blockCached.Attn.Heads[h].Wk.Data)
		copy(blockUncached.Attn.Heads[h].Wv.Data, blockCached.Attn.Heads[h].Wv.Data)
	}
	copy(blockUncached.FFN.W1.Data, blockCached.FFN.W1.Data)
	copy(blockUncached.FFN.W2.Data, blockCached.FFN.W2.Data)
	copy(blockUncached.FFN.B1, blockCached.FFN.B1)
	copy(blockUncached.FFN.B2, blockCached.FFN.B2)
	copy(blockUncached.AttnNorm.Gamma, blockCached.AttnNorm.Gamma)
	copy(blockUncached.AttnNorm.Beta, blockCached.AttnNorm.Beta)
	copy(blockUncached.FFNNorm.Gamma, blockCached.FFNNorm.Gamma)
	copy(blockUncached.FFNNorm.Beta, blockCached.FFNNorm.Beta)

	// 3-token input
	input := NewTensor(3, dModel)
	for i := range input.Data {
		input.Data[i] = float32(i) * 0.02
	}

	// Uncached: all at once
	uncachedOut, err := blockUncached.Forward(input)
	if err != nil {
		t.Fatalf("Uncached forward failed: %v", err)
	}

	// Cached: token-by-token
	var lastOut *Tensor
	for i := range 3 {
		tokenInput := NewTensor(1, dModel)
		copy(tokenInput.Data, input.Row(i))
		lastOut, err = blockCached.ForwardCached(tokenInput)
		if err != nil {
			t.Fatalf("Cached forward failed at token %d: %v", i, err)
		}
	}

	// Last token output should match
	for j := range dModel {
		if !almostEqual(lastOut.Get(0, j), uncachedOut.Get(2, j), 1e-3) {
			t.Errorf("Dim %d: cached=%f, uncached=%f",
				j, lastOut.Get(0, j), uncachedOut.Get(2, j))
		}
	}
}

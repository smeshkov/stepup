package model

import (
	"math"
	"math/rand"
)

// initWeights populates a tensor with small random values using a single allocation.
func initWeights(rows, cols int) *Tensor {
	t := NewTensor(rows, cols)
	for i := range t.Data {
		t.Data[i] = rand.Float32()*0.02 - 0.01
	}
	return t
}

// Head represents a single Attention head.
type Head struct {
	DHead  int
	Wq     *Tensor // Projects from [DModel] to [DHead]
	Wk     *Tensor
	Wv     *Tensor
	KCache *Tensor // Accumulated Key matrices from previous tokens
	VCache *Tensor // Accumulated Value matrices from previous tokens
}

func NewHead(dModel, dHead int) *Head {
	return &Head{
		DHead: dHead,
		Wq:    initWeights(dModel, dHead),
		Wk:    initWeights(dModel, dHead),
		Wv:    initWeights(dModel, dHead),
	}
}

// Forward executes Attention for this specific head
func (h *Head) Forward(input *Tensor) *Tensor {
	// 1. Project input [seqLen, dModel] * [dModel, dHead] -> [seqLen, dHead]
	Q := matMul(input, h.Wq)
	K := matMul(input, h.Wk)
	V := matMul(input, h.Wv)

	// 2. Scores [seqLen, dHead] * [dHead, seqLen] -> [seqLen, seqLen]
	K_T := transpose(K)
	scores := matMul(Q, K_T)

	// 3. Scale and Softmax
	scaleFactor := float32(math.Sqrt(float64(h.DHead)))
	scaleAndSoftmax(scores, scaleFactor)

	// 4. Multiply by Values [seqLen, seqLen] * [seqLen, dHead] -> [seqLen, dHead]
	out := matMul(scores, V)

	return out
}

// ForwardCached runs attention using the KV cache for efficient autoregressive generation.
// It takes only the new token(s), computes their K/V, appends to the cache, and runs
// attention against the full cached history. Returns output for the new tokens only.
func (h *Head) ForwardCached(input *Tensor) *Tensor {
	// 1. Compute Q, K, V for the NEW tokens only
	Q := matMul(input, h.Wq) // [newTokens, dHead]
	K := matMul(input, h.Wk) // [newTokens, dHead]
	V := matMul(input, h.Wv) // [newTokens, dHead]

	// 2. Append new K and V to the cache
	if h.KCache == nil {
		h.KCache = K
		h.VCache = V
	} else {
		h.KCache = h.KCache.AppendRows(K)
		h.VCache = h.VCache.AppendRows(V)
	}

	// 3. Compute attention scores: Q_new * K_all^T -> [newTokens, totalSeqLen]
	K_T := transpose(h.KCache)
	scores := matMul(Q, K_T)

	// 4. Scale and Softmax
	scaleFactor := float32(math.Sqrt(float64(h.DHead)))
	scaleAndSoftmax(scores, scaleFactor)

	// 5. Weighted sum over all cached Values: [newTokens, totalSeqLen] * [totalSeqLen, dHead]
	out := matMul(scores, h.VCache)

	return out
}

// ResetCache clears the KV cache, preparing the head for a new sequence.
func (h *Head) ResetCache() {
	h.KCache = nil
	h.VCache = nil
}

// MultiHeadAttention manages multiple independent Heads and mixes their outputs.
type MultiHeadAttention struct {
	NumHeads int
	DModel   int
	DHead    int
	Heads    []*Head
	Wo       *Tensor // Final output projection [dModel, dModel]
}

func NewMultiHeadAttention(numHeads, dModel int) *MultiHeadAttention {
	if dModel%numHeads != 0 {
		panic("dModel must be cleanly divisible by numHeads")
	}

	dHead := dModel / numHeads
	heads := make([]*Head, numHeads)
	for i := range numHeads {
		heads[i] = NewHead(dModel, dHead)
	}

	return &MultiHeadAttention{
		NumHeads: numHeads,
		DModel:   dModel,
		DHead:    dHead,
		Heads:    heads,
		Wo:       initWeights(dModel, dModel),
	}
}

// Forward runs all heads, concatenates results, and applies the Wo projection.
func (mha *MultiHeadAttention) Forward(input *Tensor) (*Tensor, error) {
	seqLen := input.Rows

	// 1. Collect outputs from all heads
	// Each head returns a tensor of shape [seqLen, dHead]
	headOutputs := make([]*Tensor, mha.NumHeads)
	for i, head := range mha.Heads {
		headOutputs[i] = head.Forward(input)
	}

	// 2. Concatenate the outputs
	// We want to stitch [seqLen, dHead] * NumHeads back into [seqLen, dModel]
	concatOut := NewTensor(seqLen, mha.DModel)
	for i := range seqLen {
		colOffset := 0
		for h := range mha.NumHeads {
			for j := range mha.DHead {
				concatOut.Set(i, colOffset+j, headOutputs[h].Get(i, j))
			}
			colOffset += mha.DHead
		}
	}

	// 3. Final Linear Projection (Mix the insights together)
	// [seqLen, dModel] * [dModel, dModel] -> [seqLen, dModel]
	finalOut := matMul(concatOut, mha.Wo)

	return finalOut, nil
}

// ForwardCached runs all heads with KV caching for autoregressive generation.
func (mha *MultiHeadAttention) ForwardCached(input *Tensor) (*Tensor, error) {
	newTokens := input.Rows

	headOutputs := make([]*Tensor, mha.NumHeads)
	for i, head := range mha.Heads {
		headOutputs[i] = head.ForwardCached(input)
	}

	// Concatenate head outputs: [newTokens, dHead] * NumHeads -> [newTokens, dModel]
	concatOut := NewTensor(newTokens, mha.DModel)
	for i := range newTokens {
		colOffset := 0
		for h := range mha.NumHeads {
			for j := range mha.DHead {
				concatOut.Set(i, colOffset+j, headOutputs[h].Get(i, j))
			}
			colOffset += mha.DHead
		}
	}

	finalOut := matMul(concatOut, mha.Wo)
	return finalOut, nil
}

// ResetCache clears the KV cache on all heads.
func (mha *MultiHeadAttention) ResetCache() {
	for _, head := range mha.Heads {
		head.ResetCache()
	}
}

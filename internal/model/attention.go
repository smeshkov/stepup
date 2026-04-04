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
	DHead int
	Wq    *Tensor // Projects from [DModel] to [DHead]
	Wk    *Tensor
	Wv    *Tensor
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

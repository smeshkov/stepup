package model

import (
	"math"
	"math/rand"
)

// initWeights populates a matrix with small random values
func initWeights(rows, cols int) [][]float32 {
	w := make([][]float32, rows)
	for i := range rows {
		w[i] = make([]float32, cols)
		for j := range cols {
			w[i][j] = rand.Float32()*0.02 - 0.01
		}
	}
	return w
}

// Head represents a single Attention head.
type Head struct {
	DHead int
	Wq    [][]float32 // Projects from [DModel] to [DHead]
	Wk    [][]float32
	Wv    [][]float32
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
func (h *Head) Forward(input [][]float32) [][]float32 {
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
	Wo       [][]float32 // Final output projection [dModel, dModel]
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
func (mha *MultiHeadAttention) Forward(input [][]float32) ([][]float32, error) {
	seqLen := len(input)

	// 1. Collect outputs from all heads
	// Each head returns a tensor of shape [seqLen, dHead]
	headOutputs := make([][][]float32, mha.NumHeads)
	for i, head := range mha.Heads {
		headOutputs[i] = head.Forward(input)
	}

	// 2. Concatenate the outputs
	// We want to stitch [seqLen, dHead] * NumHeads back into [seqLen, dModel]
	concatOut := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		concatOut[i] = make([]float32, mha.DModel)

		colOffset := 0
		for h := 0; h < mha.NumHeads; h++ {
			for j := 0; j < mha.DHead; j++ {
				concatOut[i][colOffset+j] = headOutputs[h][i][j]
			}
			colOffset += mha.DHead
		}
	}

	// 3. Final Linear Projection (Mix the insights together)
	// [seqLen, dModel] * [dModel, dModel] -> [seqLen, dModel]
	finalOut := matMul(concatOut, mha.Wo)

	return finalOut, nil
}

package model

import (
	"math"
	"math/rand"
)

// SelfAttention holds the learnable weight matrices that create Q, K, and V.
type SelfAttention struct {
	DModel int
	Wq     [][]float32
	Wk     [][]float32
	Wv     [][]float32
}

// initWeights populates a matrix with small random values (Xavier initialization simulation)
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

func NewSelfAttention(dModel int) *SelfAttention {
	return &SelfAttention{
		DModel: dModel,
		Wq:     initWeights(dModel, dModel),
		Wk:     initWeights(dModel, dModel),
		Wv:     initWeights(dModel, dModel),
	}
}

// Forward executes the Attention equation: softmax((Q * K^T) / sqrt(d_k)) * V
func (sa *SelfAttention) Forward(input [][]float32) ([][]float32, error) {
	// 1. Project input into Queries, Keys, and Values
	// input shape:  [seqLen, dModel]
	// weight shape: [dModel, dModel]
	// result shape: [seqLen, dModel]
	Q := matMul(input, sa.Wq)
	K := matMul(input, sa.Wk)
	V := matMul(input, sa.Wv)

	// 2. Calculate Attention Scores (Q * K^T)
	// Q shape:   [seqLen, dModel]
	// K^T shape: [dModel, seqLen]
	// scores shape: [seqLen, seqLen]  <-- This matrix represents how much every word attends to every other word!
	K_T := transpose(K)
	scores := matMul(Q, K_T)

	// 3. Scale and Softmax
	// d_k is the dimension of the key vectors (in single-head, this is just dModel)
	scaleFactor := float32(math.Sqrt(float64(sa.DModel)))
	scaleAndSoftmax(scores, scaleFactor)

	// 4. Multiply by Values (scores * V)
	// scores shape: [seqLen, seqLen]
	// V shape:      [seqLen, dModel]
	// out shape:    [seqLen, dModel]
	out := matMul(scores, V)

	return out, nil
}

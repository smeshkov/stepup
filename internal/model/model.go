package model

import (
	"errors"
	"math/rand"
)

// Embedding represents our lookup table mapping Token IDs to dense vectors.
type Embedding struct {
	VocabSize int
	DModel    int
	// Weights is our 2D matrix. In Go, we use float32 for ML to save memory,
	// as float64 doubles RAM usage with negligible accuracy gains in inference.
	Weights [][]float32
}

func NewEmbedding(vocabSize, dModel int) *Embedding {
	weights := make([][]float32, vocabSize)
	for i := range vocabSize {
		weights[i] = make([]float32, dModel)
		// In a real scenario, these are loaded from a trained model file (.safetensors).
		// For initialization before training, we populate with small random numbers.
		for j := range dModel {
			weights[i][j] = rand.Float32()*0.02 - 0.01
		}
	}
	return &Embedding{
		VocabSize: vocabSize,
		DModel:    dModel,
		Weights:   weights,
	}
}

// Forward performs the lookup. It takes a sequence of token IDs and returns
// a sequence of dense vectors (a 2D matrix).
func (e *Embedding) Forward(tokenIDs []int) ([][]float32, error) {
	seqLen := len(tokenIDs)
	out := make([][]float32, seqLen)

	for i, id := range tokenIDs {
		if id < 0 || id >= e.VocabSize {
			return nil, errors.New("token ID out of vocabulary bounds")
		}
		// Zero-copy reference to the row in our lookup table
		out[i] = e.Weights[id]
	}
	return out, nil
}

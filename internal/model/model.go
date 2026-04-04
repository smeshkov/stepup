package model

import (
	"errors"
	"math/rand"
)

// Embedding represents our lookup table mapping Token IDs to dense vectors.
// To learn more about embeddings read https://huggingface.co/spaces/hesamation/primer-llm-embedding
type Embedding struct {
	VocabSize int
	DModel    int
	// Weights is our matrix backed by a single contiguous allocation.
	Weights *Tensor
}

func NewEmbedding(vocabSize, dModel int) *Embedding {
	weights := NewTensor(vocabSize, dModel)
	// In a real scenario, these are loaded from a trained model file (.safetensors).
	// For initialization before training, we populate with small random numbers.
	for i := range weights.Data {
		weights.Data[i] = rand.Float32()*0.02 - 0.01
	}
	return &Embedding{
		VocabSize: vocabSize,
		DModel:    dModel,
		Weights:   weights,
	}
}

// Forward performs the lookup. It takes a sequence of token IDs and returns
// a sequence of dense vectors as a Tensor.
func (e *Embedding) Forward(tokenIDs []int) (*Tensor, error) {
	seqLen := len(tokenIDs)
	out := NewTensor(seqLen, e.DModel)

	for i, id := range tokenIDs {
		if id < 0 || id >= e.VocabSize {
			return nil, errors.New("token ID out of vocabulary bounds")
		}
		// Copy the weight row into the output tensor
		copy(out.Row(i), e.Weights.Row(id))
	}
	return out, nil
}

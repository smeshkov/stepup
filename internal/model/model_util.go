package model

import (
	"math"
)

// PrecomputePositionalEncoding generates the static sine/cosine matrix.
func PrecomputePositionalEncoding(maxSeqLen, dModel int) *Tensor {
	pe := NewTensor(maxSeqLen, dModel)

	for pos := range maxSeqLen {
		for i := 0; i < dModel; i += 2 {
			// Calculate the frequency denominator
			denominator := math.Pow(10000.0, float64(i)/float64(dModel))
			theta := float64(pos) / denominator

			// Even dimensions get Sine
			pe.Set(pos, i, float32(math.Sin(theta)))

			// Odd dimensions get Cosine
			if i+1 < dModel {
				pe.Set(pos, i+1, float32(math.Cos(theta)))
			}
		}
	}

	return pe
}

// PrepareInput takes raw token IDs and produces the final embedded sequence.
func PrepareInput(tokenIDs []int, embLayer *Embedding, peMatrix *Tensor) (*Tensor, error) {
	seqLen := len(tokenIDs)

	// 1. Get the Token Embeddings
	tokenEmbeddings, err := embLayer.Forward(tokenIDs)
	if err != nil {
		return nil, err
	}

	dModel := embLayer.DModel
	finalInput := NewTensor(seqLen, dModel)

	// 2. Add the Positional Encoding to the Token Embeddings
	for pos := range seqLen {
		for i := range dModel {
			// The critical math: Meaning (Token) + Order (Position)
			finalInput.Set(pos, i, tokenEmbeddings.Get(pos, i)+peMatrix.Get(pos, i))
		}
	}

	return finalInput, nil
}

package model

import (
	"math"
)

// PrecomputePositionalEncoding generates the static sine/cosine matrix.
func PrecomputePositionalEncoding(maxSeqLen, dModel int) [][]float32 {
	pe := make([][]float32, maxSeqLen)

	for pos := range maxSeqLen {
		pe[pos] = make([]float32, dModel)

		for i := 0; i < dModel; i += 2 {
			// Calculate the frequency denominator
			denominator := math.Pow(10000.0, float64(i)/float64(dModel))
			theta := float64(pos) / denominator

			// Even dimensions get Sine
			pe[pos][i] = float32(math.Sin(theta))

			// Odd dimensions get Cosine
			if i+1 < dModel {
				pe[pos][i+1] = float32(math.Cos(theta))
			}
		}
	}

	return pe
}

// PrepareInput takes raw token IDs and produces the final embedded sequence.
func PrepareInput(tokenIDs []int, embLayer *Embedding, peMatrix [][]float32) ([][]float32, error) {
	seqLen := len(tokenIDs)

	// 1. Get the Token Embeddings
	tokenEmbeddings, err := embLayer.Forward(tokenIDs)
	if err != nil {
		return nil, err
	}

	dModel := embLayer.DModel
	finalInput := make([][]float32, seqLen)

	// 2. Add the Positional Encoding to the Token Embeddings
	for pos := 0; pos < seqLen; pos++ {
		finalInput[pos] = make([]float32, dModel)

		for i := range dModel {
			// The critical math: Meaning (Token) + Order (Position)
			finalInput[pos][i] = tokenEmbeddings[pos][i] + peMatrix[pos][i]
		}
	}

	return finalInput, nil
}

package model

import (
	"math"
	"testing"
)

func TestSelfAttention_Forward(t *testing.T) {
	dModel := 16
	seqLen := 5

	attn := NewSelfAttention(dModel)

	// Create a dummy input tensor: 5 tokens, 16 dimensions
	input := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		input[i] = make([]float32, dModel)
		for j := 0; j < dModel; j++ {
			// Populate with arbitrary small numbers
			input[i][j] = float32(i) * 0.1
		}
	}

	output, err := attn.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// 1. Dimensionality Check
	// The output of Self-Attention MUST perfectly match the input dimensions
	if len(output) != seqLen {
		t.Fatalf("Expected output sequence length %d, got %d", seqLen, len(output))
	}
	if len(output[0]) != dModel {
		t.Fatalf("Expected output model dimension %d, got %d", dModel, len(output[0]))
	}

	// 2. Numerical Stability Check
	// Ensure the matrix multiplications and Softmax didn't explode into NaNs
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			if math.IsNaN(float64(output[i][j])) {
				t.Fatalf("Detected NaN in attention output at [%d][%d]", i, j)
			}
			if math.IsInf(float64(output[i][j]), 0) {
				t.Fatalf("Detected Infinity in attention output at [%d][%d]", i, j)
			}
		}
	}
}

func TestInitWeights(t *testing.T) {
	rows, cols := 10, 10
	weights := initWeights(rows, cols)

	if len(weights) != rows || len(weights[0]) != cols {
		t.Fatalf("initWeights dimension mismatch")
	}

	// Ensure weights are not purely zeros (they should be slightly randomized)
	allZero := true
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if weights[i][j] != 0.0 {
				allZero = false
				break
			}
		}
	}

	if allZero {
		t.Fatal("Weights initialized strictly to zeros, expected small random distribution")
	}
}

package model

import (
	"math"
	"testing"
)

func TestSelfAttention_Forward(t *testing.T) {
	dModel := 16
	seqLen := 5

	attn := NewMultiHeadAttention(4, dModel)

	// Create a dummy input tensor: 5 tokens, 16 dimensions
	input := NewTensor(seqLen, dModel)
	for i := range seqLen {
		for j := range dModel {
			// Populate with arbitrary small numbers
			input.Set(i, j, float32(i)*0.1)
		}
	}

	output, err := attn.Forward(input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// 1. Dimensionality Check
	// The output of Self-Attention MUST perfectly match the input dimensions
	if output.Rows != seqLen {
		t.Fatalf("Expected output sequence length %d, got %d", seqLen, output.Rows)
	}
	if output.Cols != dModel {
		t.Fatalf("Expected output model dimension %d, got %d", dModel, output.Cols)
	}

	// 2. Numerical Stability Check
	// Ensure the matrix multiplications and Softmax didn't explode into NaNs
	for i := range seqLen {
		for j := range dModel {
			val := float64(output.Get(i, j))
			if math.IsNaN(val) {
				t.Fatalf("Detected NaN in attention output at [%d][%d]", i, j)
			}
			if math.IsInf(val, 0) {
				t.Fatalf("Detected Infinity in attention output at [%d][%d]", i, j)
			}
		}
	}
}

func TestInitWeights(t *testing.T) {
	rows, cols := 10, 10
	weights := initWeights(rows, cols)

	if weights.Rows != rows || weights.Cols != cols {
		t.Fatalf("initWeights dimension mismatch")
	}

	// Ensure weights are not purely zeros (they should be slightly randomized)
	allZero := true
	for _, v := range weights.Data {
		if v != 0.0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Fatal("Weights initialized strictly to zeros, expected small random distribution")
	}
}

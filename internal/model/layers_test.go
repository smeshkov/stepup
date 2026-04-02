package model

import (
	"math"
	"testing"
)

func TestLayerNorm_Forward(t *testing.T) {
	dModel := 4
	ln := NewLayerNorm(dModel)

	// Create an input tensor with extreme, unbalanced numbers
	input := [][]float32{{100.0, 200.0, -50.0, 10.0}}

	output := ln.Forward(input)

	// Verify the output mathematically
	for i := range output {
		var sum float32 = 0
		for _, val := range output[i] {
			sum += val
		}
		mean := sum / float32(dModel)

		// The mean of the normalized output MUST be extremely close to 0
		if !almostEqual(mean, 0.0, 1e-4) {
			t.Errorf("LayerNorm failed: expected mean 0.0, got %f", mean)
		}

		var varianceSum float32 = 0
		for _, val := range output[i] {
			diff := val - mean
			varianceSum += diff * diff
		}
		variance := varianceSum / float32(dModel)

		// The variance of the normalized output MUST be extremely close to 1.0
		if !almostEqual(variance, 1.0, 1e-4) {
			t.Errorf("LayerNorm failed: expected variance 1.0, got %f", variance)
		}
	}
}

func TestFeedForward_Forward(t *testing.T) {
	dModel := 16
	seqLen := 5
	ff := NewFeedForward(dModel)

	// Create a dummy input tensor
	input := make([][]float32, seqLen)
	for i := range seqLen {
		input[i] = make([]float32, dModel)
		for j := range dModel {
			input[i][j] = 0.5 // Arbitrary starting value
		}
	}

	output := ff.Forward(input)

	// 1. Dimensionality Check: Ensure it successfully expanded to dFF (64) and contracted back to 16
	if len(output) != seqLen || len(output[0]) != dModel {
		t.Fatalf("FeedForward dimensions wrong. Expected %dx%d, got %dx%d", seqLen, dModel, len(output), len(output[0]))
	}

	// 2. Numerical Stability
	for i := range output {
		for j := range output[i] {
			if math.IsNaN(float64(output[i][j])) {
				t.Fatalf("Detected NaN in FeedForward output at [%d][%d]", i, j)
			}
		}
	}
}

func TestTransformerBlock_Forward(t *testing.T) {
	dModel := 16
	numHeads := 4
	seqLen := 3

	block := NewTransformerBlock(numHeads, dModel)

	// Create dummy input
	input := make([][]float32, seqLen)
	for i := range seqLen {
		input[i] = make([]float32, dModel)
	}

	output, err := block.Forward(input)
	if err != nil {
		t.Fatalf("TransformerBlock failed: %v", err)
	}

	// 1. Ensure the block preserves the sequence length and model dimension exactly
	if len(output) != seqLen || len(output[0]) != dModel {
		t.Fatalf("TransformerBlock dimensions wrong. Expected %dx%d, got %dx%d", seqLen, dModel, len(output), len(output[0]))
	}

	// 2. Ensure no NaNs from deep block calculation
	for i := range output {
		for j := range output[i] {
			if math.IsNaN(float64(output[i][j])) {
				t.Fatalf("Detected NaN in TransformerBlock output at [%d][%d]", i, j)
			}
		}
	}
}

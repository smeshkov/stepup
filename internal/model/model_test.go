package model

import (
	"math"
	"testing"
)

// almostEqual handles floating-point comparisons
func almostEqual(a, b, epsilon float32) bool {
	return float32(math.Abs(float64(a-b))) <= epsilon
}

func TestEmbedding_Forward(t *testing.T) {
	vocabSize := 10
	dModel := 4
	emb := NewEmbedding(vocabSize, dModel)

	// Manually overwrite weights with predictable numbers for testing
	// e.g., Token ID 2 will have weights [20, 21, 22, 23]
	for i := range vocabSize {
		for j := range dModel {
			emb.Weights[i][j] = float32(i*10 + j)
		}
	}

	// Test a valid sequence
	inputIDs := []int{2, 5}
	output, err := emb.Forward(inputIDs)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(output) != 2 || len(output[0]) != dModel {
		t.Fatalf("Incorrect output dimensions. Got %dx%d, expected 2x4", len(output), len(output[0]))
	}

	// Verify Token ID 2
	expectedToken2 := []float32{20, 21, 22, 23}
	for j := range dModel {
		if !almostEqual(output[0][j], expectedToken2[j], 1e-5) {
			t.Errorf("Embedding mismatch for token 2 at dim %d: got %f, want %f", j, output[0][j], expectedToken2[j])
		}
	}

	// Test Out of Bounds protection
	_, err = emb.Forward([]int{99})
	if err == nil {
		t.Fatal("Expected an error for out-of-bounds token ID, got nil")
	}
}

func TestPositionalEncoding(t *testing.T) {
	maxSeqLen := 5
	dModel := 4

	pe := PrecomputePositionalEncoding(maxSeqLen, dModel)

	// Check dimensions
	if len(pe) != maxSeqLen || len(pe[0]) != dModel {
		t.Fatalf("Incorrect PE dimensions. Got %dx%d, expected 5x4", len(pe), len(pe[0]))
	}

	// Mathematically verify Position 0
	// PE(0, 0) = sin(0) = 0
	// PE(0, 1) = cos(0) = 1
	if !almostEqual(pe[0][0], 0.0, 1e-5) {
		t.Errorf("PE(0,0) should be 0, got %f", pe[0][0])
	}
	if !almostEqual(pe[0][1], 1.0, 1e-5) {
		t.Errorf("PE(0,1) should be 1, got %f", pe[0][1])
	}

	// Mathematically verify Position 1, Dimension 0
	// PE(1, 0) = sin(1 / 10000^(0/4)) = sin(1) ≈ 0.84147
	expectedSin1 := float32(math.Sin(1.0))
	if !almostEqual(pe[1][0], expectedSin1, 1e-5) {
		t.Errorf("PE(1,0) should be ~0.84147, got %f", pe[1][0])
	}
}

func TestPrepareInput(t *testing.T) {
	vocabSize := 10
	dModel := 4
	maxSeqLen := 5

	emb := NewEmbedding(vocabSize, dModel)
	peMatrix := PrecomputePositionalEncoding(maxSeqLen, dModel)

	// Mock embedding weights for ID 1 to be all 1.0s
	for j := range dModel {
		emb.Weights[1][j] = 1.0
	}

	// Prepare input for a single token sequence: [1]
	finalInput, err := PrepareInput([]int{1}, emb, peMatrix)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// At Position 0, the PE adds [sin(0), cos(0), sin(0), cos(0)] -> [0, 1, 0, 1]
	// Token 1 embedding is [1, 1, 1, 1]
	// Expected result: [1+0, 1+1, 1+0, 1+1] -> [1, 2, 1, 2]
	expected := []float32{1.0, 2.0, 1.0, 2.0}

	for j := range dModel {
		if !almostEqual(finalInput[0][j], expected[j], 1e-5) {
			t.Errorf("PrepareInput mismatch at dim %d: got %f, want %f", j, finalInput[0][j], expected[j])
		}
	}
}

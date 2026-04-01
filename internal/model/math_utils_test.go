package model

import (
	"testing"
)

func TestMatMul(t *testing.T) {
	// A is 2x3
	a := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
	}
	// B is 3x2
	b := [][]float32{
		{7, 8},
		{9, 10},
		{11, 12},
	}

	expected := [][]float32{
		{58, 64},
		{139, 154},
	}

	result := matMul(a, b)

	if len(result) != 2 || len(result[0]) != 2 {
		t.Fatalf("Expected 2x2 matrix, got %dx%d", len(result), len(result[0]))
	}

	for i := 0; i < len(expected); i++ {
		for j := 0; j < len(expected[i]); j++ {
			if !almostEqual(result[i][j], expected[i][j], 1e-5) {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, expected[i][j], result[i][j])
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	// 2x3 matrix
	a := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
	}

	expected := [][]float32{
		{1, 4},
		{2, 5},
		{3, 6},
	}

	result := transpose(a)

	if len(result) != 3 || len(result[0]) != 2 {
		t.Fatalf("Expected 3x2 matrix, got %dx%d", len(result), len(result[0]))
	}

	for i := 0; i < len(expected); i++ {
		for j := 0; j < len(expected[i]); j++ {
			if !almostEqual(result[i][j], expected[i][j], 1e-5) {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, expected[i][j], result[i][j])
			}
		}
	}
}

func TestScaleAndSoftmax(t *testing.T) {
	// Test matrix with a single row
	matrix := [][]float32{{2.0, 4.0}}
	scale := float32(2.0)

	// Step 1: Scale -> [1.0, 2.0]
	// Step 2: Softmax -> exp(1-2), exp(2-2) -> exp(-1), exp(0) -> ~0.3678, 1.0
	// Sum = ~1.3678
	// Probabilities = [0.3678/1.3678, 1.0/1.3678] -> [0.26894, 0.73105]
	expected := []float32{0.26894, 0.73105}

	scaleAndSoftmax(matrix, scale)

	var sum float32 = 0
	for j, val := range matrix[0] {
		if !almostEqual(val, expected[j], 1e-4) {
			t.Errorf("Softmax mismatch at index %d: expected %f, got %f", j, expected[j], val)
		}
		sum += val
	}

	// Softmax probabilities must always sum to exactly 1.0
	if !almostEqual(sum, 1.0, 1e-5) {
		t.Errorf("Softmax probabilities did not sum to 1.0, got %f", sum)
	}
}

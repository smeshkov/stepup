package model

import (
	"testing"
)

func TestMatMul(t *testing.T) {
	// A is 2x3
	a := NewTensor(2, 3)
	a.Data = []float32{1, 2, 3, 4, 5, 6}

	// B is 3x2
	b := NewTensor(3, 2)
	b.Data = []float32{7, 8, 9, 10, 11, 12}

	expected := []float32{58, 64, 139, 154}

	result := matMul(a, b)

	if result.Rows != 2 || result.Cols != 2 {
		t.Fatalf("Expected 2x2 matrix, got %dx%d", result.Rows, result.Cols)
	}

	for i := range 2 {
		for j := range 2 {
			if !almostEqual(result.Get(i, j), expected[i*2+j], 1e-5) {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, expected[i*2+j], result.Get(i, j))
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	// 2x3 matrix
	a := NewTensor(2, 3)
	a.Data = []float32{1, 2, 3, 4, 5, 6}

	expected := []float32{1, 4, 2, 5, 3, 6} // 3x2

	result := transpose(a)

	if result.Rows != 3 || result.Cols != 2 {
		t.Fatalf("Expected 3x2 matrix, got %dx%d", result.Rows, result.Cols)
	}

	for i := range 3 {
		for j := range 2 {
			if !almostEqual(result.Get(i, j), expected[i*2+j], 1e-5) {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, expected[i*2+j], result.Get(i, j))
			}
		}
	}
}

func TestScaleAndSoftmax(t *testing.T) {
	// Test tensor with a single row
	matrix := NewTensor(1, 2)
	matrix.Data = []float32{2.0, 4.0}
	scale := float32(2.0)

	// Step 1: Scale -> [1.0, 2.0]
	// Step 2: Softmax -> exp(1-2), exp(2-2) -> exp(-1), exp(0) -> ~0.3678, 1.0
	// Sum = ~1.3678
	// Probabilities = [0.3678/1.3678, 1.0/1.3678] -> [0.26894, 0.73105]
	expected := []float32{0.26894, 0.73105}

	scaleAndSoftmax(matrix, scale)

	var sum float32
	row := matrix.Row(0)
	for j, val := range row {
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

func TestAddTensors(t *testing.T) {
	a := NewTensor(2, 2)
	a.Data = []float32{1, 2, 3, 4}
	b := NewTensor(2, 2)
	b.Data = []float32{5, 6, 7, 8}
	expected := []float32{6, 8, 10, 12}

	result := addTensors(a, b)

	for i := range 2 {
		for j := range 2 {
			if !almostEqual(result.Get(i, j), expected[i*2+j], 1e-5) {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, expected[i*2+j], result.Get(i, j))
			}
		}
	}
}

func TestAddBias(t *testing.T) {
	matrix := NewTensor(2, 2)
	matrix.Data = []float32{1, 2, 3, 4}
	bias := []float32{10, 20}
	expected := []float32{11, 22, 13, 24}

	// addBias mutates the tensor in place
	addBias(matrix, bias)

	for i := range 2 {
		for j := range 2 {
			if !almostEqual(matrix.Get(i, j), expected[i*2+j], 1e-5) {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, expected[i*2+j], matrix.Get(i, j))
			}
		}
	}
}

func TestRelu(t *testing.T) {
	matrix := NewTensor(2, 2)
	matrix.Data = []float32{-1.5, 0.0, 2.5, -99.9}
	expected := []float32{0.0, 0.0, 2.5, 0.0}

	// relu mutates the tensor in place
	relu(matrix)

	for i := range 2 {
		for j := range 2 {
			if !almostEqual(matrix.Get(i, j), expected[i*2+j], 1e-5) {
				t.Errorf("Mismatch at [%d][%d]: expected %f, got %f", i, j, expected[i*2+j], matrix.Get(i, j))
			}
		}
	}
}

package model

import (
	"testing"
)

func TestNewTensor(t *testing.T) {
	tensor := NewTensor(3, 4)

	if tensor.Rows != 3 {
		t.Errorf("Expected Rows 3, got %d", tensor.Rows)
	}
	if tensor.Cols != 4 {
		t.Errorf("Expected Cols 4, got %d", tensor.Cols)
	}
	if len(tensor.Data) != 12 {
		t.Errorf("Expected Data length 12, got %d", len(tensor.Data))
	}

	// All values should be zero-initialized
	for i, v := range tensor.Data {
		if v != 0 {
			t.Errorf("Expected zero at index %d, got %f", i, v)
		}
	}
}

func TestTensor_GetSet(t *testing.T) {
	tensor := NewTensor(2, 3)

	tensor.Set(0, 0, 1.0)
	tensor.Set(0, 2, 3.0)
	tensor.Set(1, 1, 5.0)

	if tensor.Get(0, 0) != 1.0 {
		t.Errorf("Expected 1.0 at (0,0), got %f", tensor.Get(0, 0))
	}
	if tensor.Get(0, 2) != 3.0 {
		t.Errorf("Expected 3.0 at (0,2), got %f", tensor.Get(0, 2))
	}
	if tensor.Get(1, 1) != 5.0 {
		t.Errorf("Expected 5.0 at (1,1), got %f", tensor.Get(1, 1))
	}
	// Untouched cell should still be zero
	if tensor.Get(1, 0) != 0.0 {
		t.Errorf("Expected 0.0 at (1,0), got %f", tensor.Get(1, 0))
	}
}

func TestTensor_Row(t *testing.T) {
	tensor := NewTensor(2, 3)
	tensor.Data = []float32{1, 2, 3, 4, 5, 6}

	row0 := tensor.Row(0)
	row1 := tensor.Row(1)

	expected0 := []float32{1, 2, 3}
	expected1 := []float32{4, 5, 6}

	for i, v := range row0 {
		if v != expected0[i] {
			t.Errorf("Row 0 mismatch at %d: expected %f, got %f", i, expected0[i], v)
		}
	}
	for i, v := range row1 {
		if v != expected1[i] {
			t.Errorf("Row 1 mismatch at %d: expected %f, got %f", i, expected1[i], v)
		}
	}

	// Verify zero-copy: mutating the row slice should affect the tensor
	row0[0] = 99.0
	if tensor.Get(0, 0) != 99.0 {
		t.Error("Row() should return a zero-copy slice; mutation did not propagate")
	}
}

func TestTensor_Zero(t *testing.T) {
	tensor := NewTensor(2, 3)
	tensor.Data = []float32{1, 2, 3, 4, 5, 6}

	tensor.Zero()

	for i, v := range tensor.Data {
		if v != 0 {
			t.Errorf("Expected 0 at index %d after Zero(), got %f", i, v)
		}
	}
}

func TestTensor_SingleAllocation(t *testing.T) {
	// Verify that a 1000x1000 tensor uses exactly 1 backing array
	// (not 1001 allocations like [][]float32 would)
	tensor := NewTensor(1000, 1000)

	if len(tensor.Data) != 1_000_000 {
		t.Errorf("Expected 1,000,000 elements, got %d", len(tensor.Data))
	}

	// Verify stride math works at boundaries
	tensor.Set(999, 999, 42.0)
	if tensor.Get(999, 999) != 42.0 {
		t.Error("Stride math failed at boundary (999, 999)")
	}

	tensor.Set(0, 0, 7.0)
	if tensor.Get(0, 0) != 7.0 {
		t.Error("Stride math failed at boundary (0, 0)")
	}
}

package model

import (
	"math"
)

// matMul multiplies two tensors: A (m x n) * B (n x p) = C (m x p)
func matMul(a, b *Tensor) *Tensor {
	m := a.Rows
	n := a.Cols
	p := b.Cols

	c := NewTensor(m, p)
	for i := range m {
		for j := range p {
			var sum float32
			for k := range n {
				sum += a.Get(i, k) * b.Get(k, j)
			}
			c.Set(i, j, sum)
		}
	}
	return c
}

// transpose flips a tensor over its diagonal: A (m x n) -> A^T (n x m)
func transpose(a *Tensor) *Tensor {
	c := NewTensor(a.Cols, a.Rows)
	for i := range a.Rows {
		for j := range a.Cols {
			c.Set(j, i, a.Get(i, j))
		}
	}
	return c
}

// scaleAndSoftmax applies the scaling factor and then converts rows into probability distributions
func scaleAndSoftmax(t *Tensor, scale float32) {
	for i := range t.Rows {
		row := t.Row(i)

		// 1. Scale
		for j := range row {
			row[j] /= scale
		}

		// 2. Softmax (with numerical stability fix)
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}

		var sum float32
		for j, v := range row {
			exp := float32(math.Exp(float64(v - maxVal)))
			row[j] = exp
			sum += exp
		}

		// Normalize to get probabilities that sum to 1.0
		for j := range row {
			row[j] /= sum
		}
	}
}

// addTensors performs element-wise addition: C = A + B
func addTensors(a, b *Tensor) *Tensor {
	c := NewTensor(a.Rows, a.Cols)
	for i := range c.Data {
		c.Data[i] = a.Data[i] + b.Data[i]
	}
	return c
}

// addBias applies a 1D bias vector to every row of a tensor
func addBias(t *Tensor, bias []float32) {
	for i := range t.Rows {
		row := t.Row(i)
		for j := range row {
			row[j] += bias[j]
		}
	}
}

// relu applies the Rectified Linear Unit activation: max(0, x)
func relu(t *Tensor) {
	for i := range t.Data {
		if t.Data[i] < 0 {
			t.Data[i] = 0
		}
	}
}

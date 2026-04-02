package model

import (
	"math"
)

// matMul multiplies two 2D matrices: A (m x n) * B (n x p) = C (m x p)
func matMul(a, b [][]float32) [][]float32 {
	m := len(a)
	n := len(a[0])
	p := len(b[0])

	c := make([][]float32, m)
	for i := range m {
		c[i] = make([]float32, p)
		for j := range p {
			var sum float32
			for k := range n {
				sum += a[i][k] * b[k][j]
			}
			c[i][j] = sum
		}
	}
	return c
}

// transpose flips a matrix over its diagonal: A (m x n) -> A^T (n x m)
func transpose(a [][]float32) [][]float32 {
	m := len(a)
	n := len(a[0])

	c := make([][]float32, n)
	for i := range n {
		c[i] = make([]float32, m)
		for j := range m {
			c[i][j] = a[j][i]
		}
	}
	return c
}

// scaleAndSoftmax applies the scaling factor and then converts rows into probability distributions
func scaleAndSoftmax(matrix [][]float32, scale float32) {
	for i := range matrix {
		// 1. Scale
		for j := range matrix[i] {
			matrix[i][j] /= scale
		}

		// 2. Softmax (with numerical stability fix)
		// We find the max value in the row and subtract it to prevent math.Exp() from overflowing
		maxVal := matrix[i][0]
		for _, v := range matrix[i] {
			if v > maxVal {
				maxVal = v
			}
		}

		var sum float32 = 0
		for j, v := range matrix[i] {
			exp := float32(math.Exp(float64(v - maxVal)))
			matrix[i][j] = exp
			sum += exp
		}

		// Normalize to get probabilities that sum to 1.0
		for j := range matrix[i] {
			matrix[i][j] /= sum
		}
	}
}

// addMatrices performs element-wise addition: C = A + B
func addMatrices(a, b [][]float32) [][]float32 {
	m := len(a)
	n := len(a[0])
	c := make([][]float32, m)
	for i := 0; i < m; i++ {
		c[i] = make([]float32, n)
		for j := 0; j < n; j++ {
			c[i][j] = a[i][j] + b[i][j]
		}
	}
	return c
}

// addBias applies a 1D bias vector to every row of a 2D matrix
func addBias(matrix [][]float32, bias []float32) {
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] += bias[j]
		}
	}
}

// relu applies the Rectified Linear Unit activation: max(0, x)
func relu(matrix [][]float32) {
	for i := range matrix {
		for j := range matrix[i] {
			if matrix[i][j] < 0 {
				matrix[i][j] = 0
			}
		}
	}
}

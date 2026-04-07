package model

// Tensor is a high-performance 2D matrix backed by a single contiguous 1D slice.
// This prevents the GC Death Spiral caused by thousands of small []float32 allocations
// during forward and backward passes. A Tensor of R rows and C columns requires
// exactly 1 heap allocation instead of R+1.
type Tensor struct {
	Rows int
	Cols int
	Data []float32 // Single contiguous block: index = row*Cols + col
}

// NewTensor allocates a zero-initialized Tensor with a single backing array.
func NewTensor(rows, cols int) *Tensor {
	return &Tensor{
		Rows: rows,
		Cols: cols,
		Data: make([]float32, rows*cols),
	}
}

// Get returns the value at (row, col) using stride math.
func (t *Tensor) Get(row, col int) float32 {
	return t.Data[row*t.Cols+col]
}

// Set writes a value at (row, col).
func (t *Tensor) Set(row, col int, val float32) {
	t.Data[row*t.Cols+col] = val
}

// Row returns a zero-copy slice view of a single row.
func (t *Tensor) Row(row int) []float32 {
	start := row * t.Cols
	return t.Data[start : start+t.Cols]
}

// Zero resets all values to 0, allowing the tensor to be reused without reallocation.
func (t *Tensor) Zero() {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// AppendRows appends all rows from other to the bottom of t, returning a new Tensor.
// The column count must match.
func (t *Tensor) AppendRows(other *Tensor) *Tensor {
	out := NewTensor(t.Rows+other.Rows, t.Cols)
	copy(out.Data, t.Data)
	copy(out.Data[len(t.Data):], other.Data)
	return out
}

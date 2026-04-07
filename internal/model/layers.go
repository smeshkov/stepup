package model

import "math"

// LayerNorm stabilizes the neural network mathematically
type LayerNorm struct {
	Gamma []float32 // Learnable scale
	Beta  []float32 // Learnable shift
	Eps   float32   // Small constant to prevent division by zero
}

func NewLayerNorm(dModel int) *LayerNorm {
	gamma := make([]float32, dModel)
	beta := make([]float32, dModel)
	for i := range dModel {
		gamma[i] = 1.0 // Default scale is 1
		beta[i] = 0.0  // Default shift is 0
	}
	return &LayerNorm{
		Gamma: gamma,
		Beta:  beta,
		Eps:   1e-5,
	}
}

func (ln *LayerNorm) Forward(input *Tensor) *Tensor {
	seqLen := input.Rows
	dModel := input.Cols
	out := NewTensor(seqLen, dModel)

	for i := range seqLen {
		row := input.Row(i)

		// 1. Calculate Mean
		var sum float32
		for j := range dModel {
			sum += row[j]
		}
		mean := sum / float32(dModel)

		// 2. Calculate Variance
		var varianceSum float32
		for j := range dModel {
			diff := row[j] - mean
			varianceSum += diff * diff
		}
		variance := varianceSum / float32(dModel)

		// 3. Normalize, Scale, and Shift
		stdDev := float32(math.Sqrt(float64(variance + ln.Eps)))
		for j := range dModel {
			normalized := (row[j] - mean) / stdDev
			out.Set(i, j, normalized*ln.Gamma[j]+ln.Beta[j])
		}
	}
	return out
}

// FeedForward processes each token's representation independently
type FeedForward struct {
	W1 *Tensor // Expands dimensionality (dModel -> dFF)
	B1 []float32
	W2 *Tensor // Projects back down (dFF -> dModel)
	B2 []float32
}

func NewFeedForward(dModel int) *FeedForward {
	dFF := dModel * 4 // Standard expansion factor
	return &FeedForward{
		W1: initWeights(dModel, dFF),
		B1: make([]float32, dFF),
		W2: initWeights(dFF, dModel),
		B2: make([]float32, dModel),
	}
}

func (ff *FeedForward) Forward(input *Tensor) *Tensor {
	// 1. Linear Expansion: [seqLen, dModel] * [dModel, dFF] -> [seqLen, dFF]
	hidden := matMul(input, ff.W1)
	addBias(hidden, ff.B1)

	// 2. Non-linearity
	relu(hidden)

	// 3. Linear Projection: [seqLen, dFF] * [dFF, dModel] -> [seqLen, dModel]
	out := matMul(hidden, ff.W2)
	addBias(out, ff.B2)

	return out
}

// TransformerBlock connects Attention and FFN with Normalization and Residuals
type TransformerBlock struct {
	AttnNorm *LayerNorm
	Attn     *MultiHeadAttention
	FFNNorm  *LayerNorm
	FFN      *FeedForward
}

func NewTransformerBlock(numHeads, dModel int) *TransformerBlock {
	return &TransformerBlock{
		AttnNorm: NewLayerNorm(dModel),
		Attn:     NewMultiHeadAttention(numHeads, dModel),
		FFNNorm:  NewLayerNorm(dModel),
		FFN:      NewFeedForward(dModel),
	}
}

func (tb *TransformerBlock) Forward(input *Tensor) (*Tensor, error) {
	// Block 1: Attention with Residual Connection
	// Formula: x = x + Attention(LayerNorm(x))
	normalized1 := tb.AttnNorm.Forward(input)
	attnOut, err := tb.Attn.Forward(normalized1)
	if err != nil {
		return nil, err
	}
	residual1 := addTensors(input, attnOut)

	// Block 2: FFN with Residual Connection
	// Formula: x = x + FFN(LayerNorm(x))
	normalized2 := tb.FFNNorm.Forward(residual1)
	ffnOut := tb.FFN.Forward(normalized2)
	finalOut := addTensors(residual1, ffnOut)

	return finalOut, nil
}

// ForwardCached runs the transformer block with KV caching for autoregressive generation.
// Only new token(s) are passed through; attention uses cached K/V from prior steps.
func (tb *TransformerBlock) ForwardCached(input *Tensor) (*Tensor, error) {
	// Block 1: Attention with Residual Connection
	normalized1 := tb.AttnNorm.Forward(input)
	attnOut, err := tb.Attn.ForwardCached(normalized1)
	if err != nil {
		return nil, err
	}
	residual1 := addTensors(input, attnOut)

	// Block 2: FFN with Residual Connection
	normalized2 := tb.FFNNorm.Forward(residual1)
	ffnOut := tb.FFN.Forward(normalized2)
	finalOut := addTensors(residual1, ffnOut)

	return finalOut, nil
}

// ResetCache clears the KV cache in the attention layer.
func (tb *TransformerBlock) ResetCache() {
	tb.Attn.ResetCache()
}

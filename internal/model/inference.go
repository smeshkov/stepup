package model

import (
	"sync"
)

// SamplingStrategy selects how tokens are sampled during inference.
type SamplingStrategy int

const (
	// StrategyGreedy always picks the token with the highest logit.
	StrategyGreedy SamplingStrategy = iota
	// StrategyTemperature samples from the temperature-scaled distribution.
	StrategyTemperature
	// StrategyTopP uses nucleus sampling with a cumulative probability threshold.
	StrategyTopP
)

// SamplingConfig controls token selection during inference.
type SamplingConfig struct {
	Strategy    SamplingStrategy
	Temperature float32 // Used by Temperature and TopP strategies
	TopP        float32 // Used by TopP strategy only
}

// InferRequest represents a single inference request in a batch.
type InferRequest struct {
	ID       int
	TokenIDs []int
	Sampling SamplingConfig
}

// InferResult is the output of processing an inference request.
type InferResult struct {
	ID      int
	TokenID int       // The sampled next-token ID
	Logits  []float32 // Raw logits from the last token position
	Error   error
}

// InferenceEngine holds shared model components and processes batched
// inference requests concurrently using a goroutine worker pool.
//
// The engine's Forward pass is read-only (no KV caching), so the model
// weights are safe to share across goroutines without synchronization.
type InferenceEngine struct {
	Embedding  *Embedding
	PosEnc     *Tensor
	Block      *TransformerBlock
	MaxWorkers int
}

// NewInferenceEngine creates an engine with shared model weights.
// maxWorkers controls the maximum number of concurrent goroutines.
func NewInferenceEngine(vocabSize, dModel, numHeads, maxSeqLen, maxWorkers int) *InferenceEngine {
	return &InferenceEngine{
		Embedding:  NewEmbedding(vocabSize, dModel),
		PosEnc:     PrecomputePositionalEncoding(maxSeqLen, dModel),
		Block:      NewTransformerBlock(numHeads, dModel),
		MaxWorkers: maxWorkers,
	}
}

// Infer processes a single request through the full pipeline:
// tokenIDs → embedding + position → transformer → sample.
func (e *InferenceEngine) Infer(req InferRequest) InferResult {
	// 1. Embedding + Positional Encoding
	input, err := PrepareInput(req.TokenIDs, e.Embedding, e.PosEnc)
	if err != nil {
		return InferResult{ID: req.ID, Error: err}
	}

	// 2. Transformer forward pass (read-only, safe for concurrent use)
	output, err := e.Block.Forward(input)
	if err != nil {
		return InferResult{ID: req.ID, Error: err}
	}

	// 3. Extract logits from the last token position
	lastRow := output.Row(output.Rows - 1)
	logits := make([]float32, len(lastRow))
	copy(logits, lastRow)

	// 4. Sample next token
	tokenID := sampleWith(logits, req.Sampling)

	return InferResult{
		ID:      req.ID,
		TokenID: tokenID,
		Logits:  logits,
	}
}

// RunBatch processes multiple inference requests concurrently.
// It fans out goroutines bounded by MaxWorkers (semaphore pattern)
// and collects results in order.
func (e *InferenceEngine) RunBatch(requests []InferRequest) []InferResult {
	results := make([]InferResult, len(requests))

	sem := make(chan struct{}, e.MaxWorkers)
	var wg sync.WaitGroup

	for i, req := range requests {
		wg.Add(1)
		sem <- struct{}{} // Acquire worker slot

		go func(idx int, r InferRequest) {
			defer wg.Done()
			defer func() { <-sem }() // Release worker slot

			results[idx] = e.Infer(r)
		}(i, req)
	}

	wg.Wait()

	return results
}

// sampleWith applies the configured sampling strategy to logits.
func sampleWith(logits []float32, cfg SamplingConfig) int {
	switch cfg.Strategy {
	case StrategyTemperature:
		return SampleTemperature(logits, cfg.Temperature)
	case StrategyTopP:
		return SampleTopP(logits, cfg.Temperature, cfg.TopP)
	default:
		return SampleGreedy(logits)
	}
}

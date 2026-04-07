package model

import (
	"sync"
	"testing"
)

func newTestEngine(maxWorkers int) *InferenceEngine {
	vocabSize := 266 // 256 bytes + 10 merges (matches example)
	dModel := 16
	numHeads := 4
	maxSeqLen := 128
	return NewInferenceEngine(vocabSize, dModel, numHeads, maxSeqLen, maxWorkers)
}

func TestInfer_Greedy(t *testing.T) {
	engine := newTestEngine(1)

	result := engine.Infer(InferRequest{
		ID:       1,
		TokenIDs: []int{10, 20, 30},
		Sampling: SamplingConfig{Strategy: StrategyGreedy},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	if result.ID != 1 {
		t.Fatalf("expected ID 1, got %d", result.ID)
	}
	if result.TokenID < 0 || result.TokenID >= 16 {
		t.Fatalf("token ID %d out of dModel range", result.TokenID)
	}
	if len(result.Logits) != 16 {
		t.Fatalf("expected 16 logits, got %d", len(result.Logits))
	}
}

func TestInfer_InvalidToken(t *testing.T) {
	engine := newTestEngine(1)

	result := engine.Infer(InferRequest{
		ID:       2,
		TokenIDs: []int{9999}, // Out of vocabulary
		Sampling: SamplingConfig{Strategy: StrategyGreedy},
	})

	if result.Error == nil {
		t.Fatal("expected error for out-of-bounds token ID")
	}
}

func TestInfer_Temperature(t *testing.T) {
	engine := newTestEngine(1)

	result := engine.Infer(InferRequest{
		ID:       3,
		TokenIDs: []int{10, 20},
		Sampling: SamplingConfig{Strategy: StrategyTemperature, Temperature: 1.0},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	if result.TokenID < 0 || result.TokenID >= 16 {
		t.Fatalf("token ID %d out of range", result.TokenID)
	}
}

func TestInfer_TopP(t *testing.T) {
	engine := newTestEngine(1)

	result := engine.Infer(InferRequest{
		ID:       4,
		TokenIDs: []int{10, 20},
		Sampling: SamplingConfig{Strategy: StrategyTopP, Temperature: 1.0, TopP: 0.9},
	})

	if result.Error != nil {
		t.Fatalf("unexpected error: %v", result.Error)
	}
	if result.TokenID < 0 || result.TokenID >= 16 {
		t.Fatalf("token ID %d out of range", result.TokenID)
	}
}

func TestRunBatch_ConcurrentResults(t *testing.T) {
	engine := newTestEngine(4)

	requests := make([]InferRequest, 20)
	for i := range requests {
		requests[i] = InferRequest{
			ID:       i,
			TokenIDs: []int{10, 20, 30},
			Sampling: SamplingConfig{Strategy: StrategyGreedy},
		}
	}

	results := engine.RunBatch(requests)

	if len(results) != 20 {
		t.Fatalf("expected 20 results, got %d", len(results))
	}

	// Greedy on same input should produce the same token every time
	firstToken := results[0].TokenID
	for i, r := range results {
		if r.Error != nil {
			t.Fatalf("request %d failed: %v", i, r.Error)
		}
		if r.ID != i {
			t.Fatalf("expected ID %d, got %d", i, r.ID)
		}
		if r.TokenID != firstToken {
			t.Fatalf("greedy results diverged: request 0 got %d, request %d got %d",
				firstToken, i, r.TokenID)
		}
	}
}

func TestRunBatch_OrderPreserved(t *testing.T) {
	engine := newTestEngine(2)

	requests := []InferRequest{
		{ID: 100, TokenIDs: []int{10}, Sampling: SamplingConfig{Strategy: StrategyGreedy}},
		{ID: 200, TokenIDs: []int{20}, Sampling: SamplingConfig{Strategy: StrategyGreedy}},
		{ID: 300, TokenIDs: []int{30}, Sampling: SamplingConfig{Strategy: StrategyGreedy}},
	}

	results := engine.RunBatch(requests)

	for i, r := range results {
		if r.ID != requests[i].ID {
			t.Fatalf("result %d: expected ID %d, got %d", i, requests[i].ID, r.ID)
		}
	}
}

func TestRunBatch_SingleWorker(t *testing.T) {
	engine := newTestEngine(1) // Serial execution

	requests := make([]InferRequest, 5)
	for i := range requests {
		requests[i] = InferRequest{
			ID:       i,
			TokenIDs: []int{10, 20},
			Sampling: SamplingConfig{Strategy: StrategyGreedy},
		}
	}

	results := engine.RunBatch(requests)

	for i, r := range results {
		if r.Error != nil {
			t.Fatalf("request %d failed: %v", i, r.Error)
		}
	}
}

func TestRunBatch_MixedStrategies(t *testing.T) {
	engine := newTestEngine(3)

	requests := []InferRequest{
		{ID: 0, TokenIDs: []int{10, 20}, Sampling: SamplingConfig{Strategy: StrategyGreedy}},
		{ID: 1, TokenIDs: []int{10, 20}, Sampling: SamplingConfig{Strategy: StrategyTemperature, Temperature: 1.0}},
		{ID: 2, TokenIDs: []int{10, 20}, Sampling: SamplingConfig{Strategy: StrategyTopP, Temperature: 1.0, TopP: 0.9}},
	}

	results := engine.RunBatch(requests)

	for i, r := range results {
		if r.Error != nil {
			t.Fatalf("request %d (strategy %d) failed: %v", i, requests[i].Sampling.Strategy, r.Error)
		}
		if r.TokenID < 0 || r.TokenID >= 16 {
			t.Fatalf("request %d: token ID %d out of range", i, r.TokenID)
		}
	}
}

func TestRunBatch_Empty(t *testing.T) {
	engine := newTestEngine(4)

	results := engine.RunBatch(nil)

	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}

func TestRunBatch_ConcurrencySafety(t *testing.T) {
	engine := newTestEngine(8)

	// Run the batch multiple times concurrently to stress-test thread safety
	var wg sync.WaitGroup
	for range 5 {
		wg.Add(1)
		go func() {
			defer wg.Done()

			requests := make([]InferRequest, 10)
			for i := range requests {
				requests[i] = InferRequest{
					ID:       i,
					TokenIDs: []int{10, 20, 30},
					Sampling: SamplingConfig{Strategy: StrategyGreedy},
				}
			}

			results := engine.RunBatch(requests)
			for _, r := range results {
				if r.Error != nil {
					t.Errorf("concurrent batch failed: %v", r.Error)
				}
			}
		}()
	}
	wg.Wait()
}

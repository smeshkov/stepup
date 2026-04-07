package model

import (
	"math"
	"testing"
)

func TestSampleGreedy(t *testing.T) {
	logits := []float32{1.0, 3.0, 2.0, 0.5}
	got := SampleGreedy(logits)
	if got != 1 {
		t.Errorf("SampleGreedy: expected index 1, got %d", got)
	}
}

func TestSampleGreedy_NegativeLogits(t *testing.T) {
	logits := []float32{-5.0, -1.0, -3.0}
	got := SampleGreedy(logits)
	if got != 1 {
		t.Errorf("SampleGreedy: expected index 1, got %d", got)
	}
}

func TestSampleTemperature_LowTemp(t *testing.T) {
	// Very low temperature should behave like greedy (nearly deterministic)
	logits := []float32{1.0, 5.0, 2.0}
	counts := make(map[int]int)
	for range 1000 {
		idx := SampleTemperature(logits, 0.01)
		counts[idx]++
	}
	// With temperature 0.01, index 1 (logit=5) should dominate
	if counts[1] < 990 {
		t.Errorf("SampleTemperature(0.01): expected index 1 to dominate, got counts=%v", counts)
	}
}

func TestSampleTemperature_HighTemp(t *testing.T) {
	// High temperature should spread probability more evenly
	logits := []float32{1.0, 2.0, 3.0}
	counts := make(map[int]int)
	n := 10000
	for range n {
		idx := SampleTemperature(logits, 10.0)
		counts[idx]++
	}
	// All indices should get meaningful share (at least 20% each)
	for i := range 3 {
		ratio := float64(counts[i]) / float64(n)
		if ratio < 0.2 {
			t.Errorf("SampleTemperature(10.0): index %d got only %.1f%%, expected >20%%", i, ratio*100)
		}
	}
}

func TestSampleTopP_SmallP(t *testing.T) {
	// With p=0.1, only the top token should be in the nucleus
	logits := []float32{0.0, 10.0, 0.0, 0.0}
	counts := make(map[int]int)
	for range 1000 {
		idx := SampleTopP(logits, 1.0, 0.1)
		counts[idx]++
	}
	if counts[1] < 990 {
		t.Errorf("SampleTopP(p=0.1): expected index 1 to dominate, got counts=%v", counts)
	}
}

func TestSampleTopP_LargeP(t *testing.T) {
	// With p=0.99, nearly all tokens should be in the nucleus
	logits := []float32{1.0, 1.0, 1.0, 1.0}
	counts := make(map[int]int)
	n := 10000
	for range n {
		idx := SampleTopP(logits, 1.0, 0.99)
		counts[idx]++
	}
	// Uniform logits → each should get ~25%
	for i := range 4 {
		ratio := float64(counts[i]) / float64(n)
		if ratio < 0.15 || ratio > 0.35 {
			t.Errorf("SampleTopP(p=0.99): index %d got %.1f%%, expected ~25%%", i, ratio*100)
		}
	}
}

func TestSoftmax_SumsToOne(t *testing.T) {
	logits := []float32{2.0, 1.0, 0.1, -1.0, 3.0}
	probs := softmax(logits, 1.0)
	var sum float32
	for _, p := range probs {
		sum += p
		if p < 0 {
			t.Errorf("softmax: got negative probability %f", p)
		}
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax: probabilities sum to %f, expected 1.0", sum)
	}
}

func TestSoftmax_TemperatureScaling(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}

	cold := softmax(logits, 0.1) // sharp distribution
	warm := softmax(logits, 10.0) // flat distribution

	// Cold: highest logit (idx 2) should have much higher probability
	if cold[2] < 0.95 {
		t.Errorf("softmax(T=0.1): expected idx 2 prob > 0.95, got %f", cold[2])
	}

	// Warm: distribution should be nearly uniform
	for i, p := range warm {
		if p < 0.2 || p > 0.45 {
			t.Errorf("softmax(T=10): idx %d prob=%f, expected near uniform", i, p)
		}
	}
}

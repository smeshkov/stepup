package model

import (
	"math"
	"math/rand/v2"
	"sort"
)

// SampleGreedy returns the index of the highest logit (argmax).
// This is the most deterministic strategy — always picks the most likely token.
func SampleGreedy(logits []float32) int {
	bestIdx := 0
	bestVal := logits[0]
	for i := range len(logits) {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	return bestIdx
}

// SampleTemperature divides logits by temperature, applies softmax, and samples
// from the resulting distribution. Lower temperature → more deterministic,
// higher temperature → more creative. Temperature must be > 0.
func SampleTemperature(logits []float32, temperature float32) int {
	probs := softmax(logits, temperature)
	return sampleFromProbs(probs)
}

// SampleTopP (nucleus sampling) sorts tokens by probability, keeps only the
// smallest set whose cumulative probability exceeds p, then samples from that
// nucleus. This balances diversity with quality — unlikely tokens are excluded.
func SampleTopP(logits []float32, temperature float32, p float32) int {
	probs := softmax(logits, temperature)

	// Build index-probability pairs and sort descending by probability
	type indexedProb struct {
		index int
		prob  float32
	}
	indexed := make([]indexedProb, len(probs))
	for i, prob := range probs {
		indexed[i] = indexedProb{index: i, prob: prob}
	}
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Accumulate until we exceed the threshold p
	var cumulative float32
	cutoff := 0
	for i, ip := range indexed {
		cumulative += ip.prob
		cutoff = i + 1
		if cumulative >= p {
			break
		}
	}

	// Renormalize the nucleus probabilities
	nucleus := indexed[:cutoff]
	var sum float32
	for _, ip := range nucleus {
		sum += ip.prob
	}

	// Sample from the nucleus
	r := rand.Float32() * sum
	var acc float32
	for _, ip := range nucleus {
		acc += ip.prob
		if r <= acc {
			return ip.index
		}
	}
	return nucleus[len(nucleus)-1].index
}

// softmax applies temperature scaling and converts logits to probabilities.
func softmax(logits []float32, temperature float32) []float32 {
	scaled := make([]float32, len(logits))

	// Find max for numerical stability
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// Exp with temperature scaling and stability shift
	var sum float32
	for i, v := range logits {
		exp := float32(math.Exp(float64((v - maxVal) / temperature)))
		scaled[i] = exp
		sum += exp
	}

	// Normalize
	for i := range scaled {
		scaled[i] /= sum
	}
	return scaled
}

// sampleFromProbs draws a random index from a probability distribution.
func sampleFromProbs(probs []float32) int {
	r := rand.Float32()
	var acc float32
	for i, p := range probs {
		acc += p
		if r <= acc {
			return i
		}
	}
	return len(probs) - 1
}

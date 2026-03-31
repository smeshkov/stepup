package tokenizer

// Pair represents a bigram of token IDs.
// We use a fixed-size array [2]int so it can be used as a map key in Go.
type Pair [2]int

// Tokenizer holds the state of our learned BPE rules.
type Tokenizer struct {
	// Merges maps a pair of tokens to their new combined Token ID
	Merges map[Pair]int
	// Vocab maps the integer ID back to its raw bytes for decoding
	Vocab map[int][]byte
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Merges: make(map[Pair]int),
		Vocab:  make(map[int][]byte),
	}
}

// TrainProduction takes raw text, enforces regex boundaries, and trains the BPE rules.
func (t *Tokenizer) TrainProduction(text string, targetVocabSize int) error {
	// 1. Regex Split
	stringChunks, err := chunkText(text)
	if err != nil {
		return err
	}

	// 2. Convert string chunks into byte ID chunks
	chunks := make([][]int, len(stringChunks))
	for i, chunkStr := range stringChunks {
		rawBytes := []byte(chunkStr)
		chunkIDs := make([]int, len(rawBytes))
		for j, b := range rawBytes {
			chunkIDs[j] = int(b)
		}
		chunks[i] = chunkIDs
	}

	numMerges := targetVocabSize - 256
	nextID := 256

	// 3. The Chunked Training Loop
	for range numMerges {
		stats := getChunkedStats(chunks)

		var bestPair Pair
		maxCount := -1
		for pair, count := range stats {
			if count > maxCount {
				maxCount = count
				bestPair = pair
			}
		}

		if maxCount < 2 {
			break
		}

		// Apply the merge independently to each isolated chunk
		for j, chunk := range chunks {
			chunks[j] = merge(chunk, bestPair, nextID)
		}

		t.Merges[bestPair] = nextID
		nextID++
	}

	return nil
}

// Encode takes a raw string and converts it into a sequence of token IDs
// using the previously trained BPE rules.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	// 1. Regex Split
	stringChunks, err := chunkText(text)
	if err != nil {
		return nil, err
	}

	var finalIDs []int

	// 2. Process each chunk
	for _, chunkStr := range stringChunks {
		// Convert to raw bytes
		rawBytes := []byte(chunkStr)
		chunkIDs := make([]int, len(rawBytes))
		for i, b := range rawBytes {
			chunkIDs[i] = int(b)
		}

		// 3. Apply learned merges
		// We must apply merges in the exact order they were learned (by ID).
		// In a production system, you'd pre-sort your merge rules.
		// For this implementation, we will iteratively search for applicable merges.
		for {
			var bestPair Pair
			minID := -1 // We want to apply the oldest merge (lowest ID) first

			// Find which adjacent pair in our current chunk is the "oldest" known merge
			for i := 0; i < len(chunkIDs)-1; i++ {
				pair := Pair{chunkIDs[i], chunkIDs[i+1]}
				if id, exists := t.Merges[pair]; exists {
					if minID == -1 || id < minID {
						minID = id
						bestPair = pair
					}
				}
			}

			// If no known pairs are found, this chunk is fully compressed
			if minID == -1 {
				break
			}

			// Apply the merge and loop again
			chunkIDs = merge(chunkIDs, bestPair, minID)
		}

		// 4. Append the compressed chunk to our final array
		finalIDs = append(finalIDs, chunkIDs...)
	}

	return finalIDs, nil
}

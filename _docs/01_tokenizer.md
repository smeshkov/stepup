A neural network cannot read strings. It only understands numbers (specifically, continuous vectors). Byte-Pair Encoding (BPE) is the algorithm that compresses raw, unstructured strings into a sequence of discrete integers. Extracting this kind of structural meaning from raw textual data streams shares a lot of DNA with generating metadata tags from source text—it requires efficient string traversal, mapping, and zero-copy manipulation.

### The BPE Concept

BPE is a data compression algorithm adapted for AI. 
1. We start by breaking all training text down into its raw UTF-8 bytes (vocabulary size = 256).
2. We scan the text and find the most frequently occurring adjacent pair of tokens.
3. We merge that pair into a *new* single token (vocabulary size = 257) and replace all instances of the pair in our text with the new ID.
4. We repeat this process $k$ times until we reach our target vocabulary size (e.g., Llama 3 has a vocabulary of 128,000).



### Step 1: The Core Data Structures in Go

To build this, we need a way to track pairs of tokens and a map to store our learned merge rules. 

```go
package tokenizer

// Pair represents a bigram of token IDs.
// We use a fixed-size array [2]int so it can be used as a map key in Go.
type Pair [2]int

// Tokenizer holds the state of our learned BPE rules.
type Tokenizer struct {
	// Merges maps a pair of tokens to their new combined Token ID
	Merges map[Pair]int
	// Vocab maps the integer ID back to its raw bytes for decoding
	Vocab  map[int][]byte
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Merges: make(map[Pair]int),
		Vocab:  make(map[int][]byte),
	}
}
```

### Step 2: Counting and Merging

The engine of the BPE trainer relies on two operations: counting the frequencies of pairs, and replacing those pairs. 

As an experienced Go engineer, you'll immediately spot the memory allocation trap here. In the `Merge` function, blindly using `append` would cause the garbage collector to thrash during training on gigabytes of text. Because a merged slice will *always* be equal to or smaller than the input slice, we can pre-allocate the exact maximum capacity to avoid reallocation.

```go
// getStats counts the frequency of every adjacent pair in the token slice
func getStats(ids []int) map[Pair]int {
	counts := make(map[Pair]int)
	for i := 0; i < len(ids)-1; i++ {
		pair := Pair{ids[i], ids[i+1]}
		counts[pair]++
	}
	return counts
}

// merge iterates through the IDs and replaces the target pair with the newID
func merge(ids []int, pair Pair, newID int) []int {
	// Pre-allocate to avoid GC pressure. The new slice will never exceed len(ids).
	merged := make([]int, 0, len(ids))
	
	i := 0
	for i < len(ids) {
		// If we find the pair, append the new ID and skip the next token
		if i < len(ids)-1 && ids[i] == pair[0] && ids[i+1] == pair[1] {
			merged = append(merged, newID)
			i += 2 
		} else {
			merged = append(merged, ids[i])
			i++
		}
	}
	return merged
}
```

### Step 3: The Training Loop

To actually train our tokenizer, we take a large string of raw text, convert it to its raw UTF-8 byte representations (integers 0-255), and loop the merge process.

```go
// Train runs the BPE algorithm to build a vocabulary of targetSize
func (t *Tokenizer) Train(text string, targetVocabSize int) {
	// 1. Convert raw string to initial byte IDs (0-255)
	rawBytes := []byte(text)
	ids := make([]int, len(rawBytes))
	for i, b := range rawBytes {
		ids[i] = int(b)
	}

	numMerges := targetVocabSize - 256
	nextID := 256 // Start assigning new IDs after the base 256 bytes

	for i := 0; i < numMerges; i++ {
		stats := getStats(ids)
		
		// Find the most frequent pair
		var bestPair Pair
		maxCount := -1
		for pair, count := range stats {
			if count > maxCount {
				maxCount = count
				bestPair = pair
			}
		}

		// If no pairs occur more than once, we can't compress further
		if maxCount < 2 {
			break
		}

		// Apply the merge
		ids = merge(ids, bestPair, nextID)
		
		// Record the learned rule
		t.Merges[bestPair] = nextID
		nextID++
	}
}
```

### The Real-World Catch: Pre-tokenization

If you run the code above on a raw text corpus, you will hit a problem. The algorithm will start merging characters across spaces and punctuation. It might merge the end of the word "Hello" with a comma, creating a token for `o,`. 

To prevent this, modern models like GPT-4 and Llama 3 use a **regex pre-tokenizer**. Before running BPE, they use regular expressions to forcefully split the text into isolated chunks (e.g., separating all punctuation, numbers, and distinct words) so that the BPE algorithm is mathematically blocked from merging tokens across those logical boundaries.

Should we implement the regex pre-tokenization logic to make this production-accurate, or are you ready to take our integer arrays and move into the deep math of the Embedding layer?

To make our tokenizer production-accurate (specifically, mimicking the behavior of OpenAI's `tiktoken` or Llama 3's tokenizer), we run into a fascinating quirk of the Go language: **Go's standard `regexp` package does not support lookaround assertions.**

Go's `regexp` uses the RE2 engine. It guarantees linear time $O(n)$ execution to prevent regex denial-of-service (ReDoS) attacks. However, the official regex patterns used by modern LLMs require lookaheads (like `(?!\S)`) to correctly split spaces, punctuation, and word boundaries. 

To achieve production accuracy in Go, we must bypass the standard library and use a PCRE-compatible regex engine. The industry standard for this in the Go ecosystem is `github.com/dlclark/regexp2`.

### Step 1: The GPT-4 Split Pattern

Here is the actual regex pattern used by GPT-4 (the `cl100k_base` encoding). It forces the text to break into distinct categories: contractions, letters, numbers, punctuation, and spaces.

```go
package tokenizer

import (
	"github.com/dlclark/regexp2"
)

// This is the precise regex used by GPT-4 and modern models.
// It ensures BPE never merges a word with adjacent punctuation or cross-sentence spaces.
const gpt4Pattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

var splitRegex = regexp2.MustCompile(gpt4Pattern, regexp2.None)

// chunkText uses the PCRE regex engine to slice the raw string into semantic boundaries.
func chunkText(text string) ([]string, error) {
	var chunks []string
	m, err := splitRegex.FindStringMatch(text)
	if err != nil {
		return nil, err
	}
	
	for m != nil {
		chunks = append(chunks, m.String())
		m, err = splitRegex.FindNextMatch(m)
		if err != nil {
			return nil, err
		}
	}
	return chunks, nil
}
```

### Step 2: Architecting the Chunked BPE Trainer

Because we pre-tokenized the text into a slice of strings, our BPE trainer must adapt. Instead of operating on one massive `[]int`, we now operate on a `[][]int` (a slice of integer slices, where each sub-slice is a chunk). 

This structural boundary is what physically prevents the BPE algorithm from merging tokens across chunks.

```go
// We need to update our getStats function to iterate over multiple chunks
func getChunkedStats(chunks [][]int) map[Pair]int {
	counts := make(map[Pair]int)
	for _, chunk := range chunks {
		for i := 0; i < len(chunk)-1; i++ {
			pair := Pair{chunk[i], chunk[i+1]}
			counts[pair]++
		}
	}
	return counts
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
	for i := 0; i < numMerges; i++ {
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
```

### The Production Inference Loop

Once the model is trained and `t.Merges` is populated, the actual inference—taking a new string from a user and converting it to LLM-ready tokens—follows the exact same pipeline:

1. **Regex Split:** Run the user's prompt through `regexp2` to get the string chunks.
2. **Byte Conversion:** Convert those strings to raw byte IDs.
3. **Apply Merges:** Iterate through your `t.Merges` rules (in the exact order they were learned, which requires sorting your rules by `nextID`) and aggressively apply them to the chunks.
4. **Flatten:** Concatenate the processed `[][]int` back into a single 1D `[]int` array. 

That single 1D array of integers is the final, production-ready input that gets shipped across the network to the GPU to be processed by the Embedding layer.
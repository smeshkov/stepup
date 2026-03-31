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

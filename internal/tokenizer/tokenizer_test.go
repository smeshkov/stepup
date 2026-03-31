package tokenizer

import (
	"reflect"
	"testing"
)

func TestTokenizer_EndToEnd(t *testing.T) {
	// A small, highly repetitive string to ensure BPE finds clear patterns.
	// "aaab" repeats multiple times.
	trainingText := "aaab aaab aaab aaab, Hello world! Hello world!"

	tok := NewTokenizer()

	// 256 (base bytes) + 5 custom merges = target size of 261
	targetVocabSize := 261

	err := tok.TrainProduction(trainingText, targetVocabSize)
	if err != nil {
		t.Fatalf("Failed to train tokenizer: %v", err)
	}

	// Verify that merges were actually learned
	if len(tok.Merges) == 0 {
		t.Fatal("Expected tokenizer to learn merges, but Merges map is empty")
	}

	// --- Testing the Encoding Phase ---

	// Let's test it on a string it has seen before
	testStr := "aaab"
	encodedIDs, err := tok.Encode(testStr)
	if err != nil {
		t.Fatalf("Failed to encode text: %v", err)
	}

	// "aaab" is 4 bytes. Without BPE, it would be 4 tokens.
	// Because it appeared so frequently in training, BPE should have compressed it.
	if len(encodedIDs) >= len(testStr) {
		t.Errorf("Expected encoded IDs length to be compressed (less than %d), got %d", len(testStr), len(encodedIDs))
	}

	t.Logf("Original text: %s", testStr)
	t.Logf("Encoded Token IDs: %v", encodedIDs)
}

func TestRegexChunking(t *testing.T) {
	// Testing our PCRE regex implementation
	text := "Hello, world! It's 2024."

	chunks, err := chunkText(text)
	if err != nil {
		t.Fatalf("Regex chunking failed: %v", err)
	}

	// Based on the GPT-4 regex pattern, we expect specific splits.
	// E.g., "Hello", ",", " world", "!", " It", "'s", " 202", "4", "."
	expectedChunks := []string{"Hello", ",", " world", "!", " It", "'s", " ", "202", "4", "."}

	if !reflect.DeepEqual(chunks, expectedChunks) {
		t.Errorf("Chunking failed.\nExpected: %q\nGot: %q", expectedChunks, chunks)
	}
}

package main

import (
	"fmt"
	"log"

	"github.com/smeshkov/stepup/internal/model"
	"github.com/smeshkov/stepup/internal/tokenizer"
)

func main() {
	fmt.Println("=== 🚀 Starting LLM Engine Pipeline ===")

	// ---------------------------------------------------------
	// PHASE 1: Tokenization (Text -> Discrete IDs)
	// ---------------------------------------------------------
	fmt.Println("\n[1/4] Booting and Training Tokenizer...")
	tok := tokenizer.NewTokenizer()

	// A small corpus just so the tokenizer learns *something* locally
	trainingData := "The quick brown fox jumps over the lazy dog. The dog barks."

	// We start with 256 base bytes and learn 10 new token merges
	targetVocab := 256 + 10
	if err := tok.TrainProduction(trainingData, targetVocab); err != nil {
		log.Fatalf("Failed to train tokenizer: %v", err)
	}

	fmt.Println("[2/4] Encoding User Prompt...")
	prompt := "The dog."
	tokenIDs, err := tok.Encode(prompt)
	if err != nil {
		log.Fatalf("Failed to encode prompt: %v", err)
	}

	fmt.Printf("  Raw Prompt: %q\n", prompt)
	fmt.Printf("  Token IDs:  %v\n", tokenIDs)

	// ---------------------------------------------------------
	// PHASE 2: The Math Engine (Discrete IDs -> Geometry)
	// ---------------------------------------------------------
	fmt.Println("\n[3/4] Initializing Embedding Engine...")

	// dModel is the size of our neural network's "brain".
	// GPT-3 uses 12,288. We will use 16 so it's readable in the terminal.
	dModel := 16
	maxSeqLen := 1024

	emb := model.NewEmbedding(targetVocab, dModel)
	peMatrix := model.PrecomputePositionalEncoding(maxSeqLen, dModel)

	fmt.Println("[4/4] Executing Forward Pass (Text -> Tensor)...")
	// This is where the magic happens: combining meaning (Embedding) with order (Position)
	tensor, err := model.PrepareInput(tokenIDs, emb, peMatrix)
	if err != nil {
		log.Fatalf("Failed to prepare input: %v", err)
	}

	// ---------------------------------------------------------
	// RESULTS: The Output Tensor
	// ---------------------------------------------------------
	fmt.Println("\n=== 🎯 Pipeline Complete ===")
	fmt.Printf("Input Sequence Length : %d tokens\n", len(tensor))
	fmt.Printf("Model Dimension       : %d features per token\n", len(tensor[0]))

	fmt.Println("\nFirst Token Vector (Meaning + Position):")
	// Print the first few floats of the first token to prove we have geometry
	for i, val := range tensor[0] {
		if i < 4 {
			fmt.Printf("  Dim %d: %f\n", i, val)
		}
	}
	fmt.Println("  ... (truncated)")

	fmt.Println("\nNext step: Feed this tensor into the Self-Attention block!")
}

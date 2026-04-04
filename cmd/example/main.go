package main

import (
	"fmt"
	"log"

	"github.com/smeshkov/stepup/internal/autograd"
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

	// ---------------------------------------------------------
	// PHASE 3: The Transformer Block
	// ---------------------------------------------------------
	fmt.Println("\n[5/5] Executing Complete Transformer Block...")

	numHeads := 4
	block := model.NewTransformerBlock(numHeads, dModel)

	// 'tensor' is the contextualized output from PrepareInput
	finalBlockOutput, err := block.Forward(tensor)
	if err != nil {
		log.Fatalf("Transformer Block failed: %v", err)
	}

	fmt.Println("\n=== 🧠 First Transformer Block Complete ===")
	fmt.Printf("Input  Shape : %d tokens x %d dimensions\n", len(tensor), len(tensor[0]))
	fmt.Printf("Output Shape : %d tokens x %d dimensions\n", len(finalBlockOutput), len(finalBlockOutput[0]))

	fmt.Println("\nFinal Processed Vector for Token 0:")
	for i, val := range finalBlockOutput[0] {
		if i < 4 {
			fmt.Printf("  Dim %d: %f\n", i, val)
		}
	}
	fmt.Println("  ... (truncated)")

	// ---------------------------------------------------------
	// PHASE 4: Autograd Classification (Cross-Entropy + AdamW)
	// ---------------------------------------------------------
	fmt.Println("\n=== 🎓 PHASE 4: Autograd Classification (Cross-Entropy + AdamW) ===")

	// Simulate a vocabulary of 3 possible words.
	// These are our learnable "Logits" (the raw output of a neural network before Softmax)
	logits := []*autograd.Value{
		autograd.NewValue(2.0), // Token 0 prediction
		autograd.NewValue(1.0), // Token 1 prediction
		autograd.NewValue(0.1), // Token 2 prediction (Currently the lowest probability)
	}

	// We want the model to learn that Token 2 is the correct answer
	targetIndex := 2

	// Initialize AdamW optimizer with our 3 parameters
	optimizer := autograd.NewAdamW(logits, 0.1) // Higher learning rate for quick demo

	epochs := 100

	fmt.Printf("Initial Logits: [%.3f, %.3f, %.3f]\n", logits[0].Data, logits[1].Data, logits[2].Data)

	for epoch := 1; epoch <= epochs; epoch++ {
		// 1. Forward Pass: Calculate Cross Entropy Loss
		loss := autograd.CrossEntropyLoss(logits, targetIndex)

		// 2. Backward Pass: Flush old gradients, calculate new ones
		optimizer.ZeroGrad()
		loss.Backward()

		// 3. Optimize: Adjust the logits using AdamW
		optimizer.Step()

		if epoch == 1 || epoch%20 == 0 {
			fmt.Printf("Epoch %3d | Loss: %7.4f | Logits: [%6.3f, %6.3f, %6.3f]\n",
				epoch, loss.Data, logits[0].Data, logits[1].Data, logits[2].Data)
		}
	}

	fmt.Println("\n=== 🏁 Training Complete ===")
	fmt.Printf("Final Logits: [%.3f, %.3f, %.3f]\n", logits[0].Data, logits[1].Data, logits[2].Data)
	fmt.Println("Notice how the Logit for Token 2 surged to the top, while 0 and 1 were suppressed.")
}

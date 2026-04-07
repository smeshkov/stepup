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
	fmt.Printf("Input Sequence Length : %d tokens\n", tensor.Rows)
	fmt.Printf("Model Dimension       : %d features per token\n", tensor.Cols)

	fmt.Println("\nFirst Token Vector (Meaning + Position):")
	// Print the first few floats of the first token to prove we have geometry
	firstRow := tensor.Row(0)
	for i := range 4 {
		fmt.Printf("  Dim %d: %f\n", i, firstRow[i])
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
	fmt.Printf("Input  Shape : %d tokens x %d dimensions\n", tensor.Rows, tensor.Cols)
	fmt.Printf("Output Shape : %d tokens x %d dimensions\n", finalBlockOutput.Rows, finalBlockOutput.Cols)

	fmt.Println("\nFinal Processed Vector for Token 0:")
	finalRow := finalBlockOutput.Row(0)
	for i := range 4 {
		fmt.Printf("  Dim %d: %f\n", i, finalRow[i])
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

	// ---------------------------------------------------------
	// PHASE 5: Sampling Strategies
	// ---------------------------------------------------------
	fmt.Println("\n=== 🎲 PHASE 5: Sampling Strategies ===")

	// Use the trained logits as raw scores for sampling
	rawLogits := []float32{float32(logits[0].Data), float32(logits[1].Data), float32(logits[2].Data)}
	fmt.Printf("Logits: [%.3f, %.3f, %.3f]\n\n", rawLogits[0], rawLogits[1], rawLogits[2])

	// 1. Greedy: always picks the highest logit
	greedy := model.SampleGreedy(rawLogits)
	fmt.Printf("Greedy       → Token %d (always picks the most likely)\n", greedy)

	// 2. Temperature sampling at different temperatures
	fmt.Println("\nTemperature sampling (10 samples each):")
	for _, temp := range []float32{0.1, 1.0, 5.0} {
		samples := make([]int, 10)
		for i := range samples {
			samples[i] = model.SampleTemperature(rawLogits, temp)
		}
		fmt.Printf("  T=%.1f → %v\n", temp, samples)
	}

	// 3. Top-P (nucleus) sampling
	fmt.Println("\nTop-P sampling (10 samples each):")
	for _, p := range []float32{0.5, 0.9, 0.99} {
		samples := make([]int, 10)
		for i := range samples {
			samples[i] = model.SampleTopP(rawLogits, 1.0, p)
		}
		fmt.Printf("  p=%.2f → %v\n", p, samples)
	}
}

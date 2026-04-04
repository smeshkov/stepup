package autograd

import (
	"math"
	"testing"
)

func TestCrossEntropyLoss_CorrectClassHighLogit(t *testing.T) {
	// When the correct class has the highest logit, loss should be low
	logits := []*Value{NewValue(10.0), NewValue(1.0), NewValue(0.5)}
	targetIndex := 0

	loss := CrossEntropyLoss(logits, targetIndex)

	// Softmax will assign most probability to index 0, so -ln(P) should be near 0
	if loss.Data < 0 || loss.Data > 0.001 {
		t.Errorf("Expected loss near 0 for correct high-confidence prediction, got %f", loss.Data)
	}
}

func TestCrossEntropyLoss_CorrectClassLowLogit(t *testing.T) {
	// When the correct class has a low logit, loss should be high
	logits := []*Value{NewValue(0.1), NewValue(10.0), NewValue(10.0)}
	targetIndex := 0

	loss := CrossEntropyLoss(logits, targetIndex)

	// The correct class has very little probability, so loss should be high
	if loss.Data < 5.0 {
		t.Errorf("Expected high loss for wrong prediction, got %f", loss.Data)
	}
}

func TestCrossEntropyLoss_UniformLogits(t *testing.T) {
	// When all logits are equal, each class gets 1/N probability
	// Loss should be -ln(1/3) = ln(3) ≈ 1.0986
	logits := []*Value{NewValue(1.0), NewValue(1.0), NewValue(1.0)}
	targetIndex := 1

	loss := CrossEntropyLoss(logits, targetIndex)

	expected := float32(math.Log(3.0))
	if math.Abs(float64(loss.Data-expected)) > 0.01 {
		t.Errorf("Expected loss ≈ %f for uniform logits, got %f", expected, loss.Data)
	}
}

func TestCrossEntropyLoss_Backward(t *testing.T) {
	// Verify that gradients flow back through the loss computation
	logits := []*Value{NewValue(2.0), NewValue(1.0), NewValue(0.5)}
	targetIndex := 0

	loss := CrossEntropyLoss(logits, targetIndex)
	loss.Backward()

	// The gradient of the target logit should be negative (increasing it decreases the loss)
	if logits[targetIndex].Grad >= 0 {
		t.Errorf("Expected negative gradient for target logit, got %f", logits[targetIndex].Grad)
	}

	// The gradients of non-target logits should be positive (increasing them increases the loss)
	for i, logit := range logits {
		if i != targetIndex && logit.Grad <= 0 {
			t.Errorf("Expected positive gradient for non-target logit[%d], got %f", i, logit.Grad)
		}
	}
}

func TestCrossEntropyLoss_TwoClasses(t *testing.T) {
	// Binary classification case
	logits := []*Value{NewValue(3.0), NewValue(-3.0)}
	targetIndex := 0

	loss := CrossEntropyLoss(logits, targetIndex)

	// With a large gap, the correct class probability is near 1, so loss near 0
	if loss.Data < 0 || loss.Data > 0.01 {
		t.Errorf("Expected loss near 0, got %f", loss.Data)
	}
}

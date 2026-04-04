package autograd

import (
	"math"
	"testing"
)

func TestAdamW_NewAdamW(t *testing.T) {
	params := []*Value{NewValue(1.0), NewValue(2.0)}
	opt := NewAdamW(params, 0.001)

	if opt.LearningRate != 0.001 {
		t.Errorf("Expected LearningRate 0.001, got %f", opt.LearningRate)
	}
	if opt.Beta1 != 0.9 {
		t.Errorf("Expected Beta1 0.9, got %f", opt.Beta1)
	}
	if opt.Beta2 != 0.999 {
		t.Errorf("Expected Beta2 0.999, got %f", opt.Beta2)
	}
	if opt.t != 0 {
		t.Errorf("Expected initial timestep 0, got %d", opt.t)
	}
}

func TestAdamW_ZeroGrad(t *testing.T) {
	a := NewValue(1.0)
	b := NewValue(2.0)
	a.Grad = 5.0
	b.Grad = 3.0

	opt := NewAdamW([]*Value{a, b}, 0.001)
	opt.ZeroGrad()

	if a.Grad != 0.0 {
		t.Errorf("Expected a.Grad to be 0.0 after ZeroGrad, got %f", a.Grad)
	}
	if b.Grad != 0.0 {
		t.Errorf("Expected b.Grad to be 0.0 after ZeroGrad, got %f", b.Grad)
	}
}

func TestAdamW_StepReducesLoss(t *testing.T) {
	// Simulate a simple optimization: minimize L = x^2
	// Gradient of x^2 is 2x
	x := NewValue(5.0)
	opt := NewAdamW([]*Value{x}, 0.1)

	initialData := x.Data

	for range 50 {
		opt.ZeroGrad()

		// Forward: L = x * x
		loss := x.Mul(x)

		// Backward
		loss.Backward()

		// Step
		opt.Step()
	}

	// After 50 steps, x should be much closer to 0
	if math.Abs(float64(x.Data)) >= math.Abs(float64(initialData)) {
		t.Errorf("Expected x to decrease toward 0, but went from %f to %f", initialData, x.Data)
	}
	if math.Abs(float64(x.Data)) > 1.0 {
		t.Errorf("Expected x near 0 after 50 steps, got %f", x.Data)
	}
}

func TestAdamW_TimestepIncrements(t *testing.T) {
	x := NewValue(1.0)
	opt := NewAdamW([]*Value{x}, 0.01)

	if opt.t != 0 {
		t.Fatalf("Expected initial timestep 0, got %d", opt.t)
	}

	x.Grad = 1.0
	opt.Step()
	if opt.t != 1 {
		t.Errorf("Expected timestep 1 after one step, got %d", opt.t)
	}

	x.Grad = 1.0
	opt.Step()
	if opt.t != 2 {
		t.Errorf("Expected timestep 2 after two steps, got %d", opt.t)
	}
}

func TestAdamW_WeightDecay(t *testing.T) {
	// With zero gradient, only weight decay should apply
	x := NewValue(10.0)
	opt := NewAdamW([]*Value{x}, 0.01)
	opt.WeightDecay = 0.1

	x.Grad = 0.0
	opt.Step()

	// Weight decay: x -= lr * wd * x = 0.01 * 0.1 * 10 = 0.01
	// With zero gradient, m and v stay 0, so the Adam update is 0
	expected := float32(10.0 - 0.01*0.1*10.0)
	if math.Abs(float64(x.Data-expected)) > 1e-5 {
		t.Errorf("Expected x ≈ %f after weight decay only, got %f", expected, x.Data)
	}
}

func TestAdamW_MultipleParams(t *testing.T) {
	// Minimize L = a^2 + b^2
	a := NewValue(3.0)
	b := NewValue(-4.0)
	opt := NewAdamW([]*Value{a, b}, 0.1)

	for range 100 {
		opt.ZeroGrad()

		aSquared := a.Mul(a)
		bSquared := b.Mul(b)
		loss := aSquared.Add(bSquared)
		loss.Backward()

		opt.Step()
	}

	// Both should converge near 0
	if math.Abs(float64(a.Data)) > 0.5 {
		t.Errorf("Expected a near 0, got %f", a.Data)
	}
	if math.Abs(float64(b.Data)) > 0.5 {
		t.Errorf("Expected b near 0, got %f", b.Data)
	}
}

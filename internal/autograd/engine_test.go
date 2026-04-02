package autograd

import (
	"math"
	"testing"
)

func TestAutograd(t *testing.T) {
	// Let's build a simple equation: L = (a * b) + c
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := NewValue(4.0)

	// Forward pass (This silently builds the DAG)
	ab := a.Mul(b) // 2 * 3 = 6
	L := ab.Add(c) // 6 + 4 = 10

	if L.Data != 10.0 {
		t.Fatalf("Expected L to be 10.0, got %f", L.Data)
	}

	// Backward pass (Calculates the derivatives via Chain Rule)
	L.Backward()

	// Let's verify the Calculus!
	// L = (a * b) + c

	// dL/dc: Since L = ab + c, a change in c directly changes L by the same amount. Expected: 1.0
	if math.Abs(float64(c.Grad-1.0)) > 1e-5 {
		t.Errorf("Expected c.Grad to be 1.0, got %f", c.Grad)
	}

	// dL/da: Since L = (a * 3) + 4, a change in 'a' is multiplied by 3. Expected: 3.0 (which is b.Data)
	if math.Abs(float64(a.Grad-3.0)) > 1e-5 {
		t.Errorf("Expected a.Grad to be 3.0, got %f", a.Grad)
	}

	// dL/db: Since L = (2 * b) + 4, expected: 2.0 (which is a.Data)
	if math.Abs(float64(b.Grad-2.0)) > 1e-5 {
		t.Errorf("Expected b.Grad to be 2.0, got %f", b.Grad)
	}
}

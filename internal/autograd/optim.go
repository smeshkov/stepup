package autograd

import "math"

type AdamW struct {
	Params       []*Value
	LearningRate float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32
	WeightDecay  float32

	// Caches for the running moments of each parameter
	m map[*Value]float32
	v map[*Value]float32
	t int // Timestep (epoch counter)
}

func NewAdamW(params []*Value, lr float32) *AdamW {
	return &AdamW{
		Params:       params,
		LearningRate: lr,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  0.01,
		m:            make(map[*Value]float32),
		v:            make(map[*Value]float32),
		t:            0,
	}
}

// ZeroGrad wipes the gradients clean before the next backward pass
func (opt *AdamW) ZeroGrad() {
	for _, p := range opt.Params {
		p.Grad = 0.0
	}
}

// Step applies the AdamW update rule to every parameter
func (opt *AdamW) Step() {
	opt.t++

	// Bias correction factors
	bc1 := float32(1.0 - math.Pow(float64(opt.Beta1), float64(opt.t)))
	bc2 := float32(1.0 - math.Pow(float64(opt.Beta2), float64(opt.t)))

	for _, p := range opt.Params {
		grad := p.Grad

		// Update biased first moment estimate
		opt.m[p] = opt.Beta1*opt.m[p] + (1.0-opt.Beta1)*grad

		// Update biased second raw moment estimate
		opt.v[p] = opt.Beta2*opt.v[p] + (1.0-opt.Beta2)*(grad*grad)

		// Compute bias-corrected moments
		mHat := opt.m[p] / bc1
		vHat := opt.v[p] / bc2

		// Weight Decay applied completely independently of the gradient (The "W" in AdamW)
		p.Data -= opt.LearningRate * opt.WeightDecay * p.Data

		// The core Adam update
		denominator := float32(math.Sqrt(float64(vHat))) + opt.Epsilon
		p.Data -= opt.LearningRate * (mHat / denominator)
	}
}

package autograd

// CrossEntropyLoss calculates the logarithmic loss for a classification task.
// It takes a slice of unnormalized logits and the integer index of the correct class.
func CrossEntropyLoss(logits []*Value, targetIndex int) *Value {
	// 1. Calculate the denominator of Softmax: sum(e^logit)
	// We find the max logit first for numerical stability (preventing e^1000 overflow)
	maxLogit := logits[0].Data
	for _, l := range logits {
		if l.Data > maxLogit {
			maxLogit = l.Data
		}
	}
	maxVal := NewValue(maxLogit)

	var sumExp *Value
	exps := make([]*Value, len(logits))

	for i, logit := range logits {
		// e^(logit - max)
		e := logit.Sub(maxVal).Exp()
		exps[i] = e
		if sumExp == nil {
			sumExp = e
		} else {
			sumExp = sumExp.Add(e)
		}
	}

	// 2. Calculate the probability of the target class: P = e^target_logit / sumExp
	targetProb := exps[targetIndex].Div(sumExp)

	// 3. Cross Entropy Loss = -ln(P)
	// We add a tiny epsilon to prevent ln(0) which results in -Infinity
	epsilon := NewValue(1e-7)
	logProb := targetProb.Add(epsilon).Log()

	// Multiply by -1
	negOne := NewValue(-1.0)
	loss := logProb.Mul(negOne)

	return loss
}

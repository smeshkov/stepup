package autograd

// Value is a node in our computational graph.
type Value struct {
	Data     float32  // The actual mathematical value (the forward pass)
	Grad     float32  // The derivative of the loss with respect to this value (the backward pass)
	children []*Value // The nodes that created this value
	backward func()   // The closure that executes the chain rule for this specific operation
}

// NewValue creates a raw, starting node (like a weight or input) with no children.
func NewValue(data float32) *Value {
	return &Value{
		Data:     data,
		Grad:     0.0,
		children: nil,
		backward: func() {}, // Empty closure by default
	}
}

// Add performs v + other and builds the graph connection
func (v *Value) Add(other *Value) *Value {
	out := &Value{
		Data:     v.Data + other.Data,
		Grad:     0.0,
		children: []*Value{v, other},
	}

	// The calculus of addition: the gradient flows equally to both children
	out.backward = func() {
		v.Grad += 1.0 * out.Grad
		other.Grad += 1.0 * out.Grad
	}

	return out
}

// Mul performs v * other and builds the graph connection
func (v *Value) Mul(other *Value) *Value {
	out := &Value{
		Data:     v.Data * other.Data,
		Grad:     0.0,
		children: []*Value{v, other},
	}

	// The calculus of multiplication: the gradient is multiplied by the sibling's data
	out.backward = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}

	return out
}

// Backward initiates the backpropagation process from this node
func (v *Value) Backward() {
	// The derivative of a node with respect to itself is always 1
	v.Grad = 1.0

	// 1. Build the topological order
	var topo []*Value
	visited := make(map[*Value]bool)

	var buildTopo func(node *Value)
	buildTopo = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node.children {
				buildTopo(child)
			}
			// Append after processing children to get bottom-up order
			topo = append(topo, node)
		}
	}

	buildTopo(v)

	// 2. Go one variable at a time and apply the chain rule in reverse order
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}

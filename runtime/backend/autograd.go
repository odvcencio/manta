package backend

import (
	"fmt"
	"math"
	"slices"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

// GradTensor is a runtime tensor value tracked by the reference reverse-mode
// autodiff path. Gradients are accumulated in f32 storage regardless of the
// forward dtype.
type GradTensor struct {
	Name         string
	Value        *Tensor
	Grad         *Tensor
	RequiresGrad bool

	parents  []*GradTensor
	backward func(*GradTensor) error
}

// NewGradTensor creates a differentiable tensor leaf.
func NewGradTensor(name string, value *Tensor, requiresGrad bool) *GradTensor {
	return &GradTensor{
		Name:         name,
		Value:        cloneOrNil(value),
		RequiresGrad: requiresGrad,
	}
}

// Backward seeds output with a scalar gradient of 1 and propagates gradients to
// all reachable leaves.
func Backward(output *GradTensor) error {
	if output == nil || output.Value == nil {
		return fmt.Errorf("autograd backward expects an output tensor")
	}
	seed := zeroGradLike(output.Value)
	for i := range seed.F32 {
		seed.F32[i] = 1
	}
	return BackwardWithGradient(output, seed)
}

// BackwardWithGradient propagates a caller-provided output gradient.
func BackwardWithGradient(output *GradTensor, grad *Tensor) error {
	if output == nil || output.Value == nil {
		return fmt.Errorf("autograd backward expects an output tensor")
	}
	if grad == nil || !sameTensorShape(output.Value, grad) || len(grad.F32) != output.Value.Elements() {
		return fmt.Errorf("autograd seed gradient shape %v does not match output %v", shapeOf(grad), output.Value.Shape)
	}
	if err := accumulateGrad(output, grad); err != nil {
		return err
	}
	order := topoGradTensors(output)
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node == nil || node.Grad == nil || node.backward == nil {
			continue
		}
		if err := node.backward(node); err != nil {
			return fmt.Errorf("backward %q: %w", node.Name, err)
		}
	}
	return nil
}

// GradRequest describes one reference-autograd execution of a Manta entrypoint.
type GradRequest struct {
	Entry           string
	Inputs          map[string]*Tensor
	Weights         map[string]*Tensor
	Loss            string
	TrainableInputs map[string]bool
}

// GradResult contains forward outputs and accumulated gradients for trainable
// params and explicitly trainable inputs.
type GradResult struct {
	Outputs        map[string]*Tensor
	Gradients      map[string]*Tensor
	InputGradients map[string]*Tensor
	Trace          []TraceStep
}

// ExecuteAutograd runs a Manta pipeline through the reference reverse-mode path.
// It is intentionally backend-independent; device backward kernels can reuse
// the same op contracts while this path keeps correctness tests deterministic.
func ExecuteAutograd(mod *mantaartifact.Module, req GradRequest) (GradResult, error) {
	if mod == nil {
		return GradResult{}, fmt.Errorf("nil module")
	}
	entryName := req.Entry
	if entryName == "" {
		if len(mod.EntryPoints) == 0 {
			return GradResult{}, fmt.Errorf("module %q has no entrypoints", mod.Name)
		}
		entryName = mod.EntryPoints[0].Name
	}
	entry, ok := entryPointByName(mod, entryName)
	if !ok {
		return GradResult{}, fmt.Errorf("unknown entrypoint %q", entryName)
	}

	env := map[string]*GradTensor{}
	for _, param := range mod.Params {
		weight, ok := req.Weights[param.Name]
		if !ok {
			return GradResult{}, fmt.Errorf("missing weight %q", param.Name)
		}
		env[param.Name] = NewGradTensor(param.Name, weight, param.Trainable)
	}
	for _, input := range entry.Inputs {
		value, ok := req.Inputs[input.Name]
		if !ok {
			return GradResult{}, fmt.Errorf("entrypoint %q missing input %q", entry.Name, input.Name)
		}
		env[input.Name] = NewGradTensor(input.Name, value, req.TrainableInputs[input.Name])
	}

	result := GradResult{
		Outputs:        map[string]*Tensor{},
		Gradients:      map[string]*Tensor{},
		InputGradients: map[string]*Tensor{},
		Trace:          make([]TraceStep, 0, len(stepsForEntry(mod, entry.Name))),
	}
	for _, step := range stepsForEntry(mod, entry.Name) {
		values, err := executeAutogradStep(step, env)
		if err != nil {
			return GradResult{}, fmt.Errorf("entrypoint %q step %q: %w", entry.Name, step.Name, err)
		}
		result.Trace = append(result.Trace, TraceStep{
			Entry:   step.Entry,
			Kind:    step.Kind,
			Name:    step.Name,
			Inputs:  cloneStrings(step.Inputs),
			Outputs: cloneStrings(step.Outputs),
		})
		if step.Kind == mantaartifact.StepReturn {
			for _, name := range step.Outputs {
				value, ok := env[name]
				if !ok || value.Value == nil {
					return GradResult{}, fmt.Errorf("return references unknown value %q", name)
				}
				result.Outputs[name] = value.Value.Clone()
			}
			continue
		}
		for i, name := range step.Outputs {
			if i >= len(values) {
				return GradResult{}, fmt.Errorf("step produced %d values for %d outputs", len(values), len(step.Outputs))
			}
			values[i].Name = name
			env[name] = values[i]
		}
	}

	lossName := req.Loss
	if lossName == "" {
		lossName = "loss"
	}
	loss, ok := env[lossName]
	if !ok {
		return GradResult{}, fmt.Errorf("loss value %q not found", lossName)
	}
	if loss.Value == nil || len(loss.Value.F32) != 1 {
		return GradResult{}, fmt.Errorf("loss value %q must be a scalar tensor", lossName)
	}
	if err := Backward(loss); err != nil {
		return GradResult{}, err
	}

	for _, param := range mod.Params {
		if !param.Trainable {
			continue
		}
		if node, ok := env[param.Name]; ok && node.Grad != nil {
			result.Gradients[param.Name] = node.Grad.Clone()
		}
	}
	for name, trainable := range req.TrainableInputs {
		if !trainable {
			continue
		}
		if node, ok := env[name]; ok && node.Grad != nil {
			result.InputGradients[name] = node.Grad.Clone()
		}
	}
	return result, nil
}

func executeAutogradStep(step mantaartifact.Step, env map[string]*GradTensor) ([]*GradTensor, error) {
	switch step.Kind {
	case mantaartifact.StepReturn:
		return nil, nil
	case mantaartifact.StepAlias:
		if len(step.Inputs) != 1 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("alias expects 1 input and 1 output")
		}
		input, err := gradInput(env, step.Inputs[0])
		if err != nil {
			return nil, err
		}
		return []*GradTensor{aliasGradTensor(input)}, nil
	case mantaartifact.StepConv2D:
		if len(step.Inputs) < 2 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("conv2d expects at least 2 inputs and 1 output")
		}
		input, weight, bias, err := gradInput2Optional(env, step.Inputs)
		if err != nil {
			return nil, err
		}
		out, err := Conv2DGrad(input, weight, bias, step.Attributes)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepConv2DTrans:
		if len(step.Inputs) < 2 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("conv2d_transpose expects at least 2 inputs and 1 output")
		}
		input, weight, bias, err := gradInput2Optional(env, step.Inputs)
		if err != nil {
			return nil, err
		}
		out, err := Conv2DTransposeGrad(input, weight, bias, step.Attributes)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepGDN, mantaartifact.StepIGDN:
		if len(step.Inputs) < 1 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("%s expects at least 1 input and 1 output", step.Kind)
		}
		input, err := gradInput(env, step.Inputs[0])
		if err != nil {
			return nil, err
		}
		beta, gamma, err := optionalGradInputs(env, step.Inputs, 1, 2)
		if err != nil {
			return nil, err
		}
		out, err := GDNGrad(input, beta, gamma, step.Kind == mantaartifact.StepIGDN)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepTurboQEncode:
		if len(step.Inputs) != 1 || len(step.Outputs) != 2 {
			return nil, fmt.Errorf("turboquant_encode expects 1 input and 2 outputs")
		}
		input, err := gradInput(env, step.Inputs[0])
		if err != nil {
			return nil, err
		}
		coords, norms, err := TurboQuantEncodeGrad(input, step.Attributes)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{coords, norms}, nil
	case mantaartifact.StepTurboQDecode:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("turboquant_decode expects 2 inputs and 1 output")
		}
		coords, err := gradInput(env, step.Inputs[0])
		if err != nil {
			return nil, err
		}
		norms, err := gradInput(env, step.Inputs[1])
		if err != nil {
			return nil, err
		}
		out, err := TurboQuantDecodeGrad(coords, norms, step.Attributes)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepCrossEntropy:
		if len(step.Inputs) < 1 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("cross_entropy_factorized expects at least 1 input and 1 output")
		}
		codes, err := gradInput(env, step.Inputs[0])
		if err != nil {
			return nil, err
		}
		logits, _, err := optionalGradInputs(env, step.Inputs, 1, -1)
		if err != nil {
			return nil, err
		}
		out, err := CrossEntropyFactorizedGrad(codes, logits, step.Attributes)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepMSELoss:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("mse_loss expects 2 inputs and 1 output")
		}
		lhs, rhs, err := gradInput2(env, step.Inputs)
		if err != nil {
			return nil, err
		}
		out, err := MSELossGrad(lhs, rhs)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepMSSSIMLoss:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("ms_ssim_loss expects 2 inputs and 1 output")
		}
		lhs, rhs, err := gradInput2(env, step.Inputs)
		if err != nil {
			return nil, err
		}
		out, err := MSSSIMLossGrad(lhs, rhs)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepScalarAdd:
		inputs := make([]*GradTensor, 0, len(step.Inputs))
		for _, name := range step.Inputs {
			input, err := gradInput(env, name)
			if err != nil {
				return nil, err
			}
			inputs = append(inputs, input)
		}
		out, err := ScalarAddGrad(inputs...)
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	case mantaartifact.StepRDLoss:
		if len(step.Inputs) != 2 || len(step.Outputs) != 1 {
			return nil, fmt.Errorf("rate_distortion_loss expects 2 inputs and 1 output")
		}
		distortion, rate, err := gradInput2(env, step.Inputs)
		if err != nil {
			return nil, err
		}
		out, err := RateDistortionLossGrad(distortion, rate, attrFloat(step.Attributes, "lambda", 1))
		if err != nil {
			return nil, err
		}
		return []*GradTensor{out}, nil
	default:
		return nil, fmt.Errorf("autograd does not support step kind %q", step.Kind)
	}
}

// Conv2DGrad applies conv2d and records gradients for input, weight, and bias.
func Conv2DGrad(input, weight, bias *GradTensor, attrs map[string]string) (*GradTensor, error) {
	if input == nil || weight == nil {
		return nil, fmt.Errorf("conv2d autograd expects input and weight")
	}
	out, err := conv2DTensor(input.Value, weight.Value, valueOrNil(bias), attrs)
	if err != nil {
		return nil, err
	}
	parents := compactParents(input, weight, bias)
	node := gradOpTensor("conv2d", out, parents, func(self *GradTensor) error {
		gradIn, gradW, gradB, err := conv2DBackwardTensors(input.Value, weight.Value, valueOrNil(bias), self.Grad, attrs)
		if err != nil {
			return err
		}
		if err := accumulateGrad(input, gradIn); err != nil {
			return err
		}
		if err := accumulateGrad(weight, gradW); err != nil {
			return err
		}
		if bias != nil {
			return accumulateGrad(bias, gradB)
		}
		return nil
	})
	return node, nil
}

// Conv2DTransposeGrad applies conv2d_transpose and records gradients.
func Conv2DTransposeGrad(input, weight, bias *GradTensor, attrs map[string]string) (*GradTensor, error) {
	if input == nil || weight == nil {
		return nil, fmt.Errorf("conv2d_transpose autograd expects input and weight")
	}
	out, err := conv2DTransposeTensor(input.Value, weight.Value, valueOrNil(bias), attrs)
	if err != nil {
		return nil, err
	}
	parents := compactParents(input, weight, bias)
	node := gradOpTensor("conv2d_transpose", out, parents, func(self *GradTensor) error {
		gradIn, gradW, gradB, err := conv2DTransposeBackwardTensors(input.Value, weight.Value, valueOrNil(bias), self.Grad, attrs)
		if err != nil {
			return err
		}
		if err := accumulateGrad(input, gradIn); err != nil {
			return err
		}
		if err := accumulateGrad(weight, gradW); err != nil {
			return err
		}
		if bias != nil {
			return accumulateGrad(bias, gradB)
		}
		return nil
	})
	return node, nil
}

// GDNGrad applies GDN or IGDN and records gradients for input, beta, and gamma.
func GDNGrad(input, beta, gamma *GradTensor, inverse bool) (*GradTensor, error) {
	if input == nil {
		return nil, fmt.Errorf("gdn autograd expects input")
	}
	out, err := gdnTensor(input.Value, valueOrNil(beta), valueOrNil(gamma), inverse)
	if err != nil {
		return nil, err
	}
	name := "gdn"
	if inverse {
		name = "igdn"
	}
	parents := compactParents(input, beta, gamma)
	node := gradOpTensor(name, out, parents, func(self *GradTensor) error {
		gradIn, gradBeta, gradGamma, err := gdnBackwardTensors(input.Value, valueOrNil(beta), valueOrNil(gamma), self.Grad, inverse)
		if err != nil {
			return err
		}
		if err := accumulateGrad(input, gradIn); err != nil {
			return err
		}
		if beta != nil {
			if err := accumulateGrad(beta, gradBeta); err != nil {
				return err
			}
		}
		if gamma != nil {
			return accumulateGrad(gamma, gradGamma)
		}
		return nil
	})
	return node, nil
}

// TurboQuantEncodeGrad applies TurboQuant encode. The backward path implements
// the v1 STE by passing coordinate gradients back to the input and mapping
// q_norm gradients onto the input vector magnitude.
func TurboQuantEncodeGrad(input *GradTensor, attrs map[string]string) (*GradTensor, *GradTensor, error) {
	if input == nil {
		return nil, nil, fmt.Errorf("turboquant_encode autograd expects input")
	}
	coords, norms, err := turboQuantEncodeTensor(input.Value, attrs)
	if err != nil {
		return nil, nil, err
	}
	coordNode := gradOpTensor("turboquant_encode.coords", coords, []*GradTensor{input}, func(self *GradTensor) error {
		return accumulateGrad(input, self.Grad)
	})
	normNode := gradOpTensor("turboquant_encode.norms", norms, []*GradTensor{input}, func(self *GradTensor) error {
		grad, err := turboQuantNormSTEGrad(input.Value, self.Grad)
		if err != nil {
			return err
		}
		return accumulateGrad(input, grad)
	})
	return coordNode, normNode, nil
}

// TurboQuantDecodeGrad applies TurboQuant decode. The backward path is the
// matching STE half: output gradient is copied to coordinate gradients.
func TurboQuantDecodeGrad(coords, norms *GradTensor, attrs map[string]string) (*GradTensor, error) {
	if coords == nil || norms == nil {
		return nil, fmt.Errorf("turboquant_decode autograd expects coords and norms")
	}
	out, err := turboQuantDecodeTensor(coords.Value, norms.Value, attrs)
	if err != nil {
		return nil, err
	}
	node := gradOpTensor("turboquant_decode", out, []*GradTensor{coords, norms}, func(self *GradTensor) error {
		if err := accumulateGrad(coords, self.Grad); err != nil {
			return err
		}
		return accumulateGrad(norms, zeroGradLike(norms.Value))
	})
	return node, nil
}

// CrossEntropyFactorizedGrad applies the Mirage factorized entropy loss and
// records gradients for logits/parameters plus a finite-difference surrogate
// for code indices. The code-index surrogate is what lets the rate term reach
// the TurboQuant STE path and then the analysis network.
func CrossEntropyFactorizedGrad(codes, logits *GradTensor, attrs map[string]string) (*GradTensor, error) {
	if codes == nil {
		return nil, fmt.Errorf("cross_entropy_factorized autograd expects codes")
	}
	out, err := crossEntropyFactorizedTensor(codes.Value, valueOrNil(logits), attrs)
	if err != nil {
		return nil, err
	}
	parents := compactParents(codes, logits)
	node := gradOpTensor("cross_entropy_factorized", out, parents, func(self *GradTensor) error {
		scale := scalarGrad(self.Grad)
		codeGrad, err := crossEntropyCodeGrad(codes.Value, valueOrNil(logits), attrs, scale)
		if err != nil {
			return err
		}
		if err := accumulateGrad(codes, codeGrad); err != nil {
			return err
		}
		if logits == nil || logits.Value == nil {
			return nil
		}
		grad, err := crossEntropyLogitsGrad(codes.Value, logits.Value, attrs, scale)
		if err != nil {
			return err
		}
		return accumulateGrad(logits, grad)
	})
	return node, nil
}

// MSELossGrad applies mean squared error and records gradients for both inputs.
func MSELossGrad(lhs, rhs *GradTensor) (*GradTensor, error) {
	if lhs == nil || rhs == nil {
		return nil, fmt.Errorf("mse_loss autograd expects two tensors")
	}
	out, err := mseLossTensor(lhs.Value, rhs.Value)
	if err != nil {
		return nil, err
	}
	node := gradOpTensor("mse_loss", out, []*GradTensor{lhs, rhs}, func(self *GradTensor) error {
		scale := scalarGrad(self.Grad)
		gradLHS, gradRHS, err := mseBackwardTensors(lhs.Value, rhs.Value, scale)
		if err != nil {
			return err
		}
		if err := accumulateGrad(lhs, gradLHS); err != nil {
			return err
		}
		return accumulateGrad(rhs, gradRHS)
	})
	return node, nil
}

// MSSSIMLossGrad keeps the forward MS-SSIM loss in the graph. Backward support
// is deliberately rejected if this loss becomes the differentiated objective.
func MSSSIMLossGrad(lhs, rhs *GradTensor) (*GradTensor, error) {
	if lhs == nil || rhs == nil {
		return nil, fmt.Errorf("ms_ssim_loss autograd expects two tensors")
	}
	out, err := msSSIMLossTensor(lhs.Value, rhs.Value)
	if err != nil {
		return nil, err
	}
	node := gradOpTensor("ms_ssim_loss", out, []*GradTensor{lhs, rhs}, func(*GradTensor) error {
		return fmt.Errorf("ms_ssim_loss backward is not implemented")
	})
	return node, nil
}

// ScalarAddGrad applies scalar_add and propagates the scalar gradient to all
// scalar inputs.
func ScalarAddGrad(inputs ...*GradTensor) (*GradTensor, error) {
	tensors := make([]*Tensor, 0, len(inputs))
	for _, input := range inputs {
		if input == nil {
			return nil, fmt.Errorf("scalar_add autograd expects non-nil inputs")
		}
		tensors = append(tensors, input.Value)
	}
	out, err := scalarAddTensor(tensors...)
	if err != nil {
		return nil, err
	}
	node := gradOpTensor("scalar_add", out, compactParents(inputs...), func(self *GradTensor) error {
		scale := scalarGrad(self.Grad)
		for _, input := range inputs {
			if err := accumulateGrad(input, NewTensorF32([]int{1}, []float32{scale})); err != nil {
				return err
			}
		}
		return nil
	})
	return node, nil
}

// RateDistortionLossGrad applies distortion + lambda*rate.
func RateDistortionLossGrad(distortion, rate *GradTensor, lambda float64) (*GradTensor, error) {
	if distortion == nil || rate == nil {
		return nil, fmt.Errorf("rate_distortion_loss autograd expects distortion and rate")
	}
	out, err := rateDistortionLossTensor(distortion.Value, rate.Value, lambda)
	if err != nil {
		return nil, err
	}
	node := gradOpTensor("rate_distortion_loss", out, []*GradTensor{distortion, rate}, func(self *GradTensor) error {
		scale := scalarGrad(self.Grad)
		if err := accumulateGrad(distortion, NewTensorF32([]int{1}, []float32{scale})); err != nil {
			return err
		}
		return accumulateGrad(rate, NewTensorF32([]int{1}, []float32{float32(float64(scale) * lambda)}))
	})
	return node, nil
}

func turboQuantNormSTEGrad(input, normGrad *Tensor) (*Tensor, error) {
	if input == nil || normGrad == nil {
		return nil, fmt.Errorf("turboquant norm STE expects input and norm grad")
	}
	if len(input.Shape) != 4 || len(normGrad.Shape) != 3 {
		return nil, fmt.Errorf("turboquant norm STE expects input NCHW and norm grad NHW")
	}
	n, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	if normGrad.Shape[0] != n || normGrad.Shape[1] != height || normGrad.Shape[2] != width {
		return nil, fmt.Errorf("turboquant norm STE grad shape %v does not match input %v", normGrad.Shape, input.Shape)
	}
	grad := zeroGradLike(input)
	const eps = 1e-12
	qScale := 255.0 / (qNormLogMax - qNormLogMin)
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				g := float64(normGrad.F32[(b*height+y)*width+x])
				if g == 0 {
					continue
				}
				normSq := 0.0
				for c := 0; c < channels; c++ {
					v := float64(input.F32[offset4(input.Shape, b, c, y, x)])
					normSq += v * v
				}
				if normSq <= eps {
					share := float32(g * qScale / float64(channels))
					for c := 0; c < channels; c++ {
						grad.F32[offset4(grad.Shape, b, c, y, x)] += share
					}
					continue
				}
				scale := g * qScale / normSq
				for c := 0; c < channels; c++ {
					idx := offset4(input.Shape, b, c, y, x)
					grad.F32[idx] += float32(scale * float64(input.F32[idx]))
				}
			}
		}
	}
	return grad, nil
}

func conv2DBackwardTensors(input, weight, bias, gradOut *Tensor, attrs map[string]string) (*Tensor, *Tensor, *Tensor, error) {
	if input == nil || weight == nil || gradOut == nil {
		return nil, nil, nil, fmt.Errorf("conv2d backward expects input, weight, and grad output")
	}
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(gradOut.Shape) != 4 {
		return nil, nil, nil, fmt.Errorf("conv2d backward expects NCHW/OIHW tensors")
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	outC, inPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	groups := attrInt(attrs, "groups", 1)
	strideH, strideW := attrInt(attrs, "stride_h", attrInt(attrs, "stride", 1)), attrInt(attrs, "stride_w", attrInt(attrs, "stride", 1))
	padH, padW := attrInt(attrs, "pad_h", attrInt(attrs, "padding", 0)), attrInt(attrs, "pad_w", attrInt(attrs, "padding", 0))
	dilH, dilW := attrInt(attrs, "dilation_h", attrInt(attrs, "dilation", 1)), attrInt(attrs, "dilation_w", attrInt(attrs, "dilation", 1))
	if groups <= 0 || inPerGroup*groups != inC || outC%groups != 0 {
		return nil, nil, nil, fmt.Errorf("conv2d backward invalid grouped shape")
	}
	outH, outW := gradOut.Shape[2], gradOut.Shape[3]
	gradIn := zeroGradLike(input)
	gradW := zeroGradLike(weight)
	var gradB *Tensor
	if bias != nil {
		gradB = zeroGradLike(bias)
	}
	outPerGroup := outC / groups
	for b := 0; b < n; b++ {
		for g := 0; g < groups; g++ {
			for ocg := 0; ocg < outPerGroup; ocg++ {
				oc := g*outPerGroup + ocg
				for oy := 0; oy < outH; oy++ {
					for ox := 0; ox < outW; ox++ {
						goVal := gradOut.F32[offset4(gradOut.Shape, b, oc, oy, ox)]
						if gradB != nil {
							gradB.F32[oc] += goVal
						}
						for icg := 0; icg < inPerGroup; icg++ {
							ic := g*inPerGroup + icg
							for ky := 0; ky < kH; ky++ {
								iy := oy*strideH + ky*dilH - padH
								if iy < 0 || iy >= inH {
									continue
								}
								for kx := 0; kx < kW; kx++ {
									ix := ox*strideW + kx*dilW - padW
									if ix < 0 || ix >= inW {
										continue
									}
									wIdx := ((oc*inPerGroup+icg)*kH+ky)*kW + kx
									inIdx := offset4(input.Shape, b, ic, iy, ix)
									gradIn.F32[inIdx] += goVal * weight.F32[wIdx]
									gradW.F32[wIdx] += goVal * input.F32[inIdx]
								}
							}
						}
					}
				}
			}
		}
	}
	return gradIn, gradW, gradB, nil
}

func conv2DTransposeBackwardTensors(input, weight, bias, gradOut *Tensor, attrs map[string]string) (*Tensor, *Tensor, *Tensor, error) {
	if input == nil || weight == nil || gradOut == nil {
		return nil, nil, nil, fmt.Errorf("conv2d_transpose backward expects input, weight, and grad output")
	}
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(gradOut.Shape) != 4 {
		return nil, nil, nil, fmt.Errorf("conv2d_transpose backward expects NCHW/IOHW tensors")
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	weightInC, outPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	if weightInC != inC {
		return nil, nil, nil, fmt.Errorf("conv2d_transpose backward input/weight channels mismatch")
	}
	groups := attrInt(attrs, "groups", 1)
	if groups <= 0 || inC%groups != 0 {
		return nil, nil, nil, fmt.Errorf("conv2d_transpose backward invalid groups")
	}
	strideH, strideW := attrInt(attrs, "stride_h", attrInt(attrs, "stride", 1)), attrInt(attrs, "stride_w", attrInt(attrs, "stride", 1))
	padH, padW := attrInt(attrs, "pad_h", attrInt(attrs, "padding", 0)), attrInt(attrs, "pad_w", attrInt(attrs, "padding", 0))
	dilH, dilW := attrInt(attrs, "dilation_h", attrInt(attrs, "dilation", 1)), attrInt(attrs, "dilation_w", attrInt(attrs, "dilation", 1))
	outC, outH, outW := gradOut.Shape[1], gradOut.Shape[2], gradOut.Shape[3]
	if outC != outPerGroup*groups {
		return nil, nil, nil, fmt.Errorf("conv2d_transpose backward output channels mismatch")
	}
	gradIn := zeroGradLike(input)
	gradW := zeroGradLike(weight)
	var gradB *Tensor
	if bias != nil {
		gradB = zeroGradLike(bias)
	}
	inPerGroup := inC / groups
	for b := 0; b < n; b++ {
		for g := 0; g < groups; g++ {
			for icg := 0; icg < inPerGroup; icg++ {
				ic := g*inPerGroup + icg
				for iy := 0; iy < inH; iy++ {
					for ix := 0; ix < inW; ix++ {
						inVal := input.F32[offset4(input.Shape, b, ic, iy, ix)]
						for ocg := 0; ocg < outPerGroup; ocg++ {
							oc := g*outPerGroup + ocg
							for ky := 0; ky < kH; ky++ {
								oy := iy*strideH + ky*dilH - padH
								if oy < 0 || oy >= outH {
									continue
								}
								for kx := 0; kx < kW; kx++ {
									ox := ix*strideW + kx*dilW - padW
									if ox < 0 || ox >= outW {
										continue
									}
									goVal := gradOut.F32[offset4(gradOut.Shape, b, oc, oy, ox)]
									wIdx := ((ic*outPerGroup+ocg)*kH+ky)*kW + kx
									gradIn.F32[offset4(input.Shape, b, ic, iy, ix)] += goVal * weight.F32[wIdx]
									gradW.F32[wIdx] += goVal * inVal
								}
							}
						}
					}
				}
			}
		}
	}
	if gradB != nil {
		for b := 0; b < n; b++ {
			for oc := 0; oc < outC; oc++ {
				for y := 0; y < outH; y++ {
					for x := 0; x < outW; x++ {
						gradB.F32[oc] += gradOut.F32[offset4(gradOut.Shape, b, oc, y, x)]
					}
				}
			}
		}
	}
	return gradIn, gradW, gradB, nil
}

func gdnBackwardTensors(input, beta, gamma, gradOut *Tensor, inverse bool) (*Tensor, *Tensor, *Tensor, error) {
	if input == nil || gradOut == nil || len(input.Shape) != 4 || !sameTensorShape(input, gradOut) {
		return nil, nil, nil, fmt.Errorf("gdn backward expects matching NCHW tensors")
	}
	n, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	gradIn := zeroGradLike(input)
	var gradBeta *Tensor
	if beta != nil {
		gradBeta = zeroGradLike(beta)
	}
	var gradGamma *Tensor
	if gamma != nil {
		gradGamma = zeroGradLike(gamma)
	}
	for b := 0; b < n; b++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				sums := make([]float64, channels)
				for c := 0; c < channels; c++ {
					sum := float64(betaValue(beta, c))
					for j := 0; j < channels; j++ {
						v := float64(input.F32[offset4(input.Shape, b, j, y, x)])
						sum += float64(gammaValue(gamma, c, j, channels)) * v * v
					}
					if sum < 1e-12 {
						sum = 1e-12
					}
					sums[c] = sum
				}
				for c := 0; c < channels; c++ {
					xC := float64(input.F32[offset4(input.Shape, b, c, y, x)])
					goC := float64(gradOut.F32[offset4(gradOut.Shape, b, c, y, x)])
					sqrtS := math.Sqrt(sums[c])
					invSqrt := 1 / sqrtS
					for k := 0; k < channels; k++ {
						xK := float64(input.F32[offset4(input.Shape, b, k, y, x)])
						gck := float64(gammaValue(gamma, c, k, channels))
						partial := 0.0
						if c == k {
							if inverse {
								partial += sqrtS
							} else {
								partial += invSqrt
							}
						}
						if inverse {
							partial += xC * gck * xK * invSqrt
						} else {
							partial -= xC * gck * xK * invSqrt / sums[c]
						}
						gradIn.F32[offset4(gradIn.Shape, b, k, y, x)] += float32(goC * partial)
					}
					if gradBeta != nil {
						gradBeta.F32[c%len(gradBeta.F32)] += float32(goC * gdnBetaPartial(xC, invSqrt, inverse))
					}
					if gradGamma != nil {
						for j := 0; j < channels; j++ {
							xJ := float64(input.F32[offset4(input.Shape, b, j, y, x)])
							idx := gammaGradIndex(gradGamma, c, j, channels)
							if idx >= 0 {
								gradGamma.F32[idx] += float32(goC * gdnGammaPartial(xC, xJ, invSqrt, inverse))
							}
						}
					}
				}
			}
		}
	}
	return gradIn, gradBeta, gradGamma, nil
}

func mseBackwardTensors(lhs, rhs *Tensor, scale float32) (*Tensor, *Tensor, error) {
	if lhs == nil || rhs == nil || !lhs.EqualShape(rhs) || len(lhs.F32) != len(rhs.F32) {
		return nil, nil, fmt.Errorf("mse backward expects matching tensors")
	}
	gradLHS := zeroGradLike(lhs)
	gradRHS := zeroGradLike(rhs)
	if len(lhs.F32) == 0 {
		return gradLHS, gradRHS, nil
	}
	coef := 2 * scale / float32(len(lhs.F32))
	for i := range lhs.F32 {
		diff := lhs.F32[i] - rhs.F32[i]
		gradLHS.F32[i] = coef * diff
		gradRHS.F32[i] = -coef * diff
	}
	return gradLHS, gradRHS, nil
}

func crossEntropyLogitsGrad(codes, logits *Tensor, attrs map[string]string, scale float32) (*Tensor, error) {
	if codes == nil || logits == nil {
		return nil, fmt.Errorf("cross entropy backward expects codes and logits")
	}
	grad := zeroGradLike(logits)
	if attrs != nil && attrs["distribution"] == "log_normal" {
		return logNormalParamsGrad(codes, logits, attrs, scale)
	}
	levels := attrInt(attrs, "levels", 0)
	if levels <= 0 {
		if bits := attrInt(attrs, "bits", bitsForQTensor(codes)); bits > 0 {
			levels = 1 << bits
		} else {
			levels = 256
		}
	}
	bits := attrInt(attrs, "bits", bitsForQTensor(codes))
	switch factorizationAttr(attrs) {
	case "bit-plane":
		if bits <= 0 {
			return nil, fmt.Errorf("cross entropy bit-plane backward requires bits")
		}
		for offset, raw := range codes.F32 {
			idx := clampInt(int(math.Round(float64(raw))), 0, (1<<bits)-1)
			for bit := 0; bit < bits; bit++ {
				shift := bits - 1 - bit
				bitValue := (idx >> shift) & 1
				if ok := addBitPlaneGrad(grad, logits, codes.Shape, offset, bit, bitValue, bits, scale); !ok {
					addFallbackBitPlaneGrad(grad, logits, offset, bit, bitValue, bits, scale)
				}
			}
		}
	default:
		for offset, raw := range codes.F32 {
			idx := clampInt(int(math.Round(float64(raw))), 0, levels-1)
			if ok := addCategoricalGrad(grad, logits, codes.Shape, offset, idx, levels, attrs, scale); !ok {
				addFallbackCategoricalGrad(grad, logits, offset, idx, levels, scale)
			}
		}
	}
	return grad, nil
}

func crossEntropyCodeGrad(codes, logits *Tensor, attrs map[string]string, scale float32) (*Tensor, error) {
	if codes == nil {
		return nil, fmt.Errorf("cross entropy code backward expects codes")
	}
	if attrs != nil && attrs["distribution"] == "log_normal" {
		return logNormalCodeGrad(codes, logits, attrs, scale)
	}
	grad := zeroGradLike(codes)
	levels := attrInt(attrs, "levels", 0)
	if levels <= 0 {
		if bits := attrInt(attrs, "bits", bitsForQTensor(codes)); bits > 0 {
			levels = 1 << bits
		} else {
			levels = 256
		}
	}
	bits := attrInt(attrs, "bits", bitsForQTensor(codes))
	if levels <= 1 {
		return grad, nil
	}
	switch factorizationAttr(attrs) {
	case "bit-plane":
		if bits <= 0 {
			return nil, fmt.Errorf("cross entropy bit-plane code backward requires bits")
		}
		levels = 1 << bits
		for offset, raw := range codes.F32 {
			idx := clampInt(int(math.Round(float64(raw))), 0, levels-1)
			grad.F32[offset] = finiteDifferenceCodeLoss(func(sym int) float64 {
				return bitPlaneCodeLoss(logits, codes.Shape, offset, sym, bits)
			}, idx, levels, scale)
		}
	default:
		for offset, raw := range codes.F32 {
			idx := clampInt(int(math.Round(float64(raw))), 0, levels-1)
			grad.F32[offset] = finiteDifferenceCodeLoss(func(sym int) float64 {
				return categoricalCodeLoss(logits, codes.Shape, offset, sym, levels, attrs)
			}, idx, levels, scale)
		}
	}
	return grad, nil
}

func finiteDifferenceCodeLoss(lossAt func(int) float64, idx, levels int, scale float32) float32 {
	switch {
	case levels <= 1:
		return 0
	case idx <= 0:
		return float32(float64(scale) * (lossAt(1) - lossAt(0)))
	case idx >= levels-1:
		return float32(float64(scale) * (lossAt(levels-1) - lossAt(levels-2)))
	default:
		return float32(float64(scale) * 0.5 * (lossAt(idx+1) - lossAt(idx-1)))
	}
}

func categoricalCodeLoss(logits *Tensor, codeShape []int, offset, idx, levels int, attrs map[string]string) float64 {
	p := probabilityForCode(logits, codeShape, offset, idx, levels, attrs)
	if p < 1e-12 {
		p = 1e-12
	}
	return -math.Log2(p)
}

func bitPlaneCodeLoss(logits *Tensor, codeShape []int, offset, idx, bitWidth int) float64 {
	total := 0.0
	for bit := 0; bit < bitWidth; bit++ {
		shift := bitWidth - 1 - bit
		bitValue := (idx >> shift) & 1
		p := probabilityForBit(logits, codeShape, offset, bit, bitValue, bitWidth)
		if p < 1e-12 {
			p = 1e-12
		}
		total += -math.Log2(p)
	}
	return total
}

func logNormalParamsGrad(codes, params *Tensor, attrs map[string]string, scale float32) (*Tensor, error) {
	if len(codes.Shape) != 3 || len(params.Shape) != 4 {
		return nil, fmt.Errorf("log-normal backward expects codes NHW and params N2HW")
	}
	if params.Shape[0] != codes.Shape[0] || params.Shape[1] < 2 || params.Shape[2] != codes.Shape[1] || params.Shape[3] != codes.Shape[2] {
		return nil, fmt.Errorf("log-normal backward param shape %v does not match codes %v", params.Shape, codes.Shape)
	}
	grad := zeroGradLike(params)
	for i, raw := range codes.F32 {
		n, y, x := unpackOffset3(codes.Shape, i)
		muIdx := offset4(params.Shape, n, 0, y, x)
		sigmaIdx := offset4(params.Shape, n, 1, y, x)
		mu := float64(params.F32[muIdx])
		rawSigma := float64(params.F32[sigmaIdx])
		sigma := normSigma(rawSigma, attrs)
		if sigma <= 0 || math.IsNaN(sigma) || math.IsInf(sigma, 0) {
			return nil, fmt.Errorf("log-normal backward invalid sigma at offset %d", i)
		}
		sym := clampInt(int(math.Round(float64(raw))), 0, 255)
		lo, hi := qNormLogBounds(sym)
		a := (hi - mu) / sigma
		b := (lo - mu) / sigma
		p := normalCDF(a) - normalCDF(b)
		if p < 1e-12 {
			p = 1e-12
		}
		ln2 := math.Ln2
		dMu := (normalPDF(a) - normalPDF(b)) / (sigma * p * ln2)
		dSigma := (normalPDFTimesX(a) - normalPDFTimesX(b)) / (sigma * p * ln2)
		grad.F32[muIdx] += float32(float64(scale) * dMu)
		grad.F32[sigmaIdx] += float32(float64(scale) * dSigma * sigmaParamDerivative(rawSigma, sigma, attrs))
	}
	return grad, nil
}

func logNormalCodeGrad(codes, params *Tensor, attrs map[string]string, scale float32) (*Tensor, error) {
	if params == nil {
		return nil, fmt.Errorf("log-normal code backward expects params")
	}
	if len(codes.Shape) != 3 || len(params.Shape) != 4 {
		return nil, fmt.Errorf("log-normal code backward expects codes NHW and params N2HW")
	}
	if params.Shape[0] != codes.Shape[0] || params.Shape[1] < 2 || params.Shape[2] != codes.Shape[1] || params.Shape[3] != codes.Shape[2] {
		return nil, fmt.Errorf("log-normal code backward param shape %v does not match codes %v", params.Shape, codes.Shape)
	}
	grad := zeroGradLike(codes)
	for offset, raw := range codes.F32 {
		n, y, x := unpackOffset3(codes.Shape, offset)
		mu := float64(params.F32[offset4(params.Shape, n, 0, y, x)])
		rawSigma := float64(params.F32[offset4(params.Shape, n, 1, y, x)])
		sigma := normSigma(rawSigma, attrs)
		if sigma <= 0 || math.IsNaN(sigma) || math.IsInf(sigma, 0) {
			return nil, fmt.Errorf("log-normal code backward invalid sigma at offset %d", offset)
		}
		sym := clampInt(int(math.Round(float64(raw))), 0, 255)
		grad.F32[offset] = finiteDifferenceCodeLoss(func(candidate int) float64 {
			lo, hi := qNormLogBounds(candidate)
			p := normalCDF((hi-mu)/sigma) - normalCDF((lo-mu)/sigma)
			if p < 1e-12 {
				p = 1e-12
			}
			return -math.Log2(p)
		}, sym, 256, scale)
	}
	return grad, nil
}

func addCategoricalGrad(grad, logits *Tensor, codeShape []int, offset, target, levels int, attrs map[string]string, scale float32) bool {
	if attrs != nil && attrs["logits_layout"] == "nchw_alphabet" && len(codeShape) == 4 && len(logits.Shape) == 4 {
		n, c, h, w := unpackOffset4(codeShape, offset)
		if logits.Shape[0] == codeShape[0] && logits.Shape[2] == codeShape[2] && logits.Shape[3] == codeShape[3] && logits.Shape[1] >= (c+1)*levels {
			baseChannel := c * levels
			values := make([]float64, levels)
			for i := 0; i < levels; i++ {
				values[i] = float64(logits.F32[offset4(logits.Shape, n, baseChannel+i, h, w)])
			}
			probs := softmaxFloat64(values)
			for i := 0; i < levels; i++ {
				delta := probs[i]
				if i == target {
					delta -= 1
				}
				grad.F32[offset4(grad.Shape, n, baseChannel+i, h, w)] += float32(float64(scale) * delta / math.Ln2)
			}
			return true
		}
	}
	if len(logits.F32) == levels {
		addSoftmaxGradContiguous(grad.F32, logits.F32, 0, levels, target, scale)
		return true
	}
	if len(logits.F32) >= (offset+1)*levels {
		addSoftmaxGradContiguous(grad.F32, logits.F32, offset*levels, levels, target, scale)
		return true
	}
	return false
}

func addFallbackCategoricalGrad(grad, logits *Tensor, offset, target, levels int, scale float32) {
	if len(logits.F32) == 0 {
		return
	}
	idx := offset % len(logits.F32)
	p := float64(1 / (1 + math.Exp(-float64(logits.F32[idx]))))
	observed := 0.0
	if target != 0 {
		observed = 1
	}
	grad.F32[idx] += float32(float64(scale) * (p - observed) / math.Ln2)
}

func addBitPlaneGrad(grad, logits *Tensor, codeShape []int, offset, bit, bitValue, bitWidth int, scale float32) bool {
	if len(codeShape) == 4 && len(logits.Shape) == 4 {
		n, c, h, w := unpackOffset4(codeShape, offset)
		ch := c*bitWidth*2 + bit*2
		if logits.Shape[0] == codeShape[0] && logits.Shape[2] == codeShape[2] && logits.Shape[3] == codeShape[3] && logits.Shape[1] >= ch+2 {
			aIdx := offset4(logits.Shape, n, ch, h, w)
			bIdx := offset4(logits.Shape, n, ch+1, h, w)
			a := float64(logits.F32[aIdx])
			b := float64(logits.F32[bIdx])
			probs := softmaxFloat64([]float64{a, b})
			grad.F32[aIdx] += float32(float64(scale) * (probs[0] - boolToFloat(bitValue == 0)) / math.Ln2)
			grad.F32[bIdx] += float32(float64(scale) * (probs[1] - boolToFloat(bitValue == 1)) / math.Ln2)
			return true
		}
	}
	if len(logits.F32) == bitWidth*2 {
		addSoftmaxGradContiguous(grad.F32, logits.F32, bit*2, 2, bitValue, scale)
		return true
	}
	base := (offset*bitWidth + bit) * 2
	if len(logits.F32) >= base+2 {
		addSoftmaxGradContiguous(grad.F32, logits.F32, base, 2, bitValue, scale)
		return true
	}
	return false
}

func addFallbackBitPlaneGrad(grad, logits *Tensor, offset, bit, bitValue, bitWidth int, scale float32) {
	if len(logits.F32) == 0 {
		return
	}
	idx := (offset*bitWidth + bit) % len(logits.F32)
	p := float64(1 / (1 + math.Exp(-float64(logits.F32[idx]))))
	grad.F32[idx] += float32(float64(scale) * (p - float64(bitValue)) / math.Ln2)
}

func addSoftmaxGradContiguous(grad, logits []float32, base, count, target int, scale float32) {
	maxV := float64(logits[base])
	for i := 1; i < count; i++ {
		if v := float64(logits[base+i]); v > maxV {
			maxV = v
		}
	}
	sum := 0.0
	probs := make([]float64, count)
	for i := 0; i < count; i++ {
		probs[i] = math.Exp(float64(logits[base+i]) - maxV)
		sum += probs[i]
	}
	if sum == 0 {
		return
	}
	for i := 0; i < count; i++ {
		delta := probs[i] / sum
		if i == target {
			delta -= 1
		}
		grad[base+i] += float32(float64(scale) * delta / math.Ln2)
	}
}

func softmaxFloat64(values []float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	maxV := values[0]
	for _, v := range values[1:] {
		if v > maxV {
			maxV = v
		}
	}
	out := make([]float64, len(values))
	sum := 0.0
	for i, v := range values {
		out[i] = math.Exp(v - maxV)
		sum += out[i]
	}
	if sum == 0 {
		for i := range out {
			out[i] = 1 / float64(len(out))
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func gdnBetaPartial(xC, invSqrt float64, inverse bool) float64 {
	if inverse {
		return 0.5 * xC * invSqrt
	}
	return -0.5 * xC * invSqrt * invSqrt * invSqrt
}

func gdnGammaPartial(xC, xJ, invSqrt float64, inverse bool) float64 {
	return gdnBetaPartial(xC, invSqrt, inverse) * xJ * xJ
}

func gammaGradIndex(gamma *Tensor, c, j, channels int) int {
	if gamma == nil || len(gamma.F32) == 0 {
		return -1
	}
	if len(gamma.Shape) == 2 && gamma.Shape[0] == channels && gamma.Shape[1] == channels {
		return c*channels + j
	}
	if len(gamma.F32) == channels {
		if c == j {
			return c
		}
		return -1
	}
	return (c*channels + j) % len(gamma.F32)
}

func normalPDF(x float64) float64 {
	if math.IsInf(x, 0) {
		return 0
	}
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}

func normalPDFTimesX(x float64) float64 {
	if math.IsInf(x, 0) {
		return 0
	}
	return x * normalPDF(x)
}

func sigmaParamDerivative(raw, sigma float64, attrs map[string]string) float64 {
	if attrs == nil {
		return 1
	}
	switch attrs["sigma_parameter"] {
	case "softplus":
		if raw > 32 {
			return 1
		}
		return 1 / (1 + math.Exp(-raw))
	case "exp":
		return sigma
	default:
		return 1
	}
}

func gradOpTensor(name string, value *Tensor, parents []*GradTensor, backward func(*GradTensor) error) *GradTensor {
	requiresGrad := false
	for _, parent := range parents {
		if parent != nil && parent.RequiresGrad {
			requiresGrad = true
			break
		}
	}
	return &GradTensor{
		Name:         name,
		Value:        cloneOrNil(value),
		RequiresGrad: requiresGrad,
		parents:      parents,
		backward:     backward,
	}
}

func aliasGradTensor(input *GradTensor) *GradTensor {
	return gradOpTensor("alias", input.Value, []*GradTensor{input}, func(self *GradTensor) error {
		return accumulateGrad(input, self.Grad)
	})
}

func accumulateGrad(node *GradTensor, grad *Tensor) error {
	if node == nil || grad == nil || !node.RequiresGrad {
		return nil
	}
	if node.Value == nil {
		return fmt.Errorf("cannot accumulate gradient into nil value")
	}
	if !sameTensorShape(node.Value, grad) || len(grad.F32) != node.Value.Elements() {
		return fmt.Errorf("gradient shape %v does not match value shape %v", shapeOf(grad), node.Value.Shape)
	}
	if node.Grad == nil {
		node.Grad = zeroGradLike(node.Value)
	}
	for i, v := range grad.F32 {
		node.Grad.F32[i] += v
	}
	return nil
}

func topoGradTensors(root *GradTensor) []*GradTensor {
	var out []*GradTensor
	seen := map[*GradTensor]bool{}
	var visit func(*GradTensor)
	visit = func(node *GradTensor) {
		if node == nil || seen[node] {
			return
		}
		seen[node] = true
		for _, parent := range node.parents {
			visit(parent)
		}
		out = append(out, node)
	}
	visit(root)
	return out
}

func zeroGradLike(t *Tensor) *Tensor {
	if t == nil {
		return nil
	}
	return NewTensorF32(append([]int(nil), t.Shape...), make([]float32, t.Elements()))
}

func sameTensorShape(a, b *Tensor) bool {
	return a != nil && b != nil && slices.Equal(a.Shape, b.Shape)
}

func shapeOf(t *Tensor) []int {
	if t == nil {
		return nil
	}
	return t.Shape
}

func cloneOrNil(t *Tensor) *Tensor {
	if t == nil {
		return nil
	}
	return t.Clone()
}

func valueOrNil(node *GradTensor) *Tensor {
	if node == nil {
		return nil
	}
	return node.Value
}

func scalarGrad(grad *Tensor) float32 {
	if grad == nil || len(grad.F32) == 0 {
		return 0
	}
	return grad.F32[0]
}

func compactParents(nodes ...*GradTensor) []*GradTensor {
	out := make([]*GradTensor, 0, len(nodes))
	for _, node := range nodes {
		if node != nil {
			out = append(out, node)
		}
	}
	return out
}

func gradInput(env map[string]*GradTensor, name string) (*GradTensor, error) {
	value, ok := env[name]
	if !ok || value == nil {
		return nil, fmt.Errorf("missing input %q", name)
	}
	return value, nil
}

func gradInput2(env map[string]*GradTensor, names []string) (*GradTensor, *GradTensor, error) {
	if len(names) < 2 {
		return nil, nil, fmt.Errorf("expected two inputs")
	}
	left, err := gradInput(env, names[0])
	if err != nil {
		return nil, nil, err
	}
	right, err := gradInput(env, names[1])
	if err != nil {
		return nil, nil, err
	}
	return left, right, nil
}

func gradInput2Optional(env map[string]*GradTensor, names []string) (*GradTensor, *GradTensor, *GradTensor, error) {
	left, right, err := gradInput2(env, names)
	if err != nil {
		return nil, nil, nil, err
	}
	var third *GradTensor
	if len(names) > 2 {
		third, err = gradInput(env, names[2])
		if err != nil {
			return nil, nil, nil, err
		}
	}
	return left, right, third, nil
}

func optionalGradInputs(env map[string]*GradTensor, names []string, leftIdx, rightIdx int) (*GradTensor, *GradTensor, error) {
	var left, right *GradTensor
	var err error
	if leftIdx >= 0 && leftIdx < len(names) {
		left, err = gradInput(env, names[leftIdx])
		if err != nil {
			return nil, nil, err
		}
	}
	if rightIdx >= 0 && rightIdx < len(names) {
		right, err = gradInput(env, names[rightIdx])
		if err != nil {
			return nil, nil, err
		}
	}
	return left, right, nil
}

func boolToFloat(ok bool) float64 {
	if ok {
		return 1
	}
	return 0
}

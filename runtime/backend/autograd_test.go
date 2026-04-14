package backend

import (
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

func TestExecuteAutogradConv2DMSEMatchesFiniteDifference(t *testing.T) {
	mod := autogradConvModule()
	x := NewTensorF16([]int{1, 1, 3, 3}, []float32{
		0.2, -0.1, 0.4,
		0.7, -0.3, 0.5,
		-0.6, 0.8, 0.1,
	})
	target := NewTensorF16([]int{1, 1, 2, 2}, []float32{0.3, -0.2, 0.1, 0.05})
	weight := NewTensorF16([]int{1, 1, 2, 2}, []float32{0.15, -0.25, 0.35, 0.05})
	bias := NewTensorF16([]int{1}, []float32{0.02})

	result, err := ExecuteAutograd(mod, GradRequest{
		Entry:   "train",
		Inputs:  map[string]*Tensor{"x": x, "target": target},
		Weights: map[string]*Tensor{"w": weight, "b": bias},
	})
	if err != nil {
		t.Fatal(err)
	}
	gotW := result.Gradients["w"]
	gotB := result.Gradients["b"]
	if gotW == nil || gotB == nil {
		t.Fatalf("missing gradients: %+v", result.Gradients)
	}

	const eps = 1e-3
	for i := range weight.F32 {
		want := finiteDiffWeight(t, x, target, weight, bias, i, eps)
		if !close32(gotW.F32[i], want, 3e-3) {
			t.Fatalf("weight grad[%d] = %.6f want %.6f", i, gotW.F32[i], want)
		}
	}
	wantB := finiteDiffBias(t, x, target, weight, bias, eps)
	if !close32(gotB.F32[0], wantB, 3e-3) {
		t.Fatalf("bias grad = %.6f want %.6f", gotB.F32[0], wantB)
	}
	if result.Outputs["loss"] == nil {
		t.Fatalf("missing loss output")
	}
}

func TestAutogradTurboQuantStraightThroughEstimator(t *testing.T) {
	mod := mantaartifact.NewModule("tq_ste")
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "train",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "y", Type: autogradTensorType("f16", []string{"1", "4", "1", "1"})},
			{Name: "target", Type: autogradTensorType("f16", []string{"1", "4", "1", "1"})},
		},
		Outputs: []mantaartifact.ValueBinding{{Name: "loss", Type: autogradTensorType("f32", []string{"1"})}},
	}}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "coords", DType: "q2", Shape: []string{"1", "4", "1", "1"}},
		{Name: "norms", DType: "q_norm", Shape: []string{"1", "1", "1"}},
		{Name: "y_hat", DType: "f16", Shape: []string{"1", "4", "1", "1"}},
		{Name: "loss", DType: "f32", Shape: []string{"1"}},
	}
	mod.Steps = []mantaartifact.Step{
		{Entry: "train", Kind: mantaartifact.StepTurboQEncode, Name: "encode", Inputs: []string{"y"}, Outputs: []string{"coords", "norms"}, Attributes: map[string]string{"bits": "2", "seed": "17"}},
		{Entry: "train", Kind: mantaartifact.StepTurboQDecode, Name: "decode", Inputs: []string{"coords", "norms"}, Outputs: []string{"y_hat"}, Attributes: map[string]string{"bits": "2", "seed": "17"}},
		{Entry: "train", Kind: mantaartifact.StepMSELoss, Name: "mse", Inputs: []string{"y_hat", "target"}, Outputs: []string{"loss"}},
		{Entry: "train", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"loss"}},
	}

	result, err := ExecuteAutograd(mod, GradRequest{
		Entry:           "train",
		Inputs:          map[string]*Tensor{"y": NewTensorF16([]int{1, 4, 1, 1}, []float32{0.4, -0.2, 0.6, -0.1}), "target": NewTensorF16([]int{1, 4, 1, 1}, []float32{0, 0, 0, 0})},
		TrainableInputs: map[string]bool{"y": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	grad := result.InputGradients["y"]
	if grad == nil {
		t.Fatalf("missing y gradient")
	}
	if !sameInts(grad.Shape, []int{1, 4, 1, 1}) {
		t.Fatalf("y grad shape = %v", grad.Shape)
	}
	if allZero(grad.F32) {
		t.Fatalf("STE gradient is all zero: %v", grad.F32)
	}
}

func TestCrossEntropyAutogradUpdatesCategoricalLogits(t *testing.T) {
	codes := NewGradTensor("codes", NewTensorQ2([]int{2}, []float32{2, 2}), false)
	logits := NewGradTensor("logits", NewTensorF16([]int{4}, []float32{0.1, -0.2, 0.3, 0.0}), true)
	loss, err := CrossEntropyFactorizedGrad(codes, logits, map[string]string{"bits": "2"})
	if err != nil {
		t.Fatal(err)
	}
	if err := Backward(loss); err != nil {
		t.Fatal(err)
	}
	if logits.Grad == nil {
		t.Fatalf("missing logits gradient")
	}
	sum := float32(0)
	for _, v := range logits.Grad.F32 {
		sum += v
	}
	if math.Abs(float64(sum)) > 1e-5 {
		t.Fatalf("softmax gradient sum = %.8f want 0", sum)
	}
	if logits.Grad.F32[2] >= 0 {
		t.Fatalf("target logit gradient = %.6f want negative", logits.Grad.F32[2])
	}
}

func TestCrossEntropyAutogradPropagatesCodeSurrogate(t *testing.T) {
	codes := NewGradTensor("codes", NewTensorQ2([]int{3}, []float32{0, 1, 2}), true)
	logits := NewGradTensor("logits", NewTensorF16([]int{4}, []float32{3, 1, -1, -3}), true)
	loss, err := CrossEntropyFactorizedGrad(codes, logits, map[string]string{"bits": "2"})
	if err != nil {
		t.Fatal(err)
	}
	if err := Backward(loss); err != nil {
		t.Fatal(err)
	}
	if codes.Grad == nil {
		t.Fatalf("missing code gradient")
	}
	if allZero(codes.Grad.F32) {
		t.Fatalf("code gradient is all zero: %v", codes.Grad.F32)
	}
	if codes.Grad.F32[1] <= 0 {
		t.Fatalf("middle code gradient = %.6f want positive for decreasing logits", codes.Grad.F32[1])
	}
}

func TestRateDistortionLossAutogradUsesRateWeight(t *testing.T) {
	distortion := NewGradTensor("distortion", NewTensorF32([]int{1}, []float32{2}), true)
	rate := NewGradTensor("rate", NewTensorF32([]int{1}, []float32{5}), true)
	loss, err := RateDistortionLossGrad(distortion, rate, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if got := loss.Value.F32[0]; !close32(got, 3.25, 1e-6) {
		t.Fatalf("loss = %.6f want 3.25", got)
	}
	if err := Backward(loss); err != nil {
		t.Fatal(err)
	}
	if got := distortion.Grad.F32[0]; !close32(got, 1, 1e-6) {
		t.Fatalf("distortion grad = %.6f want 1", got)
	}
	if got := rate.Grad.F32[0]; !close32(got, 0.25, 1e-6) {
		t.Fatalf("rate grad = %.6f want 0.25", got)
	}
}

func TestRateGradientFlowsThroughTurboQuantEncodeToInput(t *testing.T) {
	mod := mantaartifact.NewModule("tq_rate_ste")
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "train",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "y", Type: autogradTensorType("f16", []string{"1", "4", "1", "1"})},
			{Name: "logits", Type: autogradTensorType("f16", []string{"4"})},
		},
		Outputs: []mantaartifact.ValueBinding{{Name: "loss", Type: autogradTensorType("f32", []string{"1"})}},
	}}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "coords", DType: "q2", Shape: []string{"1", "4", "1", "1"}},
		{Name: "norms", DType: "q_norm", Shape: []string{"1", "1", "1"}},
		{Name: "loss", DType: "f32", Shape: []string{"1"}},
	}
	mod.Steps = []mantaartifact.Step{
		{Entry: "train", Kind: mantaartifact.StepTurboQEncode, Name: "encode", Inputs: []string{"y"}, Outputs: []string{"coords", "norms"}, Attributes: map[string]string{"bits": "2", "seed": "17"}},
		{Entry: "train", Kind: mantaartifact.StepCrossEntropy, Name: "rate", Inputs: []string{"coords", "logits"}, Outputs: []string{"loss"}, Attributes: map[string]string{"bits": "2"}},
		{Entry: "train", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"loss"}},
	}

	result, err := ExecuteAutograd(mod, GradRequest{
		Entry: "train",
		Inputs: map[string]*Tensor{
			"y":      NewTensorF16([]int{1, 4, 1, 1}, []float32{0.4, -0.2, 0.6, -0.1}),
			"logits": NewTensorF16([]int{4}, []float32{3, 1, -1, -3}),
		},
		TrainableInputs: map[string]bool{"y": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	grad := result.InputGradients["y"]
	if grad == nil {
		t.Fatalf("missing y gradient")
	}
	if allZero(grad.F32) {
		t.Fatalf("rate gradient did not reach y through TurboQuant STE")
	}
}

func TestGDNAutogradInputMatchesFiniteDifference(t *testing.T) {
	input := NewGradTensor("x", NewTensorF16([]int{1, 2, 1, 1}, []float32{0.4, -0.3}), true)
	beta := NewGradTensor("beta", NewTensorF16([]int{2}, []float32{1.1, 0.9}), true)
	gamma := NewGradTensor("gamma", NewTensorF16([]int{2, 2}, []float32{0.2, 0.05, 0.03, 0.25}), true)
	out, err := GDNGrad(input, beta, gamma, false)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := MSELossGrad(out, NewGradTensor("target", NewTensorF16([]int{1, 2, 1, 1}, []float32{0, 0}), false))
	if err != nil {
		t.Fatal(err)
	}
	if err := Backward(loss); err != nil {
		t.Fatal(err)
	}
	if input.Grad == nil {
		t.Fatalf("missing input gradient")
	}
	const eps = 1e-3
	for i := range input.Value.F32 {
		want := finiteDiffGDNInput(t, input.Value, beta.Value, gamma.Value, i, eps)
		if !close32(input.Grad.F32[i], want, 2e-3) {
			t.Fatalf("gdn input grad[%d] = %.6f want %.6f", i, input.Grad.F32[i], want)
		}
	}
}

func autogradConvModule() *mantaartifact.Module {
	mod := mantaartifact.NewModule("conv_autograd")
	mod.Params = []mantaartifact.Param{
		{Name: "w", Type: autogradTensorType("f16", []string{"1", "1", "2", "2"}), Binding: "weights/w", Trainable: true},
		{Name: "b", Type: autogradTensorType("f16", []string{"1"}), Binding: "weights/b", Trainable: true},
	}
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "train",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "x", Type: autogradTensorType("f16", []string{"1", "1", "3", "3"})},
			{Name: "target", Type: autogradTensorType("f16", []string{"1", "1", "2", "2"})},
		},
		Outputs: []mantaartifact.ValueBinding{{Name: "loss", Type: autogradTensorType("f32", []string{"1"})}},
	}}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "y", DType: "f16", Shape: []string{"1", "1", "2", "2"}},
		{Name: "loss", DType: "f32", Shape: []string{"1"}},
	}
	mod.Steps = []mantaartifact.Step{
		{Entry: "train", Kind: mantaartifact.StepConv2D, Name: "conv", Inputs: []string{"x", "w", "b"}, Outputs: []string{"y"}},
		{Entry: "train", Kind: mantaartifact.StepMSELoss, Name: "mse", Inputs: []string{"y", "target"}, Outputs: []string{"loss"}},
		{Entry: "train", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"loss"}},
	}
	return mod
}

func finiteDiffWeight(t *testing.T, x, target, weight, bias *Tensor, idx int, eps float32) float32 {
	t.Helper()
	plus := weight.Clone()
	minus := weight.Clone()
	plus.F32[idx] += eps
	minus.F32[idx] -= eps
	return (convMSELoss(t, x, target, plus, bias) - convMSELoss(t, x, target, minus, bias)) / (2 * eps)
}

func finiteDiffBias(t *testing.T, x, target, weight, bias *Tensor, eps float32) float32 {
	t.Helper()
	plus := bias.Clone()
	minus := bias.Clone()
	plus.F32[0] += eps
	minus.F32[0] -= eps
	return (convMSELoss(t, x, target, weight, plus) - convMSELoss(t, x, target, weight, minus)) / (2 * eps)
}

func convMSELoss(t *testing.T, x, target, weight, bias *Tensor) float32 {
	t.Helper()
	y, err := conv2DTensor(x, weight, bias, nil)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := mseLossTensor(y, target)
	if err != nil {
		t.Fatal(err)
	}
	return loss.F32[0]
}

func finiteDiffGDNInput(t *testing.T, input, beta, gamma *Tensor, idx int, eps float32) float32 {
	t.Helper()
	plus := input.Clone()
	minus := input.Clone()
	plus.F32[idx] += eps
	minus.F32[idx] -= eps
	return (gdnMSELoss(t, plus, beta, gamma) - gdnMSELoss(t, minus, beta, gamma)) / (2 * eps)
}

func gdnMSELoss(t *testing.T, input, beta, gamma *Tensor) float32 {
	t.Helper()
	y, err := gdnTensor(input, beta, gamma, false)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := mseLossTensor(y, NewTensorF16(input.Shape, make([]float32, input.Elements())))
	if err != nil {
		t.Fatal(err)
	}
	return loss.F32[0]
}

func autogradTensorType(dtype string, shape []string) mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
	}
}

func close32(got, want, tol float32) bool {
	return float32(math.Abs(float64(got-want))) <= tol
}

func allZero(values []float32) bool {
	for _, v := range values {
		if v != 0 {
			return false
		}
	}
	return true
}

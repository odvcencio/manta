//go:build linux && cgo

package cuda

import (
	"math"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAConv2DBackwardMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{2, 4, 7, 8}, patternedFloats(2*4*7*8, 0.021))
	weight := backend.NewTensorF32([]int{6, 2, 3, 2}, patternedFloats(6*2*3*2, -0.033))
	bias := backend.NewTensorF32([]int{6}, patternedFloats(6, 0.017))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: map[string]string{
		"groups":   "2",
		"stride_h": "2",
		"stride_w": "1",
		"pad_h":    "1",
		"pad_w":    "0",
	}}
	cfg, ok := planBuiltinConv2D(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.027))

	wantIn, wantW, wantB := hostConv2DBackward(input, weight, bias, gradOut, cfg)
	gotIn, gotW, gotB, err := rt.runConv2DBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d backward: %v", err)
	}
	assertTensorClose(t, gotIn, input.Shape, wantIn)
	assertTensorClose(t, gotW, weight.Shape, wantW)
	assertTensorClose(t, gotB, bias.Shape, wantB)
}

func TestCUDAConv2DBackwardMatchesFiniteDifference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 2, 4, 5}, patternedFloats(1*2*4*5, 0.019))
	weight := backend.NewTensorF32([]int{3, 2, 2, 3}, patternedFloats(3*2*2*3, -0.023))
	bias := backend.NewTensorF32([]int{3}, patternedFloats(3, 0.011))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: map[string]string{
		"stride_h": "1",
		"stride_w": "2",
		"pad_h":    "1",
		"pad_w":    "1",
	}}
	cfg, ok := planBuiltinConv2D(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.031))
	gotIn, gotW, gotB, err := rt.runConv2DBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d backward: %v", err)
	}

	const eps = 1e-3
	assertFiniteDiffTensor(t, "input", gotIn, input, eps, func() float64 {
		return conv2DFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "weight", gotW, weight, eps, func() float64 {
		return conv2DFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "bias", gotB, bias, eps, func() float64 {
		return conv2DFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
}

func TestCUDAConv2DTransposeBackwardMatchesHost(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{2, 4, 4, 5}, patternedFloats(2*4*4*5, 0.018))
	weight := backend.NewTensorF32([]int{4, 3, 3, 2}, patternedFloats(4*3*3*2, -0.026))
	bias := backend.NewTensorF32([]int{6}, patternedFloats(6, 0.013))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: map[string]string{
		"groups":           "2",
		"stride_h":         "2",
		"stride_w":         "1",
		"pad_h":            "1",
		"pad_w":            "0",
		"output_padding_h": "1",
	}}
	cfg, ok := planBuiltinConv2DTranspose(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d_transpose should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.024))

	wantIn, wantW, wantB := hostConv2DTransposeBackward(input, weight, bias, gradOut, cfg)
	gotIn, gotW, gotB, err := rt.runConv2DTransposeBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d_transpose backward: %v", err)
	}
	assertTensorClose(t, gotIn, input.Shape, wantIn)
	assertTensorClose(t, gotW, weight.Shape, wantW)
	assertTensorClose(t, gotB, bias.Shape, wantB)
}

func TestCUDAConv2DTransposeBackwardMatchesFiniteDifference(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input := backend.NewTensorF32([]int{1, 2, 3, 4}, patternedFloats(1*2*3*4, 0.017))
	weight := backend.NewTensorF32([]int{2, 3, 2, 3}, patternedFloats(2*3*2*3, -0.021))
	bias := backend.NewTensorF32([]int{3}, patternedFloats(3, 0.009))
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2DTrans, Attributes: map[string]string{
		"stride_h":         "2",
		"stride_w":         "1",
		"pad_h":            "1",
		"pad_w":            "1",
		"output_padding_h": "1",
	}}
	cfg, ok := planBuiltinConv2DTranspose(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatal("conv2d_transpose should be supported")
	}
	gradOut := backend.NewTensorF32([]int{cfg.batches, cfg.outChannels, cfg.outHeight, cfg.outWidth}, patternedFloats(cfg.batches*cfg.outChannels*cfg.outHeight*cfg.outWidth, 0.029))
	gotIn, gotW, gotB, err := rt.runConv2DTransposeBackward(input, weight, bias, gradOut, cfg)
	if err != nil {
		t.Fatalf("run conv2d_transpose backward: %v", err)
	}

	const eps = 1e-3
	assertFiniteDiffTensor(t, "transpose input", gotIn, input, eps, func() float64 {
		return conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "transpose weight", gotW, weight, eps, func() float64 {
		return conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
	assertFiniteDiffTensor(t, "transpose bias", gotB, bias, eps, func() float64 {
		return conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut, cfg)
	})
}

func TestCUDAGDNBackwardMatchesHost(t *testing.T) {
	runCUDAGDNBackwardMatchesHost(t, false)
}

func TestCUDAIGDNBackwardMatchesHost(t *testing.T) {
	runCUDAGDNBackwardMatchesHost(t, true)
}

func TestCUDAGDNBackwardMatchesFiniteDifference(t *testing.T) {
	runCUDAGDNBackwardMatchesFiniteDifference(t, false)
}

func TestCUDAIGDNBackwardMatchesFiniteDifference(t *testing.T) {
	runCUDAGDNBackwardMatchesFiniteDifference(t, true)
}

func TestCUDAImageGradAcceleratorAutogradMatchesReference(t *testing.T) {
	accel, kind, err := backend.NewPreferredImageGradAccelerator(mantaartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("image grad accelerator: %v", err)
	}
	if accel == nil {
		t.Skip("no cuda image grad accelerator available")
	}
	defer accel.Close()
	if kind != mantaartifact.BackendCUDA {
		t.Fatalf("accelerator backend = %q, want cuda", kind)
	}

	mod := imageGradAutogradModule()
	inputs := imageGradAutogradInputs()
	weights := imageGradAutogradWeights()
	want, err := backend.ExecuteAutograd(mod, backend.GradRequest{
		Entry:   "train",
		Inputs:  cloneTensorMap(inputs),
		Weights: cloneTensorMap(weights),
	})
	if err != nil {
		t.Fatalf("reference autograd: %v", err)
	}
	got, err := backend.ExecuteAutograd(mod, backend.GradRequest{
		Entry:                "train",
		Inputs:               cloneTensorMap(inputs),
		Weights:              cloneTensorMap(weights),
		ImageGradAccelerator: accel,
	})
	if err != nil {
		t.Fatalf("cuda-accelerated autograd: %v", err)
	}
	assertTensorClose(t, got.Outputs["loss"], []int{1}, want.Outputs["loss"].F32)
	for name, wantGrad := range want.Gradients {
		gotGrad := got.Gradients[name]
		if gotGrad == nil {
			t.Fatalf("missing gradient %q", name)
		}
		assertTensorClose(t, gotGrad, wantGrad.Shape, wantGrad.F32)
	}
}

func TestCUDAImageGradAcceleratorAutogradFanInMatchesReference(t *testing.T) {
	accel, kind, err := backend.NewPreferredImageGradAccelerator(mantaartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("image grad accelerator: %v", err)
	}
	if accel == nil {
		t.Skip("no cuda image grad accelerator available")
	}
	defer accel.Close()
	if kind != mantaartifact.BackendCUDA {
		t.Fatalf("accelerator backend = %q, want cuda", kind)
	}

	mod := imageGradAutogradFanInModule()
	inputs := imageGradAutogradFanInInputs()
	weights := imageGradAutogradFanInWeights()
	want, err := backend.ExecuteAutograd(mod, backend.GradRequest{
		Entry:   "train",
		Inputs:  cloneTensorMap(inputs),
		Weights: cloneTensorMap(weights),
	})
	if err != nil {
		t.Fatalf("reference autograd: %v", err)
	}
	got, err := backend.ExecuteAutograd(mod, backend.GradRequest{
		Entry:                "train",
		Inputs:               cloneTensorMap(inputs),
		Weights:              cloneTensorMap(weights),
		ImageGradAccelerator: accel,
	})
	if err != nil {
		t.Fatalf("cuda-accelerated autograd: %v", err)
	}
	assertTensorClose(t, got.Outputs["loss"], []int{1}, want.Outputs["loss"].F32)
	for name, wantGrad := range want.Gradients {
		gotGrad := got.Gradients[name]
		if gotGrad == nil {
			t.Fatalf("missing gradient %q", name)
		}
		assertTensorClose(t, gotGrad, wantGrad.Shape, wantGrad.F32)
	}
}

func TestCUDAImageGradAcceleratorKeepsForwardResident(t *testing.T) {
	accel, kind, err := backend.NewPreferredImageGradAccelerator(mantaartifact.BackendCUDA)
	if err != nil {
		t.Fatalf("image grad accelerator: %v", err)
	}
	if accel == nil {
		t.Skip("no cuda image grad accelerator available")
	}
	defer accel.Close()
	if kind != mantaartifact.BackendCUDA {
		t.Fatalf("accelerator backend = %q, want cuda", kind)
	}

	input := backend.NewTensorF32([]int{1, 2, 4, 5}, patternedFloats(1*2*4*5, 0.019))
	weight := backend.NewTensorF32([]int{3, 2, 2, 3}, patternedFloats(3*2*2*3, -0.023))
	bias := backend.NewTensorF32([]int{3}, patternedFloats(3, 0.011))
	attrs := map[string]string{"stride_h": "1", "stride_w": "1", "pad_h": "0", "pad_w": "1"}
	out, ok, err := accel.RunConv2D(input, weight, bias, attrs)
	if err != nil {
		t.Fatalf("run conv2d accelerator: %v", err)
	}
	if !ok {
		t.Fatalf("conv2d accelerator did not handle supported shape")
	}
	if out.Device != backend.DeviceCUDA || out.DevicePtr == 0 {
		t.Fatalf("accelerated conv2d output residency = (%v, %d), want CUDA ptr", out.Device, out.DevicePtr)
	}
	if len(out.F32) != 0 {
		t.Fatalf("accelerated conv2d output was materialized eagerly")
	}
	if err := accel.Materialize(out); err != nil {
		t.Fatalf("materialize output: %v", err)
	}
	step := mantaartifact.Step{Kind: mantaartifact.StepConv2D, Attributes: attrs}
	cfg, ok := planBuiltinConv2D(step, []*backend.Tensor{input, weight, bias})
	if !ok {
		t.Fatalf("conv2d plan rejected supported shape")
	}
	assertTensorClose(t, out, []int{1, 3, 3, 5}, hostConv2D(input, weight, bias, cfg))
}

func runCUDAGDNBackwardMatchesHost(t *testing.T, inverse bool) {
	t.Helper()
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input, beta, gamma, gradOut := gdnBackwardFixture()
	wantIn, wantBeta, wantGamma := hostGDNBackward(input, beta, gamma, gradOut, inverse)
	gotIn, gotBeta, gotGamma, err := rt.runGDNBackward(input, beta, gamma, gradOut, inverse)
	if err != nil {
		t.Fatalf("run gdn backward: %v", err)
	}
	assertTensorClose(t, gotIn, input.Shape, wantIn)
	assertTensorClose(t, gotBeta, beta.Shape, wantBeta)
	assertTensorClose(t, gotGamma, gamma.Shape, wantGamma)
}

func runCUDAGDNBackwardMatchesFiniteDifference(t *testing.T, inverse bool) {
	t.Helper()
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no cuda runtime available: %v", err)
	}
	if rt == nil {
		t.Skip("no cuda runtime available")
	}
	defer rt.close()

	input, beta, gamma, gradOut := gdnBackwardFixture()
	gotIn, gotBeta, gotGamma, err := rt.runGDNBackward(input, beta, gamma, gradOut, inverse)
	if err != nil {
		t.Fatalf("run gdn backward: %v", err)
	}

	const eps = 1e-3
	assertFiniteDiffTensor(t, "gdn input", gotIn, input, eps, func() float64 {
		return gdnFiniteDiffLoss(input, beta, gamma, gradOut, inverse)
	})
	assertFiniteDiffTensor(t, "gdn beta", gotBeta, beta, eps, func() float64 {
		return gdnFiniteDiffLoss(input, beta, gamma, gradOut, inverse)
	})
	assertFiniteDiffTensor(t, "gdn gamma", gotGamma, gamma, eps, func() float64 {
		return gdnFiniteDiffLoss(input, beta, gamma, gradOut, inverse)
	})
}

func patternedFloats(count int, scale float32) []float32 {
	values := make([]float32, count)
	for i := range values {
		values[i] = float32((i%23)-11) * scale
	}
	return values
}

func imageGradAutogradModule() *mantaartifact.Module {
	tensorType := func(dtype string, shape []string) mantaartifact.ValueType {
		return mantaartifact.ValueType{
			Kind:   mantaartifact.ValueTensor,
			Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
		}
	}
	mod := mantaartifact.NewModule("cuda_image_grad_autograd")
	mod.Params = []mantaartifact.Param{
		{Name: "w", Type: tensorType("f16", []string{"3", "2", "2", "2"}), Binding: "weights/w", Trainable: true},
		{Name: "b", Type: tensorType("f16", []string{"3"}), Binding: "weights/b", Trainable: true},
		{Name: "beta", Type: tensorType("f16", []string{"3"}), Binding: "weights/beta", Trainable: true},
		{Name: "gamma", Type: tensorType("f16", []string{"3", "3"}), Binding: "weights/gamma", Trainable: true},
		{Name: "wt", Type: tensorType("f16", []string{"3", "2", "2", "2"}), Binding: "weights/wt", Trainable: true},
		{Name: "bt", Type: tensorType("f16", []string{"2"}), Binding: "weights/bt", Trainable: true},
	}
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "train",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "x", Type: tensorType("f16", []string{"1", "2", "5", "6"})},
			{Name: "target", Type: tensorType("f16", []string{"1", "2", "5", "6"})},
		},
		Outputs: []mantaartifact.ValueBinding{{Name: "loss", Type: tensorType("f32", []string{"1"})}},
	}}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "y", DType: "f16", Shape: []string{"1", "3", "4", "5"}},
		{Name: "yg", DType: "f16", Shape: []string{"1", "3", "4", "5"}},
		{Name: "yhat", DType: "f16", Shape: []string{"1", "2", "5", "6"}},
		{Name: "loss", DType: "f32", Shape: []string{"1"}},
	}
	mod.Steps = []mantaartifact.Step{
		{Entry: "train", Kind: mantaartifact.StepConv2D, Name: "analysis", Inputs: []string{"x", "w", "b"}, Outputs: []string{"y"}},
		{Entry: "train", Kind: mantaartifact.StepGDN, Name: "gdn", Inputs: []string{"y", "beta", "gamma"}, Outputs: []string{"yg"}},
		{Entry: "train", Kind: mantaartifact.StepConv2DTrans, Name: "synthesis", Inputs: []string{"yg", "wt", "bt"}, Outputs: []string{"yhat"}},
		{Entry: "train", Kind: mantaartifact.StepMSELoss, Name: "mse", Inputs: []string{"yhat", "target"}, Outputs: []string{"loss"}},
		{Entry: "train", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"loss"}},
	}
	return mod
}

func imageGradAutogradFanInModule() *mantaartifact.Module {
	tensorType := func(dtype string, shape []string) mantaartifact.ValueType {
		return mantaartifact.ValueType{
			Kind:   mantaartifact.ValueTensor,
			Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
		}
	}
	mod := mantaartifact.NewModule("cuda_image_grad_autograd_fan_in")
	mod.Params = []mantaartifact.Param{
		{Name: "w0", Type: tensorType("f16", []string{"3", "2", "2", "2"}), Binding: "weights/w0", Trainable: true},
		{Name: "b0", Type: tensorType("f16", []string{"3"}), Binding: "weights/b0", Trainable: true},
		{Name: "w1", Type: tensorType("f16", []string{"2", "3", "1", "1"}), Binding: "weights/w1", Trainable: true},
		{Name: "b1", Type: tensorType("f16", []string{"2"}), Binding: "weights/b1", Trainable: true},
		{Name: "w2", Type: tensorType("f16", []string{"2", "3", "1", "1"}), Binding: "weights/w2", Trainable: true},
		{Name: "b2", Type: tensorType("f16", []string{"2"}), Binding: "weights/b2", Trainable: true},
	}
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "train",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "x", Type: tensorType("f16", []string{"1", "2", "5", "6"})},
			{Name: "target1", Type: tensorType("f16", []string{"1", "2", "4", "5"})},
			{Name: "target2", Type: tensorType("f16", []string{"1", "2", "4", "5"})},
		},
		Outputs: []mantaartifact.ValueBinding{{Name: "loss", Type: tensorType("f32", []string{"1"})}},
	}}
	mod.Buffers = []mantaartifact.Buffer{
		{Name: "y", DType: "f16", Shape: []string{"1", "3", "4", "5"}},
		{Name: "y1", DType: "f16", Shape: []string{"1", "2", "4", "5"}},
		{Name: "y2", DType: "f16", Shape: []string{"1", "2", "4", "5"}},
		{Name: "loss1", DType: "f32", Shape: []string{"1"}},
		{Name: "loss2", DType: "f32", Shape: []string{"1"}},
		{Name: "loss", DType: "f32", Shape: []string{"1"}},
	}
	mod.Steps = []mantaartifact.Step{
		{Entry: "train", Kind: mantaartifact.StepConv2D, Name: "stem", Inputs: []string{"x", "w0", "b0"}, Outputs: []string{"y"}},
		{Entry: "train", Kind: mantaartifact.StepConv2D, Name: "branch1", Inputs: []string{"y", "w1", "b1"}, Outputs: []string{"y1"}},
		{Entry: "train", Kind: mantaartifact.StepConv2D, Name: "branch2", Inputs: []string{"y", "w2", "b2"}, Outputs: []string{"y2"}},
		{Entry: "train", Kind: mantaartifact.StepMSELoss, Name: "mse1", Inputs: []string{"y1", "target1"}, Outputs: []string{"loss1"}},
		{Entry: "train", Kind: mantaartifact.StepMSELoss, Name: "mse2", Inputs: []string{"y2", "target2"}, Outputs: []string{"loss2"}},
		{Entry: "train", Kind: mantaartifact.StepScalarAdd, Name: "loss_sum", Inputs: []string{"loss1", "loss2"}, Outputs: []string{"loss"}},
		{Entry: "train", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"loss"}},
	}
	return mod
}

func imageGradAutogradInputs() map[string]*backend.Tensor {
	return map[string]*backend.Tensor{
		"x":      backend.NewTensorF16([]int{1, 2, 5, 6}, patternedFloats(1*2*5*6, 0.031)),
		"target": backend.NewTensorF16([]int{1, 2, 5, 6}, patternedFloats(1*2*5*6, -0.017)),
	}
}

func imageGradAutogradFanInInputs() map[string]*backend.Tensor {
	return map[string]*backend.Tensor{
		"x":       backend.NewTensorF16([]int{1, 2, 5, 6}, patternedFloats(1*2*5*6, 0.029)),
		"target1": backend.NewTensorF16([]int{1, 2, 4, 5}, patternedFloats(1*2*4*5, -0.015)),
		"target2": backend.NewTensorF16([]int{1, 2, 4, 5}, patternedFloats(1*2*4*5, 0.021)),
	}
}

func imageGradAutogradWeights() map[string]*backend.Tensor {
	return map[string]*backend.Tensor{
		"w":     backend.NewTensorF16([]int{3, 2, 2, 2}, patternedFloats(3*2*2*2, 0.023)),
		"b":     backend.NewTensorF16([]int{3}, []float32{0.01, -0.02, 0.03}),
		"beta":  backend.NewTensorF16([]int{3}, []float32{0.9, 1.1, 1.3}),
		"gamma": backend.NewTensorF16([]int{3, 3}, []float32{0.08, 0.01, 0.02, 0.015, 0.07, 0.025, 0.012, 0.018, 0.09}),
		"wt":    backend.NewTensorF16([]int{3, 2, 2, 2}, patternedFloats(3*2*2*2, -0.019)),
		"bt":    backend.NewTensorF16([]int{2}, []float32{0.015, -0.005}),
	}
}

func imageGradAutogradFanInWeights() map[string]*backend.Tensor {
	return map[string]*backend.Tensor{
		"w0": backend.NewTensorF16([]int{3, 2, 2, 2}, patternedFloats(3*2*2*2, 0.019)),
		"b0": backend.NewTensorF16([]int{3}, []float32{0.012, -0.018, 0.026}),
		"w1": backend.NewTensorF16([]int{2, 3, 1, 1}, patternedFloats(2*3, -0.024)),
		"b1": backend.NewTensorF16([]int{2}, []float32{0.007, -0.011}),
		"w2": backend.NewTensorF16([]int{2, 3, 1, 1}, patternedFloats(2*3, 0.027)),
		"b2": backend.NewTensorF16([]int{2}, []float32{-0.005, 0.013}),
	}
}

func cloneTensorMap(in map[string]*backend.Tensor) map[string]*backend.Tensor {
	out := make(map[string]*backend.Tensor, len(in))
	for name, tensor := range in {
		out[name] = tensor.Clone()
	}
	return out
}

func hostConv2DBackward(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DConfig) ([]float32, []float32, []float32) {
	gradIn := make([]float32, input.Elements())
	gradW := make([]float32, weight.Elements())
	var gradB []float32
	if bias != nil {
		gradB = make([]float32, bias.Elements())
	}
	for n := 0; n < cfg.batches; n++ {
		for group := 0; group < cfg.groups; group++ {
			for ocg := 0; ocg < cfg.outPerGroup; ocg++ {
				oc := group*cfg.outPerGroup + ocg
				for oy := 0; oy < cfg.outHeight; oy++ {
					for ox := 0; ox < cfg.outWidth; ox++ {
						goVal := gradOut.F32[offset4(gradOut.Shape, n, oc, oy, ox)]
						if gradB != nil {
							gradB[oc] += goVal
						}
						for icg := 0; icg < cfg.inPerGroup; icg++ {
							ic := group*cfg.inPerGroup + icg
							for ky := 0; ky < cfg.kernelH; ky++ {
								iy := oy*cfg.strideH + ky*cfg.dilationH - cfg.padH
								if iy < 0 || iy >= cfg.inHeight {
									continue
								}
								for kx := 0; kx < cfg.kernelW; kx++ {
									ix := ox*cfg.strideW + kx*cfg.dilationW - cfg.padW
									if ix < 0 || ix >= cfg.inWidth {
										continue
									}
									wIdx := ((oc*cfg.inPerGroup+icg)*cfg.kernelH+ky)*cfg.kernelW + kx
									inIdx := offset4(input.Shape, n, ic, iy, ix)
									gradIn[inIdx] += goVal * weight.F32[wIdx]
									gradW[wIdx] += goVal * input.F32[inIdx]
								}
							}
						}
					}
				}
			}
		}
	}
	return gradIn, gradW, gradB
}

func hostConv2DTransposeBackward(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DTransposeConfig) ([]float32, []float32, []float32) {
	gradIn := make([]float32, input.Elements())
	gradW := make([]float32, weight.Elements())
	var gradB []float32
	if bias != nil {
		gradB = make([]float32, bias.Elements())
	}
	for n := 0; n < cfg.batches; n++ {
		for group := 0; group < cfg.groups; group++ {
			for icg := 0; icg < cfg.inPerGroup; icg++ {
				ic := group*cfg.inPerGroup + icg
				for iy := 0; iy < cfg.inHeight; iy++ {
					for ix := 0; ix < cfg.inWidth; ix++ {
						inVal := input.F32[offset4(input.Shape, n, ic, iy, ix)]
						for ocg := 0; ocg < cfg.outPerGroup; ocg++ {
							oc := group*cfg.outPerGroup + ocg
							for ky := 0; ky < cfg.kernelH; ky++ {
								oy := iy*cfg.strideH + ky*cfg.dilationH - cfg.padH
								if oy < 0 || oy >= cfg.outHeight {
									continue
								}
								for kx := 0; kx < cfg.kernelW; kx++ {
									ox := ix*cfg.strideW + kx*cfg.dilationW - cfg.padW
									if ox < 0 || ox >= cfg.outWidth {
										continue
									}
									goVal := gradOut.F32[offset4(gradOut.Shape, n, oc, oy, ox)]
									wIdx := ((ic*cfg.outPerGroup+ocg)*cfg.kernelH+ky)*cfg.kernelW + kx
									gradIn[offset4(input.Shape, n, ic, iy, ix)] += goVal * weight.F32[wIdx]
									gradW[wIdx] += goVal * inVal
								}
							}
						}
					}
				}
			}
			if gradB != nil {
				for ocg := 0; ocg < cfg.outPerGroup; ocg++ {
					oc := group*cfg.outPerGroup + ocg
					for y := 0; y < cfg.outHeight; y++ {
						for x := 0; x < cfg.outWidth; x++ {
							gradB[oc] += gradOut.F32[offset4(gradOut.Shape, n, oc, y, x)]
						}
					}
				}
			}
		}
	}
	return gradIn, gradW, gradB
}

func conv2DFiniteDiffLoss(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DConfig) float64 {
	out := hostConv2D(input, weight, bias, cfg)
	loss := 0.0
	for i, v := range out {
		loss += float64(v * gradOut.F32[i])
	}
	return loss
}

func conv2DTransposeFiniteDiffLoss(input, weight, bias, gradOut *backend.Tensor, cfg cudaConv2DTransposeConfig) float64 {
	out := hostConv2DTranspose(input, weight, bias, cfg)
	loss := 0.0
	for i, v := range out {
		loss += float64(v * gradOut.F32[i])
	}
	return loss
}

func gdnBackwardFixture() (*backend.Tensor, *backend.Tensor, *backend.Tensor, *backend.Tensor) {
	input := backend.NewTensorF32([]int{2, 3, 4, 5}, patternedFloats(2*3*4*5, 0.037))
	beta := backend.NewTensorF32([]int{3}, []float32{0.8, 1.1, 1.4})
	gamma := backend.NewTensorF32([]int{3, 3}, []float32{
		0.090, 0.011, 0.017,
		0.013, 0.075, 0.019,
		0.007, 0.023, 0.082,
	})
	gradOut := backend.NewTensorF32([]int{2, 3, 4, 5}, patternedFloats(2*3*4*5, -0.029))
	return input, beta, gamma, gradOut
}

func hostGDNBackward(input, beta, gamma, gradOut *backend.Tensor, inverse bool) ([]float32, []float32, []float32) {
	_, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	batches := input.Shape[0]
	gradIn := make([]float32, input.Elements())
	gradBeta := make([]float32, beta.Elements())
	gradGamma := make([]float32, gamma.Elements())
	for n := 0; n < batches; n++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				sums := make([]float64, channels)
				for c := 0; c < channels; c++ {
					sum := float64(beta.F32[c])
					for j := 0; j < channels; j++ {
						v := float64(input.F32[offset4(input.Shape, n, j, y, x)])
						sum += float64(gamma.F32[c*channels+j]) * v * v
					}
					if sum < 1e-12 {
						sum = 1e-12
					}
					sums[c] = sum
				}
				for c := 0; c < channels; c++ {
					xC := float64(input.F32[offset4(input.Shape, n, c, y, x)])
					goC := float64(gradOut.F32[offset4(gradOut.Shape, n, c, y, x)])
					sqrtS := math.Sqrt(sums[c])
					invSqrt := 1 / sqrtS
					for k := 0; k < channels; k++ {
						xK := float64(input.F32[offset4(input.Shape, n, k, y, x)])
						gck := float64(gamma.F32[c*channels+k])
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
						gradIn[offset4(input.Shape, n, k, y, x)] += float32(goC * partial)
					}
					betaPartial := hostGDNBetaPartial(xC, invSqrt, inverse)
					gradBeta[c] += float32(goC * betaPartial)
					for j := 0; j < channels; j++ {
						xJ := float64(input.F32[offset4(input.Shape, n, j, y, x)])
						gradGamma[c*channels+j] += float32(goC * betaPartial * xJ * xJ)
					}
				}
			}
		}
	}
	return gradIn, gradBeta, gradGamma
}

func gdnFiniteDiffLoss(input, beta, gamma, gradOut *backend.Tensor, inverse bool) float64 {
	out := hostGDN(input, beta, gamma, inverse)
	loss := 0.0
	for i, v := range out {
		loss += float64(v * gradOut.F32[i])
	}
	return loss
}

func hostGDNBetaPartial(xC, invSqrt float64, inverse bool) float64 {
	if inverse {
		return 0.5 * xC * invSqrt
	}
	return -0.5 * xC * invSqrt * invSqrt * invSqrt
}

func assertFiniteDiffTensor(t *testing.T, name string, analytic, variable *backend.Tensor, eps float32, loss func() float64) {
	t.Helper()
	for i := range variable.F32 {
		original := variable.F32[i]
		variable.F32[i] = original + eps
		plus := loss()
		variable.F32[i] = original - eps
		minus := loss()
		variable.F32[i] = original
		numeric := (plus - minus) / float64(2*eps)
		got := float64(analytic.F32[i])
		if math.Abs(got-numeric) > 2e-2 {
			t.Fatalf("%s grad[%d] = %.6f, finite diff %.6f", name, i, got, numeric)
		}
	}
}

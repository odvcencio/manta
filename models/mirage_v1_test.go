package models

import (
	"context"
	"math"
	"strconv"
	"strings"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
)

func TestDefaultMirageV1ModuleRuns(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{})
	if err != nil {
		t.Fatal(err)
	}
	rt := mantaruntime.New(webgpu.New())
	prog, err := rt.Load(context.Background(), mod, mirageLoadOptions(t, mod)...)
	if err != nil {
		t.Fatal(err)
	}
	x := backend.NewTensorF16([]int{1, 3, 16, 16}, filled(1*3*16*16, 0.1))
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "train_step",
		Inputs: map[string]any{"x": x},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"loss", "x_hat", "c_z", "c_z_norms", "c_coords", "c_norms", "pi_logits", "norm_params", "rate", "rate_z", "rate_coords", "rate_norms", "mse", "ms_ssim"} {
		if _, ok := result.Outputs[name]; !ok {
			t.Fatalf("missing output %q", name)
		}
	}
	out := result.Outputs["x_hat"].Data.(*backend.Tensor)
	if !sameShape(out.Shape, []int{1, 3, 16, 16}) {
		t.Fatalf("x_hat shape: got %v", out.Shape)
	}

	analyzed, err := prog.Run(context.Background(), backend.Request{
		Entry:  "analyze",
		Inputs: map[string]any{"x": x},
	})
	if err != nil {
		t.Fatal(err)
	}
	hyper, err := prog.Run(context.Background(), backend.Request{
		Entry: "synthesize_hyperprior",
		Inputs: map[string]any{
			"c_z":       analyzed.Outputs["c_z"].Data.(*backend.Tensor),
			"c_z_norms": analyzed.Outputs["c_z_norms"].Data.(*backend.Tensor),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"pi_logits", "norm_params"} {
		if _, ok := hyper.Outputs[name]; !ok {
			t.Fatalf("missing hyper output %q", name)
		}
	}
	synthesized, err := prog.Run(context.Background(), backend.Request{
		Entry: "synthesize_image",
		Inputs: map[string]any{
			"c_coords": analyzed.Outputs["c_coords"].Data.(*backend.Tensor),
			"c_norms":  analyzed.Outputs["c_norms"].Data.(*backend.Tensor),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out := synthesized.Outputs["x_hat"].Data.(*backend.Tensor); !sameShape(out.Shape, []int{1, 3, 16, 16}) {
		t.Fatalf("synthesize_image x_hat shape: got %v", out.Shape)
	}
}

func TestDefaultMirageV1ModuleUsesBPPScaledRDLoss(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{
		ImageHeight: 16,
		ImageWidth:  32,
		Lambda:      0.01,
		LambdaSet:   true,
	})
	if err != nil {
		t.Fatal(err)
	}
	loss := findMirageStep(t, mod, "train_step", "loss")
	scale, err := strconv.ParseFloat(loss.Attributes["rate_scale"], 64)
	if err != nil {
		t.Fatalf("parse rate_scale: %v", err)
	}
	want := 1 / float64(16*32)
	if math.Abs(scale-want) > 1e-12 {
		t.Fatalf("rate_scale = %.12f want %.12f", scale, want)
	}
	if got := mod.Metadata["rate_unit"]; got != "bits_per_pixel" {
		t.Fatalf("rate_unit metadata = %v", got)
	}
}

func TestDefaultMirageV1ModuleAllowsExplicitZeroLambda(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{Lambda: 0, LambdaSet: true})
	if err != nil {
		t.Fatal(err)
	}
	loss := findMirageStep(t, mod, "train_step", "loss")
	if got := loss.Attributes["lambda"]; got != "0" {
		t.Fatalf("lambda attr = %q want 0", got)
	}
}

func TestDefaultMirageV1TrainStepUsesContinuousReconstructionPath(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{})
	if err != nil {
		t.Fatal(err)
	}
	trainSynthesis := findMirageStep(t, mod, "train_step", "synthesis_0")
	if got := trainSynthesis.Inputs[0]; got != "y" {
		t.Fatalf("train synthesis input = %q want y", got)
	}
	deploySynthesis := findMirageStep(t, mod, "synthesize_image", "synthesis_0")
	if got := deploySynthesis.Inputs[0]; got != "y_hat" {
		t.Fatalf("deploy synthesis input = %q want y_hat", got)
	}
}

func TestDefaultMirageV1BitPlaneModuleRuns(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{Factorization: "bit-plane"})
	if err != nil {
		t.Fatal(err)
	}
	if got := mod.Metadata["factorization"]; got != "bit-plane" {
		t.Fatalf("factorization metadata = %v", got)
	}
	rt := mantaruntime.New(webgpu.New())
	prog, err := rt.Load(context.Background(), mod, mirageLoadOptions(t, mod)...)
	if err != nil {
		t.Fatal(err)
	}
	x := backend.NewTensorF16([]int{1, 3, 16, 16}, filled(1*3*16*16, 0.1))
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "train_step",
		Inputs: map[string]any{"x": x},
	})
	if err != nil {
		t.Fatal(err)
	}
	logits := result.Outputs["pi_logits"].Data.(*backend.Tensor)
	if !sameShape(logits.Shape, []int{1, 4 * 4 * 2, 1, 1}) {
		t.Fatalf("bit-plane pi_logits shape: got %v", logits.Shape)
	}
}

func findMirageStep(t *testing.T, mod *mantaartifact.Module, entry, name string) mantaartifact.Step {
	t.Helper()
	for _, step := range mod.Steps {
		if step.Entry == entry && step.Name == name {
			return step
		}
	}
	t.Fatalf("missing step %s/%s", entry, name)
	return mantaartifact.Step{}
}

func TestDefaultMirageV1AutogradProducesTrainableGradients(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{})
	if err != nil {
		t.Fatal(err)
	}
	x := backend.NewTensorF16([]int{1, 3, 16, 16}, filled(1*3*16*16, 0.1))
	result, err := backend.ExecuteAutograd(mod, backend.GradRequest{
		Entry:   "train_step",
		Inputs:  map[string]*backend.Tensor{"x": x},
		Weights: mirageGradWeights(t, mod),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Outputs["loss"] == nil {
		t.Fatalf("missing loss output")
	}
	for _, name := range []string{"ga0_weight", "gs3_weight", "hs_logits_weight", "hs_norm_weight", "prior_z_logits"} {
		grad := result.Gradients[name]
		if grad == nil {
			t.Fatalf("missing gradient for %s", name)
		}
		if len(grad.F32) != grad.Elements() {
			t.Fatalf("%s gradient storage = %d want %d", name, len(grad.F32), grad.Elements())
		}
		if !hasFiniteNonZero(grad.F32) {
			t.Fatalf("%s gradient is not finite and non-zero: %v", name, grad.F32)
		}
	}
}

func TestDefaultMirageV1WebGPUSynthesizeUsesDecodeBuiltins(t *testing.T) {
	mod, err := DefaultMirageV1Module(MirageV1Config{})
	if err != nil {
		t.Fatal(err)
	}
	rt := mantaruntime.New(webgpu.New())
	prog, err := rt.Load(context.Background(), mod, mirageLoadOptions(t, mod)...)
	if err != nil {
		t.Fatal(err)
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry: "synthesize_image",
		Inputs: map[string]any{
			"c_coords": backend.NewTensorQ4([]int{1, 4, 1, 1}, []float32{1, 2, 3, 4}),
			"c_norms":  backend.NewTensorQNorm([]int{1, 1, 1}, []float32{128}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantVariants := map[string]bool{
		"manta_webgpu_turboquant_decode_nchw": false,
		"manta_webgpu_conv2d_transpose_nchw":  false,
		"manta_webgpu_igdn_nchw":              false,
	}
	for _, step := range result.Trace {
		if _, ok := wantVariants[step.Variant]; ok {
			wantVariants[step.Variant] = true
		}
	}
	for variant, seen := range wantVariants {
		if !seen {
			t.Fatalf("missing WebGPU builtin variant %s in trace %+v", variant, result.Trace)
		}
	}
	out := result.Outputs["x_hat"]
	if out.Metadata["execution_mode"] != "wgsl_host_reference" {
		t.Fatalf("x_hat execution_mode = %v", out.Metadata["execution_mode"])
	}
}

func mirageLoadOptions(t *testing.T, mod *mantaartifact.Module) []mantaruntime.LoadOption {
	t.Helper()
	opts := make([]mantaruntime.LoadOption, 0, len(mod.Params))
	for _, param := range mod.Params {
		shape := concreteShape(t, param.Type.Tensor.Shape)
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		value := float32(0.01)
		switch {
		case strings.Contains(param.Name, "beta"):
			value = 1
		case strings.Contains(param.Name, "gamma"):
			value = 0.001
		case strings.Contains(param.Name, "bias"):
			value = 0.05
		}
		opts = append(opts, mantaruntime.WithWeight(param.Name, backend.NewTensorF16(shape, filled(n, value))))
	}
	return opts
}

func mirageGradWeights(t *testing.T, mod *mantaartifact.Module) map[string]*backend.Tensor {
	t.Helper()
	weights := make(map[string]*backend.Tensor, len(mod.Params))
	for _, param := range mod.Params {
		shape := concreteShape(t, param.Type.Tensor.Shape)
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		value := float32(0.01)
		switch {
		case strings.Contains(param.Name, "beta"):
			value = 1
		case strings.Contains(param.Name, "gamma"):
			value = 0.001
		case strings.Contains(param.Name, "bias"):
			value = 0.05
		}
		weights[param.Name] = backend.NewTensorF16(shape, filled(n, value))
	}
	return weights
}

func concreteShape(t *testing.T, dims []string) []int {
	t.Helper()
	out := make([]int, len(dims))
	for i, dim := range dims {
		n, err := strconv.Atoi(dim)
		if err != nil {
			t.Fatalf("non-concrete Mirage test shape %q", dim)
		}
		out[i] = n
	}
	return out
}

func filled(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value + float32(i%7)*0.001
	}
	return out
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func hasFiniteNonZero(values []float32) bool {
	for _, value := range values {
		if value != 0 && !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0) {
			return true
		}
	}
	return false
}

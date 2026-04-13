package models

import (
	"context"
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

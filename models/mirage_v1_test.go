package models

import (
	"context"
	"testing"

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
	prog, err := rt.Load(context.Background(), mod,
		mantaruntime.WithWeight("ga_weight", backend.NewTensorF16([]int{4, 3, 3, 3}, filled(4*3*3*3, 0.01))),
		mantaruntime.WithWeight("ga_bias", backend.NewTensorF16([]int{4}, filled(4, 0.01))),
		mantaruntime.WithWeight("gdn_beta", backend.NewTensorF16([]int{4}, filled(4, 1))),
		mantaruntime.WithWeight("gdn_gamma", backend.NewTensorF16([]int{4, 4}, filled(4*4, 0.001))),
		mantaruntime.WithWeight("gs_weight", backend.NewTensorF16([]int{4, 3, 3, 3}, filled(4*3*3*3, 0.01))),
		mantaruntime.WithWeight("gs_bias", backend.NewTensorF16([]int{3}, filled(3, 0))),
		mantaruntime.WithWeight("prior_logits", backend.NewTensorF16([]int{16}, filled(16, 0))),
	)
	if err != nil {
		t.Fatal(err)
	}
	x := backend.NewTensorF16([]int{1, 3, 8, 8}, filled(1*3*8*8, 0.1))
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  "train_step",
		Inputs: map[string]any{"x": x},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"x_hat", "rate", "mse", "ms_ssim"} {
		if _, ok := result.Outputs[name]; !ok {
			t.Fatalf("missing output %q", name)
		}
	}
	out := result.Outputs["x_hat"].Data.(*backend.Tensor)
	if !sameShape(out.Shape, []int{1, 3, 8, 8}) {
		t.Fatalf("x_hat shape: got %v", out.Shape)
	}
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

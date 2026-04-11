//go:build linux && cgo

package cuda

import (
	"testing"

	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAActivationAcceleratorTracksStats(t *testing.T) {
	accelAny, err := NewActivationAccelerator()
	if err != nil {
		t.Fatalf("new activation accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda activation accelerator available")
	}
	accel, ok := accelAny.(*activationAccelerator)
	if !ok {
		t.Fatalf("activation accelerator type = %T, want *activationAccelerator", accelAny)
	}
	defer accel.Close()

	if _, err := accel.RunGELUBackwardMul(
		backend.NewTensorF32([]int{2, 2}, []float32{0.2, -0.1, 0.05, -0.25}),
		backend.NewTensorF32([]int{2, 2}, []float32{-1.0, -0.5, 0.5, 1.0}),
	); err != nil {
		t.Fatalf("gelu backward: %v", err)
	}
	if _, err := accel.RunSoftmaxBackwardRows(
		backend.NewTensorF32([]int{2, 2}, []float32{0.3, -0.1, -0.2, 0.4}),
		backend.NewTensorF32([]int{2, 2}, []float32{0.7, 0.3, 0.25, 0.75}),
	); err != nil {
		t.Fatalf("softmax backward: %v", err)
	}
	if _, err := accel.RunLayerNormBackwardRows(
		backend.NewTensorF32([]int{2, 3}, []float32{0.2, -0.1, 0.3, -0.4, 0.25, 0.15}),
		backend.NewTensorF32([]int{2, 3}, []float32{1.1, -0.8, -0.3, 0.2, 1.1, -1.3}),
		backend.NewTensorF32([]int{2, 3}, []float32{1.2, -0.4, 0.1, 0.5, 1.0, -0.5}),
	); err != nil {
		t.Fatalf("layernorm backward: %v", err)
	}
	if err := accel.BindTensor("pre_act", backend.NewTensorF32([]int{2, 2}, []float32{-1.0, -0.5, 0.5, 1.0})); err != nil {
		t.Fatalf("bind pre_act: %v", err)
	}
	if _, err := accel.RunGELUBackwardMulWithBoundPreAct(
		backend.NewTensorF32([]int{2, 2}, []float32{0.2, -0.1, 0.05, -0.25}),
		"pre_act",
	); err != nil {
		t.Fatalf("gelu backward with bound pre_act: %v", err)
	}
	if err := accel.BindTensor("probs", backend.NewTensorF32([]int{2, 2}, []float32{0.7, 0.3, 0.25, 0.75})); err != nil {
		t.Fatalf("bind probs: %v", err)
	}
	if _, err := accel.RunSoftmaxBackwardRowsWithBoundProbs(
		backend.NewTensorF32([]int{2, 2}, []float32{0.3, -0.1, -0.2, 0.4}),
		"probs",
	); err != nil {
		t.Fatalf("softmax backward with bound probs: %v", err)
	}
	if err := accel.BindTensor("normalized", backend.NewTensorF32([]int{2, 3}, []float32{1.1, -0.8, -0.3, 0.2, 1.1, -1.3})); err != nil {
		t.Fatalf("bind normalized: %v", err)
	}
	if err := accel.BindTensor("pre", backend.NewTensorF32([]int{2, 3}, []float32{1.2, -0.4, 0.1, 0.5, 1.0, -0.5})); err != nil {
		t.Fatalf("bind pre: %v", err)
	}
	if _, err := accel.RunLayerNormBackwardRowsWithBoundInputs(
		backend.NewTensorF32([]int{2, 3}, []float32{0.2, -0.1, 0.3, -0.4, 0.25, 0.15}),
		"normalized",
		"pre",
	); err != nil {
		t.Fatalf("layernorm backward with bound inputs: %v", err)
	}

	stats := accel.Stats()
	if stats.BindCalls != 4 {
		t.Fatalf("bind calls = %d, want 4", stats.BindCalls)
	}
	if stats.GELUBackwardCalls != 2 {
		t.Fatalf("gelu backward calls = %d, want 2", stats.GELUBackwardCalls)
	}
	if stats.SoftmaxBackwardCalls != 2 {
		t.Fatalf("softmax backward calls = %d, want 2", stats.SoftmaxBackwardCalls)
	}
	if stats.LayerNormBackwardCalls != 2 {
		t.Fatalf("layernorm backward calls = %d, want 2", stats.LayerNormBackwardCalls)
	}
	if stats.UploadedBytes <= 0 {
		t.Fatalf("uploaded bytes = %d, want positive", stats.UploadedBytes)
	}
	if stats.DownloadedBytes <= 0 {
		t.Fatalf("downloaded bytes = %d, want positive", stats.DownloadedBytes)
	}
	if stats.RunNanos <= 0 {
		t.Fatalf("run nanos = %d, want positive", stats.RunNanos)
	}
	if stats.BoundTensors != 4 {
		t.Fatalf("bound tensors = %d, want 4", stats.BoundTensors)
	}
}

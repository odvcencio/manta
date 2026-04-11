//go:build linux && cgo

package cuda

import (
	"testing"

	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAOptimizerAcceleratorKeepsResidentStateAcrossUpdates(t *testing.T) {
	accelAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new optimizer accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda optimizer accelerator available")
	}
	accel, ok := accelAny.(*optimizerAccelerator)
	if !ok {
		t.Fatalf("optimizer accelerator type = %T, want *optimizerAccelerator", accelAny)
	}
	defer accel.Close()

	param := backend.NewTensorF32([]int{2, 2}, []float32{
		0.5, -0.25,
		1.0, -0.75,
	})
	mom1 := backend.NewTensorF32([]int{2, 2}, []float32{
		0.1, -0.05,
		0.2, -0.1,
	})
	mom2 := backend.NewTensorF32([]int{2, 2}, []float32{
		0.01, 0.02,
		0.03, 0.04,
	})
	cfg := backend.OptimizerUpdateConfig{
		Optimizer:    "adamw",
		Step:         1,
		LearningRate: 0.01,
		WeightDecay:  0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		Scale:        1,
	}
	if err := accel.ApplyUpdate("projection", cfg, param, mom1, mom2, backend.NewTensorF32([]int{2, 2}, []float32{
		0.2, -0.1,
		0.05, -0.15,
	})); err != nil {
		t.Fatalf("first update: %v", err)
	}
	stateA, ok := accel.resident["projection"]
	if !ok {
		t.Fatal("expected resident optimizer state after first update")
	}
	if len(accel.resident) != 1 {
		t.Fatalf("resident state count = %d, want 1", len(accel.resident))
	}

	cfg.Step = 2
	if err := accel.ApplyUpdate("projection", cfg, param, mom1, mom2, backend.NewTensorF32([]int{2, 2}, []float32{
		-0.05, 0.15,
		0.1, -0.2,
	})); err != nil {
		t.Fatalf("second update: %v", err)
	}
	stateB, ok := accel.resident["projection"]
	if !ok {
		t.Fatal("expected resident optimizer state after second update")
	}
	if len(accel.resident) != 1 {
		t.Fatalf("resident state count = %d, want 1 after reuse", len(accel.resident))
	}
	if stateA.param != stateB.param || stateA.mom1 != stateB.mom1 || stateA.mom2 != stateB.mom2 {
		t.Fatalf("expected resident buffers to be reused, before=%+v after=%+v", stateA, stateB)
	}

	for i := range mom1.F32 {
		mom1.F32[i] = 0
	}
	for i := range mom2.F32 {
		mom2.F32[i] = 0
	}
	if err := accel.SyncState("projection", param, mom1, mom2, true); err != nil {
		t.Fatalf("sync resident state: %v", err)
	}
	allZero := true
	for i := range mom1.F32 {
		if mom1.F32[i] != 0 || mom2.F32[i] != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("expected sync to materialize resident moment state")
	}
	stateC, ok := accel.resident["projection"]
	if !ok {
		t.Fatal("expected resident optimizer state after sync")
	}
	if stateB.param != stateC.param || stateB.mom1 != stateC.mom1 || stateB.mom2 != stateC.mom2 {
		t.Fatalf("expected sync to preserve resident buffers, before=%+v after=%+v", stateB, stateC)
	}
	stats := accel.Stats()
	if stats.UpdateCalls != 2 {
		t.Fatalf("update calls = %d, want 2", stats.UpdateCalls)
	}
	if stats.SyncCalls != 1 {
		t.Fatalf("sync calls = %d, want 1", stats.SyncCalls)
	}
	if stats.UploadedBytes <= 0 {
		t.Fatalf("uploaded bytes = %d, want positive", stats.UploadedBytes)
	}
	if stats.DownloadedBytes <= 0 {
		t.Fatalf("downloaded bytes = %d, want positive", stats.DownloadedBytes)
	}
	if stats.ResidentParams != 1 {
		t.Fatalf("resident params = %d, want 1", stats.ResidentParams)
	}
}

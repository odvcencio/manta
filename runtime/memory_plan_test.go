package mantaruntime

import (
	"path/filepath"
	"testing"

	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
)

func TestDefaultMemoryPlanPath(t *testing.T) {
	got := DefaultMemoryPlanPath("/tmp/tiny_embed_pooled.mll")
	if want := "/tmp/tiny_embed_pooled.memory.mll"; got != want {
		t.Fatalf("memory plan path = %q, want %q", got, want)
	}
}

func TestMemoryPlanRoundTrip(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	weights := map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorQ8([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorQ8([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	}
	want := NewMemoryPlan(bundle.Artifact, weights, MemoryPlanOptions{
		DeviceBudgetBytes: 4,
		SharedHostWeights: true,
	})
	path := filepath.Join(t.TempDir(), "tiny.memory.mll")
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write memory plan: %v", err)
	}
	got, err := ReadMemoryPlanFile(path)
	if err != nil {
		t.Fatalf("read memory plan: %v", err)
	}
	if got.Version != want.Version {
		t.Fatalf("version = %q, want %q", got.Version, want.Version)
	}
	if got.ModuleName != want.ModuleName {
		t.Fatalf("module name = %q, want %q", got.ModuleName, want.ModuleName)
	}
	if len(got.Weights) != len(want.Weights) {
		t.Fatalf("weight count = %d, want %d", len(got.Weights), len(want.Weights))
	}
	if got.TotalWeightBytes != want.TotalWeightBytes {
		t.Fatalf("total bytes = %d, want %d", got.TotalWeightBytes, want.TotalWeightBytes)
	}
}

func TestNewMemoryPlanAssignsBudgetedResidency(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	weights := map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorQ8([]int{32, 16}, make([]float32, 32*16)),
		"projection":      backend.NewTensorQ8([]int{16, 16}, make([]float32, 16*16)),
	}
	plan := NewMemoryPlan(bundle.Artifact, weights, MemoryPlanOptions{
		DeviceBudgetBytes: 300,
		SharedHostWeights: true,
	})
	if len(plan.Weights) != 2 {
		t.Fatalf("weight count = %d, want 2", len(plan.Weights))
	}
	residency := map[string]MemoryResidency{}
	for _, item := range plan.Weights {
		residency[item.Name] = item.Residency
	}
	if residency["projection"] != ResidencyDeviceResident {
		t.Fatalf("projection residency = %q, want %q", residency["projection"], ResidencyDeviceResident)
	}
	if residency["token_embedding"] != ResidencyHostShared {
		t.Fatalf("token_embedding residency = %q, want %q", residency["token_embedding"], ResidencyHostShared)
	}
}

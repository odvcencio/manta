package barruntime

import (
	"path/filepath"
	"testing"

	"github.com/odvcencio/manta/runtime/backend"
)

func TestDefaultWeightFilePath(t *testing.T) {
	got := DefaultWeightFilePath("/tmp/tiny_embed_pooled.mll")
	if want := "/tmp/tiny_embed_pooled.weights.mll"; got != want {
		t.Fatalf("weight path = %q, want %q", got, want)
	}
}

func TestWeightFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.weights.mll")
	want := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorQ8([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorQ8([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	})
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write weight file: %v", err)
	}
	got, err := ReadWeightFile(path)
	if err != nil {
		t.Fatalf("read weight file: %v", err)
	}
	if got.Version != want.Version {
		t.Fatalf("version = %q, want %q", got.Version, want.Version)
	}
	if len(got.Weights) != 2 {
		t.Fatalf("weight count = %d, want 2", len(got.Weights))
	}
	assertTensorClose(t, got.Weights["token_embedding"], []int{3, 2}, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	assertTensorClose(t, got.Weights["projection"], []int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
}

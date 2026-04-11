package metal

import (
	"context"
	"testing"

	"github.com/odvcencio/barracuda/compiler"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func TestLoadWithCacheKeyReusesCompiledLoad(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed", Preset: compiler.PresetTinyEmbed})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	b := New()
	load := func() (*executor, error) {
		exec, err := b.LoadWithCacheKey(context.Background(), bundle.Artifact, map[string]backend.WeightBinding{
			"token_embedding": {Name: "token_embedding", Data: backend.NewTensorF16([]int{3, 2}, []float32{
				1, 0,
				0, 1,
				1, 1,
			})},
			"projection": {Name: "projection", Data: backend.NewTensorF16([]int{2, 2}, []float32{
				1, 0,
				0, 1,
			})},
		}, "tiny-embed-cache")
		if err != nil {
			return nil, err
		}
		out, ok := exec.(*executor)
		if !ok {
			t.Fatalf("executor type = %T, want *executor", exec)
		}
		return out, nil
	}

	first, err := load()
	if err != nil {
		t.Fatalf("first load: %v", err)
	}
	second, err := load()
	if err != nil {
		t.Fatalf("second load: %v", err)
	}

	if len(b.loadCache) != 1 {
		t.Fatalf("load cache size = %d, want 1", len(b.loadCache))
	}
	if b.cacheMisses != 1 {
		t.Fatalf("cache misses = %d, want 1", b.cacheMisses)
	}
	if b.cacheHits != 1 {
		t.Fatalf("cache hits = %d, want 1", b.cacheHits)
	}
	if first.device != second.device {
		t.Fatalf("expected cached device runtime reuse, first=%p second=%p", first.device, second.device)
	}
}

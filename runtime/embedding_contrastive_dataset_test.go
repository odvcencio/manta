package mantaruntime

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEmbeddingContrastiveExamplesFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pairs.jsonl")
	want := []EmbeddingContrastiveExample{
		{
			QueryTokens:    []int32{1, 2},
			PositiveTokens: []int32{2, 1},
			QueryMask:      []int32{1, 1},
			PositiveMask:   []int32{1, 1},
		},
		{
			QueryTokens:    []int32{0},
			PositiveTokens: []int32{1},
		},
	}
	if err := WriteEmbeddingContrastiveExamplesFile(path, want); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	got, err := ReadEmbeddingContrastiveExamplesFile(path)
	if err != nil {
		t.Fatalf("read dataset: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("dataset len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		assertInt32SliceEqual(t, got[i].QueryTokens, want[i].QueryTokens)
		assertInt32SliceEqual(t, got[i].PositiveTokens, want[i].PositiveTokens)
		assertInt32SliceEqual(t, got[i].QueryMask, want[i].QueryMask)
		assertInt32SliceEqual(t, got[i].PositiveMask, want[i].PositiveMask)
	}
}

func TestEmbeddingContrastiveExamplesFileRejectsInvalidRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.jsonl")
	if err := os.WriteFile(path, []byte("{\"query_tokens\":[],\"positive_tokens\":[1]}\n"), 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	if _, err := ReadEmbeddingContrastiveExamplesFile(path); err == nil {
		t.Fatal("expected invalid record error")
	}
}

func assertInt32SliceEqual(t *testing.T, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("slice[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

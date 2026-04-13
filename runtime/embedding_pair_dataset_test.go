package mantaruntime

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEmbeddingPairExamplesFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pairs.jsonl")
	want := []EmbeddingPairExample{
		{
			LeftTokens:  []int32{1, 2},
			RightTokens: []int32{3, 4},
			LeftMask:    []int32{1, 1},
			RightMask:   []int32{1, 0},
			Target:      -1,
		},
		{
			LeftTokens:  []int32{5},
			RightTokens: []int32{6},
			Target:      1,
		},
	}
	if err := WriteEmbeddingPairExamplesFile(path, want); err != nil {
		t.Fatalf("write pair examples: %v", err)
	}
	got, err := ReadEmbeddingPairExamplesFile(path)
	if err != nil {
		t.Fatalf("read pair examples: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("pair count = %d, want %d", len(got), len(want))
	}
	assertInt32SliceEqual(t, got[0].LeftTokens, want[0].LeftTokens)
	assertInt32SliceEqual(t, got[0].RightTokens, want[0].RightTokens)
	assertInt32SliceEqual(t, got[0].LeftMask, want[0].LeftMask)
	assertInt32SliceEqual(t, got[0].RightMask, want[0].RightMask)
	if got[0].Target != want[0].Target || got[1].Target != want[1].Target {
		t.Fatalf("targets = %v/%v, want %v/%v", got[0].Target, got[1].Target, want[0].Target, want[1].Target)
	}
}

func TestEmbeddingPairExamplesFileAcceptsLabelAlias(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pairs.jsonl")
	if err := os.WriteFile(path, []byte(`{"left_tokens":[1],"right_tokens":[2],"label":0}`), 0o644); err != nil {
		t.Fatalf("write pair examples: %v", err)
	}
	got, err := ReadEmbeddingPairExamplesFile(path)
	if err != nil {
		t.Fatalf("read pair examples: %v", err)
	}
	if got[0].Target != 0 {
		t.Fatalf("target = %v, want 0", got[0].Target)
	}
}

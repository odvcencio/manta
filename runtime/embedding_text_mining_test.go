package barruntime

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadEmbeddingTextCorpusFileRejectsEmptyCorpus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.txt")
	if err := os.WriteFile(path, []byte("\n \n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if _, err := ReadEmbeddingTextCorpusFile(path); err == nil {
		t.Fatal("expected empty corpus error")
	}
}

func TestMineEmbeddingTextDatasetsBuildsTrainAndEvalPairs(t *testing.T) {
	lines := []string{
		"alpha beta gamma. gamma delta epsilon.",
		"delta epsilon zeta. eta theta iota.",
		"kappa lambda mu. nu xi omicron.",
		"pi rho sigma. tau upsilon phi.",
	}
	train, eval, err := MineEmbeddingTextDatasets(lines, EmbeddingTextMiningConfig{
		MinChars:  5,
		EvalPairs: 2,
		Seed:      7,
	})
	if err != nil {
		t.Fatalf("mine datasets: %v", err)
	}
	if len(train) == 0 {
		t.Fatal("expected non-empty train set")
	}
	if len(eval) < 2 {
		t.Fatalf("expected labeled eval pairs, got %d", len(eval))
	}
	var positives, negatives int
	for _, example := range eval {
		switch {
		case example.Target > 0:
			positives++
		case example.Target == 0:
			negatives++
		default:
			t.Fatalf("unexpected eval target: %v", example.Target)
		}
		if example.Query == "" || example.Right == "" {
			t.Fatalf("expected populated eval example: %#v", example)
		}
	}
	if positives == 0 || negatives == 0 {
		t.Fatalf("expected both positive and negative eval pairs, got positives=%d negatives=%d", positives, negatives)
	}
}

func TestWriteEmbeddingTextPairExamplesFileRoundTrips(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pairs.jsonl")
	examples := []EmbeddingTextPairExample{
		{Query: "alpha beta", Right: "alpha beta", Target: 1},
		{Query: "alpha beta", Right: "gamma delta", Target: 0},
	}
	if err := WriteEmbeddingTextPairExamplesFile(path, examples); err != nil {
		t.Fatalf("write pairs: %v", err)
	}
	loaded, err := ReadEmbeddingTextPairExamplesFile(path)
	if err != nil {
		t.Fatalf("read pairs: %v", err)
	}
	if len(loaded) != len(examples) {
		t.Fatalf("expected %d pairs, got %d", len(examples), len(loaded))
	}
	for i := range examples {
		if loaded[i] != examples[i] {
			t.Fatalf("pair %d mismatch: got %#v want %#v", i, loaded[i], examples[i])
		}
	}
}

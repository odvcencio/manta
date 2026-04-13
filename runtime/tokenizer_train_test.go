package mantaruntime

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTrainTokenizerFromCorpus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("banana bandana banana band\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	file, err := TrainTokenizerFromCorpus(TokenizerTrainConfig{
		CorpusPath: path,
		VocabSize:  16,
		MinFreq:    2,
	})
	if err != nil {
		t.Fatalf("train tokenizer: %v", err)
	}
	if len(file.Tokens) == 0 {
		t.Fatal("expected non-empty tokenizer tokens")
	}
	if len(file.Tokens) > 16 {
		t.Fatalf("token count = %d, want <= 16", len(file.Tokens))
	}
	if len(file.Merges) == 0 {
		t.Fatal("expected at least one merge from repeated corpus")
	}
	if file.Tokens[0] != "[PAD]" || file.Tokens[1] != "[CLS]" || file.Tokens[2] != "[SEP]" || file.Tokens[3] != "[UNK]" {
		t.Fatalf("unexpected special token prefix: %v", file.Tokens[:4])
	}
}

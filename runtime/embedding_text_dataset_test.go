package barruntime

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadEmbeddingTextPairExamplesFileAcceptsCommonSchemas(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pairs.jsonl")
	data := "" +
		"{\"query\":\"alpha\",\"positive\":\"beta\"}\n" +
		"{\"query\":\"gamma\",\"document\":\"delta\",\"label\":1}\n" +
		"{\"left\":\"epsilon\",\"right\":\"zeta\",\"label\":0}\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	got, err := ReadEmbeddingTextPairExamplesFile(path)
	if err != nil {
		t.Fatalf("read text pair dataset: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("pair count = %d, want 3", len(got))
	}
	if got[0].Query != "alpha" || got[0].Right != "beta" || got[0].Target != 1 {
		t.Fatalf("pair[0] = %+v", got[0])
	}
	if got[1].Query != "gamma" || got[1].Right != "delta" || got[1].Target != 1 {
		t.Fatalf("pair[1] = %+v", got[1])
	}
	if got[2].Query != "epsilon" || got[2].Right != "zeta" || got[2].Target != 0 {
		t.Fatalf("pair[2] = %+v", got[2])
	}
}

func TestReadEmbeddingTextContrastiveExamplesFileSkipsNegativeLabels(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pairs.jsonl")
	data := "" +
		"{\"query\":\"alpha\",\"document\":\"beta\",\"label\":1}\n" +
		"{\"left\":\"gamma\",\"right\":\"delta\",\"label\":0}\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	got, err := ReadEmbeddingTextContrastiveExamplesFile(path)
	if err != nil {
		t.Fatalf("read contrastive text dataset: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("contrastive count = %d, want 1", len(got))
	}
	if got[0].Query != "alpha" || got[0].Positive != "beta" {
		t.Fatalf("contrastive[0] = %+v", got[0])
	}
}

func TestTokenizeEmbeddingTextPairExamplesPreservesTargets(t *testing.T) {
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	out, err := TokenizeEmbeddingTextPairExamples([]EmbeddingTextPairExample{
		{Query: "ab", Right: "ab", Target: 1},
		{Query: "cd", Right: "ab", Target: 0},
	}, tokenizer)
	if err != nil {
		t.Fatalf("tokenize pair examples: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("example count = %d, want 2", len(out))
	}
	assertInt32SliceEqual(t, out[0].LeftTokens, []int32{2, 4, 5, 3})
	assertInt32SliceEqual(t, out[0].RightTokens, []int32{2, 4, 5, 3})
	if out[0].Target != 1 || out[1].Target != 0 {
		t.Fatalf("targets = %v, %v; want 1, 0", out[0].Target, out[1].Target)
	}
}

func TestTokenizeEmbeddingTextPairExamplesCachesRepeatedTextsInternally(t *testing.T) {
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	cache := embeddingTextTokenCache{}
	out, err := tokenizeEmbeddingTextPairExamples([]EmbeddingTextPairExample{
		{Query: "ab", Right: "cd", Target: 1},
		{Query: "ab", Right: "cd", Target: 0},
		{Query: "cd", Right: "ab", Target: 1},
	}, tokenizer, cache, false)
	if err != nil {
		t.Fatalf("tokenize pair examples: %v", err)
	}
	if len(cache) != 2 {
		t.Fatalf("cache size = %d, want 2", len(cache))
	}
	if len(out) != 3 {
		t.Fatalf("example count = %d, want 3", len(out))
	}
	if &out[0].LeftTokens[0] != &out[1].LeftTokens[0] {
		t.Fatal("repeated left text did not reuse token slice")
	}
	if &out[0].RightTokens[0] != &out[1].RightTokens[0] {
		t.Fatal("repeated right text did not reuse token slice")
	}
	if &out[0].LeftTokens[0] != &out[2].RightTokens[0] {
		t.Fatal("reused text across left/right fields did not share token slice")
	}
	assertInt32SliceEqual(t, out[0].LeftTokens, []int32{2, 4, 5, 3})
	assertInt32SliceEqual(t, out[0].RightTokens, []int32{2, 6, 7, 3})
}

func TestTokenizeEmbeddingTextContrastiveExamplesCachesRepeatedTextsInternally(t *testing.T) {
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	cache := embeddingTextTokenCache{}
	out, err := tokenizeEmbeddingTextContrastiveExamples([]EmbeddingTextContrastiveExample{
		{Query: "ab", Positive: "cd"},
		{Query: "cd", Positive: "ab"},
		{Query: "ab", Positive: "ab"},
	}, tokenizer, cache, false)
	if err != nil {
		t.Fatalf("tokenize contrastive examples: %v", err)
	}
	if len(cache) != 2 {
		t.Fatalf("cache size = %d, want 2", len(cache))
	}
	if len(out) != 3 {
		t.Fatalf("example count = %d, want 3", len(out))
	}
	if &out[0].QueryTokens[0] != &out[1].PositiveTokens[0] {
		t.Fatal("reused text across query/positive fields did not share token slice")
	}
	if &out[0].QueryMask[0] != &out[2].PositiveMask[0] {
		t.Fatal("reused mask did not share cache slice")
	}
	assertInt32SliceEqual(t, out[0].QueryTokens, []int32{2, 4, 5, 3})
	assertInt32SliceEqual(t, out[0].PositiveTokens, []int32{2, 6, 7, 3})
}

func TestTokenizeEmbeddingTextExamplesClonePublicOutputs(t *testing.T) {
	tokenizer := newEmbeddingTextDatasetTestTokenizer(t)
	pairs, err := TokenizeEmbeddingTextPairExamples([]EmbeddingTextPairExample{
		{Query: "ab", Right: "cd", Target: 1},
		{Query: "ab", Right: "cd", Target: 0},
	}, tokenizer)
	if err != nil {
		t.Fatalf("tokenize pair examples: %v", err)
	}
	pairs[0].LeftTokens[1] = 99
	pairs[0].LeftMask[0] = 0
	if pairs[1].LeftTokens[1] == 99 {
		t.Fatal("public pair token output shared slices across examples")
	}
	if pairs[1].LeftMask[0] == 0 {
		t.Fatal("public pair mask output shared slices across examples")
	}

	contrastive, err := TokenizeEmbeddingTextContrastiveExamples([]EmbeddingTextContrastiveExample{
		{Query: "ab", Positive: "cd"},
		{Query: "ab", Positive: "cd"},
	}, tokenizer)
	if err != nil {
		t.Fatalf("tokenize contrastive examples: %v", err)
	}
	contrastive[0].QueryTokens[1] = 99
	contrastive[0].QueryMask[0] = 0
	if contrastive[1].QueryTokens[1] == 99 {
		t.Fatal("public contrastive token output shared slices across examples")
	}
	if contrastive[1].QueryMask[0] == 0 {
		t.Fatal("public contrastive mask output shared slices across examples")
	}
}

func newEmbeddingTextDatasetTestTokenizer(t *testing.T) *BPETokenizer {
	t.Helper()
	file := TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	tokenizer, err := NewBPETokenizer(file, TokenizerManifest{VocabSize: 8, MaxSequence: 8})
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	return tokenizer
}

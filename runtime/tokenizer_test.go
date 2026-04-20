package mantaruntime

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestDefaultTokenizerPath(t *testing.T) {
	got := DefaultTokenizerPath("/tmp/tiny_train_embed.mll")
	if want := "/tmp/tiny_train_embed.tokenizer.mll"; got != want {
		t.Fatalf("tokenizer path = %q, want %q", got, want)
	}
}

func TestTokenizerFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.tokenizer.mll")
	want := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "ab"},
		Merges:       []TokenizerMerge{{Left: "a", Right: "b"}},
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
	}
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	got, err := ReadTokenizerFile(path)
	if err != nil {
		t.Fatalf("read tokenizer: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokenizer mismatch:\nwant: %+v\ngot:  %+v", want, got)
	}
}

func TestBPETokenizerEncodeAppliesSpecialsAndTruncation(t *testing.T) {
	file := TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "ab"},
		Merges:  []TokenizerMerge{{Left: "a", Right: "b"}},
	}
	tokenizer, err := NewBPETokenizer(file, TokenizerManifest{VocabSize: 7, MaxSequence: 3})
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	ids, mask, err := tokenizer.Encode("ab")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	wantIDs := []int32{2, 6, 3}
	wantMask := []int32{1, 1, 1}
	assertInt32SliceEqual(t, ids, wantIDs)
	assertInt32SliceEqual(t, mask, wantMask)
}

func TestBPETokenizerEncodeAppliesRankedOverlappingMerges(t *testing.T) {
	file := TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens: []string{
			"[PAD]", "[UNK]", "[CLS]", "[SEP]",
			"a", "b", "c", "ab", "abc",
		},
		Merges: []TokenizerMerge{
			{Left: "a", Right: "b"},
			{Left: "ab", Right: "c"},
		},
	}
	tokenizer, err := NewBPETokenizer(file, TokenizerManifest{VocabSize: 9, MaxSequence: 8})
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	ids, mask, err := tokenizer.Encode("ab abc")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	// "ab abc" is two words. Each word is BPE-merged independently;
	// no separator token is emitted between words. Expect:
	//   [CLS]=2, "ab"=7, "abc"=8, [SEP]=3
	assertInt32SliceEqual(t, ids, []int32{2, 7, 8, 3})
	assertInt32SliceEqual(t, mask, []int32{1, 1, 1, 1})
}

func TestNormalizedWordsSegmentsOnNonAlnum(t *testing.T) {
	got := normalizedWords("SARS-CoV-2 vaccines, mRNA-1273!")
	want := []string{"sars", "cov", "2", "vaccines", "mrna", "1273"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("normalizedWords mismatch:\nwant: %v\ngot:  %v", want, got)
	}
}

func TestBPETokenizerEncodeSkipsWordSeparators(t *testing.T) {
	// Regression test for the "every word boundary becomes [UNK]" bug.
	// The encoder must never emit the unknown id just because of
	// whitespace or punctuation between words, and must never insert
	// a separator token between adjacent words.
	file := TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c"},
	}
	tokenizer, err := NewBPETokenizer(file, TokenizerManifest{VocabSize: 7, MaxSequence: 16})
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	ids, _, err := tokenizer.Encode("a b-c")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	// Expect [CLS]=2, "a"=4, "b"=5, "c"=6, [SEP]=3 with no [UNK]=1.
	assertInt32SliceEqual(t, ids, []int32{2, 4, 5, 6, 3})
	for _, id := range ids {
		if id == 1 {
			t.Fatalf("encoder produced [UNK] for a word separator: %v", ids)
		}
	}
}

func TestTokenizeEmbeddingTextContrastiveExamples(t *testing.T) {
	file := TokenizerFile{
		Version: TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "r", "e", "d"},
	}
	tokenizer, err := NewBPETokenizer(file, TokenizerManifest{VocabSize: 7, MaxSequence: 8})
	if err != nil {
		t.Fatalf("new tokenizer: %v", err)
	}
	out, err := TokenizeEmbeddingTextContrastiveExamples([]EmbeddingTextContrastiveExample{
		{Query: "red", Positive: "red"},
	}, tokenizer)
	if err != nil {
		t.Fatalf("tokenize examples: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("example count = %d, want 1", len(out))
	}
	assertInt32SliceEqual(t, out[0].QueryTokens, []int32{2, 4, 5, 6, 3})
	assertInt32SliceEqual(t, out[0].PositiveTokens, []int32{2, 4, 5, 6, 3})
	assertInt32SliceEqual(t, out[0].QueryMask, []int32{1, 1, 1, 1, 1})
	assertInt32SliceEqual(t, out[0].PositiveMask, []int32{1, 1, 1, 1, 1})
}

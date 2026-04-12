package barruntime

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type EmbeddingTextContrastiveExample struct {
	Query    string
	Positive string
}

type EmbeddingTextPairExample struct {
	Query  string
	Right  string
	Target float32
}

type embeddingTextContrastiveRecord struct {
	Query    string   `json:"query"`
	Positive string   `json:"positive"`
	Document string   `json:"document"`
	Left     string   `json:"left"`
	Right    string   `json:"right"`
	Label    *float64 `json:"label,omitempty"`
}

func ReadEmbeddingTextContrastiveExamplesFile(path string) ([]EmbeddingTextContrastiveExample, error) {
	pairs, err := ReadEmbeddingTextPairExamplesFile(path)
	if err != nil {
		return nil, err
	}
	out := make([]EmbeddingTextContrastiveExample, 0, len(pairs))
	for _, pair := range pairs {
		if pair.Target <= 0 {
			continue
		}
		out = append(out, EmbeddingTextContrastiveExample{
			Query:    pair.Query,
			Positive: pair.Right,
		})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("text contrastive dataset has no positive pairs")
	}
	return out, nil
}

func ReadEmbeddingTextPairExamplesFile(path string) ([]EmbeddingTextPairExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingTextPairExample
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record embeddingTextContrastiveRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		example, err := record.pairExample()
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, example)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("text pair dataset is empty")
	}
	return out, nil
}

func WriteEmbeddingTextContrastiveExamplesFile(path string, examples []EmbeddingTextContrastiveExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("text contrastive dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingTextContrastiveRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

type embeddingTokenizedText struct {
	tokens []int32
	mask   []int32
}

type embeddingTextTokenCache map[string]embeddingTokenizedText

func (c embeddingTextTokenCache) encode(text string, tokenizer *BPETokenizer) (embeddingTokenizedText, error) {
	if cached, ok := c[text]; ok {
		return cached, nil
	}
	tokens, mask, err := tokenizer.Encode(text)
	if err != nil {
		return embeddingTokenizedText{}, err
	}
	tokenized := embeddingTokenizedText{tokens: tokens, mask: mask}
	c[text] = tokenized
	return tokenized, nil
}

func cloneTokenizedText(in embeddingTokenizedText) embeddingTokenizedText {
	return embeddingTokenizedText{
		tokens: append([]int32(nil), in.tokens...),
		mask:   append([]int32(nil), in.mask...),
	}
}

func TokenizeEmbeddingTextPairExamples(examples []EmbeddingTextPairExample, tokenizer *BPETokenizer) ([]EmbeddingPairExample, error) {
	return tokenizeEmbeddingTextPairExamples(examples, tokenizer, embeddingTextTokenCache{}, true)
}

func tokenizeEmbeddingTextPairExamples(examples []EmbeddingTextPairExample, tokenizer *BPETokenizer, cache embeddingTextTokenCache, cloneOutput bool) ([]EmbeddingPairExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("text pair dataset is empty")
	}
	if tokenizer == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if cache == nil {
		cache = embeddingTextTokenCache{}
	}
	out := make([]EmbeddingPairExample, 0, len(examples))
	for i, example := range examples {
		query, err := cache.encode(example.Query, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d query: %w", i, err)
		}
		right, err := cache.encode(example.Right, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d right: %w", i, err)
		}
		if cloneOutput {
			query = cloneTokenizedText(query)
			right = cloneTokenizedText(right)
		}
		out = append(out, EmbeddingPairExample{
			LeftTokens:  query.tokens,
			RightTokens: right.tokens,
			LeftMask:    query.mask,
			RightMask:   right.mask,
			Target:      example.Target,
		})
	}
	return out, nil
}

func TokenizeEmbeddingTextContrastiveExamples(examples []EmbeddingTextContrastiveExample, tokenizer *BPETokenizer) ([]EmbeddingContrastiveExample, error) {
	return tokenizeEmbeddingTextContrastiveExamples(examples, tokenizer, embeddingTextTokenCache{}, true)
}

func tokenizeEmbeddingTextContrastiveExamples(examples []EmbeddingTextContrastiveExample, tokenizer *BPETokenizer, cache embeddingTextTokenCache, cloneOutput bool) ([]EmbeddingContrastiveExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("text contrastive dataset is empty")
	}
	if tokenizer == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if cache == nil {
		cache = embeddingTextTokenCache{}
	}
	out := make([]EmbeddingContrastiveExample, 0, len(examples))
	for i, example := range examples {
		query, err := cache.encode(example.Query, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d query: %w", i, err)
		}
		positive, err := cache.encode(example.Positive, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d positive: %w", i, err)
		}
		if cloneOutput {
			query = cloneTokenizedText(query)
			positive = cloneTokenizedText(positive)
		}
		out = append(out, EmbeddingContrastiveExample{
			QueryTokens:    query.tokens,
			PositiveTokens: positive.tokens,
			QueryMask:      query.mask,
			PositiveMask:   positive.mask,
		})
	}
	return out, nil
}

func newEmbeddingTextContrastiveRecord(example EmbeddingTextContrastiveExample) (embeddingTextContrastiveRecord, error) {
	if strings.TrimSpace(example.Query) == "" {
		return embeddingTextContrastiveRecord{}, fmt.Errorf("query is empty")
	}
	if strings.TrimSpace(example.Positive) == "" {
		return embeddingTextContrastiveRecord{}, fmt.Errorf("positive is empty")
	}
	return embeddingTextContrastiveRecord{
		Query:    example.Query,
		Positive: example.Positive,
	}, nil
}

func (r embeddingTextContrastiveRecord) pairExample() (EmbeddingTextPairExample, error) {
	query := firstNonEmpty(r.Query, r.Left)
	right := firstNonEmpty(r.Positive, r.Document, r.Right)
	target := float32(1)
	if r.Label != nil {
		target = float32(*r.Label)
	}
	record, err := newEmbeddingTextContrastiveRecord(EmbeddingTextContrastiveExample{
		Query:    query,
		Positive: right,
	})
	if err != nil {
		return EmbeddingTextPairExample{}, err
	}
	return EmbeddingTextPairExample{
		Query:  record.Query,
		Right:  record.Positive,
		Target: target,
	}, nil
}

func (r embeddingTextContrastiveRecord) example() (EmbeddingTextContrastiveExample, error) {
	pair, err := r.pairExample()
	if err != nil {
		return EmbeddingTextContrastiveExample{}, err
	}
	if pair.Target <= 0 {
		return EmbeddingTextContrastiveExample{}, fmt.Errorf("positive is empty")
	}
	return EmbeddingTextContrastiveExample{
		Query:    pair.Query,
		Positive: pair.Right,
	}, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

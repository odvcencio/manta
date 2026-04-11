package barruntime

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// EmbeddingContrastiveExample is one query-positive training example.
type EmbeddingContrastiveExample struct {
	QueryTokens    []int32
	PositiveTokens []int32
	QueryMask      []int32
	PositiveMask   []int32
}

type embeddingContrastiveRecord struct {
	QueryTokens    []int32 `json:"query_tokens"`
	PositiveTokens []int32 `json:"positive_tokens"`
	QueryMask      []int32 `json:"query_mask,omitempty"`
	PositiveMask   []int32 `json:"positive_mask,omitempty"`
}

// ReadEmbeddingContrastiveExamplesFile reads a JSONL contrastive dataset.
func ReadEmbeddingContrastiveExamplesFile(path string) ([]EmbeddingContrastiveExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingContrastiveExample
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record embeddingContrastiveRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		example, err := record.example()
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, example)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("contrastive dataset is empty")
	}
	return out, nil
}

// WriteEmbeddingContrastiveExamplesFile writes a JSONL contrastive dataset.
func WriteEmbeddingContrastiveExamplesFile(path string, examples []EmbeddingContrastiveExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("contrastive dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingContrastiveRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

func newEmbeddingContrastiveRecord(example EmbeddingContrastiveExample) (embeddingContrastiveRecord, error) {
	if len(example.QueryTokens) == 0 {
		return embeddingContrastiveRecord{}, fmt.Errorf("query_tokens are empty")
	}
	if len(example.PositiveTokens) == 0 {
		return embeddingContrastiveRecord{}, fmt.Errorf("positive_tokens are empty")
	}
	if len(example.QueryMask) > 0 && len(example.QueryMask) != len(example.QueryTokens) {
		return embeddingContrastiveRecord{}, fmt.Errorf("query_mask length %d does not match query_tokens length %d", len(example.QueryMask), len(example.QueryTokens))
	}
	if len(example.PositiveMask) > 0 && len(example.PositiveMask) != len(example.PositiveTokens) {
		return embeddingContrastiveRecord{}, fmt.Errorf("positive_mask length %d does not match positive_tokens length %d", len(example.PositiveMask), len(example.PositiveTokens))
	}
	return embeddingContrastiveRecord{
		QueryTokens:    append([]int32(nil), example.QueryTokens...),
		PositiveTokens: append([]int32(nil), example.PositiveTokens...),
		QueryMask:      append([]int32(nil), example.QueryMask...),
		PositiveMask:   append([]int32(nil), example.PositiveMask...),
	}, nil
}

func (r embeddingContrastiveRecord) example() (EmbeddingContrastiveExample, error) {
	_, err := newEmbeddingContrastiveRecord(EmbeddingContrastiveExample{
		QueryTokens:    r.QueryTokens,
		PositiveTokens: r.PositiveTokens,
		QueryMask:      r.QueryMask,
		PositiveMask:   r.PositiveMask,
	})
	if err != nil {
		return EmbeddingContrastiveExample{}, err
	}
	return EmbeddingContrastiveExample{
		QueryTokens:    append([]int32(nil), r.QueryTokens...),
		PositiveTokens: append([]int32(nil), r.PositiveTokens...),
		QueryMask:      append([]int32(nil), r.QueryMask...),
		PositiveMask:   append([]int32(nil), r.PositiveMask...),
	}, nil
}

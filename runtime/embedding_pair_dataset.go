package mantaruntime

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type embeddingPairRecord struct {
	LeftTokens  []int32  `json:"left_tokens"`
	RightTokens []int32  `json:"right_tokens"`
	LeftMask    []int32  `json:"left_mask,omitempty"`
	RightMask   []int32  `json:"right_mask,omitempty"`
	Target      *float64 `json:"target,omitempty"`
	Label       *float64 `json:"label,omitempty"`
}

func ReadEmbeddingPairExamplesFile(path string) ([]EmbeddingPairExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingPairExample
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record embeddingPairRecord
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
		return nil, fmt.Errorf("pair dataset is empty")
	}
	return out, nil
}

func WriteEmbeddingPairExamplesFile(path string, examples []EmbeddingPairExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("pair dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingPairRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

func newEmbeddingPairRecord(example EmbeddingPairExample) (embeddingPairRecord, error) {
	if len(example.LeftTokens) == 0 {
		return embeddingPairRecord{}, fmt.Errorf("left_tokens are empty")
	}
	if len(example.RightTokens) == 0 {
		return embeddingPairRecord{}, fmt.Errorf("right_tokens are empty")
	}
	if len(example.LeftMask) > 0 && len(example.LeftMask) != len(example.LeftTokens) {
		return embeddingPairRecord{}, fmt.Errorf("left_mask length %d does not match left_tokens length %d", len(example.LeftMask), len(example.LeftTokens))
	}
	if len(example.RightMask) > 0 && len(example.RightMask) != len(example.RightTokens) {
		return embeddingPairRecord{}, fmt.Errorf("right_mask length %d does not match right_tokens length %d", len(example.RightMask), len(example.RightTokens))
	}
	target := float64(example.Target)
	return embeddingPairRecord{
		LeftTokens:  append([]int32(nil), example.LeftTokens...),
		RightTokens: append([]int32(nil), example.RightTokens...),
		LeftMask:    append([]int32(nil), example.LeftMask...),
		RightMask:   append([]int32(nil), example.RightMask...),
		Target:      &target,
	}, nil
}

func (r embeddingPairRecord) example() (EmbeddingPairExample, error) {
	target := float32(1)
	if r.Target != nil {
		target = float32(*r.Target)
	} else if r.Label != nil {
		target = float32(*r.Label)
	}
	_, err := newEmbeddingPairRecord(EmbeddingPairExample{
		LeftTokens:  r.LeftTokens,
		RightTokens: r.RightTokens,
		LeftMask:    r.LeftMask,
		RightMask:   r.RightMask,
		Target:      target,
	})
	if err != nil {
		return EmbeddingPairExample{}, err
	}
	return EmbeddingPairExample{
		LeftTokens:  append([]int32(nil), r.LeftTokens...),
		RightTokens: append([]int32(nil), r.RightTokens...),
		LeftMask:    append([]int32(nil), r.LeftMask...),
		RightMask:   append([]int32(nil), r.RightMask...),
		Target:      target,
	}, nil
}

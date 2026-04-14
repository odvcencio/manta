package mantaruntime

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// EmbeddingHardNegativeExample is one query-positive example with explicit hard negatives.
type EmbeddingHardNegativeExample struct {
	QueryTokens    []int32
	PositiveTokens []int32
	NegativeTokens [][]int32
	QueryMask      []int32
	PositiveMask   []int32
	NegativeMasks  [][]int32
}

type EmbeddingTextHardNegativeExample struct {
	Query     string
	Positive  string
	Negatives []string
}

type embeddingHardNegativeRecord struct {
	QueryTokens    []int32   `json:"query_tokens"`
	PositiveTokens []int32   `json:"positive_tokens"`
	NegativeTokens [][]int32 `json:"negative_tokens,omitempty"`
	QueryMask      []int32   `json:"query_mask,omitempty"`
	PositiveMask   []int32   `json:"positive_mask,omitempty"`
	NegativeMasks  [][]int32 `json:"negative_masks,omitempty"`
}

type embeddingTextHardNegativeRecord struct {
	Query     string   `json:"query"`
	Positive  string   `json:"positive"`
	Document  string   `json:"document,omitempty"`
	Negatives []string `json:"negatives,omitempty"`
}

// ReadEmbeddingHardNegativeExamplesFile reads tokenized hard-negative JSONL.
func ReadEmbeddingHardNegativeExamplesFile(path string) ([]EmbeddingHardNegativeExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingHardNegativeExample
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record embeddingHardNegativeRecord
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
		return nil, fmt.Errorf("hard-negative dataset is empty")
	}
	return out, nil
}

// WriteEmbeddingHardNegativeExamplesFile writes tokenized hard-negative JSONL.
func WriteEmbeddingHardNegativeExamplesFile(path string, examples []EmbeddingHardNegativeExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("hard-negative dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingHardNegativeRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

// BuildEmbeddingHardNegativeExamplesFromPairs groups labeled pair examples by query.
func BuildEmbeddingHardNegativeExamplesFromPairs(pairs []EmbeddingPairExample, maxNegatives int) ([]EmbeddingHardNegativeExample, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("pair dataset is empty")
	}
	type queryGroup struct {
		queryTokens []int32
		queryMask   []int32
		positives   []embeddingTokenizedText
		negatives   []embeddingTokenizedText
	}
	groups := map[string]*queryGroup{}
	order := []string{}
	for _, pair := range pairs {
		key := embeddingBatchSequenceKey(pair.LeftTokens, pair.LeftMask)
		group := groups[key]
		if group == nil {
			group = &queryGroup{
				queryTokens: append([]int32(nil), pair.LeftTokens...),
				queryMask:   append([]int32(nil), pair.LeftMask...),
			}
			groups[key] = group
			order = append(order, key)
		}
		item := embeddingTokenizedText{
			tokens: append([]int32(nil), pair.RightTokens...),
			mask:   append([]int32(nil), pair.RightMask...),
		}
		if pair.Target > 0 {
			group.positives = append(group.positives, item)
		} else {
			group.negatives = append(group.negatives, item)
		}
	}
	out := []EmbeddingHardNegativeExample{}
	for _, key := range order {
		group := groups[key]
		if len(group.positives) == 0 || len(group.negatives) == 0 {
			continue
		}
		limit := maxNegatives
		if limit <= 0 || limit > len(group.negatives) {
			limit = len(group.negatives)
		}
		for i, positive := range group.positives {
			negatives := make([][]int32, 0, limit)
			negativeMasks := make([][]int32, 0, limit)
			for j := 0; j < limit; j++ {
				negative := group.negatives[(i+j)%len(group.negatives)]
				negatives = append(negatives, append([]int32(nil), negative.tokens...))
				negativeMasks = append(negativeMasks, append([]int32(nil), negative.mask...))
			}
			out = append(out, EmbeddingHardNegativeExample{
				QueryTokens:    append([]int32(nil), group.queryTokens...),
				PositiveTokens: append([]int32(nil), positive.tokens...),
				NegativeTokens: negatives,
				QueryMask:      append([]int32(nil), group.queryMask...),
				PositiveMask:   append([]int32(nil), positive.mask...),
				NegativeMasks:  negativeMasks,
			})
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("hard-negative dataset has no positive query groups with negatives")
	}
	return out, nil
}

// ReadEmbeddingTextHardNegativeExamplesFile reads grouped text hard-negative JSONL.
func ReadEmbeddingTextHardNegativeExamplesFile(path string) ([]EmbeddingTextHardNegativeExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []EmbeddingTextHardNegativeExample
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record embeddingTextHardNegativeRecord
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
		return nil, fmt.Errorf("text hard-negative dataset is empty")
	}
	return out, nil
}

func WriteEmbeddingTextHardNegativeExamplesFile(path string, examples []EmbeddingTextHardNegativeExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("text hard-negative dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingTextHardNegativeRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

func BuildEmbeddingTextHardNegativeExamplesFromPairs(pairs []EmbeddingTextPairExample, maxNegatives int) ([]EmbeddingTextHardNegativeExample, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("text pair dataset is empty")
	}
	type queryGroup struct {
		positives []string
		negatives []string
	}
	groups := map[string]*queryGroup{}
	order := []string{}
	for _, pair := range pairs {
		key := pair.Query
		group := groups[key]
		if group == nil {
			group = &queryGroup{}
			groups[key] = group
			order = append(order, key)
		}
		if pair.Target > 0 {
			group.positives = append(group.positives, pair.Right)
		} else {
			group.negatives = append(group.negatives, pair.Right)
		}
	}
	out := []EmbeddingTextHardNegativeExample{}
	for _, query := range order {
		group := groups[query]
		if len(group.positives) == 0 || len(group.negatives) == 0 {
			continue
		}
		limit := maxNegatives
		if limit <= 0 || limit > len(group.negatives) {
			limit = len(group.negatives)
		}
		for i, positive := range group.positives {
			negatives := make([]string, 0, limit)
			for j := 0; j < limit; j++ {
				negatives = append(negatives, group.negatives[(i+j)%len(group.negatives)])
			}
			out = append(out, EmbeddingTextHardNegativeExample{
				Query:     query,
				Positive:  positive,
				Negatives: negatives,
			})
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("text hard-negative dataset has no positive query groups with negatives")
	}
	return out, nil
}

func TokenizeEmbeddingTextHardNegativeExamples(examples []EmbeddingTextHardNegativeExample, tokenizer *BPETokenizer) ([]EmbeddingHardNegativeExample, error) {
	return tokenizeEmbeddingTextHardNegativeExamples(examples, tokenizer, embeddingTextTokenCache{}, true)
}

func tokenizeEmbeddingTextHardNegativeExamples(examples []EmbeddingTextHardNegativeExample, tokenizer *BPETokenizer, cache embeddingTextTokenCache, cloneOutput bool) ([]EmbeddingHardNegativeExample, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("text hard-negative dataset is empty")
	}
	if tokenizer == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	if cache == nil {
		cache = embeddingTextTokenCache{}
	}
	out := make([]EmbeddingHardNegativeExample, 0, len(examples))
	for i, example := range examples {
		query, err := cache.encode(example.Query, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d query: %w", i, err)
		}
		positive, err := cache.encode(example.Positive, tokenizer)
		if err != nil {
			return nil, fmt.Errorf("example %d positive: %w", i, err)
		}
		negatives := make([][]int32, 0, len(example.Negatives))
		negativeMasks := make([][]int32, 0, len(example.Negatives))
		for j, rawNegative := range example.Negatives {
			negative, err := cache.encode(rawNegative, tokenizer)
			if err != nil {
				return nil, fmt.Errorf("example %d negative %d: %w", i, j, err)
			}
			if cloneOutput {
				negative = cloneTokenizedText(negative)
			}
			negatives = append(negatives, negative.tokens)
			negativeMasks = append(negativeMasks, negative.mask)
		}
		if cloneOutput {
			query = cloneTokenizedText(query)
			positive = cloneTokenizedText(positive)
		}
		out = append(out, EmbeddingHardNegativeExample{
			QueryTokens:    query.tokens,
			PositiveTokens: positive.tokens,
			NegativeTokens: negatives,
			QueryMask:      query.mask,
			PositiveMask:   positive.mask,
			NegativeMasks:  negativeMasks,
		})
	}
	return out, nil
}

func limitHardNegativeExamples(examples []EmbeddingHardNegativeExample, maxNegatives int) []EmbeddingHardNegativeExample {
	if maxNegatives <= 0 {
		return examples
	}
	out := make([]EmbeddingHardNegativeExample, len(examples))
	for i, example := range examples {
		out[i] = example
		if len(out[i].NegativeTokens) > maxNegatives {
			out[i].NegativeTokens = out[i].NegativeTokens[:maxNegatives]
		}
		if len(out[i].NegativeMasks) > maxNegatives {
			out[i].NegativeMasks = out[i].NegativeMasks[:maxNegatives]
		}
	}
	return out
}

func newEmbeddingHardNegativeRecord(example EmbeddingHardNegativeExample) (embeddingHardNegativeRecord, error) {
	if len(example.QueryTokens) == 0 {
		return embeddingHardNegativeRecord{}, fmt.Errorf("query_tokens are empty")
	}
	if len(example.PositiveTokens) == 0 {
		return embeddingHardNegativeRecord{}, fmt.Errorf("positive_tokens are empty")
	}
	if len(example.QueryMask) > 0 && len(example.QueryMask) != len(example.QueryTokens) {
		return embeddingHardNegativeRecord{}, fmt.Errorf("query_mask length %d does not match query_tokens length %d", len(example.QueryMask), len(example.QueryTokens))
	}
	if len(example.PositiveMask) > 0 && len(example.PositiveMask) != len(example.PositiveTokens) {
		return embeddingHardNegativeRecord{}, fmt.Errorf("positive_mask length %d does not match positive_tokens length %d", len(example.PositiveMask), len(example.PositiveTokens))
	}
	if len(example.NegativeTokens) == 0 {
		return embeddingHardNegativeRecord{}, fmt.Errorf("negative_tokens are empty")
	}
	if len(example.NegativeMasks) > 0 && len(example.NegativeMasks) != len(example.NegativeTokens) {
		return embeddingHardNegativeRecord{}, fmt.Errorf("negative_masks length %d does not match negative_tokens length %d", len(example.NegativeMasks), len(example.NegativeTokens))
	}
	for i, tokens := range example.NegativeTokens {
		if len(tokens) == 0 {
			return embeddingHardNegativeRecord{}, fmt.Errorf("negative_tokens[%d] are empty", i)
		}
		if len(example.NegativeMasks) > i && len(example.NegativeMasks[i]) > 0 && len(example.NegativeMasks[i]) != len(tokens) {
			return embeddingHardNegativeRecord{}, fmt.Errorf("negative_masks[%d] length %d does not match negative_tokens[%d] length %d", i, len(example.NegativeMasks[i]), i, len(tokens))
		}
	}
	return embeddingHardNegativeRecord{
		QueryTokens:    append([]int32(nil), example.QueryTokens...),
		PositiveTokens: append([]int32(nil), example.PositiveTokens...),
		NegativeTokens: cloneInt32Matrix(example.NegativeTokens),
		QueryMask:      append([]int32(nil), example.QueryMask...),
		PositiveMask:   append([]int32(nil), example.PositiveMask...),
		NegativeMasks:  cloneInt32Matrix(example.NegativeMasks),
	}, nil
}

func (r embeddingHardNegativeRecord) example() (EmbeddingHardNegativeExample, error) {
	record, err := newEmbeddingHardNegativeRecord(EmbeddingHardNegativeExample{
		QueryTokens:    r.QueryTokens,
		PositiveTokens: r.PositiveTokens,
		NegativeTokens: r.NegativeTokens,
		QueryMask:      r.QueryMask,
		PositiveMask:   r.PositiveMask,
		NegativeMasks:  r.NegativeMasks,
	})
	if err != nil {
		return EmbeddingHardNegativeExample{}, err
	}
	return EmbeddingHardNegativeExample{
		QueryTokens:    record.QueryTokens,
		PositiveTokens: record.PositiveTokens,
		NegativeTokens: record.NegativeTokens,
		QueryMask:      record.QueryMask,
		PositiveMask:   record.PositiveMask,
		NegativeMasks:  record.NegativeMasks,
	}, nil
}

func newEmbeddingTextHardNegativeRecord(example EmbeddingTextHardNegativeExample) (embeddingTextHardNegativeRecord, error) {
	if strings.TrimSpace(example.Query) == "" {
		return embeddingTextHardNegativeRecord{}, fmt.Errorf("query is empty")
	}
	if strings.TrimSpace(example.Positive) == "" {
		return embeddingTextHardNegativeRecord{}, fmt.Errorf("positive is empty")
	}
	if len(example.Negatives) == 0 {
		return embeddingTextHardNegativeRecord{}, fmt.Errorf("negatives are empty")
	}
	for i, negative := range example.Negatives {
		if strings.TrimSpace(negative) == "" {
			return embeddingTextHardNegativeRecord{}, fmt.Errorf("negative %d is empty", i)
		}
	}
	return embeddingTextHardNegativeRecord{
		Query:     example.Query,
		Positive:  example.Positive,
		Negatives: append([]string(nil), example.Negatives...),
	}, nil
}

func (r embeddingTextHardNegativeRecord) example() (EmbeddingTextHardNegativeExample, error) {
	positive := firstNonEmpty(r.Positive, r.Document)
	record, err := newEmbeddingTextHardNegativeRecord(EmbeddingTextHardNegativeExample{
		Query:     r.Query,
		Positive:  positive,
		Negatives: r.Negatives,
	})
	if err != nil {
		return EmbeddingTextHardNegativeExample{}, err
	}
	return EmbeddingTextHardNegativeExample{
		Query:     record.Query,
		Positive:  record.Positive,
		Negatives: record.Negatives,
	}, nil
}

func cloneInt32Matrix(in [][]int32) [][]int32 {
	if len(in) == 0 {
		return nil
	}
	out := make([][]int32, len(in))
	for i := range in {
		out[i] = append([]int32(nil), in[i]...)
	}
	return out
}

package mantaruntime

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
)

type EmbeddingTextMiningConfig struct {
	MinChars  int
	MaxPairs  int
	EvalPairs int
	Seed      int64
}

func DefaultMinedTrainPairsPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".mined-train.jsonl")
}

func DefaultMinedEvalPairsPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".mined-eval.jsonl")
}

func ReadEmbeddingTextCorpusFile(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := normalizeMiningText(scanner.Text())
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(lines) == 0 {
		return nil, fmt.Errorf("text corpus is empty")
	}
	return lines, nil
}

func MineEmbeddingTextDatasetsFromCorpusFile(path string, cfg EmbeddingTextMiningConfig) ([]EmbeddingTextContrastiveExample, []EmbeddingTextPairExample, error) {
	lines, err := ReadEmbeddingTextCorpusFile(path)
	if err != nil {
		return nil, nil, err
	}
	return MineEmbeddingTextDatasets(lines, cfg)
}

func MineEmbeddingTextDatasets(lines []string, cfg EmbeddingTextMiningConfig) ([]EmbeddingTextContrastiveExample, []EmbeddingTextPairExample, error) {
	cfg = normalizedEmbeddingTextMiningConfig(cfg)
	positives := minePositiveTextPairs(lines, cfg.MinChars)
	if len(positives) == 0 {
		return nil, nil, fmt.Errorf("no contrastive text pairs could be mined from corpus")
	}
	if cfg.MaxPairs > 0 && len(positives) > cfg.MaxPairs {
		rng := rand.New(rand.NewSource(cfg.Seed))
		rng.Shuffle(len(positives), func(i, j int) {
			positives[i], positives[j] = positives[j], positives[i]
		})
		positives = append([]EmbeddingTextContrastiveExample(nil), positives[:cfg.MaxPairs]...)
	}
	evalPairs := cfg.EvalPairs
	if evalPairs > len(positives)/2 {
		evalPairs = len(positives) / 2
	}
	if evalPairs == 0 && len(positives) > 2 {
		evalPairs = 1
	}
	train := append([]EmbeddingTextContrastiveExample(nil), positives...)
	var eval []EmbeddingTextPairExample
	if evalPairs > 0 {
		evalPositives := append([]EmbeddingTextContrastiveExample(nil), positives[:evalPairs]...)
		train = append([]EmbeddingTextContrastiveExample(nil), positives[evalPairs:]...)
		if len(train) == 0 {
			train = evalPositives[:len(evalPositives)-1]
			evalPositives = evalPositives[len(evalPositives)-1:]
		}
		eval = make([]EmbeddingTextPairExample, 0, len(evalPositives)*2)
		for _, item := range evalPositives {
			eval = append(eval, EmbeddingTextPairExample{
				Query:  item.Query,
				Right:  item.Positive,
				Target: 1,
			})
		}
		negativePool := append([]EmbeddingTextContrastiveExample(nil), positives...)
		for i, item := range evalPositives {
			neg := selectNegativeTextPair(item, negativePool, i)
			if neg == "" {
				continue
			}
			eval = append(eval, EmbeddingTextPairExample{
				Query:  item.Query,
				Right:  neg,
				Target: 0,
			})
		}
	}
	if len(train) == 0 {
		return nil, nil, fmt.Errorf("mined corpus produced no training pairs")
	}
	return train, eval, nil
}

func WriteEmbeddingTextPairExamplesFile(path string, examples []EmbeddingTextPairExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("text pair dataset is empty")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for i, example := range examples {
		record, err := newEmbeddingTextPairRecord(example)
		if err != nil {
			return fmt.Errorf("example %d: %w", i, err)
		}
		if err := enc.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

func newEmbeddingTextPairRecord(example EmbeddingTextPairExample) (embeddingTextContrastiveRecord, error) {
	if strings.TrimSpace(example.Query) == "" {
		return embeddingTextContrastiveRecord{}, fmt.Errorf("query is empty")
	}
	if strings.TrimSpace(example.Right) == "" {
		return embeddingTextContrastiveRecord{}, fmt.Errorf("right is empty")
	}
	label := float64(example.Target)
	return embeddingTextContrastiveRecord{
		Query:    example.Query,
		Positive: example.Right,
		Label:    &label,
	}, nil
}

func normalizedEmbeddingTextMiningConfig(cfg EmbeddingTextMiningConfig) EmbeddingTextMiningConfig {
	if cfg.MinChars <= 0 {
		cfg.MinChars = 8
	}
	if cfg.Seed == 0 {
		cfg.Seed = 1
	}
	return cfg
}

func minePositiveTextPairs(lines []string, minChars int) []EmbeddingTextContrastiveExample {
	seen := map[string]bool{}
	var out []EmbeddingTextContrastiveExample
	add := func(query, positive string) {
		query = normalizeMiningText(query)
		positive = normalizeMiningText(positive)
		if len(query) < minChars || len(positive) < minChars || query == "" || positive == "" || query == positive {
			return
		}
		key := query + "\x00" + positive
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, EmbeddingTextContrastiveExample{Query: query, Positive: positive})
	}
	for _, line := range lines {
		segments := splitMiningSegments(line, minChars)
		for i := 0; i+1 < len(segments); i++ {
			add(segments[i], segments[i+1])
		}
	}
	for i := 0; i+1 < len(lines); i++ {
		add(lines[i], lines[i+1])
	}
	return out
}

func splitMiningSegments(line string, minChars int) []string {
	fields := strings.FieldsFunc(line, func(r rune) bool {
		switch r {
		case '.', '!', '?', ';', ':':
			return true
		default:
			return false
		}
	})
	out := make([]string, 0, len(fields))
	for _, field := range fields {
		field = normalizeMiningText(field)
		if len(field) >= minChars {
			out = append(out, field)
		}
	}
	return out
}

func normalizeMiningText(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	return strings.Join(strings.Fields(text), " ")
}

func selectNegativeTextPair(target EmbeddingTextContrastiveExample, pool []EmbeddingTextContrastiveExample, offset int) string {
	if len(pool) == 0 {
		return ""
	}
	for i := 0; i < len(pool); i++ {
		candidate := pool[(offset+i+1)%len(pool)].Positive
		candidate = normalizeMiningText(candidate)
		if candidate != "" && candidate != target.Positive && candidate != target.Query {
			return candidate
		}
	}
	return ""
}

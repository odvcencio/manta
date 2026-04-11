package barruntime

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

type TokenizerTrainConfig struct {
	CorpusPath string
	VocabSize  int
	MinFreq    int
}

func MinimumTokenizerVocabSizeForCorpus(path string) (int, error) {
	words, err := loadCorpusWords(path)
	if err != nil {
		return 0, err
	}
	if len(words) == 0 {
		return 0, fmt.Errorf("empty corpus")
	}
	charSet := make(map[string]bool)
	for _, word := range words {
		for _, r := range word {
			charSet[string(r)] = true
		}
	}
	return 5 + len(charSet), nil
}

// TrainTokenizerFromCorpus builds a lightweight BPE tokenizer file from a raw text corpus.
func TrainTokenizerFromCorpus(cfg TokenizerTrainConfig) (TokenizerFile, error) {
	if cfg.CorpusPath == "" {
		return TokenizerFile{}, fmt.Errorf("tokenizer corpus path is required")
	}
	if cfg.VocabSize <= 0 {
		return TokenizerFile{}, fmt.Errorf("tokenizer vocab size must be positive")
	}
	if cfg.VocabSize < 5 {
		return TokenizerFile{}, fmt.Errorf("tokenizer vocab size %d is too small for Barracuda special tokens", cfg.VocabSize)
	}
	if cfg.MinFreq <= 0 {
		cfg.MinFreq = 2
	}
	words, err := loadCorpusWords(cfg.CorpusPath)
	if err != nil {
		return TokenizerFile{}, err
	}
	if len(words) == 0 {
		return TokenizerFile{}, fmt.Errorf("empty corpus")
	}

	type wordEntry struct {
		tokens []string
		freq   int
	}
	wordMap := make(map[string]*wordEntry)
	for _, word := range words {
		if entry, ok := wordMap[word]; ok {
			entry.freq++
			continue
		}
		chars := make([]string, 0, len(word))
		for _, r := range word {
			chars = append(chars, string(r))
		}
		wordMap[word] = &wordEntry{tokens: chars, freq: 1}
	}

	charSet := make(map[string]bool)
	for _, entry := range wordMap {
		for _, tok := range entry.tokens {
			charSet[tok] = true
		}
	}
	baseTokens := make([]string, 0, len(charSet))
	for tok := range charSet {
		baseTokens = append(baseTokens, tok)
	}
	sort.Strings(baseTokens)

	merges := make([]TokenizerMerge, 0)
	specials := []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"}
	if len(specials)+len(baseTokens) > cfg.VocabSize {
		return TokenizerFile{}, fmt.Errorf("tokenizer vocab size %d is too small for %d special+base tokens", cfg.VocabSize, len(specials)+len(baseTokens))
	}
	for len(specials)+len(baseTokens)+len(merges) < cfg.VocabSize {
		type pair struct{ left, right string }
		pairFreqs := make(map[pair]int)
		for _, entry := range wordMap {
			for i := 0; i < len(entry.tokens)-1; i++ {
				pairFreqs[pair{left: entry.tokens[i], right: entry.tokens[i+1]}] += entry.freq
			}
		}
		if len(pairFreqs) == 0 {
			break
		}

		var (
			bestPair pair
			bestFreq int
			found    bool
		)
		for candidate, freq := range pairFreqs {
			if freq < cfg.MinFreq {
				continue
			}
			if !found || freq > bestFreq || (freq == bestFreq && pairLess(candidate, bestPair)) {
				bestPair = candidate
				bestFreq = freq
				found = true
			}
		}
		if !found {
			break
		}

		merged := bestPair.left + bestPair.right
		for _, entry := range wordMap {
			entry.tokens = applyMerge(entry.tokens, bestPair.left, bestPair.right)
		}
		merges = append(merges, TokenizerMerge{Left: bestPair.left, Right: bestPair.right})
		if !containsString(baseTokens, merged) {
			baseTokens = append(baseTokens, merged)
		}
	}

	tokens := make([]string, 0, len(specials)+len(baseTokens))
	tokens = append(tokens, specials...)
	tokens = append(tokens, baseTokens...)
	file := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       tokens,
		Merges:       merges,
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
	}
	return file, file.Validate()
}

func loadCorpusWords(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var words []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		for _, word := range strings.Fields(line) {
			cleaned := normalizeWord(word)
			if cleaned != "" {
				words = append(words, cleaned)
			}
		}
	}
	return words, scanner.Err()
}

func normalizeWord(word string) string {
	var b strings.Builder
	for _, r := range word {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

func pairLess(a, b struct{ left, right string }) bool {
	if a.left != b.left {
		return a.left < b.left
	}
	return a.right < b.right
}

func containsString(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

// tokenstats: one-shot diagnostic for manta-embed-v1 tokenization on BEIR corpora.
//
// For each dataset, reads queries.jsonl and corpus.jsonl, tokenizes every doc
// with the production tokenizer (no max_seq cap), and reports:
//   - pieces/word ratio (fragmentation)
//   - unknown-token rate (OOV)
//   - % of docs that would be truncated at the production cap (160)
//   - % of content lost to truncation
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	mantaruntime "github.com/odvcencio/manta/runtime"
)

type beirDoc struct {
	ID    string `json:"_id"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

type stats struct {
	docs         int
	totalPieces  int64
	totalWords   int64
	totalUnks    int64
	truncated    int
	lostPieces   int64
	maxPieces    int
	histBuckets  [8]int // 0-32,32-64,64-128,128-160,160-256,256-512,512-1024,1024+
	longestWords int
}

func (s *stats) add(pieces, unks, words int) {
	s.docs++
	s.totalPieces += int64(pieces)
	s.totalUnks += int64(unks)
	s.totalWords += int64(words)
	if pieces > s.maxPieces {
		s.maxPieces = pieces
	}
	if words > s.longestWords {
		s.longestWords = words
	}
	if pieces > 160 {
		s.truncated++
		s.lostPieces += int64(pieces - 160)
	}
	switch {
	case pieces <= 32:
		s.histBuckets[0]++
	case pieces <= 64:
		s.histBuckets[1]++
	case pieces <= 128:
		s.histBuckets[2]++
	case pieces <= 160:
		s.histBuckets[3]++
	case pieces <= 256:
		s.histBuckets[4]++
	case pieces <= 512:
		s.histBuckets[5]++
	case pieces <= 1024:
		s.histBuckets[6]++
	default:
		s.histBuckets[7]++
	}
}

func (s *stats) report(label string) {
	if s.docs == 0 {
		fmt.Printf("  %-20s no docs\n", label)
		return
	}
	piecesPerDoc := float64(s.totalPieces) / float64(s.docs)
	piecesPerWord := 0.0
	if s.totalWords > 0 {
		piecesPerWord = float64(s.totalPieces) / float64(s.totalWords)
	}
	unkRate := 100 * float64(s.totalUnks) / float64(s.totalPieces)
	truncPct := 100 * float64(s.truncated) / float64(s.docs)
	lossPct := 0.0
	if s.totalPieces > 0 {
		lossPct = 100 * float64(s.lostPieces) / float64(s.totalPieces)
	}
	fmt.Printf("  %-20s n=%-6d  pieces/doc=%6.1f  max=%-5d  pieces/word=%4.2f  unk%%=%5.2f  trunc@160=%5.1f%%  content_lost=%5.1f%%\n",
		label, s.docs, piecesPerDoc, s.maxPieces, piecesPerWord, unkRate, truncPct, lossPct)
	fmt.Printf("    histogram: ≤32=%d  ≤64=%d  ≤128=%d  ≤160=%d  ≤256=%d  ≤512=%d  ≤1024=%d  >1024=%d\n",
		s.histBuckets[0], s.histBuckets[1], s.histBuckets[2], s.histBuckets[3],
		s.histBuckets[4], s.histBuckets[5], s.histBuckets[6], s.histBuckets[7])
}

func readJSONL(path string) ([]beirDoc, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)
	var out []beirDoc
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var d beirDoc
		if err := json.Unmarshal([]byte(line), &d); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, scanner.Err()
}

// buildText mirrors how the retrieval eval concatenates title + text.
func buildText(d beirDoc) string {
	if d.Title == "" {
		return d.Text
	}
	return d.Title + " " + d.Text
}

// approxWordCount counts space-separated tokens in the normalized text.
// normalizeText in runtime lowercases and replaces non-alnum with spaces.
func approxWordCount(text string) int {
	// Cheap mirror of runtime normalization: treat non-alnum as word boundary.
	n := 0
	inWord := false
	for _, r := range text {
		isAlnum := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
		if isAlnum {
			if !inWord {
				n++
				inWord = true
			}
		} else {
			inWord = false
		}
	}
	return n
}

func main() {
	var artifact, datasetRoot, out string
	flag.StringVar(&artifact, "artifact", "", "sealed .mll artifact (for sibling tokenizer + manifest)")
	flag.StringVar(&datasetRoot, "datasets", "datasets/manta-embed-v1/raw", "BEIR raw datasets root")
	flag.StringVar(&out, "out", "", "optional JSON output path")
	flag.Parse()
	if artifact == "" {
		fmt.Fprintln(os.Stderr, "usage: tokenstats --artifact <sealed.mll> [--datasets <root>]")
		os.Exit(2)
	}

	tokPath := mantaruntime.DefaultTokenizerPath(artifact)
	tokFile, err := mantaruntime.ReadTokenizerFile(tokPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read tokenizer:", err)
		os.Exit(1)
	}
	manifestPath := mantaruntime.ResolveEmbeddingManifestPath(artifact)
	manifest, err := mantaruntime.ReadEmbeddingManifestFile(manifestPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read embedding manifest:", err)
		os.Exit(1)
	}
	prodMax := manifest.Tokenizer.MaxSequence

	// Build a tokenizer with MaxSequence=0 so Encode returns full pre-truncation pieces.
	uncapped := manifest.Tokenizer
	uncapped.MaxSequence = 0
	tok, err := mantaruntime.NewBPETokenizer(tokFile, uncapped)
	if err != nil {
		fmt.Fprintln(os.Stderr, "build tokenizer:", err)
		os.Exit(1)
	}

	fmt.Printf("artifact:    %s\n", artifact)
	fmt.Printf("tokenizer:   %s\n", tokPath)
	fmt.Printf("vocab_size:  %d (manifest=%d)\n", len(tokFile.Tokens), manifest.Tokenizer.VocabSize)
	fmt.Printf("prod cap:    %d pieces\n\n", prodMax)

	type datasetReport struct {
		Dataset        string  `json:"dataset"`
		QueryStats     *stats  `json:"-"`
		DocStats       *stats  `json:"-"`
		QueryPiecesPD  float64 `json:"query_pieces_per_doc"`
		DocPiecesPD    float64 `json:"doc_pieces_per_doc"`
		QueryPiecesPW  float64 `json:"query_pieces_per_word"`
		DocPiecesPW    float64 `json:"doc_pieces_per_word"`
		QueryUnkPct    float64 `json:"query_unk_pct"`
		DocUnkPct      float64 `json:"doc_unk_pct"`
		DocTruncPct    float64 `json:"doc_trunc_pct_at_cap"`
		DocLossPct     float64 `json:"doc_content_lost_pct_at_cap"`
		DocsAboveCap   int     `json:"docs_above_cap"`
		DocsTotal      int     `json:"docs_total"`
	}
	var reports []datasetReport

	datasets := []struct {
		name    string
		corpus  string
		queries string
	}{
		{"scifact", "scifact/scifact/corpus.jsonl", "scifact/scifact/queries.jsonl"},
		{"nfcorpus", "nfcorpus/nfcorpus/corpus.jsonl", "nfcorpus/nfcorpus/queries.jsonl"},
		{"fiqa", "fiqa/fiqa/corpus.jsonl", "fiqa/fiqa/queries.jsonl"},
	}

	for _, ds := range datasets {
		corpusPath := datasetRoot + "/" + ds.corpus
		queriesPath := datasetRoot + "/" + ds.queries
		docs, err := readJSONL(corpusPath)
		if err != nil {
			// Some BEIR layouts omit the nested dir.
			alt := datasetRoot + "/" + ds.name + "/corpus.jsonl"
			docs2, err2 := readJSONL(alt)
			if err2 != nil {
				fmt.Printf("[%s] corpus read failed: %v\n", ds.name, err)
				continue
			}
			docs = docs2
			queriesPath = datasetRoot + "/" + ds.name + "/queries.jsonl"
		}
		queries, err := readJSONL(queriesPath)
		if err != nil {
			fmt.Printf("[%s] queries read failed: %v\n", ds.name, err)
			continue
		}

		qs := &stats{}
		ds_ := &stats{}
		for _, q := range queries {
			text := buildText(q)
			ids, _, err := tok.Encode(text)
			if err != nil {
				continue
			}
			unks := 0
			for _, id := range ids {
				if id == tokUnkID(tokFile, manifest.Tokenizer) {
					unks++
				}
			}
			qs.add(len(ids), unks, approxWordCount(text))
		}
		for _, d := range docs {
			text := buildText(d)
			ids, _, err := tok.Encode(text)
			if err != nil {
				continue
			}
			unks := 0
			for _, id := range ids {
				if id == tokUnkID(tokFile, manifest.Tokenizer) {
					unks++
				}
			}
			ds_.add(len(ids), unks, approxWordCount(text))
		}

		fmt.Printf("[%s]\n", ds.name)
		qs.report("queries")
		ds_.report("docs")
		fmt.Println()

		qppd := 0.0
		if qs.docs > 0 {
			qppd = float64(qs.totalPieces) / float64(qs.docs)
		}
		dppd := 0.0
		if ds_.docs > 0 {
			dppd = float64(ds_.totalPieces) / float64(ds_.docs)
		}
		qppw := 0.0
		if qs.totalWords > 0 {
			qppw = float64(qs.totalPieces) / float64(qs.totalWords)
		}
		dppw := 0.0
		if ds_.totalWords > 0 {
			dppw = float64(ds_.totalPieces) / float64(ds_.totalWords)
		}
		qunk := 0.0
		if qs.totalPieces > 0 {
			qunk = 100 * float64(qs.totalUnks) / float64(qs.totalPieces)
		}
		dunk := 0.0
		if ds_.totalPieces > 0 {
			dunk = 100 * float64(ds_.totalUnks) / float64(ds_.totalPieces)
		}
		dtrunc := 0.0
		if ds_.docs > 0 {
			dtrunc = 100 * float64(ds_.truncated) / float64(ds_.docs)
		}
		dloss := 0.0
		if ds_.totalPieces > 0 {
			dloss = 100 * float64(ds_.lostPieces) / float64(ds_.totalPieces)
		}
		reports = append(reports, datasetReport{
			Dataset:       ds.name,
			QueryStats:    qs,
			DocStats:      ds_,
			QueryPiecesPD: qppd,
			DocPiecesPD:   dppd,
			QueryPiecesPW: qppw,
			DocPiecesPW:   dppw,
			QueryUnkPct:   qunk,
			DocUnkPct:     dunk,
			DocTruncPct:   dtrunc,
			DocLossPct:    dloss,
			DocsAboveCap:  ds_.truncated,
			DocsTotal:     ds_.docs,
		})
	}

	sort.Slice(reports, func(i, j int) bool { return reports[i].Dataset < reports[j].Dataset })

	if out != "" {
		fo, err := os.Create(out)
		if err == nil {
			enc := json.NewEncoder(fo)
			enc.SetIndent("", "  ")
			_ = enc.Encode(reports)
			fo.Close()
			fmt.Printf("wrote %s\n", out)
		}
	}
}

// tokUnkID returns the tokenizer's unknown-token id (-1 if absent).
func tokUnkID(f mantaruntime.TokenizerFile, _ mantaruntime.TokenizerManifest) int32 {
	unk := f.UnknownToken
	if unk == "" {
		unk = "[UNK]"
	}
	for i, t := range f.Tokens {
		if t == unk {
			return int32(i)
		}
	}
	return -1
}

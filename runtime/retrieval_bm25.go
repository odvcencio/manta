package mantaruntime

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"
	"unicode"
)

const (
	defaultBM25K1 = 0.9
	defaultBM25B  = 0.4
)

type bm25Document struct {
	ID       string
	Length   int
	TermFreq map[string]int
}

type bm25Index struct {
	Documents []bm25Document
	DocFreq   map[string]int
	Postings  map[string][]int
	AvgLength float64
	K1        float64
	B         float64
}

// EvaluateBM25Retrieval evaluates a BEIR-style split with a lexical BM25 baseline.
func EvaluateBM25Retrieval(ctx context.Context, cfg RetrievalEvalConfig) (RetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	start := time.Now()
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpus(cfg.CorpusPath, cfg.MaxDocs)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	indexStart := time.Now()
	index, err := buildBM25Index(ctx, corpus)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	indexDuration := time.Since(indexStart)

	queryStart := time.Now()
	tokenizedQueries, err := tokenizeBM25Queries(ctx, queries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queryDuration := time.Since(queryStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeBM25RetrievalQuality(ctx, tokenizedQueries, index, qrels, cfg.TopK)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(index.Documents))
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  "bm25",
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:    cfg.CorpusPath,
			QueriesPath:   cfg.QueriesPath,
			QrelsPath:     cfg.QrelsPath,
			Documents:     len(index.Documents),
			Queries:       evaluatedQueries,
			RelevantPairs: relevantPairs,
			ScoredPairs:   scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			BatchSize:  cfg.BatchSize,
			TopK:       cfg.TopK,
			MaxDocs:    cfg.MaxDocs,
			MaxQueries: cfg.MaxQueries,
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: indexDuration.Seconds(),
			QueryEmbedSeconds:    queryDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(index.Documents)), indexDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(tokenizedQueries)), queryDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
		},
	}, nil
}

func buildBM25Index(ctx context.Context, records []retrievalTextRecord) (bm25Index, error) {
	index := bm25Index{
		Documents: make([]bm25Document, 0, len(records)),
		DocFreq:   map[string]int{},
		Postings:  map[string][]int{},
		K1:        defaultBM25K1,
		B:         defaultBM25B,
	}
	var totalLength int
	for i, record := range records {
		if err := ctx.Err(); err != nil {
			return bm25Index{}, err
		}
		tokens := tokenizeBM25Text(record.Text)
		if len(tokens) == 0 {
			tokens = []string{""}
		}
		tf := make(map[string]int, len(tokens))
		seen := map[string]bool{}
		for _, token := range tokens {
			tf[token]++
			if !seen[token] {
				index.DocFreq[token]++
				index.Postings[token] = append(index.Postings[token], len(index.Documents))
				seen[token] = true
			}
		}
		index.Documents = append(index.Documents, bm25Document{
			ID:       record.ID,
			Length:   len(tokens),
			TermFreq: tf,
		})
		totalLength += len(tokens)
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return bm25Index{}, err
			}
		}
	}
	if len(index.Documents) > 0 {
		index.AvgLength = float64(totalLength) / float64(len(index.Documents))
	}
	return index, nil
}

type bm25Query struct {
	ID     string
	Tokens []string
}

func tokenizeBM25Queries(ctx context.Context, records []retrievalTextRecord) ([]bm25Query, error) {
	out := make([]bm25Query, len(records))
	for i, record := range records {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		out[i] = bm25Query{ID: record.ID, Tokens: tokenizeBM25Text(record.Text)}
	}
	return out, nil
}

func computeBM25RetrievalQuality(ctx context.Context, queries []bm25Query, index bm25Index, qrels retrievalQrels, topK int) (RetrievalEvalQualityMetrics, int, int, int, int, error) {
	docIDSet := make(map[string]bool, len(index.Documents))
	for _, doc := range index.Documents {
		docIDSet[doc.ID] = true
	}
	if topK < 100 {
		topK = 100
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
		}
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			} else {
				skippedRelevantDocs++
			}
		}
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		scores := topBM25Scores(query.Tokens, index, topK)
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		totals.NDCGAt10 += ndcgAt(scores, filteredRels, 10)
		totals.MRRAt10 += mrrAt(scores, filteredRels, 10)
		totals.RecallAt10 += recallAt(scores, filteredRels, 10)
		totals.RecallAt100 += recallAt(scores, filteredRels, 100)
	}
	if evaluatedQueries > 0 {
		denom := float64(evaluatedQueries)
		totals.NDCGAt10 /= denom
		totals.MRRAt10 /= denom
		totals.RecallAt10 /= denom
		totals.RecallAt100 /= denom
	}
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, nil
}

func topBM25Scores(queryTokens []string, index bm25Index, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(index.Documents) {
		topK = len(index.Documents)
	}
	h := make(retrievalScoreHeap, 0, topK)
	candidates, candidateSet := bm25CandidateDocIndices(queryTokens, index)
	for _, docIndex := range candidates {
		doc := index.Documents[docIndex]
		score := retrievalScoredDoc{ID: doc.ID, Score: float32(scoreBM25Document(queryTokens, doc, index))}
		pushBM25Score(&h, score, topK)
	}
	if len(h) < topK {
		for docIndex, doc := range index.Documents {
			if candidateSet[docIndex] {
				continue
			}
			pushBM25Score(&h, retrievalScoredDoc{ID: doc.ID}, topK)
		}
	}
	scores := []retrievalScoredDoc(h)
	slices.SortFunc(scores, func(a, b retrievalScoredDoc) int {
		if retrievalScoreBetter(a, b) {
			return -1
		}
		if retrievalScoreBetter(b, a) {
			return 1
		}
		return 0
	})
	return scores
}

func bm25CandidateDocIndices(queryTokens []string, index bm25Index) ([]int, map[int]bool) {
	candidateSet := map[int]bool{}
	candidates := []int{}
	for _, token := range queryTokens {
		if token == "" {
			continue
		}
		for _, docIndex := range index.Postings[token] {
			if candidateSet[docIndex] {
				continue
			}
			candidateSet[docIndex] = true
			candidates = append(candidates, docIndex)
		}
	}
	return candidates, candidateSet
}

func pushBM25Score(h *retrievalScoreHeap, score retrievalScoredDoc, topK int) {
	if topK <= 0 {
		return
	}
	if len(*h) < topK {
		heap.Push(h, score)
		return
	}
	if retrievalScoreBetter(score, (*h)[0]) {
		(*h)[0] = score
		heap.Fix(h, 0)
	}
}

func topBM25NonPositiveScores(queryTokens []string, positiveIDs map[string]bool, index bm25Index, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(index.Documents) {
		topK = len(index.Documents)
	}
	h := make(retrievalScoreHeap, 0, topK)
	candidates, _ := bm25CandidateDocIndices(queryTokens, index)
	for _, docIndex := range candidates {
		doc := index.Documents[docIndex]
		if positiveIDs[doc.ID] {
			continue
		}
		score := retrievalScoredDoc{ID: doc.ID, Score: float32(scoreBM25Document(queryTokens, doc, index))}
		pushBM25Score(&h, score, topK)
	}
	scores := []retrievalScoredDoc(h)
	slices.SortFunc(scores, func(a, b retrievalScoredDoc) int {
		if retrievalScoreBetter(a, b) {
			return -1
		}
		if retrievalScoreBetter(b, a) {
			return 1
		}
		return 0
	})
	return scores
}

func topBM25NonPositiveTexts(queryTokens []string, positiveIDs map[string]bool, index bm25Index, docText map[string]string, topK int) []string {
	scores := topBM25NonPositiveScores(queryTokens, positiveIDs, index, topK)
	texts := make([]string, 0, len(scores))
	seen := map[string]bool{}
	for _, score := range scores {
		text := strings.TrimSpace(docText[score.ID])
		if text == "" || seen[text] {
			continue
		}
		seen[text] = true
		texts = append(texts, text)
	}
	return texts
}

func scoreBM25Document(queryTokens []string, doc bm25Document, index bm25Index) float64 {
	if len(queryTokens) == 0 || len(index.Documents) == 0 || index.AvgLength == 0 {
		return 0
	}
	queryFreq := make(map[string]int, len(queryTokens))
	for _, token := range queryTokens {
		if token != "" {
			queryFreq[token]++
		}
	}
	var score float64
	nDocs := float64(len(index.Documents))
	lengthNorm := index.K1 * (1 - index.B + index.B*float64(doc.Length)/index.AvgLength)
	for token, qtf := range queryFreq {
		tf := doc.TermFreq[token]
		if tf == 0 {
			continue
		}
		df := float64(index.DocFreq[token])
		idf := math.Log(1 + (nDocs-df+0.5)/(df+0.5))
		tfWeight := (float64(tf) * (index.K1 + 1)) / (float64(tf) + lengthNorm)
		score += float64(qtf) * idf * tfWeight
	}
	return score
}

func tokenizeBM25Text(text string) []string {
	text = strings.ToLower(text)
	tokens := []string{}
	var b strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
			continue
		}
		if b.Len() > 0 {
			tokens = append(tokens, b.String())
			b.Reset()
		}
	}
	if b.Len() > 0 {
		tokens = append(tokens, b.String())
	}
	return tokens
}

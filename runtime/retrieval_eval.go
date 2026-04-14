package mantaruntime

import (
	"bufio"
	"container/heap"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/odvcencio/manta/runtime/backend"
)

const RetrievalEvalMetricsSchema = "manta.embedding_retrieval_metrics.v1"

// RetrievalEvalConfig describes a BEIR-style retrieval eval.
type RetrievalEvalConfig struct {
	DatasetName  string
	ArtifactPath string
	CorpusPath   string
	QueriesPath  string
	QrelsPath    string
	BatchSize    int
	TopK         int
	MaxDocs      int
	MaxQueries   int
}

// RetrievalEvalMetrics records standard retrieval metrics for one dataset split.
type RetrievalEvalMetrics struct {
	Schema        string                      `json:"schema"`
	Dataset       string                      `json:"dataset"`
	Artifact      string                      `json:"artifact,omitempty"`
	Backend       string                      `json:"backend,omitempty"`
	Inputs        RetrievalEvalInputMetrics   `json:"inputs"`
	Config        RetrievalEvalConfigMetrics  `json:"config"`
	Quality       RetrievalEvalQualityMetrics `json:"quality"`
	Throughput    RetrievalEvalThroughput     `json:"throughput"`
	SkippedCounts RetrievalEvalSkippedCounts  `json:"skipped_counts,omitempty"`
}

type RetrievalEvalInputMetrics struct {
	CorpusPath    string `json:"corpus_path,omitempty"`
	QueriesPath   string `json:"queries_path,omitempty"`
	QrelsPath     string `json:"qrels_path,omitempty"`
	Documents     int    `json:"documents"`
	Queries       int    `json:"queries"`
	RelevantPairs int    `json:"relevant_pairs"`
	ScoredPairs   int64  `json:"scored_pairs"`
}

type RetrievalEvalConfigMetrics struct {
	BatchSize  int `json:"batch_size"`
	TopK       int `json:"top_k"`
	MaxDocs    int `json:"max_docs,omitempty"`
	MaxQueries int `json:"max_queries,omitempty"`
}

type RetrievalEvalQualityMetrics struct {
	NDCGAt10    float64 `json:"ndcg_at_10"`
	MRRAt10     float64 `json:"mrr_at_10"`
	RecallAt10  float64 `json:"recall_at_10"`
	RecallAt100 float64 `json:"recall_at_100"`
}

type RetrievalEvalThroughput struct {
	ElapsedSeconds       float64 `json:"elapsed_seconds"`
	DocumentEmbedSeconds float64 `json:"document_embed_seconds"`
	QueryEmbedSeconds    float64 `json:"query_embed_seconds"`
	ScoreSeconds         float64 `json:"score_seconds"`
	DocumentsPerSecond   float64 `json:"documents_per_second"`
	QueriesPerSecond     float64 `json:"queries_per_second"`
	ScoresPerSecond      float64 `json:"scores_per_second"`
}

type RetrievalEvalSkippedCounts struct {
	QueriesWithoutText         int `json:"queries_without_text,omitempty"`
	RelevantDocsWithoutText    int `json:"relevant_docs_without_text,omitempty"`
	QueriesWithoutRelevantDocs int `json:"queries_without_relevant_docs,omitempty"`
}

type retrievalTextRecord struct {
	ID   string
	Text string
}

type tokenizedRetrievalTextRecord struct {
	Index  int
	ID     string
	Tokens []int32
}

type retrievalVectorRecord struct {
	ID     string
	Vector []float32
}

type retrievalQrels map[string]map[string]float64

// BEIRRetrievalPaths resolves the conventional corpus/query/qrels files under a dataset directory.
func BEIRRetrievalPaths(datasetDir, split string) (corpusPath, queriesPath, qrelsPath string) {
	if split == "" {
		split = "test"
	}
	return filepath.Join(datasetDir, "corpus.jsonl"), filepath.Join(datasetDir, "queries.jsonl"), filepath.Join(datasetDir, "qrels", split+".tsv")
}

// EvaluateEmbeddingRetrieval evaluates an embedding model against BEIR-style corpus, query, and qrels files.
func EvaluateEmbeddingRetrieval(ctx context.Context, model *EmbeddingModel, cfg RetrievalEvalConfig) (RetrievalEvalMetrics, error) {
	if model == nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embedding model is not loaded")
	}
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

	docStart := time.Now()
	docVectors, err := embedRetrievalTexts(ctx, model, corpus, cfg.BatchSize)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embed corpus: %w", err)
	}
	docDuration := time.Since(docStart)

	queryStart := time.Now()
	queryVectors, err := embedRetrievalTexts(ctx, model, queries, cfg.BatchSize)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embed queries: %w", err)
	}
	queryDuration := time.Since(queryStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant := computeRetrievalQuality(queryVectors, docVectors, qrels, cfg.TopK)
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(docVectors))
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  string(model.Backend()),
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:    cfg.CorpusPath,
			QueriesPath:   cfg.QueriesPath,
			QrelsPath:     cfg.QrelsPath,
			Documents:     len(docVectors),
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
			DocumentEmbedSeconds: docDuration.Seconds(),
			QueryEmbedSeconds:    queryDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(docVectors)), docDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(queryVectors)), queryDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
		},
	}, nil
}

func normalizeRetrievalEvalConfig(cfg RetrievalEvalConfig) RetrievalEvalConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "retrieval"
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 100
	}
	if cfg.TopK < 100 {
		cfg.TopK = 100
	}
	return cfg
}

type beirJSONRecord struct {
	ID    string         `json:"_id"`
	Title string         `json:"title,omitempty"`
	Text  string         `json:"text"`
	Meta  map[string]any `json:"metadata,omitempty"`
}

func readBEIRCorpus(path string, limit int) ([]retrievalTextRecord, error) {
	return readBEIRTextFile(path, nil, limit)
}

func readBEIRQueries(path string, qrels retrievalQrels, limit int) ([]retrievalTextRecord, int, error) {
	needed := make(map[string]bool, len(qrels))
	for id := range qrels {
		needed[id] = true
	}
	records, err := readBEIRTextFile(path, needed, limit)
	if err != nil {
		return nil, 0, err
	}
	return records, len(needed) - len(records), nil
}

func readBEIRTextFile(path string, ids map[string]bool, limit int) ([]retrievalTextRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	out := []retrievalTextRecord{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		if limit > 0 && len(out) >= limit {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record beirJSONRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if record.ID == "" {
			continue
		}
		if ids != nil && !ids[record.ID] {
			continue
		}
		text := strings.TrimSpace(strings.Join([]string{record.Title, record.Text}, "\n"))
		if text == "" {
			continue
		}
		out = append(out, retrievalTextRecord{ID: record.ID, Text: text})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func readBEIRQrels(path string) (retrievalQrels, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	qrels := retrievalQrels{}
	scanner := bufio.NewScanner(file)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		if lineNo == 1 && len(parts) >= 2 && strings.Contains(strings.ToLower(parts[0]), "query") {
			continue
		}
		if len(parts) < 3 {
			return nil, fmt.Errorf("%s:%d: expected query-id, corpus-id, score", path, lineNo)
		}
		docField, scoreField := 1, 2
		if len(parts) >= 4 {
			docField, scoreField = 2, 3
		}
		score, err := strconv.ParseFloat(parts[scoreField], 64)
		if err != nil {
			return nil, fmt.Errorf("%s:%d: score: %w", path, lineNo, err)
		}
		if score <= 0 {
			continue
		}
		queryID := strings.TrimSpace(parts[0])
		docID := strings.TrimSpace(parts[docField])
		if queryID == "" || docID == "" {
			continue
		}
		if qrels[queryID] == nil {
			qrels[queryID] = map[string]float64{}
		}
		qrels[queryID][docID] = score
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(qrels) == 0 {
		return nil, fmt.Errorf("qrels file has no positive relevance rows: %s", path)
	}
	return qrels, nil
}

func embedRetrievalTexts(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord, batchSize int) ([]retrievalVectorRecord, error) {
	if batchSize <= 0 {
		batchSize = 64
	}
	tokenized, lengths, err := tokenizeRetrievalTexts(ctx, model, records)
	if err != nil {
		return nil, err
	}
	out := make([]retrievalVectorRecord, len(records))
	for _, length := range lengths {
		group := tokenized[length]
		for start := 0; start < len(group); start += batchSize {
			end := start + batchSize
			if end > len(group) {
				end = len(group)
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			chunk := group[start:end]
			batches := make([][]int32, len(chunk))
			for i, record := range chunk {
				batches[i] = record.Tokens
			}
			result, err := model.EmbedBatch(ctx, batches)
			if err != nil {
				return nil, err
			}
			rows, err := embeddingRowViews(result.Embeddings, len(chunk))
			if err != nil {
				return nil, err
			}
			for i, row := range rows {
				record := chunk[i]
				out[record.Index] = retrievalVectorRecord{
					ID:     record.ID,
					Vector: normalizeRetrievalVector(row),
				}
			}
		}
	}
	return out, nil
}

func embeddingRowViews(t *backend.Tensor, wantRows int) ([][]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("embedding tensor is nil")
	}
	if len(t.F32) == 0 {
		return nil, fmt.Errorf("embedding tensor has no float data")
	}
	switch len(t.Shape) {
	case 1:
		if wantRows != 1 {
			return nil, fmt.Errorf("embedding tensor shape %v cannot provide %d rows", t.Shape, wantRows)
		}
		return [][]float32{t.F32}, nil
	case 2:
		rows, cols := t.Shape[0], t.Shape[1]
		if rows != wantRows {
			return nil, fmt.Errorf("embedding tensor rows = %d, want %d", rows, wantRows)
		}
		if len(t.F32) < rows*cols {
			return nil, fmt.Errorf("embedding tensor has %d values, want at least %d", len(t.F32), rows*cols)
		}
		out := make([][]float32, rows)
		for i := 0; i < rows; i++ {
			out[i] = t.F32[i*cols : (i+1)*cols]
		}
		return out, nil
	default:
		return nil, fmt.Errorf("embedding tensor shape %v is not rank 1 or 2", t.Shape)
	}
}

func tokenizeRetrievalTexts(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord) (map[int][]tokenizedRetrievalTextRecord, []int, error) {
	tokenized, err := tokenizeRetrievalTextRecords(ctx, model, records)
	if err != nil {
		return nil, nil, err
	}
	groups := make(map[int][]tokenizedRetrievalTextRecord)
	lengths := []int{}
	for _, record := range tokenized {
		length := len(record.Tokens)
		if len(groups[length]) == 0 {
			lengths = append(lengths, length)
		}
		groups[length] = append(groups[length], record)
	}
	slices.Sort(lengths)
	return groups, lengths, nil
}

func tokenizeRetrievalTextRecords(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord) ([]tokenizedRetrievalTextRecord, error) {
	if len(records) == 0 {
		return nil, nil
	}
	workers := min(runtime.GOMAXPROCS(0), len(records))
	if workers <= 1 || len(records) < 128 {
		return tokenizeRetrievalTextRecordsSerial(ctx, model, records)
	}
	out := make([]tokenizedRetrievalTextRecord, len(records))
	jobs := make(chan int)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error
	setErr := func(err error) {
		if err == nil {
			return
		}
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
	}
	hasErr := func() bool {
		errMu.Lock()
		ok := firstErr != nil
		errMu.Unlock()
		return ok
	}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				if err := ctx.Err(); err != nil {
					setErr(err)
					continue
				}
				tokens, _, err := model.TokenizeText(records[i].Text)
				if err != nil {
					setErr(fmt.Errorf("text %d: %w", i, err))
					continue
				}
				out[i] = tokenizedRetrievalTextRecord{
					Index:  i,
					ID:     records[i].ID,
					Tokens: tokens,
				}
			}
		}()
	}
	for i := range records {
		if err := ctx.Err(); err != nil {
			setErr(err)
			break
		}
		if hasErr() {
			break
		}
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	errMu.Lock()
	err := firstErr
	errMu.Unlock()
	if err != nil {
		return nil, err
	}
	return out, nil
}

func tokenizeRetrievalTextRecordsSerial(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord) ([]tokenizedRetrievalTextRecord, error) {
	out := make([]tokenizedRetrievalTextRecord, len(records))
	for i, record := range records {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tokens, _, err := model.TokenizeText(record.Text)
		if err != nil {
			return nil, fmt.Errorf("text %d: %w", i, err)
		}
		out[i] = tokenizedRetrievalTextRecord{
			Index:  i,
			ID:     record.ID,
			Tokens: tokens,
		}
	}
	return out, nil
}

func normalizeRetrievalVector(in []float32) []float32 {
	out := append([]float32(nil), in...)
	var sum float64
	for _, v := range out {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return out
	}
	scale := float32(1 / math.Sqrt(sum))
	for i := range out {
		out[i] *= scale
	}
	return out
}

type retrievalScoredDoc struct {
	ID    string
	Score float32
}

func computeRetrievalQuality(queries, docs []retrievalVectorRecord, qrels retrievalQrels, topK int) (RetrievalEvalQualityMetrics, int, int, int, int) {
	docIDSet := make(map[string]bool, len(docs))
	for _, doc := range docs {
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
		scores := topRetrievalScores(query.Vector, docs, topK)
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
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant
}

func topRetrievalScores(query []float32, docs []retrievalVectorRecord, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, doc := range docs {
		score := retrievalScoredDoc{ID: doc.ID, Score: dotRetrievalVectors(query, doc.Vector)}
		if len(h) < topK {
			heap.Push(&h, score)
			continue
		}
		if retrievalScoreBetter(score, h[0]) {
			h[0] = score
			heap.Fix(&h, 0)
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

func retrievalScoreBetter(a, b retrievalScoredDoc) bool {
	if a.Score > b.Score {
		return true
	}
	if a.Score < b.Score {
		return false
	}
	return a.ID < b.ID
}

type retrievalScoreHeap []retrievalScoredDoc

func (h retrievalScoreHeap) Len() int {
	return len(h)
}

func (h retrievalScoreHeap) Less(i, j int) bool {
	return retrievalScoreBetter(h[j], h[i])
}

func (h retrievalScoreHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *retrievalScoreHeap) Push(x any) {
	*h = append(*h, x.(retrievalScoredDoc))
}

func (h *retrievalScoreHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func dotRetrievalVectors(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func ndcgAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	dcg := 0.0
	limit := min(k, len(scores))
	for i := 0; i < limit; i++ {
		rel := rels[scores[i].ID]
		if rel <= 0 {
			continue
		}
		dcg += rel / math.Log2(float64(i)+2)
	}
	idcg := idealDCGAt(rels, k)
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func idealDCGAt(rels map[string]float64, k int) float64 {
	values := make([]float64, 0, len(rels))
	for _, rel := range rels {
		if rel > 0 {
			values = append(values, rel)
		}
	}
	slices.SortFunc(values, func(a, b float64) int {
		if a > b {
			return -1
		}
		if a < b {
			return 1
		}
		return 0
	})
	dcg := 0.0
	limit := min(k, len(values))
	for i := 0; i < limit; i++ {
		dcg += values[i] / math.Log2(float64(i)+2)
	}
	return dcg
}

func mrrAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	limit := min(k, len(scores))
	for i := 0; i < limit; i++ {
		if rels[scores[i].ID] > 0 {
			return 1 / float64(i+1)
		}
	}
	return 0
}

func recallAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	if len(rels) == 0 {
		return 0
	}
	hits := 0
	limit := min(k, len(scores))
	for i := 0; i < limit; i++ {
		if rels[scores[i].ID] > 0 {
			hits++
		}
	}
	return float64(hits) / float64(len(rels))
}

func ratePerSecond(count float64, duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return count / duration.Seconds()
}

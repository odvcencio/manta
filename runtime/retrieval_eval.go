package mantaruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
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
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant := computeRetrievalQuality(queryVectors, docVectors, qrels)
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
	out := make([]retrievalVectorRecord, 0, len(records))
	for start := 0; start < len(records); start += batchSize {
		end := start + batchSize
		if end > len(records) {
			end = len(records)
		}
		texts := make([]string, end-start)
		for i := range texts {
			texts[i] = records[start+i].Text
		}
		result, err := model.EmbedTextBatch(ctx, texts)
		if err != nil {
			return nil, err
		}
		rows, err := embeddingRows(result.Embeddings, len(texts))
		if err != nil {
			return nil, err
		}
		for i, row := range rows {
			out = append(out, retrievalVectorRecord{
				ID:     records[start+i].ID,
				Vector: normalizeRetrievalVector(row),
			})
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

func computeRetrievalQuality(queries, docs []retrievalVectorRecord, qrels retrievalQrels) (RetrievalEvalQualityMetrics, int, int, int, int) {
	docIDSet := make(map[string]bool, len(docs))
	for _, doc := range docs {
		docIDSet[doc.ID] = true
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	scores := make([]retrievalScoredDoc, len(docs))
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
		for i, doc := range docs {
			scores[i] = retrievalScoredDoc{ID: doc.ID, Score: dotRetrievalVectors(query.Vector, doc.Vector)}
		}
		slices.SortFunc(scores, func(a, b retrievalScoredDoc) int {
			if a.Score > b.Score {
				return -1
			}
			if a.Score < b.Score {
				return 1
			}
			return strings.Compare(a.ID, b.ID)
		})
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

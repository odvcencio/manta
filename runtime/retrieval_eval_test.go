package mantaruntime

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/metal"
)

func TestComputeRetrievalQualityPerfectRanking(t *testing.T) {
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1})},
	}
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{0, 1})},
		{ID: "d3", Vector: normalizeRetrievalVector([]float32{0.2, 0.1})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries := computeRetrievalQuality(queries, docs, qrels, 100)
	if queriesCount != 2 || relevantPairs != 2 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if quality.NDCGAt10 != 1 || quality.MRRAt10 != 1 || quality.RecallAt10 != 1 || quality.RecallAt100 != 1 {
		t.Fatalf("quality = %+v, want perfect ranking", quality)
	}
}

func TestComputeRetrievalQualityUsesBoundedTopK(t *testing.T) {
	queries := []retrievalVectorRecord{{ID: "q", Vector: []float32{1}}}
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		docs[i] = retrievalVectorRecord{
			ID:     fmt.Sprintf("d%03d", i),
			Vector: []float32{float32(200 - i)},
		}
	}
	qrels := retrievalQrels{
		"q": {
			"d000": 1,
			"d009": 1,
			"d099": 1,
			"d100": 1,
		},
	}

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries := computeRetrievalQuality(queries, docs, qrels, 100)
	if queriesCount != 1 || relevantPairs != 4 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if quality.MRRAt10 != 1 {
		t.Fatalf("mrr@10 = %v, want 1", quality.MRRAt10)
	}
	if quality.RecallAt10 != 0.5 {
		t.Fatalf("recall@10 = %v, want 0.5", quality.RecallAt10)
	}
	if quality.RecallAt100 != 0.75 {
		t.Fatalf("recall@100 = %v, want 0.75", quality.RecallAt100)
	}
}

func TestEmbedRetrievalTextsGroupsByTokenLengthAndPreservesOrder(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyMaskedEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}
	if err := model.attachTokenizer(tinyEmbeddingTokenizerFile()); err != nil {
		t.Fatalf("attach tokenizer: %v", err)
	}
	records := []retrievalTextRecord{
		{ID: "long-1", Text: "aa"},
		{ID: "short", Text: "a"},
		{ID: "long-2", Text: "aa"},
	}

	got, err := embedRetrievalTexts(context.Background(), model, records, 2)
	if err != nil {
		t.Fatalf("embed retrieval texts: %v", err)
	}
	if len(got) != len(records) {
		t.Fatalf("embedded rows = %d, want %d", len(got), len(records))
	}
	for i, record := range records {
		if got[i].ID != record.ID {
			t.Fatalf("row %d id = %q, want %q", i, got[i].ID, record.ID)
		}
		want, err := model.EmbedText(context.Background(), record.Text)
		if err != nil {
			t.Fatalf("embed text %q: %v", record.Text, err)
		}
		wantRows, err := embeddingRows(want.Embeddings, 1)
		if err != nil {
			t.Fatalf("embedding rows: %v", err)
		}
		wantVector := normalizeRetrievalVector(wantRows[0])
		for j, wantValue := range wantVector {
			if diff := math.Abs(float64(got[i].Vector[j] - wantValue)); diff > 1e-5 {
				t.Fatalf("row %d vector[%d] = %v, want %v", i, j, got[i].Vector[j], wantValue)
			}
		}
	}
}

func TestReadBEIRRetrievalFiles(t *testing.T) {
	dir := t.TempDir()
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","title":"Title","text":"Document body"}`+"\n"+
			`{"_id":"d2","text":"Other document"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"document query"}`+"\n"+
			`{"_id":"q2","text":"unused query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	corpusPath, queriesPath, gotQrelsPath := BEIRRetrievalPaths(dir, "test")
	if gotQrelsPath != qrelsPath {
		t.Fatalf("qrels path = %q, want %q", gotQrelsPath, qrelsPath)
	}
	qrels, err := readBEIRQrels(gotQrelsPath)
	if err != nil {
		t.Fatalf("read qrels: %v", err)
	}
	corpus, err := readBEIRCorpus(corpusPath, 0)
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	queries, skipped, err := readBEIRQueries(queriesPath, qrels, 0)
	if err != nil {
		t.Fatalf("read queries: %v", err)
	}
	if len(corpus) != 2 || corpus[0].Text != "Title\nDocument body" {
		t.Fatalf("corpus = %+v", corpus)
	}
	if len(queries) != 1 || queries[0].ID != "q1" || skipped != 0 {
		t.Fatalf("queries = %+v skipped=%d", queries, skipped)
	}
}

func TestReadBEIRQrelsAcceptsTRECFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.tsv")
	if err := os.WriteFile(path, []byte("q1\tQ0\td1\t2\nq1\tQ0\td2\t0\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	qrels, err := readBEIRQrels(path)
	if err != nil {
		t.Fatalf("read qrels: %v", err)
	}
	if got := qrels["q1"]["d1"]; got != 2 {
		t.Fatalf("qrels[q1][d1] = %v, want 2", got)
	}
	if _, ok := qrels["q1"]["d2"]; ok {
		t.Fatalf("non-positive qrel was retained: %+v", qrels)
	}
}

func TestEvaluateBM25RetrievalRanksLexicalMatch(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha alpha finance"}`+"\n"+
			`{"_id":"d2","text":"beta medicine"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha finance"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	metrics, err := EvaluateBM25Retrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny",
		CorpusPath:  corpusPath,
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		TopK:        100,
	})
	if err != nil {
		t.Fatalf("evaluate bm25: %v", err)
	}
	if metrics.Backend != "bm25" || metrics.Dataset != "tiny" {
		t.Fatalf("metrics identity = backend:%q dataset:%q", metrics.Backend, metrics.Dataset)
	}
	if metrics.Inputs.Documents != 2 || metrics.Inputs.Queries != 1 || metrics.Inputs.ScoredPairs != 2 {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 || metrics.Quality.RecallAt10 != 1 || metrics.Quality.RecallAt100 != 1 {
		t.Fatalf("quality = %+v, want perfect lexical ranking", metrics.Quality)
	}
}

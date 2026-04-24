// retrievaldump: per-query diagnostic dump for an embedding model.
//
// For each query, embeds query + corpus, computes top-K retrieved docs by
// cosine similarity, cross-references qrels, and prints structured output so
// failure modes can be inspected.
//
// Usage:
//
//	retrievaldump --artifact sealed.mll --dataset-dir /path/to/beir-dataset \
//	    [--split test] [--max-queries 10] [--failing-only] [--top-k 10]
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/directml"
	"github.com/odvcencio/manta/runtime/backends/metal"
	"github.com/odvcencio/manta/runtime/backends/vulkan"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
)

func tensorRows(t *backend.Tensor, wantRows int) ([][]float32, error) {
	if t == nil || len(t.F32) == 0 {
		return nil, fmt.Errorf("tensor has no float data")
	}
	switch len(t.Shape) {
	case 1:
		if wantRows != 1 {
			return nil, fmt.Errorf("shape %v cannot provide %d rows", t.Shape, wantRows)
		}
		return [][]float32{t.F32}, nil
	case 2:
		rows, cols := t.Shape[0], t.Shape[1]
		if rows != wantRows {
			return nil, fmt.Errorf("rows=%d, want %d", rows, wantRows)
		}
		out := make([][]float32, rows)
		for i := 0; i < rows; i++ {
			out[i] = t.F32[i*cols : (i+1)*cols]
		}
		return out, nil
	default:
		return nil, fmt.Errorf("shape %v not rank 1 or 2", t.Shape)
	}
}

type beirDoc struct {
	ID    string `json:"_id"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

func main() {
	var artifact, datasetDir, split string
	var maxQueries, topK int
	var failingOnly bool
	flag.StringVar(&artifact, "artifact", "", "sealed model .mll")
	flag.StringVar(&datasetDir, "dataset-dir", "", "BEIR dataset directory (with corpus.jsonl, queries.jsonl, qrels/)")
	flag.StringVar(&split, "split", "test", "qrels split")
	flag.IntVar(&maxQueries, "max-queries", 10, "limit queries to inspect")
	flag.IntVar(&topK, "top-k", 10, "retrieval depth to dump")
	flag.BoolVar(&failingOnly, "failing-only", true, "only show queries where no qrel hits top-K")
	flag.Parse()
	if artifact == "" || datasetDir == "" {
		fmt.Fprintln(os.Stderr, "usage: retrievaldump --artifact sealed.mll --dataset-dir <beir-dir> [--split test] [--max-queries N] [--top-k K] [--failing-only]")
		os.Exit(2)
	}

	corpusPath, queriesPath, qrelsPath := mantaruntime.BEIRRetrievalPaths(datasetDir, split)

	docs, err := readDocs(corpusPath)
	must(err, "read corpus")
	queries, err := readDocs(queriesPath)
	must(err, "read queries")
	qrels, err := readQrels(qrelsPath)
	must(err, "read qrels")

	fmt.Printf("artifact:    %s\n", artifact)
	fmt.Printf("dataset:     %s\n", datasetDir)
	fmt.Printf("corpus:      %d docs\n", len(docs))
	fmt.Printf("queries:     %d (keeping only those in qrels)\n", len(queries))
	fmt.Printf("qrels:       %d queries with %d relevant pairs\n", countQueries(qrels), countPairs(qrels))

	rt := mantaruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifact)
	must(err, "load model")

	fmt.Println("embedding corpus...")
	docVectors := make(map[string][]float32, len(docs))
	for i := 0; i < len(docs); i += 256 {
		end := i + 256
		if end > len(docs) {
			end = len(docs)
		}
		texts := make([]string, end-i)
		for k := 0; k < end-i; k++ {
			texts[k] = buildText(docs[i+k])
		}
		result, err := model.EmbedTextBatch(context.Background(), texts)
		must(err, "embed corpus batch")
		rows, err := tensorRows(result.Embeddings, len(texts))
		must(err, "tensor rows")
		for k, vec := range rows {
			// Clone so we don't share backing memory across batches.
			v := append([]float32(nil), vec...)
			docVectors[docs[i+k].ID] = normalize(v)
		}
	}

	// Build doc-ID to title map for labeling.
	docText := make(map[string]string, len(docs))
	for _, d := range docs {
		docText[d.ID] = d.Title
		if docText[d.ID] == "" {
			docText[d.ID] = firstN(d.Text, 80)
		}
	}

	qrelQueries := make([]beirDoc, 0, len(queries))
	qrelSet := make(map[string]bool, len(qrels))
	for qid := range qrels {
		qrelSet[qid] = true
	}
	for _, q := range queries {
		if qrelSet[q.ID] {
			qrelQueries = append(qrelQueries, q)
		}
	}
	sort.Slice(qrelQueries, func(i, j int) bool { return qrelQueries[i].ID < qrelQueries[j].ID })

	fmt.Printf("inspecting %d queries (failing_only=%v)...\n\n", maxQueries, failingOnly)
	shown := 0
	for _, q := range qrelQueries {
		if shown >= maxQueries {
			break
		}
		qText := buildText(q)
		qResult, err := model.EmbedTextBatch(context.Background(), []string{qText})
		must(err, "embed query")
		qRows, err := tensorRows(qResult.Embeddings, 1)
		must(err, "query tensor rows")
		qVec := normalize(append([]float32(nil), qRows[0]...))

		scored := make([]scoredDoc, 0, len(docVectors))
		for docID, dv := range docVectors {
			scored = append(scored, scoredDoc{ID: docID, Score: dot(qVec, dv)})
		}
		sort.Slice(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
		if topK > len(scored) {
			topK = len(scored)
		}
		top := scored[:topK]

		rels := qrels[q.ID]
		hits := 0
		for _, s := range top {
			if _, ok := rels[s.ID]; ok {
				hits++
			}
		}

		if failingOnly && hits > 0 {
			continue
		}
		shown++

		fmt.Printf("=== query %s ===\n", q.ID)
		fmt.Printf("  text:       %s\n", firstN(qText, 120))
		fmt.Printf("  #relevant:  %d (in corpus)\n", len(rels))
		fmt.Printf("  top-%d hits=%d\n", topK, hits)
		fmt.Printf("  top-%d:\n", topK)
		for i, s := range top {
			mark := "  "
			if _, ok := rels[s.ID]; ok {
				mark = "* "
			}
			title := docText[s.ID]
			if title == "" {
				title = "(no title)"
			}
			fmt.Printf("    %s%2d. score=%.4f id=%s  %s\n", mark, i+1, s.Score, s.ID, firstN(title, 90))
		}
		// Also show a sample of the relevant docs.
		if len(rels) > 0 {
			fmt.Printf("  actual relevant docs (sample, up to 5):\n")
			relIDs := make([]string, 0, len(rels))
			for id := range rels {
				relIDs = append(relIDs, id)
			}
			sort.Strings(relIDs)
			for i, id := range relIDs {
				if i >= 5 {
					break
				}
				// Compute rank of this doc in the full score list for reference.
				rank := -1
				for r, s := range scored {
					if s.ID == id {
						rank = r + 1
						break
					}
				}
				title := docText[id]
				if title == "" {
					title = "(no title)"
				}
				fmt.Printf("    - rank=%d id=%s  %s\n", rank, id, firstN(title, 90))
			}
		}
		fmt.Println()
	}
	fmt.Printf("done. shown=%d\n", shown)
}

type scoredDoc struct {
	ID    string
	Score float32
}

func buildText(d beirDoc) string {
	if d.Title == "" {
		return d.Text
	}
	return d.Title + " " + d.Text
}

func firstN(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func normalize(v []float32) []float32 {
	var sumsq float64
	for _, x := range v {
		sumsq += float64(x) * float64(x)
	}
	if sumsq == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(sumsq))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func dot(a, b []float32) float32 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return float32(s)
}

func readDocs(path string) ([]beirDoc, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<25)
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

func readQrels(path string) (map[string]map[string]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	out := map[string]map[string]float64{}
	scanner := bufio.NewScanner(f)
	first := true
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 3 {
			continue
		}
		if first && parts[0] == "query-id" {
			first = false
			continue
		}
		first = false
		q, d := parts[0], parts[1]
		rel, err := strconv.ParseFloat(parts[2], 64)
		if err != nil {
			continue
		}
		if rel <= 0 {
			continue
		}
		if out[q] == nil {
			out[q] = map[string]float64{}
		}
		out[q][d] = rel
	}
	return out, scanner.Err()
}

func countQueries(qrels map[string]map[string]float64) int { return len(qrels) }

func countPairs(qrels map[string]map[string]float64) int {
	n := 0
	for _, m := range qrels {
		n += len(m)
	}
	return n
}

func must(err error, label string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", label, err)
		os.Exit(1)
	}
}

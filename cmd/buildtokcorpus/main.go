// buildtokcorpus emits a tokenizer training corpus that covers the full
// vocabulary of one or more BEIR-format datasets.
//
// For each dataset directory passed in, reads queries.jsonl and corpus.jsonl
// and writes one normalized text line per query/document. Each line is
// title + " " + text (when title is present), so the BPE trainer sees the
// same string the production tokenizer will encode at retrieval time.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type beirDoc struct {
	ID    string `json:"_id"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

func emit(out *bufio.Writer, line string) error {
	line = strings.ReplaceAll(strings.TrimSpace(line), "\n", " ")
	if line == "" {
		return nil
	}
	if _, err := out.WriteString(line); err != nil {
		return err
	}
	return out.WriteByte('\n')
}

func process(path string, out *bufio.Writer) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)
	n := 0
	for scanner.Scan() {
		raw := scanner.Bytes()
		if len(raw) == 0 {
			continue
		}
		var d beirDoc
		if err := json.Unmarshal(raw, &d); err != nil {
			return n, fmt.Errorf("%s: %w", path, err)
		}
		text := d.Text
		if d.Title != "" {
			text = d.Title + " " + d.Text
		}
		if err := emit(out, text); err != nil {
			return n, err
		}
		n++
	}
	return n, scanner.Err()
}

// resolveBEIR finds the standard BEIR layout under a dataset root.
// Tries <root>/<name>/{queries,corpus}.jsonl first, then <root>/{queries,corpus}.jsonl.
func resolveBEIR(root string) (queries, corpus string, err error) {
	candidates := []struct{ q, c string }{
		{filepath.Join(root, filepath.Base(root), "queries.jsonl"), filepath.Join(root, filepath.Base(root), "corpus.jsonl")},
		{filepath.Join(root, "queries.jsonl"), filepath.Join(root, "corpus.jsonl")},
	}
	for _, pair := range candidates {
		if _, err1 := os.Stat(pair.q); err1 != nil {
			continue
		}
		if _, err2 := os.Stat(pair.c); err2 != nil {
			continue
		}
		return pair.q, pair.c, nil
	}
	return "", "", fmt.Errorf("no queries.jsonl/corpus.jsonl found under %s", root)
}

func main() {
	var output string
	flag.StringVar(&output, "out", "", "output path (default stdout)")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: buildtokcorpus [--out path] <beir-dataset-dir>...")
		os.Exit(2)
	}

	var sink *os.File
	if output == "" {
		sink = os.Stdout
	} else {
		f, err := os.Create(output)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer f.Close()
		sink = f
	}
	w := bufio.NewWriterSize(sink, 1<<20)
	defer w.Flush()

	totals := map[string][2]int{}
	for _, root := range flag.Args() {
		queries, corpus, err := resolveBEIR(root)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		nq, err := process(queries, w)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		nd, err := process(corpus, w)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		totals[filepath.Base(root)] = [2]int{nq, nd}
	}
	if err := w.Flush(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for name, c := range totals {
		fmt.Fprintf(os.Stderr, "%s: queries=%d docs=%d\n", name, c[0], c[1])
	}
	if output != "" {
		fmt.Fprintf(os.Stderr, "wrote %s\n", output)
	}
}

package mantaruntime

import (
	"context"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/compiler"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/cuda"
	"github.com/odvcencio/manta/runtime/backends/metal"
)

func TestEmbeddingManifestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "embed_pooled.embedding.mll")
	want := tinyEmbeddingManifest()
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("manifest mismatch:\nwant: %+v\ngot:  %+v", want, got)
	}
}

func TestDefaultEmbeddingManifestPath(t *testing.T) {
	got := DefaultEmbeddingManifestPath("/tmp/tiny_embed_pooled.mll")
	if want := "/tmp/tiny_embed_pooled.embedding.mll"; got != want {
		t.Fatalf("manifest path = %q, want %q", got, want)
	}
}

func TestLoadEmbeddingValidatesManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_score", Preset: compiler.PresetTinyScore})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	_, err = rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyScoreWeights()...)
	if err == nil {
		t.Fatal("expected embedding manifest validation error")
	}
	if !strings.Contains(err.Error(), `unknown entrypoint "embed_pooled"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbeddingModelEmbed(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if got := model.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestEmbeddingModelEmbedBatch(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.EmbedBatch(context.Background(), [][]int32{
		{0, 2},
		{1, 0},
	})
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0.5, 0.5,
	})
}

func TestEmbeddingModelRejectsTokenOutOfRange(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	_, err = model.Embed(context.Background(), []int32{0, 3})
	if err == nil {
		t.Fatal("expected token range error")
	}
	if !strings.Contains(err.Error(), "outside vocab_size 3") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEmbeddingModelRejectsSequenceTooLong(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	_, err = model.Embed(context.Background(), []int32{0, 1, 2})
	if err == nil {
		t.Fatal("expected max sequence error")
	}
	if !strings.Contains(err.Error(), "exceeds max_sequence 2") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestLoadEmbeddingFileUsesManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed_pooled.mll")
	manifestPath := filepath.Join(dir, "tiny_embed_pooled.embedding.mll")
	if err := mantaartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(manifestPath); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	manifest, err := ReadEmbeddingManifestFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingFile(context.Background(), artifactPath, manifest, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding file: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed from file: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestLoadEmbeddingBundleUsesSiblingManifest(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed_pooled.mll")
	if err := mantaartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingBundle(context.Background(), artifactPath, tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding bundle: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed from bundle: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
}

func TestLoadEmbeddingPackageUsesSiblingWeights(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_pooled", Preset: compiler.PresetTinyEmbedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed_pooled.mll")
	if err := mantaartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyEmbeddingManifest().WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	})
	if err := weights.WriteFile(DefaultWeightFilePath(artifactPath)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	if err := tinyEmbeddingTokenizerFile().WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	plan := NewMemoryPlan(bundle.Artifact, weights.Weights, MemoryPlanOptions{
		DeviceBudgetBytes: 6,
		SharedHostWeights: true,
	})
	if err := plan.WriteFile(DefaultMemoryPlanPath(artifactPath)); err != nil {
		t.Fatalf("write memory plan: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("load embedding package: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0, 2})
	if err != nil {
		t.Fatalf("embed from package: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0.8535534, 0.35355338,
	})
	if !model.HasTokenizer() {
		t.Fatal("expected packaged embedding model to load sibling tokenizer")
	}
	tokens, mask, err := model.TokenizeText("a")
	if err != nil {
		t.Fatalf("tokenize text: %v", err)
	}
	if !reflect.DeepEqual(tokens, []int32{2}) || !reflect.DeepEqual(mask, []int32{1}) {
		t.Fatalf("tokenized text = tokens %v mask %v, want [2] [1]", tokens, mask)
	}
	textResult, err := model.EmbedText(context.Background(), "a")
	if err != nil {
		t.Fatalf("embed text: %v", err)
	}
	tokenResult, err := model.Embed(context.Background(), []int32{2})
	if err != nil {
		t.Fatalf("embed tokenized text: %v", err)
	}
	assertTensorClose(t, textResult.Embeddings, tokenResult.Embeddings.Shape, tokenResult.Embeddings.F32)
	batchTextResult, err := model.EmbedTextBatch(context.Background(), []string{"a", "a"})
	if err != nil {
		t.Fatalf("embed text batch: %v", err)
	}
	if !reflect.DeepEqual(batchTextResult.Embeddings.Shape, []int{2, 2}) {
		t.Fatalf("text batch embedding shape = %v, want [2 2]", batchTextResult.Embeddings.Shape)
	}
	if got := model.MemoryPlan(); got == nil {
		t.Fatal("expected memory plan on loaded embedding package")
	} else if got.ModuleName != bundle.Artifact.Name {
		t.Fatalf("memory plan module name = %q, want %q", got.ModuleName, bundle.Artifact.Name)
	}
}

func TestLoadEmbeddingPackageAcceptsTrainingPackageManifest(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.02)
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.mll")
	if _, err := trainer.WriteTrainingPackage(path); err != nil {
		t.Fatalf("write training package: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		t.Fatalf("load embedding package from training bundle: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{0})
	if err != nil {
		t.Fatalf("embed from training bundle: %v", err)
	}
	if result.Embeddings == nil || len(result.Embeddings.Shape) != 1 {
		t.Fatalf("embedding shape = %v, want rank-1 tensor", result.Embeddings.Shape)
	}
}

func TestEmbeddingModelEmbedMasked(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyMaskedEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.Embed(context.Background(), []int32{1})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{
		0, 1,
	})
}

func TestEmbeddingModelEmbedBatchPadsRaggedWithMask(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyMaskedEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	result, err := model.EmbedBatch(context.Background(), [][]int32{
		{0, 2},
		{1},
	})
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2}, []float32{
		0.8535534, 0.35355338,
		0, 1,
	})
}

func TestEmbeddingModelEmbedBatchGroupsRaggedAttentionInputs(t *testing.T) {
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param attn_q: f16[D, D] @weight("weights/attn_q")
param attn_k: f16[D, D] @weight("weights/attn_k")
param attn_v: f16[D, D] @weight("weights/attn_v")
param attn_o: f16[D, D] @weight("weights/attn_o")
param projection: f16[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden = gather(token_embedding, tokens)
    let q = @matmul(hidden, attn_q)
    let k = @matmul(hidden, attn_k)
    let v = @matmul(hidden, attn_v)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, attn_o)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden = gather(token_embedding, tokens)
    let q = @matmul(hidden, attn_q)
    let k = @matmul(hidden, attn_k)
    let v = @matmul(hidden, attn_v)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, attn_o)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_attention_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyAttentionEmbeddingManifest(), tinyAttentionEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}

	longResult, err := model.Embed(context.Background(), []int32{0, 1})
	if err != nil {
		t.Fatalf("embed long: %v", err)
	}
	shortResult, err := model.Embed(context.Background(), []int32{1})
	if err != nil {
		t.Fatalf("embed short: %v", err)
	}
	batchResult, err := model.EmbedBatch(context.Background(), [][]int32{
		{0, 1},
		{1},
	})
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	want := append(append([]float32(nil), longResult.Embeddings.F32...), shortResult.Embeddings.F32...)
	assertTensorClose(t, batchResult.Embeddings, []int{2, 2}, want)
}

func tinyEmbeddingManifest() EmbeddingManifest {
	return EmbeddingManifest{
		Name:                "tiny_embed_pooled",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "f16",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   3,
			MaxSequence: 2,
			PadID:       0,
		},
	}
}

func tinyEmbeddingTokenizerFile() TokenizerFile {
	return TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a"},
		UnknownToken: "[UNK]",
	}
}

func tinyMaskedEmbeddingManifest() EmbeddingManifest {
	manifest := tinyEmbeddingManifest()
	manifest.Name = "tiny_embed_masked_pooled"
	manifest.MaskInput = "attention_mask"
	return manifest
}

func tinyAttentionEmbeddingManifest() EmbeddingManifest {
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Name = "tiny_attention_embed"
	manifest.AttentionQueryParam = "attn_q"
	manifest.AttentionKeyParam = "attn_k"
	manifest.AttentionValueParam = "attn_v"
	manifest.AttentionOutputParam = "attn_o"
	return manifest
}

func tinyAttentionEmbedWeights() []LoadOption {
	identity := backend.NewTensorF16([]int{2, 2}, []float32{
		1, 0,
		0, 1,
	})
	return []LoadOption{
		WithWeight("token_embedding", backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		})),
		WithWeight("attn_q", identity),
		WithWeight("attn_k", identity),
		WithWeight("attn_v", identity),
		WithWeight("attn_o", identity),
		WithWeight("projection", identity),
	}
}

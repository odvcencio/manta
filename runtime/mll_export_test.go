package barruntime

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/compiler"
	mll "github.com/odvcencio/mll"
)

func TestDefaultMLLPath(t *testing.T) {
	path := DefaultMLLPath("/tmp/model.barr")
	if path != "/tmp/model.mll" {
		t.Fatalf("DefaultMLLPath = %q, want %q", path, "/tmp/model.mll")
	}
}

func TestExportPackageToMLLWritesSealedContainer(t *testing.T) {
	path := writeMLLTrainableArtifact(t)
	if _, err := InitializeEmbeddingTrainerPackage(path, EmbeddingTrainInitOptions{
		Seed:       1,
		ShapeSizes: map[string]int{"D": 4, "E": 3},
	}); err != nil {
		t.Fatalf("initialize training package: %v", err)
	}

	outPath, err := ExportPackageToMLL(path, "")
	if err != nil {
		t.Fatalf("ExportPackageToMLL: %v", err)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("expected exported MLL file %q: %v", outPath, err)
	}

	reader, err := mll.ReadFile(outPath, mll.WithDigestVerification())
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if reader.Profile() != mll.ProfileSealed {
		t.Fatalf("profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}

	for _, tag := range [][4]byte{mll.TagHEAD, mll.TagSTRG, mll.TagDIMS, mll.TagTYPE, mll.TagPARM, mll.TagENTR, mll.TagMEMP, mll.TagTNSR, barr.MLLTagXBAR} {
		if _, ok := reader.Section(tag); !ok {
			t.Fatalf("missing section %q", string(tag[:]))
		}
	}

	typeBody, _ := reader.Section(mll.TagTYPE)
	typeSection, err := mll.ReadTypeSection(typeBody)
	if err != nil {
		t.Fatalf("ReadTypeSection: %v", err)
	}
	parmBody, _ := reader.Section(mll.TagPARM)
	parmSection, err := mll.ReadParmSection(parmBody)
	if err != nil {
		t.Fatalf("ReadParmSection: %v", err)
	}
	if len(parmSection.Decls) != 2 {
		t.Fatalf("param count = %d, want 2", len(parmSection.Decls))
	}
	firstTypeRef := parmSection.Decls[0].TypeRef
	if firstTypeRef.Tag != mll.TagTYPE {
		t.Fatalf("first param type tag = %q, want TYPE", string(firstTypeRef.Tag[:]))
	}
	if got := typeSection.Decls[firstTypeRef.Index].DType; got != mll.DTypeQ8 {
		t.Fatalf("logical param dtype = %d, want %d", got, mll.DTypeQ8)
	}

	mempBody, _ := reader.Section(mll.TagMEMP)
	mempSection, err := mll.ReadMempSection(mempBody)
	if err != nil {
		t.Fatalf("ReadMempSection: %v", err)
	}
	if len(mempSection.Entries) != 2 {
		t.Fatalf("memory plan count = %d, want 2", len(mempSection.Entries))
	}

	tnsrBody, _ := reader.Section(mll.TagTNSR)
	tnsrSection, err := mll.ReadTnsrSection(tnsrBody)
	if err != nil {
		t.Fatalf("ReadTnsrSection: %v", err)
	}
	if len(tnsrSection.Tensors) != 2 {
		t.Fatalf("tensor count = %d, want 2", len(tnsrSection.Tensors))
	}
	if got := tnsrSection.Tensors[0].DType; got != mll.DTypeF32 {
		t.Fatalf("stored tensor dtype = %d, want widened f32 (%d)", got, mll.DTypeF32)
	}

	xbarBody, _ := reader.Section(barr.MLLTagXBAR)
	meta, err := barr.DecodeMLLMetadata(xbarBody)
	if err != nil {
		t.Fatalf("decode XBAR: %v", err)
	}
	if meta.ModuleName != "tiny_train_embed" {
		t.Fatalf("XBAR module_name = %q", meta.ModuleName)
	}
	if meta.LogicalTensorDType["token_embedding"] != "q8" {
		t.Fatalf("logical dtype for token_embedding = %q, want q8", meta.LogicalTensorDType["token_embedding"])
	}
	if _, ok := meta.JSONFiles["embedding_manifest"]; !ok {
		t.Fatalf("expected embedding_manifest JSON metadata")
	}
	if _, ok := meta.JSONFiles["package_manifest"]; !ok {
		t.Fatalf("expected package_manifest JSON metadata")
	}
}

func TestExportPackageToMLLRequiresWeightsForParams(t *testing.T) {
	path := writeMLLTrainableArtifact(t)
	if _, err := ExportPackageToMLL(path, ""); err == nil {
		t.Fatalf("expected missing weight file error")
	}
}

func writeMLLTrainableArtifact(t *testing.T) string {
	t.Helper()
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.barr")
	if err := barr.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(path)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}

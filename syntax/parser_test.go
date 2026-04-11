package syntax

import "testing"

func TestParseTinyEmbedModule(t *testing.T) {
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

kernel l2_normalize(x: f16[T, E]) -> f16[T, E] {
    return normalize(x)
}

pipeline embed(tokens: i32[T]) -> f16[T, E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    return l2_normalize(projected)
}
`)

	file, diags := Parse("tiny_embed", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	if got := len(file.Decls); got != 4 {
		t.Fatalf("decl count = %d, want 4", got)
	}
	param0, ok := file.Decls[0].(*ParamDecl)
	if !ok || param0.Name != "token_embedding" {
		t.Fatalf("unexpected first decl: %#v", file.Decls[0])
	}
	pipe, ok := file.Decls[3].(*CallableDecl)
	if !ok || pipe.Kind != CallablePipeline || pipe.Name != "embed" {
		t.Fatalf("unexpected pipeline decl: %#v", file.Decls[3])
	}
	if got := len(pipe.Body); got != 3 {
		t.Fatalf("pipeline body length = %d, want 3", got)
	}
}

func TestParseBatchEmbedModule(t *testing.T) {
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

kernel l2_normalize(x: f16[B, T, E]) -> f16[B, T, E] {
    return normalize(x)
}

pipeline embed_batch(tokens: i32[B, T]) -> f16[B, T, E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    return l2_normalize(projected)
}
`)

	file, diags := Parse("batch_embed", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	if got := len(file.Decls); got != 4 {
		t.Fatalf("decl count = %d, want 4", got)
	}
	pipe, ok := file.Decls[3].(*CallableDecl)
	if !ok || pipe.Kind != CallablePipeline || pipe.Name != "embed_batch" {
		t.Fatalf("unexpected pipeline decl: %#v", file.Decls[3])
	}
	if got := len(pipe.Params[0].Type.Shape); got != 2 {
		t.Fatalf("tokens rank = %d, want 2", got)
	}
	if got := len(pipe.Result.Shape); got != 3 {
		t.Fatalf("result rank = %d, want 3", got)
	}
}

func TestParsePooledEmbedModule(t *testing.T) {
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T]) -> f16[E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> f16[B, E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized)
}
`)

	file, diags := Parse("pooled_embed", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	if got := len(file.Decls); got != 4 {
		t.Fatalf("decl count = %d, want 4", got)
	}
	pipe0 := file.Decls[2].(*CallableDecl)
	pipe1 := file.Decls[3].(*CallableDecl)
	if got := pipe0.Name; got != "embed_pooled" {
		t.Fatalf("first pooled pipeline name = %q, want embed_pooled", got)
	}
	if got := len(pipe0.Result.Shape); got != 1 {
		t.Fatalf("embed_pooled rank = %d, want 1", got)
	}
	if got := pipe1.Name; got != "embed_pooled_batch" {
		t.Fatalf("second pooled pipeline name = %q, want embed_pooled_batch", got)
	}
	if got := len(pipe1.Result.Shape); got != 2 {
		t.Fatalf("embed_pooled_batch rank = %d, want 2", got)
	}
}

func TestParseTrainableParamAnnotation(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    return mean_pool(hidden, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    return mean_pool(hidden, attention_mask)
}
`)

	file, diags := Parse("trainable_embed", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	param0, ok := file.Decls[0].(*ParamDecl)
	if !ok {
		t.Fatalf("first decl = %#v, want *ParamDecl", file.Decls[0])
	}
	if !param0.Trainable {
		t.Fatal("expected token_embedding param to be trainable")
	}
}

func TestParseMaskedPooledEmbedModule(t *testing.T) {
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let batch_hidden = gather(token_embedding, tokens)
    let batch_projected = @matmul(batch_hidden, projection)
    let batch_normalized = normalize(batch_projected)
    return mean_pool(batch_normalized, attention_mask)
}
`)

	file, diags := Parse("masked_pooled_embed", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	if got := len(file.Decls); got != 4 {
		t.Fatalf("decl count = %d, want 4", got)
	}
	pipe0 := file.Decls[2].(*CallableDecl)
	pipe1 := file.Decls[3].(*CallableDecl)
	if got := len(pipe0.Params); got != 2 {
		t.Fatalf("embed_pooled param count = %d, want 2", got)
	}
	if got := pipe0.Params[1].Name; got != "attention_mask" {
		t.Fatalf("embed_pooled mask param = %q, want attention_mask", got)
	}
	if got := len(pipe1.Params[1].Type.Shape); got != 2 {
		t.Fatalf("embed_pooled_batch mask rank = %d, want 2", got)
	}
}

func TestParseKVCachePipeline(t *testing.T) {
	src := []byte(`
param wq: f16[D, D] @weight("weights/wq")

pipeline decode_step(x: f16[T, D], cache: kv_cache) -> f16[T, D] {
    let q = @matmul(x, wq)
    let q2 = rope(q)
    let past = kv_read(cache)
    kv_write(cache, q2)
    return softmax(q2 + past)
}
`)

	file, diags := Parse("decode_step", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	pipe := file.Decls[1].(*CallableDecl)
	if got := pipe.Params[1].Type.Name; got != "kv_cache" {
		t.Fatalf("cache param type = %q, want kv_cache", got)
	}
}

func TestParseRerankTopKPipeline(t *testing.T) {
	src := []byte(`
param docs: q4[N, D] @weight("weights/docs")

pipeline rerank(query: f16[D]) -> (top_ids: i32[2], top_scores: f32[2]) {
    let scores = cosine(query, docs)
    let top_ids = topk(scores, 2)
    let top_scores = gather(scores, top_ids)
    return top_ids, top_scores
}
`)

	file, diags := Parse("rerank", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	pipe := file.Decls[1].(*CallableDecl)
	if got := pipe.Name; got != "rerank" {
		t.Fatalf("pipeline name = %q, want rerank", got)
	}
	if got := len(pipe.Results); got != 2 {
		t.Fatalf("pipeline result count = %d, want 2", got)
	}
	if got := pipe.Results[0].Name; got != "top_ids" {
		t.Fatalf("first result name = %q, want top_ids", got)
	}
	if got := pipe.Results[1].Name; got != "top_scores" {
		t.Fatalf("second result name = %q, want top_scores", got)
	}
	ret, ok := pipe.Body[3].(*ReturnStmt)
	if !ok {
		t.Fatalf("return stmt = %#v, want *ReturnStmt", pipe.Body[3])
	}
	if got := len(ret.Exprs); got != 2 {
		t.Fatalf("return expr count = %d, want 2", got)
	}
	id0, ok := ret.Exprs[0].(*IdentExpr)
	if !ok || id0.Name != "top_ids" {
		t.Fatalf("return expr[0] = %#v, want top_ids ident", ret.Exprs[0])
	}
	id1, ok := ret.Exprs[1].(*IdentExpr)
	if !ok || id1.Name != "top_scores" {
		t.Fatalf("return expr[1] = %#v, want top_scores ident", ret.Exprs[1])
	}
}

func TestParseBatchCandidatesPipeline(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> (top_candidate_ids: i64[Q, 2], top_scores: f32[Q, 2], top_docs: q4[Q, 2, D]) {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return top_candidate_ids, top_scores, top_docs
}
`)

	file, diags := Parse("rerank_candidates_batch", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	pipe := file.Decls[0].(*CallableDecl)
	if got := pipe.Name; got != "rerank_candidates_batch" {
		t.Fatalf("pipeline name = %q, want rerank_candidates_batch", got)
	}
	if got := len(pipe.Params); got != 3 {
		t.Fatalf("pipeline param count = %d, want 3", got)
	}
	if got := len(pipe.Results); got != 3 {
		t.Fatalf("pipeline result count = %d, want 3", got)
	}
	if got := pipe.Results[2].Type.Name; got != "q4" {
		t.Fatalf("third result dtype = %q, want q4", got)
	}
}

func TestParsePackedCandidatesPipeline(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_packed(query: f16[D], docs: q4[N, D], candidate_ids: i64[N]) -> candidate_pack[2, D] {
    let scores = cosine(query, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`)

	file, diags := Parse("rerank_candidates_packed", src)
	if len(diags) > 0 {
		t.Fatalf("unexpected diagnostics:\n%s", DebugString(diags))
	}
	pipe := file.Decls[0].(*CallableDecl)
	if got := pipe.Result.Name; got != "candidate_pack" {
		t.Fatalf("pipeline result type = %q, want candidate_pack", got)
	}
	if got := len(pipe.Result.Shape); got != 2 {
		t.Fatalf("pipeline result rank = %d, want 2", got)
	}
}

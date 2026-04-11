package compiler

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/ir/hir"
	"github.com/odvcencio/barracuda/ir/lir"
	"github.com/odvcencio/barracuda/ir/mir"
	"github.com/odvcencio/barracuda/syntax"
)

// Options configures compilation.
type Options struct {
	ModuleName string
	Preset     Preset
}

// Preset selects a built-in bootstrap module shape.
type Preset string

const (
	PresetTinyEmbed             Preset = "tiny_embed"
	PresetTinyEmbedPooled       Preset = "tiny_embed_pooled"
	PresetTinyEmbedMaskedPooled Preset = "tiny_embed_masked_pooled"
	PresetTinyDecode            Preset = "tiny_decode"
	PresetTinyScore             Preset = "tiny_score"
	PresetTinyRerank            Preset = "tiny_rerank"
	PresetTinySelect            Preset = "tiny_select"
	PresetTinyRetrieve          Preset = "tiny_retrieve"
	PresetTinyCandidates        Preset = "tiny_candidates"
	PresetTinyBatchCandidates   Preset = "tiny_batch_candidates"
	PresetTinyPackedCandidates  Preset = "tiny_packed_candidates"
)

// Bundle captures the compiler pipeline outputs.
type Bundle struct {
	Source   *syntax.File
	HIR      *hir.Module
	MIR      *mir.Module
	LIR      *lir.Plan
	Artifact *barr.Module
}

// Build runs the current stub pipeline and produces a minimal executable artifact.
func Build(src []byte, opts Options) (*Bundle, error) {
	name := opts.ModuleName
	if name == "" {
		name = "tiny_embed"
	}
	preset := opts.Preset
	if preset == "" {
		preset = PresetTinyEmbed
	}
	if len(src) == 0 {
		src = []byte(sourceForPreset(preset))
	}

	file, diags := syntax.Parse(name, src)
	if len(diags) > 0 {
		return nil, &DiagnosticsError{Diagnostics: diags}
	}
	if sema := semanticDiagnostics(file); len(sema) > 0 {
		return nil, &DiagnosticsError{Diagnostics: sema}
	}

	h := lowerHIR(file)
	m := lowerMIR(file)
	l, err := lowerLIR(file, h, m)
	if err != nil {
		return nil, err
	}
	a, err := lowerArtifact(file, h, l)
	if err != nil {
		return nil, err
	}

	if err := a.Validate(); err != nil {
		return nil, err
	}

	return &Bundle{
		Source:   file,
		HIR:      h,
		MIR:      m,
		LIR:      l,
		Artifact: a,
	}, nil
}

func lowerHIR(file *syntax.File) *hir.Module {
	mod := &hir.Module{Name: file.ModuleName}
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *syntax.ParamDecl:
			mod.Params = append(mod.Params, hir.Param{
				Name:      d.Name,
				Binding:   d.Binding,
				Type:      lowerTensorType(d.Type),
				Trainable: d.Trainable,
			})
		case *syntax.CallableDecl:
			entry := hir.EntryPoint{
				Name: d.Name,
				Kind: lowerEntryKind(d.Kind),
			}
			for _, param := range d.Params {
				entry.Inputs = append(entry.Inputs, hir.Value{Name: param.Name, Type: lowerType(param.Type)})
			}
			for _, result := range callableResultFields(d) {
				entry.Outputs = append(entry.Outputs, hir.Value{Name: result.Name, Type: lowerType(result.Type)})
			}
			mod.EntryPoints = append(mod.EntryPoints, entry)
		}
	}
	return mod
}

func lowerMIR(file *syntax.File) *mir.Module {
	mod := &mir.Module{Name: file.ModuleName}
	kernels := callableNameSet(file, syntax.CallableKernel)
	for _, decl := range file.Decls {
		callable, ok := decl.(*syntax.CallableDecl)
		if !ok {
			continue
		}
		for _, stmt := range callable.Body {
			mod.Ops = append(mod.Ops, collectOpsFromStmt(callable, stmt, kernels)...)
		}
	}
	return mod
}

func lowerLIR(file *syntax.File, h *hir.Module, m *mir.Module) (*lir.Plan, error) {
	plan := &lir.Plan{Name: m.Name}
	for _, param := range h.Params {
		plan.Params = append(plan.Params, lir.Param{
			Name:      param.Name,
			Binding:   param.Binding,
			DType:     param.Type.DType,
			Shape:     dimTexts(param.Type.Shape),
			Trainable: param.Trainable,
		})
	}
	env := newModuleTypeEnv(file)
	kernelNames := callableNameSet(file, syntax.CallableKernel)
	kernelSeen := map[string]bool{}
	for _, decl := range file.Decls {
		callable, ok := decl.(*syntax.CallableDecl)
		if !ok || callable.Kind != syntax.CallableKernel {
			continue
		}
		kernel, err := lowerKernelCallable(callable, kernelNames)
		if err != nil {
			return nil, err
		}
		plan.Kernels = append(plan.Kernels, kernel)
		kernelSeen[kernel.Name] = true
	}
	for _, decl := range file.Decls {
		callable, ok := decl.(*syntax.CallableDecl)
		if !ok || callable.Kind != syntax.CallablePipeline {
			continue
		}
		local := env.forCallable(callable)
		for _, param := range callable.Params {
			plan.Buffers = appendIfMissingBuffer(plan.Buffers, bufferForValue(param.Name, lowerType(param.Type), true))
		}
		for _, stmt := range callable.Body {
			steps, buffers, newKernels, err := lowerStmtToPlan(stmt, callable, local, kernelNames)
			if err != nil {
				return nil, err
			}
			plan.Steps = append(plan.Steps, steps...)
			for _, buf := range buffers {
				plan.Buffers = appendIfMissingBuffer(plan.Buffers, buf)
			}
			for _, kernel := range newKernels {
				if !kernelSeen[kernel.Name] {
					plan.Kernels = append(plan.Kernels, kernel)
					kernelSeen[kernel.Name] = true
				}
			}
		}
	}
	return plan, nil
}

func lowerArtifact(file *syntax.File, h *hir.Module, plan *lir.Plan) (*barr.Module, error) {
	mod := barr.NewModule(plan.Name)
	for _, param := range plan.Params {
		mod.Params = append(mod.Params, barr.Param{
			Name:      param.Name,
			Binding:   param.Binding,
			Trainable: param.Trainable,
			Type: barr.ValueType{
				Kind:   barr.ValueTensor,
				Tensor: &barr.TensorType{DType: param.DType, Shape: param.Shape},
			},
		})
	}
	for _, entry := range h.EntryPoints {
		if entry.Kind != hir.EntryPointPipeline {
			continue
		}
		out := barr.EntryPoint{Name: entry.Name, Kind: barr.EntryPointPipeline}
		for _, input := range entry.Inputs {
			out.Inputs = append(out.Inputs, barr.ValueBinding{Name: input.Name, Type: lowerArtifactValueType(input.Type)})
		}
		for _, output := range entry.Outputs {
			out.Outputs = append(out.Outputs, barr.ValueBinding{Name: output.Name, Type: lowerArtifactValueType(output.Type)})
		}
		mod.EntryPoints = append(mod.EntryPoints, out)
	}
	for _, buf := range plan.Buffers {
		mod.Buffers = append(mod.Buffers, barr.Buffer{
			Name:         buf.Name,
			DType:        buf.DType,
			Shape:        buf.Shape,
			StorageClass: buf.StorageClass,
		})
	}
	for _, kernel := range plan.Kernels {
		out := barr.Kernel{
			Name: kernel.Name,
			Hints: barr.ScheduleHints{
				Tile:        append([]int(nil), kernel.Hints.Tile...),
				VectorWidth: kernel.Hints.VectorWidth,
				Subgroup:    kernel.Hints.Subgroup,
				Memory:      kernel.Hints.Memory,
			},
			Variants: emitKernelVariants(kernel),
		}
		for _, input := range kernel.Inputs {
			out.Inputs = append(out.Inputs, lowerArtifactKernelValue(input))
		}
		for _, output := range kernel.Outputs {
			out.Outputs = append(out.Outputs, lowerArtifactKernelValue(output))
		}
		for _, op := range kernel.Body {
			out.Body = append(out.Body, lowerArtifactKernelOp(op))
		}
		mod.Kernels = append(mod.Kernels, out)
	}
	mod.Steps = append(mod.Steps, plan.Steps...)
	mod.Requirements.Capabilities = inferModuleCapabilities(mod)
	mod.Metadata = map[string]any{
		"source_decls": len(file.Decls),
	}
	return mod, nil
}

func inferModuleCapabilities(mod *barr.Module) []string {
	if mod == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	add := func(capability string) {
		if capability == "" || seen[capability] {
			return
		}
		seen[capability] = true
		out = append(out, capability)
	}

	for _, param := range mod.Params {
		addCapabilitiesFromValueType(add, param.Type)
	}
	for _, entry := range mod.EntryPoints {
		for _, input := range entry.Inputs {
			addCapabilitiesFromValueType(add, input.Type)
		}
		for _, output := range entry.Outputs {
			addCapabilitiesFromValueType(add, output.Type)
		}
	}
	for _, buf := range mod.Buffers {
		switch buf.DType {
		case "kv_cache":
			add(barr.CapabilityKVCache)
		case "candidate_pack":
			add(barr.CapabilityCandidatePack)
		}
	}
	for _, kernel := range mod.Kernels {
		for _, op := range kernel.Body {
			if op.Op == "mean_pool" && len(op.Inputs) > 1 {
				add(barr.CapabilityMaskedMeanPool)
			}
		}
	}
	for _, step := range mod.Steps {
		switch step.Kind {
		case barr.StepKVRead, barr.StepKVWrite:
			add(barr.CapabilityKVCache)
		case barr.StepPack:
			add(barr.CapabilityCandidatePack)
		}
	}
	return out
}

func addCapabilitiesFromValueType(add func(string), v barr.ValueType) {
	switch v.Kind {
	case barr.ValueKVCache:
		add(barr.CapabilityKVCache)
	case barr.ValueCandidatePack:
		add(barr.CapabilityCandidatePack)
	}
}

func sourceForPreset(preset Preset) string {
	switch preset {
	case PresetTinyEmbed:
		return `
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
`
	case PresetTinyEmbedPooled:
		return `
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T]) -> f16[E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> f16[B, E] {
    let batch_hidden = gather(token_embedding, tokens)
    let batch_projected = @matmul(batch_hidden, projection)
    let batch_normalized = normalize(batch_projected)
    return mean_pool(batch_normalized)
}
`
	case PresetTinyEmbedMaskedPooled:
		return `
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
`
	case PresetTinyDecode:
		return `
param wq: f16[D, D] @weight("weights/wq")

pipeline decode_step(x: f16[T, D], cache: kv_cache) -> f16[T, D] {
    let q = @matmul(x, wq)
    let q2 = rope(q)
    let past = kv_read(cache)
    kv_write(cache, q2)
    return softmax(q2 + past)
}
`
	case PresetTinyScore:
		return `
param docs: q4[N, D] @weight("weights/docs")

pipeline score(query: f16[D]) -> f32[N] {
    return cosine(query, docs)
}
`
	case PresetTinyRerank:
		return `
param docs: q4[N, D] @weight("weights/docs")

pipeline rerank(query: f16[D]) -> (top_ids: i32[2], top_scores: f32[2]) {
    let scores = cosine(query, docs)
    let top_ids = topk(scores, 2)
    let top_scores = gather(scores, top_ids)
    return top_ids, top_scores
}
`
	case PresetTinySelect:
		return `
param docs: q4[N, D] @weight("weights/docs")

pipeline select_scores(query: f16[D]) -> f32[2] {
    let scores = cosine(query, docs)
    let ids = topk(scores, 2)
    return gather(scores, ids)
}
`
	case PresetTinyRetrieve:
		return `
param docs: q4[N, D] @weight("weights/docs")

pipeline retrieve(query: f16[D]) -> (top_ids: i32[2], top_scores: f32[2], top_docs: q4[2, D]) {
    let scores = cosine(query, docs)
    let top_ids = topk(scores, 2)
    let top_scores = gather(scores, top_ids)
    let top_docs = gather(docs, top_ids)
    return top_ids, top_scores, top_docs
}
`
	case PresetTinyCandidates:
		return `
pipeline rerank_candidates(query: f16[D], docs: q4[N, D], candidate_ids: i64[N]) -> (top_candidate_ids: i64[2], top_scores: f32[2], top_docs: q4[2, D]) {
    let scores = cosine(query, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return top_candidate_ids, top_scores, top_docs
}
`
	case PresetTinyBatchCandidates:
		return `
pipeline rerank_candidates_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> (top_candidate_ids: i64[Q, 2], top_scores: f32[Q, 2], top_docs: q4[Q, 2, D]) {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return top_candidate_ids, top_scores, top_docs
}
`
	case PresetTinyPackedCandidates:
		return `
pipeline rerank_candidates_packed(query: f16[D], docs: q4[N, D], candidate_ids: i64[N]) -> candidate_pack[2, D] {
    let scores = cosine(query, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`
	default:
		return ""
	}
}

func lowerTensorType(t syntax.TypeRef) hir.TensorType {
	return hir.TensorType{DType: t.Name, Shape: lowerDims(t.Shape)}
}

func lowerDims(in []syntax.DimExpr) []hir.DimExpr {
	out := make([]hir.DimExpr, 0, len(in))
	for _, dim := range in {
		out = append(out, hir.DimExpr{Name: dim.Text})
	}
	return out
}

func lowerEntryKind(kind syntax.CallableKind) hir.EntryPointKind {
	if kind == syntax.CallableKernel {
		return hir.EntryPointKernel
	}
	return hir.EntryPointPipeline
}

func lowerType(t syntax.TypeRef) hir.Type {
	if t.Name == "kv_cache" {
		return hir.Type{Kind: hir.TypeKVCache}
	}
	if t.Name == "candidate_pack" {
		return hir.Type{Kind: hir.TypeCandidatePack, CandidatePack: &hir.CandidatePackType{Shape: lowerDims(t.Shape)}}
	}
	return hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: t.Name, Shape: lowerDims(t.Shape)}}
}

func lowerArtifactValueType(t hir.Type) barr.ValueType {
	switch t.Kind {
	case hir.TypeKVCache:
		return barr.ValueType{Kind: barr.ValueKVCache}
	case hir.TypeCandidatePack:
		return barr.ValueType{
			Kind:          barr.ValueCandidatePack,
			CandidatePack: &barr.CandidatePackType{Shape: dimTexts(t.CandidatePack.Shape)},
		}
	default:
		return barr.ValueType{
			Kind:   barr.ValueTensor,
			Tensor: &barr.TensorType{DType: t.Tensor.DType, Shape: dimTexts(t.Tensor.Shape)},
		}
	}
}

func lowerArtifactKernelValue(v lir.Value) barr.ValueBinding {
	switch v.Kind {
	case lir.ValueKVCache:
		return barr.ValueBinding{Name: v.Name, Type: barr.ValueType{Kind: barr.ValueKVCache}}
	default:
		return barr.ValueBinding{
			Name: v.Name,
			Type: barr.ValueType{
				Kind:   barr.ValueTensor,
				Tensor: &barr.TensorType{DType: v.DType, Shape: append([]string(nil), v.Shape...)},
			},
		}
	}
}

func lowerArtifactKernelOp(op lir.KernelOp) barr.KernelOp {
	kind := barr.KernelOpKind(op.Kind)
	return barr.KernelOp{
		Kind:       kind,
		Name:       op.Name,
		Op:         op.Op,
		Inputs:     append([]string(nil), op.Inputs...),
		Outputs:    append([]string(nil), op.Outputs...),
		Attributes: cloneStringMap(op.Attributes),
	}
}

type kernelVariantEmitter struct {
	backend barr.BackendKind
	entry   func(lir.Kernel) string
	source  func(lir.Kernel) string
}

type nativeKernelEmitter struct {
	rowKernel     func(lir.Kernel, string) string
	scoreKernel   func(lir.Kernel, string) string
	geluKernel    func(lir.Kernel) string
	binaryKernel  func(lir.Kernel, string) string
	dequantKernel func(lir.Kernel) string
	binaryExpr    func(string) string
	normalizeBody string
	layerNormBody string
	softmaxBody   string
	ropeBody      string
	scoreBodies   map[string]string
}

var kernelVariantEmitters = []kernelVariantEmitter{
	{
		backend: barr.BackendCUDA,
		entry: func(kernel lir.Kernel) string {
			return kernel.Name + "_cuda"
		},
		source: emitCUDAKernelSource,
	},
	{
		backend: barr.BackendMetal,
		entry: func(kernel lir.Kernel) string {
			return kernel.Name + "_metal"
		},
		source: emitMetalKernelSource,
	},
}

var cudaNativeEmitter = nativeKernelEmitter{
	rowKernel:     emitCUDARowKernel,
	scoreKernel:   emitCUDAScoreKernel,
	geluKernel:    emitCUDAGELUKernel,
	binaryKernel:  emitCUDAElementwiseBinaryKernel,
	dequantKernel: emitCUDADequantKernel,
	binaryExpr:    cudaBinaryExpr,
	normalizeBody: `
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = in0[base + c];
        sum += v * v;
    }
    float norm = sqrtf(sum);
    if (norm == 0.0f) norm = 1.0f;
    for (int c = 0; c < cols; ++c) {
        out0[base + c] = in0[base + c] / norm;
    }
`,
	layerNormBody: `
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
        mean += in0[base + c];
    }
    mean = mean / (float)cols;
    float variance = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float centered = in0[base + c] - mean;
        variance += centered * centered;
    }
    variance = variance / (float)cols;
    float inv_std = rsqrtf(variance + 1.0e-5f);
    for (int c = 0; c < cols; ++c) {
        out0[base + c] = (in0[base + c] - mean) * inv_std;
    }
`,
	softmaxBody: `
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    float max_v = in0[base];
    for (int c = 1; c < cols; ++c) {
        float v = in0[base + c];
        if (v > max_v) max_v = v;
    }
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = expf(in0[base + c] - max_v);
        out0[base + c] = v;
        sum += v;
    }
    if (sum == 0.0f) sum = 1.0f;
    for (int c = 0; c < cols; ++c) {
        out0[base + c] = out0[base + c] / sum;
    }
`,
	ropeBody: `
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    for (int c = 0; c + 1 < cols; c += 2) {
        float theta = ((float)row) / powf(10000.0f, ((float)c) / (float)cols);
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        float x0 = in0[base + c];
        float x1 = in0[base + c + 1];
        out0[base + c] = x0 * cos_theta - x1 * sin_theta;
        out0[base + c + 1] = x0 * sin_theta + x1 * cos_theta;
    }
    if ((cols & 1) != 0) {
        out0[base + cols - 1] = in0[base + cols - 1];
    }
`,
	scoreBodies: map[string]string{
		"dot": `
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    float dot = 0.0f;
    for (int c = 0; c < cols; ++c) {
        dot += query[c] * docs[base + c];
    }
    out0[row] = dot;
`,
		"cosine": `
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    float dot = 0.0f;
    float query_norm = 0.0f;
    float doc_norm = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float q = query[c];
        float d = docs[base + c];
        dot += q * d;
        query_norm += q * q;
        doc_norm += d * d;
    }
    float denom = sqrtf(query_norm) * sqrtf(doc_norm);
    out0[row] = denom == 0.0f ? 0.0f : dot / denom;
`,
		"l2_distance": `
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    int base = row * cols;
    float dist = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float diff = query[c] - docs[base + c];
        dist += diff * diff;
    }
    out0[row] = sqrtf(dist);
`,
	},
}

var metalNativeEmitter = nativeKernelEmitter{
	rowKernel:     emitMetalRowKernel,
	scoreKernel:   emitMetalScoreKernel,
	geluKernel:    emitMetalGELUKernel,
	binaryKernel:  emitMetalElementwiseBinaryKernel,
	dequantKernel: emitMetalDequantKernel,
	binaryExpr:    metalBinaryExpr,
	normalizeBody: `
    if ((int)gid >= rows) return;
    int base = ((int)gid) * cols;
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = in0[base + c];
        sum += v * v;
    }
    float norm = sqrt(sum);
    if (norm == 0.0f) norm = 1.0f;
    for (int c = 0; c < cols; ++c) {
        out0[base + c] = in0[base + c] / norm;
    }
`,
	layerNormBody: `
    if ((int)gid >= rows) return;
    int base = ((int)gid) * cols;
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
        mean += in0[base + c];
    }
    mean = mean / (float)cols;
    float variance = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float centered = in0[base + c] - mean;
        variance += centered * centered;
    }
    variance = variance / (float)cols;
    float inv_std = rsqrt(variance + 1.0e-5f);
    for (int c = 0; c < cols; ++c) {
        out0[base + c] = (in0[base + c] - mean) * inv_std;
    }
`,
	softmaxBody: `
    if ((int)gid >= rows) return;
    int base = ((int)gid) * cols;
    float max_v = in0[base];
    for (int c = 1; c < cols; ++c) {
        float v = in0[base + c];
        if (v > max_v) max_v = v;
    }
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float v = exp(in0[base + c] - max_v);
        out0[base + c] = v;
        sum += v;
    }
    if (sum == 0.0f) sum = 1.0f;
    for (int c = 0; c < cols; ++c) {
        out0[base + c] = out0[base + c] / sum;
    }
`,
	ropeBody: `
    if ((int)gid >= rows) return;
    int base = ((int)gid) * cols;
    for (int c = 0; c + 1 < cols; c += 2) {
        float theta = ((float)gid) / pow(10000.0f, ((float)c) / (float)cols);
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        float x0 = in0[base + c];
        float x1 = in0[base + c + 1];
        out0[base + c] = x0 * cos_theta - x1 * sin_theta;
        out0[base + c + 1] = x0 * sin_theta + x1 * cos_theta;
    }
    if ((cols & 1) != 0) {
        out0[base + cols - 1] = in0[base + cols - 1];
    }
`,
	scoreBodies: map[string]string{
		"dot": `
    if ((int)gid >= rows) return;
    int base = ((int)gid) * cols;
    float dot = 0.0f;
    for (int c = 0; c < cols; ++c) {
        dot += query[c] * docs[base + c];
    }
    out0[gid] = dot;
`,
		"cosine": `
    if ((int)gid >= rows) return;
    int base = ((int)gid) * cols;
    float dot = 0.0f;
    float query_norm = 0.0f;
    float doc_norm = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float q = query[c];
        float d = docs[base + c];
        dot += q * d;
        query_norm += q * q;
        doc_norm += d * d;
    }
    float denom = sqrt(query_norm) * sqrt(doc_norm);
    out0[gid] = denom == 0.0f ? 0.0f : dot / denom;
`,
		"l2_distance": `
    if ((int)gid >= rows) return;
    int base = ((int)gid) * cols;
    float dist = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float diff = query[c] - docs[base + c];
        dist += diff * diff;
    }
    out0[gid] = sqrt(dist);
`,
	},
}

func emitKernelVariants(kernel lir.Kernel) []barr.KernelVariant {
	meta := map[string]string{
		"tile":         scheduleTileString(kernel.Hints.Tile),
		"vector_width": strconv.Itoa(kernel.Hints.VectorWidth),
		"subgroup":     strconv.FormatBool(kernel.Hints.Subgroup),
		"memory":       kernel.Hints.Memory,
	}
	variants := make([]barr.KernelVariant, 0, len(kernelVariantEmitters))
	for _, emitter := range kernelVariantEmitters {
		variants = append(variants, barr.KernelVariant{
			Backend: emitter.backend,
			Entry:   emitter.entry(kernel),
			Source:  emitter.source(kernel),
			Meta:    cloneStringMap(meta),
		})
	}
	return variants
}

func emitCUDAKernelSource(kernel lir.Kernel) string {
	if src, ok := emitNativeCUDAKernelSource(kernel); ok {
		return src
	}
	var b strings.Builder
	b.WriteString("#include <cuda_fp16.h>\n\n")
	b.WriteString("extern \"C\" __global__ void ")
	b.WriteString(kernel.Name)
	b.WriteString("_cuda(")
	b.WriteString(strings.Join(cudaKernelParams(kernel), ", "))
	b.WriteString(") {\n")
	b.WriteString("  // tile=")
	b.WriteString(scheduleTileString(kernel.Hints.Tile))
	b.WriteString(" vector_width=")
	b.WriteString(strconv.Itoa(kernel.Hints.VectorWidth))
	b.WriteString(" subgroup=")
	b.WriteString(strconv.FormatBool(kernel.Hints.Subgroup))
	b.WriteString(" memory=")
	b.WriteString(kernel.Hints.Memory)
	b.WriteString("\n")
	for _, op := range kernel.Body {
		b.WriteString("  ")
		b.WriteString(renderKernelOpComment(op))
		b.WriteString("\n")
	}
	b.WriteString("}\n")
	return b.String()
}

func emitNativeCUDAKernelSource(kernel lir.Kernel) (string, bool) {
	return emitNativeKernelSourceWithEmitter(cudaNativeEmitter, kernel)
}

func emitNativeKernelSourceWithEmitter(emitter nativeKernelEmitter, kernel lir.Kernel) (string, bool) {
	if len(kernel.Body) < 2 {
		return "", false
	}
	op := kernel.Body[0]
	if kernel.Body[len(kernel.Body)-1].Op != "return" {
		return "", false
	}
	switch op.Op {
	case "normalize", "rmsnorm":
		if len(kernel.Inputs) != 1 || len(kernel.Outputs) != 1 {
			return "", false
		}
		return emitter.rowKernel(kernel, emitter.normalizeBody), true
	case "layernorm":
		if len(kernel.Inputs) != 1 || len(kernel.Outputs) != 1 {
			return "", false
		}
		return emitter.rowKernel(kernel, emitter.layerNormBody), true
	case "softmax":
		if len(kernel.Inputs) != 1 || len(kernel.Outputs) != 1 {
			return "", false
		}
		return emitter.rowKernel(kernel, emitter.softmaxBody), true
	case "gelu":
		if len(kernel.Inputs) != 1 || len(kernel.Outputs) != 1 {
			return "", false
		}
		return emitter.geluKernel(kernel), true
	case "rope":
		if len(kernel.Inputs) != 1 || len(kernel.Outputs) != 1 {
			return "", false
		}
		return emitter.rowKernel(kernel, emitter.ropeBody), true
	case "binary_add", "binary_sub", "binary_mul", "binary_div":
		if len(kernel.Inputs) != 2 || len(kernel.Outputs) != 1 {
			return "", false
		}
		return emitter.binaryKernel(kernel, emitter.binaryExpr(op.Op)), true
	case "dequant":
		if len(kernel.Inputs) != 1 || len(kernel.Outputs) != 1 {
			return "", false
		}
		return emitter.dequantKernel(kernel), true
	case "dot", "cosine", "l2_distance":
		if len(kernel.Inputs) != 2 || len(kernel.Outputs) != 1 {
			return "", false
		}
		body, ok := emitter.scoreBodies[op.Op]
		if !ok {
			return "", false
		}
		return emitter.scoreKernel(kernel, body), true
	default:
		return "", false
	}
}

func emitCUDARowKernel(kernel lir.Kernel, body string) string {
	var b strings.Builder
	b.WriteString("extern \"C\" __global__ void ")
	b.WriteString(kernel.Name)
	b.WriteString("_cuda(const float* in0, float* out0, int rows, int cols) {\n")
	b.WriteString(body)
	b.WriteString("}\n")
	return b.String()
}

func emitCUDAScoreKernel(kernel lir.Kernel, body string) string {
	var b strings.Builder
	b.WriteString("extern \"C\" __global__ void ")
	b.WriteString(kernel.Name)
	b.WriteString("_cuda(const float* query, const float* docs, float* out0, int rows, int cols) {\n")
	b.WriteString(body)
	b.WriteString("}\n")
	return b.String()
}

func emitCUDAGELUKernel(kernel lir.Kernel) string {
	var b strings.Builder
	b.WriteString("extern \"C\" __global__ void ")
	b.WriteString(kernel.Name)
	b.WriteString("_cuda(const float* in0, float* out0, int elements) {\n")
	b.WriteString("    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n")
	b.WriteString("    if (idx >= elements) return;\n")
	b.WriteString("    float x = in0[idx];\n")
	b.WriteString("    float cubic = x * x * x;\n")
	b.WriteString("    float inner = 0.7978845608f * (x + 0.044715f * cubic);\n")
	b.WriteString("    out0[idx] = 0.5f * x * (1.0f + tanhf(inner));\n")
	b.WriteString("}\n")
	return b.String()
}

func emitCUDAElementwiseBinaryKernel(kernel lir.Kernel, expr string) string {
	var b strings.Builder
	b.WriteString("extern \"C\" __global__ void ")
	b.WriteString(kernel.Name)
	b.WriteString("_cuda(const float* lhs, const float* rhs, float* out0, int elements) {\n")
	b.WriteString("    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n")
	b.WriteString("    if (idx >= elements) return;\n")
	b.WriteString("    out0[idx] = ")
	b.WriteString(expr)
	b.WriteString(";\n")
	b.WriteString("}\n")
	return b.String()
}

func emitCUDADequantKernel(kernel lir.Kernel) string {
	var b strings.Builder
	b.WriteString("extern \"C\" __global__ void ")
	b.WriteString(kernel.Name)
	b.WriteString("_cuda(const float* in0, float* out0, int elements) {\n")
	b.WriteString("    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n")
	b.WriteString("    if (idx >= elements) return;\n")
	b.WriteString("    out0[idx] = in0[idx];\n")
	b.WriteString("}\n")
	return b.String()
}

func cudaBinaryExpr(op string) string {
	switch op {
	case "binary_sub":
		return "lhs[idx] - rhs[idx]"
	case "binary_mul":
		return "lhs[idx] * rhs[idx]"
	case "binary_div":
		return "lhs[idx] / rhs[idx]"
	default:
		return "lhs[idx] + rhs[idx]"
	}
}

func emitMetalKernelSource(kernel lir.Kernel) string {
	if src, ok := emitNativeMetalKernelSource(kernel); ok {
		return src
	}
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\n")
	b.WriteString("kernel void ")
	b.WriteString(kernel.Name)
	b.WriteString("_metal(")
	b.WriteString(strings.Join(metalKernelParams(kernel), ", "))
	b.WriteString(") {\n")
	b.WriteString("  // tile=")
	b.WriteString(scheduleTileString(kernel.Hints.Tile))
	b.WriteString(" vector_width=")
	b.WriteString(strconv.Itoa(kernel.Hints.VectorWidth))
	b.WriteString(" subgroup=")
	b.WriteString(strconv.FormatBool(kernel.Hints.Subgroup))
	b.WriteString(" memory=")
	b.WriteString(kernel.Hints.Memory)
	b.WriteString("\n")
	for _, op := range kernel.Body {
		b.WriteString("  ")
		b.WriteString(renderKernelOpComment(op))
		b.WriteString("\n")
	}
	b.WriteString("}\n")
	return b.String()
}

func emitNativeMetalKernelSource(kernel lir.Kernel) (string, bool) {
	return emitNativeKernelSourceWithEmitter(metalNativeEmitter, kernel)
}

func emitMetalRowKernel(kernel lir.Kernel, body string) string {
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\n")
	b.WriteString("kernel void ")
	b.WriteString(kernel.Name)
	b.WriteString("_metal(")
	b.WriteString("const device float* in0 [[buffer(0)]], ")
	b.WriteString("device float* out0 [[buffer(1)]], ")
	b.WriteString("constant int& rows [[buffer(2)]], ")
	b.WriteString("constant int& cols [[buffer(3)]], ")
	b.WriteString("uint gid [[thread_position_in_grid]]) {\n")
	b.WriteString(body)
	b.WriteString("}\n")
	return b.String()
}

func emitMetalScoreKernel(kernel lir.Kernel, body string) string {
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\n")
	b.WriteString("kernel void ")
	b.WriteString(kernel.Name)
	b.WriteString("_metal(")
	b.WriteString("const device float* query [[buffer(0)]], ")
	b.WriteString("const device float* docs [[buffer(1)]], ")
	b.WriteString("device float* out0 [[buffer(2)]], ")
	b.WriteString("constant int& rows [[buffer(3)]], ")
	b.WriteString("constant int& cols [[buffer(4)]], ")
	b.WriteString("uint gid [[thread_position_in_grid]]) {\n")
	b.WriteString(body)
	b.WriteString("}\n")
	return b.String()
}

func emitMetalGELUKernel(kernel lir.Kernel) string {
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\n")
	b.WriteString("kernel void ")
	b.WriteString(kernel.Name)
	b.WriteString("_metal(")
	b.WriteString("const device float* in0 [[buffer(0)]], ")
	b.WriteString("device float* out0 [[buffer(1)]], ")
	b.WriteString("constant int& elements [[buffer(2)]], ")
	b.WriteString("uint gid [[thread_position_in_grid]]) {\n")
	b.WriteString("    if ((int)gid >= elements) return;\n")
	b.WriteString("    float x = in0[gid];\n")
	b.WriteString("    float cubic = x * x * x;\n")
	b.WriteString("    float inner = 0.7978845608f * (x + 0.044715f * cubic);\n")
	b.WriteString("    out0[gid] = 0.5f * x * (1.0f + tanh(inner));\n")
	b.WriteString("}\n")
	return b.String()
}

func emitMetalElementwiseBinaryKernel(kernel lir.Kernel, expr string) string {
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\n")
	b.WriteString("kernel void ")
	b.WriteString(kernel.Name)
	b.WriteString("_metal(")
	b.WriteString("const device float* lhs [[buffer(0)]], ")
	b.WriteString("const device float* rhs [[buffer(1)]], ")
	b.WriteString("device float* out0 [[buffer(2)]], ")
	b.WriteString("constant int& elements [[buffer(3)]], ")
	b.WriteString("uint gid [[thread_position_in_grid]]) {\n")
	b.WriteString("    if ((int)gid >= elements) return;\n")
	b.WriteString("    out0[gid] = ")
	b.WriteString(expr)
	b.WriteString(";\n")
	b.WriteString("}\n")
	return b.String()
}

func emitMetalDequantKernel(kernel lir.Kernel) string {
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\n")
	b.WriteString("kernel void ")
	b.WriteString(kernel.Name)
	b.WriteString("_metal(")
	b.WriteString("const device float* in0 [[buffer(0)]], ")
	b.WriteString("device float* out0 [[buffer(1)]], ")
	b.WriteString("constant int& elements [[buffer(2)]], ")
	b.WriteString("uint gid [[thread_position_in_grid]]) {\n")
	b.WriteString("    if ((int)gid >= elements) return;\n")
	b.WriteString("    out0[gid] = in0[gid];\n")
	b.WriteString("}\n")
	return b.String()
}

func metalBinaryExpr(op string) string {
	switch op {
	case "binary_sub":
		return "lhs[gid] - rhs[gid]"
	case "binary_mul":
		return "lhs[gid] * rhs[gid]"
	case "binary_div":
		return "lhs[gid] / rhs[gid]"
	default:
		return "lhs[gid] + rhs[gid]"
	}
}

func cudaKernelParams(kernel lir.Kernel) []string {
	params := make([]string, 0, len(kernel.Inputs)+len(kernel.Outputs))
	for _, input := range kernel.Inputs {
		params = append(params, "const "+cudaType(input)+"* "+input.Name)
	}
	for _, output := range kernel.Outputs {
		params = append(params, cudaType(output)+"* "+output.Name)
	}
	params = append(params, "int gid")
	return params
}

func metalKernelParams(kernel lir.Kernel) []string {
	params := make([]string, 0, len(kernel.Inputs)+len(kernel.Outputs)+1)
	for i, input := range kernel.Inputs {
		params = append(params, "const device "+metalType(input)+"* "+input.Name+" [[buffer("+strconv.Itoa(i)+")]]")
	}
	base := len(kernel.Inputs)
	for i, output := range kernel.Outputs {
		params = append(params, "device "+metalType(output)+"* "+output.Name+" [[buffer("+strconv.Itoa(base+i)+")]]")
	}
	params = append(params, "uint gid [[thread_position_in_grid]]")
	return params
}

func cudaType(v lir.Value) string {
	switch v.DType {
	case "f16":
		return "half"
	case "f32":
		return "float"
	case "i32":
		return "int"
	default:
		return "float"
	}
}

func metalType(v lir.Value) string {
	switch v.DType {
	case "f16":
		return "half"
	case "f32":
		return "float"
	case "i32":
		return "int"
	default:
		return "float"
	}
}

func renderKernelOpComment(op lir.KernelOp) string {
	var b strings.Builder
	b.WriteString("// ")
	b.WriteString(string(op.Kind))
	b.WriteString(" ")
	b.WriteString(op.Op)
	if len(op.Inputs) > 0 {
		b.WriteString(" in=")
		b.WriteString(strings.Join(op.Inputs, ","))
	}
	if len(op.Outputs) > 0 {
		b.WriteString(" out=")
		b.WriteString(strings.Join(op.Outputs, ","))
	}
	return b.String()
}

func scheduleTileString(tile []int) string {
	if len(tile) == 0 {
		return "[]"
	}
	parts := make([]string, 0, len(tile))
	for _, n := range tile {
		parts = append(parts, strconv.Itoa(n))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func dimTexts(dims []hir.DimExpr) []string {
	out := make([]string, 0, len(dims))
	for _, dim := range dims {
		out = append(out, dim.Name)
	}
	return out
}

func resultNameForCallable(d *syntax.CallableDecl) string {
	if d.Kind == syntax.CallablePipeline && len(d.Results) > 0 {
		return d.Results[0].Name
	}
	if d.Kind == syntax.CallableKernel {
		return "result"
	}
	switch d.Name {
	case "embed":
		return "embeddings"
	case "decode_step":
		return "logits"
	case "score":
		return "scores"
	case "rerank":
		return "top_ids"
	case "rerank_candidates_packed":
		return "candidates"
	case "select_scores":
		return "top_scores"
	default:
		return "result"
	}
}

func callableNameSet(file *syntax.File, kind syntax.CallableKind) map[string]bool {
	out := map[string]bool{}
	for _, decl := range file.Decls {
		callable, ok := decl.(*syntax.CallableDecl)
		if ok && callable.Kind == kind {
			out[callable.Name] = true
		}
	}
	return out
}

func collectOpsFromStmt(callable *syntax.CallableDecl, stmt syntax.Stmt, kernels map[string]bool) []mir.Op {
	switch s := stmt.(type) {
	case *syntax.LetStmt:
		return collectOpsFromExpr(callable.Name+"."+s.Name, s.Expr, []string{s.Name}, kernels)
	case *syntax.ReturnStmt:
		exprs := returnStmtExprs(s)
		results := callableResultFields(callable)
		out := []mir.Op{}
		for i := 0; i < len(exprs) && i < len(results); i++ {
			out = append(out, collectOpsFromExpr(callable.Name+".return."+results[i].Name, exprs[i], []string{results[i].Name}, kernels)...)
		}
		return out
	case *syntax.ExprStmt:
		return collectOpsFromExpr(callable.Name+".expr", s.Expr, nil, kernels)
	default:
		return nil
	}
}

func collectOpsFromExpr(name string, expr syntax.Expr, outputs []string, kernels map[string]bool) []mir.Op {
	call, ok := expr.(*syntax.CallExpr)
	if !ok {
		return nil
	}
	op := mir.Op{
		Name:    name,
		Kind:    lowerOpKind(call, kernels),
		Callee:  call.Callee,
		Inputs:  exprArgNames(call.Args),
		Outputs: append([]string(nil), outputs...),
	}
	return []mir.Op{op}
}

func lowerOpKind(call *syntax.CallExpr, kernels map[string]bool) mir.OpKind {
	if call.Intrinsic && call.Callee == "matmul" {
		return mir.OpMatMul
	}
	switch call.Callee {
	case "gather":
		return mir.OpGather
	case "dequant":
		return mir.OpDequant
	case "topk":
		return mir.OpTopK
	case "softmax":
		return mir.OpSoftmax
	case "rope":
		return mir.OpRoPE
	case "mean_pool":
		return mir.OpReduce
	case "kv_read":
		return mir.OpKVRead
	case "kv_write":
		return mir.OpKVWrite
	case "dot", "cosine", "l2_distance":
		return mir.OpScore
	case "rmsnorm", "normalize", "layernorm", "gelu":
		return mir.OpPointwise
	default:
		if kernels[call.Callee] {
			return mir.OpKernel
		}
		return mir.OpPointwise
	}
}

func exprArgNames(args []syntax.Expr) []string {
	out := make([]string, 0, len(args))
	for _, arg := range args {
		out = append(out, exprName(arg))
	}
	return out
}

func exprName(expr syntax.Expr) string {
	switch e := expr.(type) {
	case *syntax.IdentExpr:
		return e.Name
	case *syntax.NumberExpr:
		return e.Text
	case *syntax.StringExpr:
		return strconvSafeName(e.Value)
	case *syntax.CallExpr:
		return e.Callee
	case *syntax.BinaryExpr:
		return exprName(e.Left) + "_" + sanitizeOp(e.Op) + "_" + exprName(e.Right)
	default:
		return "value"
	}
}

func sanitizeOp(op string) string {
	switch op {
	case "+":
		return "add"
	case "-":
		return "sub"
	case "*":
		return "mul"
	case "/":
		return "div"
	default:
		return "op"
	}
}

func strconvSafeName(in string) string {
	if in == "" {
		return "string"
	}
	return strings.ReplaceAll(in, " ", "_")
}

type moduleTypeEnv struct {
	params    map[string]hir.Type
	callables map[string]*syntax.CallableDecl
}

func newModuleTypeEnv(file *syntax.File) moduleTypeEnv {
	env := moduleTypeEnv{
		params:    map[string]hir.Type{},
		callables: map[string]*syntax.CallableDecl{},
	}
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *syntax.ParamDecl:
			env.params[d.Name] = lowerType(d.Type)
		case *syntax.CallableDecl:
			env.callables[d.Name] = d
		}
	}
	return env
}

func (e moduleTypeEnv) forCallable(callable *syntax.CallableDecl) map[string]hir.Type {
	out := map[string]hir.Type{}
	for name, typ := range e.params {
		out[name] = typ
	}
	for _, param := range callable.Params {
		out[param.Name] = lowerType(param.Type)
	}
	return out
}

func lowerStmtToPlan(stmt syntax.Stmt, callable *syntax.CallableDecl, env map[string]hir.Type, kernels map[string]bool) ([]barr.Step, []lir.Buffer, []lir.Kernel, error) {
	switch s := stmt.(type) {
	case *syntax.LetStmt:
		typ, steps, buffers, kernelsAdded, err := inferExprPlan(s.Expr, s.Name, callable, env, kernels)
		if err != nil {
			return nil, nil, nil, err
		}
		env[s.Name] = typ
		buffers = appendIfMissingBuffer(buffers, bufferForValue(s.Name, typ, false))
		return withStepEntry(steps, callable.Name), buffers, kernelsAdded, nil
	case *syntax.ReturnStmt:
		exprs := returnStmtExprs(s)
		results := callableResultFields(callable)
		steps := []barr.Step{}
		buffers := []lir.Buffer{}
		kernelsAdded := []lir.Kernel{}
		outputs := make([]string, 0, len(results))
		for i := 0; i < len(exprs) && i < len(results); i++ {
			name := results[i].Name
			typ, exprSteps, exprBuffers, exprKernels, err := inferExprPlan(exprs[i], name, callable, env, kernels)
			if err != nil {
				return nil, nil, nil, err
			}
			env[name] = typ
			steps = append(steps, exprSteps...)
			for _, buf := range exprBuffers {
				buffers = appendIfMissingBuffer(buffers, buf)
			}
			buffers = appendIfMissingBuffer(buffers, bufferForValue(name, typ, false))
			kernelsAdded = appendKernelsIfMissing(kernelsAdded, exprKernels...)
			outputs = append(outputs, name)
		}
		steps = append(steps, barr.Step{Kind: barr.StepReturn, Outputs: outputs})
		return withStepEntry(steps, callable.Name), buffers, kernelsAdded, nil
	case *syntax.ExprStmt:
		_, steps, buffers, kernelsAdded, err := inferExprPlan(s.Expr, "", callable, env, kernels)
		if err != nil {
			return nil, nil, nil, err
		}
		return withStepEntry(steps, callable.Name), buffers, kernelsAdded, nil
	default:
		return nil, nil, nil, nil
	}
}

func inferExprPlan(expr syntax.Expr, output string, callable *syntax.CallableDecl, env map[string]hir.Type, kernels map[string]bool) (hir.Type, []barr.Step, []lir.Buffer, []lir.Kernel, error) {
	switch e := expr.(type) {
	case *syntax.IdentExpr:
		typ, ok := env[e.Name]
		if !ok {
			scope := "expression"
			if callable != nil {
				scope = callable.Name
			}
			return hir.Type{}, nil, nil, nil, fmt.Errorf("unknown identifier %q in %s", e.Name, scope)
		}
		if output != "" && output != e.Name {
			return typ,
				[]barr.Step{{Kind: barr.StepAlias, Name: output, Inputs: []string{e.Name}, Outputs: []string{output}}},
				[]lir.Buffer{bufferForValue(output, typ, false)},
				nil,
				nil
		}
		return typ, nil, nil, nil, nil
	case *syntax.BinaryExpr:
		leftType, err := inferExprType(e.Left, env)
		if err != nil {
			return hir.Type{}, nil, nil, nil, err
		}
		if output == "" {
			return leftType, nil, nil, nil, nil
		}
		name := binaryKernelName(e.Op)
		return leftType,
			[]barr.Step{{Kind: barr.StepLaunchKernel, Name: name, Kernel: name, Inputs: []string{exprName(e.Left), exprName(e.Right)}, Outputs: []string{output}}},
			[]lir.Buffer{bufferForValue(output, leftType, false)},
			[]lir.Kernel{synthesizeBinaryKernel(e, leftType, env)},
			nil
	case *syntax.CallExpr:
		return inferCallPlan(e, output, env, kernels)
	default:
		return hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f32"}}, nil, nil, nil, nil
	}
}

func inferCallPlan(call *syntax.CallExpr, output string, env map[string]hir.Type, kernels map[string]bool) (hir.Type, []barr.Step, []lir.Buffer, []lir.Kernel, error) {
	if !call.Intrinsic && call.Callee == "topk" {
		if len(call.Args) != 2 {
			return hir.Type{}, nil, nil, nil, fmt.Errorf("topk expects 2 args")
		}
		scoreType, scoreName, steps, buffers, newKernels, err := materializeCallArg(call.Args[0], output, call.Callee, 0, env, kernels)
		if err != nil {
			return hir.Type{}, nil, nil, nil, err
		}
		k, ok := topKLiteralExpr(call.Args[1])
		if !ok {
			return hir.Type{}, nil, nil, nil, fmt.Errorf("topk requires a positive integer literal limit")
		}
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "i32"}}
		switch {
		case isRank1HIRTensor(scoreType):
			out.Tensor.Shape = []hir.DimExpr{{Name: fmt.Sprintf("%d", k)}}
		case isRank2HIRTensor(scoreType):
			out.Tensor.Shape = []hir.DimExpr{
				{Name: scoreType.Tensor.Shape[0].Name},
				{Name: fmt.Sprintf("%d", k)},
			}
		default:
			return hir.Type{}, nil, nil, nil, fmt.Errorf("topk expects rank-1 or rank-2 score input")
		}
		steps = append(steps, barr.Step{
			Kind:       barr.StepTopK,
			Name:       call.Callee,
			Inputs:     []string{scoreName},
			Outputs:    maybeOutput(output),
			Attributes: map[string]string{"k": fmt.Sprintf("%d", k)},
		})
		buffers = appendOutputBuffer(buffers, output, out)
		return out, steps, buffers, newKernels, nil
	}

	inputs := make([]string, 0, len(call.Args))
	argTypes := make([]hir.Type, 0, len(call.Args))
	steps := []barr.Step{}
	buffers := []lir.Buffer{}
	newKernels := []lir.Kernel{}
	for _, arg := range call.Args {
		typ, err := inferExprType(arg, env)
		if err != nil {
			return hir.Type{}, nil, nil, nil, err
		}
		argTypes = append(argTypes, typ)
	}
	for i, arg := range call.Args {
		_, name, argSteps, argBuffers, argKernels, err := materializeCallArg(arg, output, call.Callee, i, env, kernels)
		if err != nil {
			return hir.Type{}, nil, nil, nil, err
		}
		steps = append(steps, argSteps...)
		for _, buf := range argBuffers {
			buffers = appendIfMissingBuffer(buffers, buf)
		}
		newKernels = appendKernelsIfMissing(newKernels, argKernels...)
		inputs = append(inputs, name)
	}
	switch {
	case call.Intrinsic && call.Callee == "matmul":
		if len(call.Args) != 2 {
			return hir.Type{}, nil, nil, nil, fmt.Errorf("@matmul expects 2 args")
		}
		lhs := argTypes[0]
		rhs := argTypes[1]
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: lhs.Tensor.DType}}
		if lhs.Tensor != nil && rhs.Tensor != nil && len(lhs.Tensor.Shape) == 2 && len(rhs.Tensor.Shape) == 2 {
			out.Tensor.Shape = []hir.DimExpr{{Name: lhs.Tensor.Shape[0].Name}, {Name: rhs.Tensor.Shape[1].Name}}
		}
		if lhs.Tensor != nil && rhs.Tensor != nil && len(lhs.Tensor.Shape) == 3 && len(rhs.Tensor.Shape) == 2 {
			out.Tensor.Shape = []hir.DimExpr{{Name: lhs.Tensor.Shape[0].Name}, {Name: lhs.Tensor.Shape[1].Name}, {Name: rhs.Tensor.Shape[1].Name}}
		}
		if lhs.Tensor != nil && rhs.Tensor != nil && len(lhs.Tensor.Shape) == 3 && len(rhs.Tensor.Shape) == 3 {
			out.Tensor.Shape = []hir.DimExpr{{Name: lhs.Tensor.Shape[0].Name}, {Name: lhs.Tensor.Shape[1].Name}, {Name: rhs.Tensor.Shape[2].Name}}
		}
		steps = append(steps, barr.Step{Kind: barr.StepMatMul, Name: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		return out, steps, buffers, newKernels, nil
	case call.Callee == "transpose":
		if len(call.Args) != 1 {
			return hir.Type{}, nil, nil, nil, fmt.Errorf("transpose expects 1 arg")
		}
		in := argTypes[0]
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: in.Tensor.DType}}
		if isRank2HIRTensor(in) {
			out.Tensor.Shape = []hir.DimExpr{{Name: in.Tensor.Shape[1].Name}, {Name: in.Tensor.Shape[0].Name}}
		}
		if isRank3HIRTensor(in) {
			out.Tensor.Shape = []hir.DimExpr{{Name: in.Tensor.Shape[0].Name}, {Name: in.Tensor.Shape[2].Name}, {Name: in.Tensor.Shape[1].Name}}
		}
		steps = append(steps, barr.Step{Kind: barr.StepTranspose, Name: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		return out, steps, buffers, newKernels, nil
	case call.Callee == "gather":
		if len(call.Args) != 2 {
			return hir.Type{}, nil, nil, nil, fmt.Errorf("gather expects 2 args")
		}
		table := argTypes[0]
		index := argTypes[1]
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: table.Tensor.DType}}
		if table.Tensor != nil && index.Tensor != nil && len(table.Tensor.Shape) == 1 && len(index.Tensor.Shape) == 1 {
			out.Tensor.Shape = []hir.DimExpr{{Name: index.Tensor.Shape[0].Name}}
		}
		if table.Tensor != nil && index.Tensor != nil && len(table.Tensor.Shape) == 2 && len(index.Tensor.Shape) == 1 {
			out.Tensor.Shape = []hir.DimExpr{{Name: index.Tensor.Shape[0].Name}, {Name: table.Tensor.Shape[1].Name}}
		}
		if table.Tensor != nil && index.Tensor != nil && len(table.Tensor.Shape) == 2 && len(index.Tensor.Shape) == 2 {
			if table.Tensor.Shape[0].Name == index.Tensor.Shape[0].Name {
				out.Tensor.Shape = []hir.DimExpr{{Name: index.Tensor.Shape[0].Name}, {Name: index.Tensor.Shape[1].Name}}
			} else {
				out.Tensor.Shape = []hir.DimExpr{{Name: index.Tensor.Shape[0].Name}, {Name: index.Tensor.Shape[1].Name}, {Name: table.Tensor.Shape[1].Name}}
			}
		}
		if table.Tensor != nil && index.Tensor != nil && len(table.Tensor.Shape) == 3 && len(index.Tensor.Shape) == 2 {
			out.Tensor.Shape = []hir.DimExpr{{Name: index.Tensor.Shape[0].Name}, {Name: index.Tensor.Shape[1].Name}, {Name: table.Tensor.Shape[2].Name}}
		}
		steps = append(steps, barr.Step{Kind: barr.StepGather, Name: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		return out, steps, buffers, newKernels, nil
	case call.Callee == "pack_candidates":
		if len(call.Args) != 3 {
			return hir.Type{}, nil, nil, nil, fmt.Errorf("pack_candidates expects 3 args")
		}
		ids := argTypes[0]
		scores := argTypes[1]
		docs := argTypes[2]
		out := hir.Type{Kind: hir.TypeCandidatePack, CandidatePack: &hir.CandidatePackType{}}
		if isRank1HIRTensor(ids) && isRank1HIRTensor(scores) && isRank2HIRTensor(docs) {
			out.CandidatePack.Shape = []hir.DimExpr{{Name: docs.Tensor.Shape[0].Name}, {Name: docs.Tensor.Shape[1].Name}}
		}
		if isRank2HIRTensor(ids) && isRank2HIRTensor(scores) && isRank3HIRTensor(docs) {
			out.CandidatePack.Shape = []hir.DimExpr{{Name: docs.Tensor.Shape[0].Name}, {Name: docs.Tensor.Shape[1].Name}, {Name: docs.Tensor.Shape[2].Name}}
		}
		steps = append(steps, barr.Step{Kind: barr.StepPack, Name: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		return out, steps, buffers, newKernels, nil
	case call.Callee == "kv_read":
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f16", Shape: []hir.DimExpr{{Name: "T"}, {Name: "D"}}}}
		steps = append(steps, barr.Step{Kind: barr.StepKVRead, Name: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		return out, steps, buffers, newKernels, nil
	case call.Callee == "kv_write":
		steps = append(steps, barr.Step{Kind: barr.StepKVWrite, Name: call.Callee, Inputs: inputs})
		return hir.Type{Kind: hir.TypeKVCache}, steps, buffers, newKernels, nil
	case call.Callee == "dequant":
		in := argTypes[0]
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f16", Shape: dimClone(in.Tensor.Shape)}}
		steps = append(steps, barr.Step{Kind: barr.StepDequant, Name: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		return out, steps, buffers, newKernels, nil
	case call.Callee == "mean_pool":
		in := argTypes[0]
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: in.Tensor.DType}}
		if isRank2HIRTensor(in) {
			out.Tensor.Shape = []hir.DimExpr{{Name: in.Tensor.Shape[1].Name}}
		}
		if isRank3HIRTensor(in) {
			out.Tensor.Shape = []hir.DimExpr{{Name: in.Tensor.Shape[0].Name}, {Name: in.Tensor.Shape[2].Name}}
		}
		steps = append(steps, barr.Step{Kind: barr.StepLaunchKernel, Name: call.Callee, Kernel: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		newKernels = appendKernelsIfMissing(newKernels, synthesizeBuiltinKernel(call, argTypes, out))
		return out, steps, buffers, newKernels, nil
	case call.Callee == "dot" || call.Callee == "cosine" || call.Callee == "l2_distance":
		out := hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f32"}}
		if len(argTypes) == 2 && isRank2HIRTensor(argTypes[1]) {
			out.Tensor.Shape = []hir.DimExpr{{Name: argTypes[1].Tensor.Shape[0].Name}}
		}
		if len(argTypes) == 2 && isRank2HIRTensor(argTypes[0]) && isRank3HIRTensor(argTypes[1]) {
			out.Tensor.Shape = []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[0].Name}, {Name: argTypes[1].Tensor.Shape[1].Name}}
		}
		steps = append(steps, barr.Step{Kind: barr.StepLaunchKernel, Name: call.Callee, Kernel: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		newKernels = appendKernelsIfMissing(newKernels, synthesizeBuiltinKernel(call, argTypes, out))
		return out, steps, buffers, newKernels, nil
	default:
		var out hir.Type
		if len(argTypes) > 0 {
			out = argTypes[0]
		}
		if kernels[call.Callee] {
			steps = append(steps, barr.Step{Kind: barr.StepLaunchKernel, Name: call.Callee, Kernel: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
			buffers = appendOutputBuffer(buffers, output, out)
			return out, steps, buffers, newKernels, nil
		}
		steps = append(steps, barr.Step{Kind: barr.StepLaunchKernel, Name: call.Callee, Kernel: call.Callee, Inputs: inputs, Outputs: maybeOutput(output)})
		buffers = appendOutputBuffer(buffers, output, out)
		newKernels = appendKernelsIfMissing(newKernels, synthesizeBuiltinKernel(call, argTypes, out))
		return out, steps, buffers, newKernels, nil
	}
}

func inferExprType(expr syntax.Expr, env map[string]hir.Type) (hir.Type, error) {
	switch e := expr.(type) {
	case *syntax.IdentExpr:
		typ, ok := env[e.Name]
		if !ok {
			return hir.Type{}, fmt.Errorf("unknown identifier %q", e.Name)
		}
		return typ, nil
	case *syntax.BinaryExpr:
		return inferExprType(e.Left, env)
	case *syntax.CallExpr:
		if !e.Intrinsic && e.Callee == "topk" {
			if len(e.Args) != 2 {
				return hir.Type{}, fmt.Errorf("topk expects 2 args")
			}
			scoreType, err := inferExprType(e.Args[0], env)
			if err != nil {
				return hir.Type{}, err
			}
			k, ok := topKLiteralExpr(e.Args[1])
			if !ok {
				return hir.Type{}, fmt.Errorf("topk requires a positive integer literal limit")
			}
			switch {
			case isRank1HIRTensor(scoreType):
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: "i32",
						Shape: []hir.DimExpr{{Name: fmt.Sprintf("%d", k)}},
					},
				}, nil
			case isRank2HIRTensor(scoreType):
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: "i32",
						Shape: []hir.DimExpr{
							{Name: scoreType.Tensor.Shape[0].Name},
							{Name: fmt.Sprintf("%d", k)},
						},
					},
				}, nil
			default:
				return hir.Type{}, fmt.Errorf("topk expects rank-1 or rank-2 score input")
			}
		}
		argTypes := make([]hir.Type, 0, len(e.Args))
		for _, arg := range e.Args {
			typ, err := inferExprType(arg, env)
			if err != nil {
				return hir.Type{}, err
			}
			argTypes = append(argTypes, typ)
		}
		if e.Intrinsic && e.Callee == "matmul" && len(argTypes) == 2 {
			if isRank2HIRTensor(argTypes[0]) {
				if !isRank2HIRTensor(argTypes[1]) {
					return hir.Type{}, fmt.Errorf("@matmul rank-2 lhs requires rank-2 rhs")
				}
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[0].Name}, {Name: argTypes[1].Tensor.Shape[1].Name}},
					},
				}, nil
			}
			if isRank3HIRTensor(argTypes[0]) {
				if isRank3HIRTensor(argTypes[1]) {
					return hir.Type{
						Kind: hir.TypeTensor,
						Tensor: &hir.TensorType{
							DType: argTypes[0].Tensor.DType,
							Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[0].Name}, {Name: argTypes[0].Tensor.Shape[1].Name}, {Name: argTypes[1].Tensor.Shape[2].Name}},
						},
					}, nil
				}
				if !isRank2HIRTensor(argTypes[1]) {
					return hir.Type{}, fmt.Errorf("@matmul rank-3 lhs requires rank-2 or rank-3 rhs")
				}
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[0].Name}, {Name: argTypes[0].Tensor.Shape[1].Name}, {Name: argTypes[1].Tensor.Shape[1].Name}},
					},
				}, nil
			}
		}
		switch e.Callee {
		case "gather":
			if len(argTypes) == 2 && isRank1HIRTensor(argTypes[0]) && isRank1HIRTensor(argTypes[1]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[1].Tensor.Shape[0].Name}},
					},
				}, nil
			}
			if len(argTypes) == 2 && isRank2HIRTensor(argTypes[0]) && isRank1HIRTensor(argTypes[1]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[1].Tensor.Shape[0].Name}, {Name: argTypes[0].Tensor.Shape[1].Name}},
					},
				}, nil
			}
			if len(argTypes) == 2 && isRank2HIRTensor(argTypes[0]) && isRank2HIRTensor(argTypes[1]) {
				if argTypes[0].Tensor.Shape[0].Name == argTypes[1].Tensor.Shape[0].Name {
					return hir.Type{
						Kind: hir.TypeTensor,
						Tensor: &hir.TensorType{
							DType: argTypes[0].Tensor.DType,
							Shape: []hir.DimExpr{{Name: argTypes[1].Tensor.Shape[0].Name}, {Name: argTypes[1].Tensor.Shape[1].Name}},
						},
					}, nil
				}
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[1].Tensor.Shape[0].Name}, {Name: argTypes[1].Tensor.Shape[1].Name}, {Name: argTypes[0].Tensor.Shape[1].Name}},
					},
				}, nil
			}
			if len(argTypes) == 2 && isRank3HIRTensor(argTypes[0]) && isRank2HIRTensor(argTypes[1]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[1].Tensor.Shape[0].Name}, {Name: argTypes[1].Tensor.Shape[1].Name}, {Name: argTypes[0].Tensor.Shape[2].Name}},
					},
				}, nil
			}
		case "pack_candidates":
			if len(argTypes) == 3 && isRank1HIRTensor(argTypes[0]) && isRank1HIRTensor(argTypes[1]) && isRank2HIRTensor(argTypes[2]) {
				return hir.Type{
					Kind: hir.TypeCandidatePack,
					CandidatePack: &hir.CandidatePackType{
						Shape: []hir.DimExpr{{Name: argTypes[2].Tensor.Shape[0].Name}, {Name: argTypes[2].Tensor.Shape[1].Name}},
					},
				}, nil
			}
			if len(argTypes) == 3 && isRank2HIRTensor(argTypes[0]) && isRank2HIRTensor(argTypes[1]) && isRank3HIRTensor(argTypes[2]) {
				return hir.Type{
					Kind: hir.TypeCandidatePack,
					CandidatePack: &hir.CandidatePackType{
						Shape: []hir.DimExpr{{Name: argTypes[2].Tensor.Shape[0].Name}, {Name: argTypes[2].Tensor.Shape[1].Name}, {Name: argTypes[2].Tensor.Shape[2].Name}},
					},
				}, nil
			}
		case "dequant":
			if len(argTypes) == 1 && argTypes[0].Tensor != nil {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: "f16",
						Shape: dimClone(argTypes[0].Tensor.Shape),
					},
				}, nil
			}
		case "mean_pool":
			if len(argTypes) >= 1 && len(argTypes) <= 2 && isRank2HIRTensor(argTypes[0]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[1].Name}},
					},
				}, nil
			}
			if len(argTypes) >= 1 && len(argTypes) <= 2 && isRank3HIRTensor(argTypes[0]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[0].Name}, {Name: argTypes[0].Tensor.Shape[2].Name}},
					},
				}, nil
			}
		case "transpose":
			if len(argTypes) == 1 && isRank2HIRTensor(argTypes[0]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[1].Name}, {Name: argTypes[0].Tensor.Shape[0].Name}},
					},
				}, nil
			}
			if len(argTypes) == 1 && isRank3HIRTensor(argTypes[0]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: argTypes[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[0].Name}, {Name: argTypes[0].Tensor.Shape[2].Name}, {Name: argTypes[0].Tensor.Shape[1].Name}},
					},
				}, nil
			}
		case "softmax", "rope", "normalize", "rmsnorm", "layernorm", "gelu":
			if len(argTypes) == 1 {
				return argTypes[0], nil
			}
		case "kv_read":
			return hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f16", Shape: []hir.DimExpr{{Name: "T"}, {Name: "D"}}}}, nil
		case "kv_write":
			return hir.Type{Kind: hir.TypeKVCache}, nil
		case "dot", "cosine", "l2_distance":
			if len(argTypes) == 2 && isRank1HIRTensor(argTypes[0]) && isRank2HIRTensor(argTypes[1]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: "f32",
						Shape: []hir.DimExpr{{Name: argTypes[1].Tensor.Shape[0].Name}},
					},
				}, nil
			}
			if len(argTypes) == 2 && isRank2HIRTensor(argTypes[0]) && isRank3HIRTensor(argTypes[1]) {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: "f32",
						Shape: []hir.DimExpr{{Name: argTypes[0].Tensor.Shape[0].Name}, {Name: argTypes[1].Tensor.Shape[1].Name}},
					},
				}, nil
			}
		}
		if len(argTypes) > 0 {
			return argTypes[0], nil
		}
		return hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f32"}}, nil
	case *syntax.NumberExpr:
		return hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f32"}}, nil
	default:
		return hir.Type{}, fmt.Errorf("unsupported expr type %T", expr)
	}
}

func maybeOutput(name string) []string {
	if name == "" {
		return nil
	}
	return []string{name}
}

func materializeCallArg(arg syntax.Expr, output, callee string, index int, env map[string]hir.Type, kernels map[string]bool) (hir.Type, string, []barr.Step, []lir.Buffer, []lir.Kernel, error) {
	switch arg.(type) {
	case *syntax.IdentExpr, *syntax.NumberExpr, *syntax.StringExpr:
		typ, err := inferExprType(arg, env)
		if err != nil {
			return hir.Type{}, "", nil, nil, nil, err
		}
		return typ, exprName(arg), nil, nil, nil, nil
	default:
		name := tempValueName(output, callee, index, arg)
		typ, steps, buffers, newKernels, err := inferExprPlan(arg, name, nil, env, kernels)
		if err != nil {
			return hir.Type{}, "", nil, nil, nil, err
		}
		return typ, name, steps, buffers, newKernels, nil
	}
}

func tempValueName(output, callee string, index int, expr syntax.Expr) string {
	base := callee
	if output != "" {
		base = output
	}
	return fmt.Sprintf("%s_arg%d_%s", base, index, exprName(expr))
}

func bufferForValue(name string, typ hir.Type, input bool) lir.Buffer {
	storage := "device_local"
	if input {
		storage = "host_visible"
	}
	if typ.Kind == hir.TypeKVCache {
		return lir.Buffer{Name: name, DType: "kv_cache", StorageClass: "unified"}
	}
	if typ.Kind == hir.TypeCandidatePack {
		shape := []string{}
		if typ.CandidatePack != nil {
			shape = dimTexts(typ.CandidatePack.Shape)
		}
		return lir.Buffer{Name: name, DType: "candidate_pack", Shape: shape, StorageClass: "host_visible"}
	}
	shape := []string{}
	if typ.Tensor != nil {
		shape = dimTexts(typ.Tensor.Shape)
	}
	return lir.Buffer{Name: name, DType: typ.Tensor.DType, Shape: shape, StorageClass: storage}
}

func containsBuffer(buffers []lir.Buffer, name string) bool {
	for _, buf := range buffers {
		if buf.Name == name {
			return true
		}
	}
	return false
}

func appendIfMissingBuffer(buffers []lir.Buffer, buf lir.Buffer) []lir.Buffer {
	if containsBuffer(buffers, buf.Name) {
		return buffers
	}
	return append(buffers, buf)
}

func appendOutputBuffer(buffers []lir.Buffer, name string, typ hir.Type) []lir.Buffer {
	if name == "" {
		return buffers
	}
	return appendIfMissingBuffer(buffers, bufferForValue(name, typ, false))
}

func withStepEntry(steps []barr.Step, entry string) []barr.Step {
	for i := range steps {
		steps[i].Entry = entry
	}
	return steps
}

func appendKernelsIfMissing(existing []lir.Kernel, additions ...lir.Kernel) []lir.Kernel {
	for _, kernel := range additions {
		if kernel.Name == "" || containsKernel(existing, kernel.Name) {
			continue
		}
		existing = append(existing, kernel)
	}
	return existing
}

func containsKernel(kernels []lir.Kernel, name string) bool {
	for _, kernel := range kernels {
		if kernel.Name == name {
			return true
		}
	}
	return false
}

func lowerLIRValue(name string, typ hir.Type) lir.Value {
	if typ.Kind == hir.TypeKVCache {
		return lir.Value{Name: name, Kind: lir.ValueKVCache}
	}
	shape := []string{}
	dtype := ""
	if typ.Tensor != nil {
		shape = dimTexts(typ.Tensor.Shape)
		dtype = typ.Tensor.DType
	}
	return lir.Value{Name: name, Kind: lir.ValueTensor, DType: dtype, Shape: shape}
}

func lowerKernelCallable(callable *syntax.CallableDecl, kernels map[string]bool) (lir.Kernel, error) {
	kernel := lir.Kernel{Name: callable.Name}
	for _, param := range callable.Params {
		kernel.Inputs = append(kernel.Inputs, lowerLIRValue(param.Name, lowerType(param.Type)))
	}
	resultName := resultNameForCallable(callable)
	kernel.Outputs = append(kernel.Outputs, lowerLIRValue(resultName, lowerType(callable.Result)))
	body, err := lowerKernelBody(callable, resultName, kernels)
	if err != nil {
		return lir.Kernel{}, err
	}
	kernel.Body = body
	kernel.Hints = scheduleHintsForKernel(callable.Name, body)
	return kernel, nil
}

func lowerKernelBody(callable *syntax.CallableDecl, resultName string, kernels map[string]bool) ([]lir.KernelOp, error) {
	ops := []lir.KernelOp{}
	for _, stmt := range callable.Body {
		switch s := stmt.(type) {
		case *syntax.LetStmt:
			exprOps, err := lowerKernelExpr(s.Expr, s.Name, kernels)
			if err != nil {
				return nil, err
			}
			ops = append(ops, exprOps...)
		case *syntax.ReturnStmt:
			if ident, ok := s.Expr.(*syntax.IdentExpr); ok {
				ops = append(ops, lir.KernelOp{Kind: lir.KernelOpReturn, Op: "return", Outputs: []string{ident.Name}})
				continue
			}
			exprOps, err := lowerKernelExpr(s.Expr, resultName, kernels)
			if err != nil {
				return nil, err
			}
			ops = append(ops, exprOps...)
			ops = append(ops, lir.KernelOp{Kind: lir.KernelOpReturn, Op: "return", Outputs: []string{resultName}})
		case *syntax.ExprStmt:
			exprOps, err := lowerKernelExpr(s.Expr, "", kernels)
			if err != nil {
				return nil, err
			}
			ops = append(ops, exprOps...)
		}
	}
	return ops, nil
}

func lowerKernelExpr(expr syntax.Expr, output string, kernels map[string]bool) ([]lir.KernelOp, error) {
	switch e := expr.(type) {
	case *syntax.IdentExpr:
		return nil, nil
	case *syntax.BinaryExpr:
		return []lir.KernelOp{{
			Kind:    lir.KernelOpPointwise,
			Name:    output,
			Op:      binaryKernelName(e.Op),
			Inputs:  []string{exprName(e.Left), exprName(e.Right)},
			Outputs: maybeOutput(output),
			Attributes: map[string]string{
				"operator": e.Op,
			},
		}}, nil
	case *syntax.CallExpr:
		return []lir.KernelOp{kernelOpForCall(e, output, kernels)}, nil
	default:
		return nil, fmt.Errorf("unsupported kernel expression %T", expr)
	}
}

func kernelOpForCall(call *syntax.CallExpr, output string, kernels map[string]bool) lir.KernelOp {
	kind := lir.KernelOpPointwise
	switch call.Callee {
	case "normalize", "rmsnorm", "layernorm", "softmax", "dot", "cosine", "l2_distance", "mean_pool":
		kind = lir.KernelOpReduce
	case "rope", "dequant", "gather", "kv_read", "kv_write":
		kind = lir.KernelOpBuiltin
	default:
		if call.Intrinsic || kernels[call.Callee] {
			kind = lir.KernelOpBuiltin
		}
	}
	attrs := map[string]string{}
	if call.Intrinsic {
		attrs["intrinsic"] = "true"
	}
	if call.Callee == "softmax" {
		attrs["axis"] = "-1"
	}
	return lir.KernelOp{
		Kind:       kind,
		Name:       output,
		Op:         call.Callee,
		Inputs:     exprArgNames(call.Args),
		Outputs:    maybeOutput(output),
		Attributes: attrs,
	}
}

func synthesizeBuiltinKernel(call *syntax.CallExpr, argTypes []hir.Type, out hir.Type) lir.Kernel {
	inputs := make([]lir.Value, 0, len(argTypes))
	names := exprArgNames(call.Args)
	for i, typ := range argTypes {
		name := fmt.Sprintf("input%d", i)
		if i < len(names) && names[i] != "" {
			name = names[i]
		}
		inputs = append(inputs, lowerLIRValue(name, typ))
	}
	resultName := "result"
	body := []lir.KernelOp{
		kernelOpForCall(call, resultName, nil),
		{Kind: lir.KernelOpReturn, Op: "return", Outputs: []string{resultName}},
	}
	return lir.Kernel{
		Name:    call.Callee,
		Inputs:  inputs,
		Outputs: []lir.Value{lowerLIRValue(resultName, out)},
		Hints:   scheduleHintsForKernel(call.Callee, body),
		Body:    body,
	}
}

func synthesizeBinaryKernel(expr *syntax.BinaryExpr, out hir.Type, env map[string]hir.Type) lir.Kernel {
	leftType, leftErr := inferExprType(expr.Left, env)
	rightType, rightErr := inferExprType(expr.Right, env)
	inputs := []lir.Value{}
	if leftErr == nil {
		inputs = append(inputs, lowerLIRValue("lhs", leftType))
	}
	if rightErr == nil {
		inputs = append(inputs, lowerLIRValue("rhs", rightType))
	}
	name := binaryKernelName(expr.Op)
	body := []lir.KernelOp{
		{
			Kind:    lir.KernelOpPointwise,
			Name:    "result",
			Op:      name,
			Inputs:  []string{"lhs", "rhs"},
			Outputs: []string{"result"},
			Attributes: map[string]string{
				"operator": expr.Op,
			},
		},
		{Kind: lir.KernelOpReturn, Op: "return", Outputs: []string{"result"}},
	}
	return lir.Kernel{
		Name:    name,
		Inputs:  inputs,
		Outputs: []lir.Value{lowerLIRValue("result", out)},
		Hints:   scheduleHintsForKernel(name, body),
		Body:    body,
	}
}

func binaryKernelName(op string) string {
	return "binary_" + sanitizeOp(op)
}

func scheduleHintsForKernel(name string, ops []lir.KernelOp) lir.ScheduleHints {
	hints := lir.ScheduleHints{
		Tile:        []int{128},
		VectorWidth: 4,
		Memory:      "device_local",
	}
	for _, op := range ops {
		switch op.Op {
		case "softmax":
			hints.Tile = []int{64}
			hints.VectorWidth = 1
			hints.Subgroup = true
			hints.Memory = "workgroup_local"
		case "normalize", "rmsnorm", "layernorm":
			hints.Tile = []int{128}
			hints.VectorWidth = 4
			hints.Subgroup = true
			hints.Memory = "workgroup_local"
		case "rope":
			hints.Tile = []int{128}
			hints.VectorWidth = 2
			hints.Subgroup = true
		case "dot", "cosine", "l2_distance":
			hints.Tile = []int{128}
			hints.VectorWidth = 4
			hints.Subgroup = true
			hints.Memory = "workgroup_local"
		case "mean_pool":
			hints.Tile = []int{64}
			hints.VectorWidth = 1
			hints.Subgroup = true
			hints.Memory = "workgroup_local"
		case "binary_add", "binary_mul", "binary_sub", "binary_div":
			hints.Tile = []int{128}
			hints.VectorWidth = 4
		}
	}
	if strings.Contains(name, "norm") {
		hints.Subgroup = true
	}
	return hints
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func dimClone(in []hir.DimExpr) []hir.DimExpr {
	out := make([]hir.DimExpr, 0, len(in))
	out = append(out, in...)
	return out
}

func topKLiteralExpr(expr syntax.Expr) (int, bool) {
	num, ok := expr.(*syntax.NumberExpr)
	if !ok {
		return 0, false
	}
	value, err := strconv.Atoi(strings.TrimSpace(num.Text))
	if err != nil || value <= 0 {
		return 0, false
	}
	return value, true
}

func isRank1HIRTensor(typ hir.Type) bool {
	return typ.Kind == hir.TypeTensor && typ.Tensor != nil && len(typ.Tensor.Shape) == 1
}

func isRank2HIRTensor(typ hir.Type) bool {
	return typ.Kind == hir.TypeTensor && typ.Tensor != nil && len(typ.Tensor.Shape) == 2
}

func isRank3HIRTensor(typ hir.Type) bool {
	return typ.Kind == hir.TypeTensor && typ.Tensor != nil && len(typ.Tensor.Shape) == 3
}

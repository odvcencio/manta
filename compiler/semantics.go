package compiler

import (
	"fmt"
	"strings"

	"github.com/odvcencio/barracuda/ir/hir"
	"github.com/odvcencio/barracuda/syntax"
)

// DiagnosticsError reports one or more source-tied compiler diagnostics.
type DiagnosticsError struct {
	Diagnostics []syntax.Diagnostic
}

func (e *DiagnosticsError) Error() string {
	if e == nil {
		return ""
	}
	return syntax.DebugString(e.Diagnostics)
}

func semanticDiagnostics(file *syntax.File) []syntax.Diagnostic {
	if file == nil {
		return nil
	}

	env := newModuleTypeEnv(file)
	diags := []syntax.Diagnostic{}
	decls := map[string]syntax.Span{}
	for _, decl := range file.Decls {
		name := declName(decl)
		if name == "" {
			continue
		}
		if prev, ok := decls[name]; ok {
			diags = append(diags, diagnosticError(decl.DeclSpan(), "duplicate declaration %q; first declared at %d:%d", name, prev.Line, prev.Column))
			continue
		}
		decls[name] = decl.DeclSpan()
	}

	for _, decl := range file.Decls {
		callable, ok := decl.(*syntax.CallableDecl)
		if !ok {
			continue
		}
		diags = append(diags, validateCallable(callable, env)...)
	}
	return diags
}

func validateCallable(callable *syntax.CallableDecl, env moduleTypeEnv) []syntax.Diagnostic {
	diags := []syntax.Diagnostic{}
	locals := env.forCallable(callable)
	localDecls := map[string]syntax.Span{}
	for _, param := range callable.Params {
		if prev, ok := localDecls[param.Name]; ok {
			diags = append(diags, diagnosticError(param.Span, "duplicate parameter %q; first declared at %d:%d", param.Name, prev.Line, prev.Column))
			continue
		}
		localDecls[param.Name] = param.Span
	}

	resultFields := callableResultFields(callable)
	hasReturn := false
	for _, stmt := range callable.Body {
		switch s := stmt.(type) {
		case *syntax.LetStmt:
			if prev, ok := localDecls[s.Name]; ok {
				diags = append(diags, diagnosticError(s.Span, "duplicate local %q; first declared at %d:%d", s.Name, prev.Line, prev.Column))
				continue
			}
			localDecls[s.Name] = s.Span
			typ, stmtDiags := inferSemanticExprType(s.Expr, callable, locals, env)
			diags = append(diags, stmtDiags...)
			if typ.Kind != "" {
				locals[s.Name] = typ
			}
		case *syntax.ReturnStmt:
			hasReturn = true
			exprs := returnStmtExprs(s)
			if len(exprs) != len(resultFields) {
				diags = append(diags, diagnosticError(s.Span, "return value count mismatch: got %d, want %d", len(exprs), len(resultFields)))
				continue
			}
			for i, expr := range exprs {
				typ, stmtDiags := inferSemanticExprType(expr, callable, locals, env)
				diags = append(diags, stmtDiags...)
				want := lowerType(resultFields[i].Type)
				if typ.Kind != "" && !sameType(typ, want) {
					diags = append(diags, diagnosticError(expr.ExprSpan(), "return value %d type mismatch: got %s, want %s", i+1, typeString(typ), typeString(want)))
				}
			}
		case *syntax.ExprStmt:
			_, stmtDiags := inferSemanticExprType(s.Expr, callable, locals, env)
			diags = append(diags, stmtDiags...)
			if !exprHasEffect(s.Expr) {
				diags = append(diags, diagnosticError(s.Span, "expression statement has no effect"))
			}
		}
	}
	if !hasReturn {
		diags = append(diags, diagnosticError(callable.Span, "%s %q must end with a return statement", callable.Kind, callable.Name))
	}
	return diags
}

func callableResultFields(callable *syntax.CallableDecl) []syntax.Field {
	if callable.Kind == syntax.CallablePipeline && len(callable.Results) > 0 {
		out := make([]syntax.Field, len(callable.Results))
		copy(out, callable.Results)
		return out
	}
	return []syntax.Field{{
		Name: resultNameForCallable(callable),
		Type: callable.Result,
		Span: callable.Span,
	}}
}

func returnStmtExprs(stmt *syntax.ReturnStmt) []syntax.Expr {
	if stmt == nil {
		return nil
	}
	if len(stmt.Exprs) > 0 {
		out := make([]syntax.Expr, len(stmt.Exprs))
		copy(out, stmt.Exprs)
		return out
	}
	if stmt.Expr == nil {
		return nil
	}
	return []syntax.Expr{stmt.Expr}
}

func inferSemanticExprType(expr syntax.Expr, callable *syntax.CallableDecl, locals map[string]hir.Type, env moduleTypeEnv) (hir.Type, []syntax.Diagnostic) {
	switch e := expr.(type) {
	case *syntax.IdentExpr:
		typ, ok := locals[e.Name]
		if !ok {
			return hir.Type{}, []syntax.Diagnostic{diagnosticError(e.Span, "unknown identifier %q", e.Name)}
		}
		return typ, nil
	case *syntax.NumberExpr:
		return hir.Type{}, []syntax.Diagnostic{diagnosticError(e.Span, "numeric literals in expressions are not lowered yet")}
	case *syntax.StringExpr:
		return hir.Type{}, []syntax.Diagnostic{diagnosticError(e.Span, "string literals are not supported in expressions")}
	case *syntax.BinaryExpr:
		left, leftDiags := inferSemanticExprType(e.Left, callable, locals, env)
		right, rightDiags := inferSemanticExprType(e.Right, callable, locals, env)
		diags := append(leftDiags, rightDiags...)
		if left.Kind == "" || right.Kind == "" {
			return left, diags
		}
		if left.Kind != hir.TypeTensor || right.Kind != hir.TypeTensor {
			diags = append(diags, diagnosticError(e.Span, "binary operator %q requires tensor operands", e.Op))
			return left, diags
		}
		if !sameTensorShape(left.Tensor, right.Tensor) {
			diags = append(diags, diagnosticError(e.Span, "binary operator %q requires equal tensor shapes; got %s and %s", e.Op, typeString(left), typeString(right)))
		}
		if left.Tensor != nil && right.Tensor != nil && left.Tensor.DType != right.Tensor.DType {
			diags = append(diags, diagnosticError(e.Span, "binary operator %q requires matching dtypes; got %s and %s", e.Op, left.Tensor.DType, right.Tensor.DType))
		}
		return left, diags
	case *syntax.CallExpr:
		return inferSemanticCallType(e, callable, locals, env)
	default:
		return hir.Type{}, []syntax.Diagnostic{diagnosticError(expr.ExprSpan(), "unsupported expression %T", expr)}
	}
}

func inferSemanticCallType(call *syntax.CallExpr, callable *syntax.CallableDecl, locals map[string]hir.Type, env moduleTypeEnv) (hir.Type, []syntax.Diagnostic) {
	if !call.Intrinsic && call.Callee == "topk" {
		return inferSemanticTopKCallType(call, callable, locals, env)
	}

	diags := []syntax.Diagnostic{}
	args := make([]hir.Type, 0, len(call.Args))
	for _, arg := range call.Args {
		typ, argDiags := inferSemanticExprType(arg, callable, locals, env)
		diags = append(diags, argDiags...)
		args = append(args, typ)
	}

	if call.Intrinsic {
		if call.Callee != "matmul" {
			diags = append(diags, diagnosticError(call.Span, "unknown intrinsic @%s", call.Callee))
			return hir.Type{}, diags
		}
		if callable.Kind == syntax.CallableKernel {
			diags = append(diags, diagnosticError(call.Span, "@matmul is not supported inside kernel bodies"))
		}
		if len(args) != 2 {
			diags = append(diags, diagnosticError(call.Span, "@matmul expects 2 arguments, got %d", len(args)))
			return hir.Type{}, diags
		}
		if !((isRank2Tensor(args[0]) || isRank3Tensor(args[0])) && (isRank2Tensor(args[1]) || isRank3Tensor(args[1]))) {
			diags = append(diags, diagnosticError(call.Span, "@matmul expects rank-2 or rank-3 tensor inputs"))
			return hir.Type{}, diags
		}
		lhs := args[0].Tensor
		rhs := args[1].Tensor
		if len(lhs.Shape) != len(rhs.Shape) && !(len(lhs.Shape) == 2 && len(rhs.Shape) == 2) && !(len(lhs.Shape) == 3 && len(rhs.Shape) == 2) {
			diags = append(diags, diagnosticError(call.Span, "@matmul expects rank-2 x rank-2, rank-3 x rank-2, or rank-3 x rank-3 inputs"))
			return hir.Type{}, diags
		}
		if len(lhs.Shape) == 2 && len(rhs.Shape) != 2 {
			diags = append(diags, diagnosticError(call.Span, "@matmul rank-2 lhs requires rank-2 rhs"))
			return hir.Type{}, diags
		}
		lhsInner := lhs.Shape[len(lhs.Shape)-1].Name
		rhsInner := rhs.Shape[len(rhs.Shape)-2].Name
		if lhsInner != rhsInner {
			diags = append(diags, diagnosticError(call.Span, "@matmul inner dimensions mismatch: %s vs %s", lhsInner, rhsInner))
		}
		shape := []hir.DimExpr{{Name: lhs.Shape[0].Name}, {Name: rhs.Shape[1].Name}}
		if len(lhs.Shape) == 3 {
			if len(rhs.Shape) == 3 {
				if lhs.Shape[0].Name != rhs.Shape[0].Name {
					diags = append(diags, diagnosticError(call.Span, "@matmul batch dimensions mismatch: %s vs %s", lhs.Shape[0].Name, rhs.Shape[0].Name))
				}
				shape = []hir.DimExpr{{Name: lhs.Shape[0].Name}, {Name: lhs.Shape[1].Name}, {Name: rhs.Shape[2].Name}}
			} else {
				shape = []hir.DimExpr{{Name: lhs.Shape[0].Name}, {Name: lhs.Shape[1].Name}, {Name: rhs.Shape[1].Name}}
			}
		}
		return hir.Type{
			Kind: hir.TypeTensor,
			Tensor: &hir.TensorType{
				DType: lhs.DType,
				Shape: shape,
			},
		}, diags
	}

	switch call.Callee {
	case "gather":
		if callable.Kind == syntax.CallableKernel {
			diags = append(diags, diagnosticError(call.Span, "gather is only supported in pipeline bodies"))
		}
		if len(args) != 2 {
			diags = append(diags, diagnosticError(call.Span, "gather expects 2 arguments, got %d", len(args)))
			return hir.Type{}, diags
		}
		if !((isRank1Tensor(args[1]) || isRank2Tensor(args[1])) && args[1].Tensor.DType == "i32") {
			diags = append(diags, diagnosticError(call.Span, "gather indices must be i32[T] or i32[Q, T]"))
			return hir.Type{}, diags
		}
		if !isRank1Tensor(args[0]) && !isRank2Tensor(args[0]) && !isRank3Tensor(args[0]) {
			diags = append(diags, diagnosticError(call.Span, "gather table must be a rank-1, rank-2, or rank-3 tensor"))
			return hir.Type{}, diags
		}
		if isRank1Tensor(args[0]) && isRank1Tensor(args[1]) {
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[1].Tensor.Shape[0].Name}},
				},
			}, diags
		}
		if isRank2Tensor(args[0]) && isRank1Tensor(args[1]) {
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[1].Tensor.Shape[0].Name}, {Name: args[0].Tensor.Shape[1].Name}},
				},
			}, diags
		}
		if isRank2Tensor(args[0]) && isRank2Tensor(args[1]) {
			if args[0].Tensor.Shape[0].Name == args[1].Tensor.Shape[0].Name {
				return hir.Type{
					Kind: hir.TypeTensor,
					Tensor: &hir.TensorType{
						DType: args[0].Tensor.DType,
						Shape: []hir.DimExpr{{Name: args[1].Tensor.Shape[0].Name}, {Name: args[1].Tensor.Shape[1].Name}},
					},
				}, diags
			}
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[1].Tensor.Shape[0].Name}, {Name: args[1].Tensor.Shape[1].Name}, {Name: args[0].Tensor.Shape[1].Name}},
				},
			}, diags
		}
		if isRank3Tensor(args[0]) && isRank2Tensor(args[1]) {
			if args[0].Tensor.Shape[0].Name != args[1].Tensor.Shape[0].Name {
				diags = append(diags, diagnosticError(call.Span, "batched gather requires matching leading dimensions; got %s and %s", args[0].Tensor.Shape[0].Name, args[1].Tensor.Shape[0].Name))
			}
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[1].Tensor.Shape[0].Name}, {Name: args[1].Tensor.Shape[1].Name}, {Name: args[0].Tensor.Shape[2].Name}},
				},
			}, diags
		}
		diags = append(diags, diagnosticError(call.Span, "unsupported gather rank combination"))
		return hir.Type{}, diags
	case "dequant":
		if len(args) != 1 {
			diags = append(diags, diagnosticError(call.Span, "dequant expects 1 argument, got %d", len(args)))
			return hir.Type{}, diags
		}
		if args[0].Kind != hir.TypeTensor {
			diags = append(diags, diagnosticError(call.Span, "dequant requires a tensor input"))
			return hir.Type{}, diags
		}
		return hir.Type{
			Kind: hir.TypeTensor,
			Tensor: &hir.TensorType{
				DType: "f16",
				Shape: cloneDims(args[0].Tensor.Shape),
			},
		}, diags
	case "mean_pool":
		if callable.Kind == syntax.CallableKernel {
			diags = append(diags, diagnosticError(call.Span, "mean_pool is only supported in pipeline bodies"))
		}
		if len(args) != 1 && len(args) != 2 {
			diags = append(diags, diagnosticError(call.Span, "mean_pool expects 1 or 2 arguments, got %d", len(args)))
			return hir.Type{}, diags
		}
		if len(args) == 2 {
			if args[1].Kind != hir.TypeTensor || args[1].Tensor == nil || args[1].Tensor.DType != "i32" {
				diags = append(diags, diagnosticError(call.Args[1].ExprSpan(), "mean_pool mask must be i32[T] or i32[B, T]"))
				return hir.Type{}, diags
			}
		}
		if isRank2Tensor(args[0]) {
			if len(args) == 2 {
				if !isRank1Tensor(args[1]) {
					diags = append(diags, diagnosticError(call.Args[1].ExprSpan(), "mean_pool mask rank must be 1 for f*[T, D] input"))
				} else if args[0].Tensor.Shape[0].Name != args[1].Tensor.Shape[0].Name {
					diags = append(diags, diagnosticError(call.Span, "mean_pool sequence dimension mismatch: input %s vs mask %s", args[0].Tensor.Shape[0].Name, args[1].Tensor.Shape[0].Name))
				}
			}
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[0].Tensor.Shape[1].Name}},
				},
			}, diags
		}
		if isRank3Tensor(args[0]) {
			if len(args) == 2 {
				if !isRank2Tensor(args[1]) {
					diags = append(diags, diagnosticError(call.Args[1].ExprSpan(), "mean_pool mask rank must be 2 for f*[B, T, D] input"))
				} else {
					if args[0].Tensor.Shape[0].Name != args[1].Tensor.Shape[0].Name {
						diags = append(diags, diagnosticError(call.Span, "mean_pool batch dimension mismatch: input %s vs mask %s", args[0].Tensor.Shape[0].Name, args[1].Tensor.Shape[0].Name))
					}
					if args[0].Tensor.Shape[1].Name != args[1].Tensor.Shape[1].Name {
						diags = append(diags, diagnosticError(call.Span, "mean_pool sequence dimension mismatch: input %s vs mask %s", args[0].Tensor.Shape[1].Name, args[1].Tensor.Shape[1].Name))
					}
				}
			}
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[0].Tensor.Shape[0].Name}, {Name: args[0].Tensor.Shape[2].Name}},
				},
			}, diags
		}
		diags = append(diags, diagnosticError(call.Span, "mean_pool expects f*[T, D] or f*[B, T, D]"))
		return hir.Type{}, diags
	case "transpose":
		if callable.Kind == syntax.CallableKernel {
			diags = append(diags, diagnosticError(call.Span, "transpose is only supported in pipeline bodies"))
		}
		if len(args) != 1 {
			diags = append(diags, diagnosticError(call.Span, "transpose expects 1 argument, got %d", len(args)))
			return hir.Type{}, diags
		}
		if isRank2Tensor(args[0]) {
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[0].Tensor.Shape[1].Name}, {Name: args[0].Tensor.Shape[0].Name}},
				},
			}, diags
		}
		if isRank3Tensor(args[0]) {
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: args[0].Tensor.DType,
					Shape: []hir.DimExpr{{Name: args[0].Tensor.Shape[0].Name}, {Name: args[0].Tensor.Shape[2].Name}, {Name: args[0].Tensor.Shape[1].Name}},
				},
			}, diags
		}
		diags = append(diags, diagnosticError(call.Span, "transpose expects f*[M, N] or f*[B, M, N]"))
		return hir.Type{}, diags
	case "softmax", "rope", "normalize", "rmsnorm", "layernorm", "gelu":
		if len(args) != 1 {
			diags = append(diags, diagnosticError(call.Span, "%s expects 1 argument, got %d", call.Callee, len(args)))
			return hir.Type{}, diags
		}
		if args[0].Kind != hir.TypeTensor {
			diags = append(diags, diagnosticError(call.Span, "%s requires a tensor input", call.Callee))
			return hir.Type{}, diags
		}
		return args[0], diags
	case "dot", "cosine", "l2_distance":
		if len(args) != 2 {
			diags = append(diags, diagnosticError(call.Span, "%s expects 2 arguments, got %d", call.Callee, len(args)))
			return hir.Type{}, diags
		}
		if isRank1Tensor(args[0]) && isRank2Tensor(args[1]) {
			query := args[0].Tensor
			docs := args[1].Tensor
			if query.Shape[0].Name != docs.Shape[1].Name {
				diags = append(diags, diagnosticError(call.Span, "%s dimension mismatch: query %s vs docs %s", call.Callee, query.Shape[0].Name, docs.Shape[1].Name))
			}
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: "f32",
					Shape: []hir.DimExpr{{Name: docs.Shape[0].Name}},
				},
			}, diags
		}
		if isRank2Tensor(args[0]) && isRank3Tensor(args[1]) {
			query := args[0].Tensor
			docs := args[1].Tensor
			if query.Shape[0].Name != docs.Shape[0].Name {
				diags = append(diags, diagnosticError(call.Span, "%s batch dimension mismatch: queries %s vs docs %s", call.Callee, query.Shape[0].Name, docs.Shape[0].Name))
			}
			if query.Shape[1].Name != docs.Shape[2].Name {
				diags = append(diags, diagnosticError(call.Span, "%s feature dimension mismatch: queries %s vs docs %s", call.Callee, query.Shape[1].Name, docs.Shape[2].Name))
			}
			return hir.Type{
				Kind: hir.TypeTensor,
				Tensor: &hir.TensorType{
					DType: "f32",
					Shape: []hir.DimExpr{{Name: docs.Shape[0].Name}, {Name: docs.Shape[1].Name}},
				},
			}, diags
		}
		diags = append(diags, diagnosticError(call.Span, "%s expects query/docs shapes f*[D] x f*[N,D] or f*[Q,D] x f*[Q,N,D]", call.Callee))
		return hir.Type{}, diags
	case "pack_candidates":
		if callable.Kind == syntax.CallableKernel {
			diags = append(diags, diagnosticError(call.Span, "pack_candidates is only supported in pipeline bodies"))
		}
		if len(args) != 3 {
			diags = append(diags, diagnosticError(call.Span, "pack_candidates expects 3 arguments, got %d", len(args)))
			return hir.Type{}, diags
		}
		if isRank1Tensor(args[0]) && isRank1Tensor(args[1]) && isRank2Tensor(args[2]) {
			if args[0].Tensor.DType != "i64" {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates ids must be i64[K]"))
			}
			if args[1].Tensor.DType != "f32" {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates scores must be f32[K]"))
			}
			if args[2].Tensor.DType != "q4" {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates docs must be q4[K, D]"))
			}
			if args[0].Tensor.Shape[0].Name != args[1].Tensor.Shape[0].Name || args[0].Tensor.Shape[0].Name != args[2].Tensor.Shape[0].Name {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates leading dimensions must match"))
			}
			return hir.Type{
				Kind: hir.TypeCandidatePack,
				CandidatePack: &hir.CandidatePackType{
					Shape: []hir.DimExpr{{Name: args[2].Tensor.Shape[0].Name}, {Name: args[2].Tensor.Shape[1].Name}},
				},
			}, diags
		}
		if isRank2Tensor(args[0]) && isRank2Tensor(args[1]) && isRank3Tensor(args[2]) {
			if args[0].Tensor.DType != "i64" {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates ids must be i64[Q, K]"))
			}
			if args[1].Tensor.DType != "f32" {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates scores must be f32[Q, K]"))
			}
			if args[2].Tensor.DType != "q4" {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates docs must be q4[Q, K, D]"))
			}
			if args[0].Tensor.Shape[0].Name != args[1].Tensor.Shape[0].Name || args[0].Tensor.Shape[0].Name != args[2].Tensor.Shape[0].Name {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates batch dimensions must match"))
			}
			if args[0].Tensor.Shape[1].Name != args[1].Tensor.Shape[1].Name || args[0].Tensor.Shape[1].Name != args[2].Tensor.Shape[1].Name {
				diags = append(diags, diagnosticError(call.Span, "pack_candidates candidate dimensions must match"))
			}
			return hir.Type{
				Kind: hir.TypeCandidatePack,
				CandidatePack: &hir.CandidatePackType{
					Shape: []hir.DimExpr{{Name: args[2].Tensor.Shape[0].Name}, {Name: args[2].Tensor.Shape[1].Name}, {Name: args[2].Tensor.Shape[2].Name}},
				},
			}, diags
		}
		diags = append(diags, diagnosticError(call.Span, "pack_candidates expects ids/scores/docs as i64[K] x f32[K] x q4[K,D] or i64[Q,K] x f32[Q,K] x q4[Q,K,D]"))
		return hir.Type{}, diags
	case "kv_read":
		if callable.Kind == syntax.CallableKernel {
			diags = append(diags, diagnosticError(call.Span, "kv_read is only supported in pipeline bodies"))
		}
		if len(args) != 1 {
			diags = append(diags, diagnosticError(call.Span, "kv_read expects 1 argument, got %d", len(args)))
			return hir.Type{}, diags
		}
		if args[0].Kind != hir.TypeKVCache {
			diags = append(diags, diagnosticError(call.Span, "kv_read requires a kv_cache input"))
			return hir.Type{}, diags
		}
		return hir.Type{Kind: hir.TypeTensor, Tensor: &hir.TensorType{DType: "f16", Shape: []hir.DimExpr{{Name: "T"}, {Name: "D"}}}}, diags
	case "kv_write":
		if callable.Kind == syntax.CallableKernel {
			diags = append(diags, diagnosticError(call.Span, "kv_write is only supported in pipeline bodies"))
		}
		if len(args) != 2 {
			diags = append(diags, diagnosticError(call.Span, "kv_write expects 2 arguments, got %d", len(args)))
			return hir.Type{}, diags
		}
		if args[0].Kind != hir.TypeKVCache {
			diags = append(diags, diagnosticError(call.Span, "kv_write first argument must be kv_cache"))
		}
		if args[1].Kind != hir.TypeTensor {
			diags = append(diags, diagnosticError(call.Span, "kv_write second argument must be a tensor"))
		}
		return hir.Type{Kind: hir.TypeKVCache}, diags
	}

	target, ok := env.callables[call.Callee]
	if !ok {
		diags = append(diags, diagnosticError(call.Span, "unknown callable %q", call.Callee))
		return hir.Type{}, diags
	}
	if target.Kind == syntax.CallablePipeline {
		diags = append(diags, diagnosticError(call.Span, "pipeline calls are not supported yet: %s", call.Callee))
		return primaryCallableResultType(target), diags
	}
	if callable.Kind == syntax.CallableKernel {
		diags = append(diags, diagnosticError(call.Span, "kernel bodies cannot call named kernels yet: %s", call.Callee))
		return primaryCallableResultType(target), diags
	}
	if len(call.Args) != len(target.Params) {
		diags = append(diags, diagnosticError(call.Span, "call to %s expects %d arguments, got %d", call.Callee, len(target.Params), len(call.Args)))
		return primaryCallableResultType(target), diags
	}
	for i, param := range target.Params {
		want := lowerType(param.Type)
		got := args[i]
		if got.Kind != "" && !sameType(got, want) {
			diags = append(diags, diagnosticError(call.Args[i].ExprSpan(), "argument %d to %s has type %s, want %s", i+1, call.Callee, typeString(got), typeString(want)))
		}
	}
	return primaryCallableResultType(target), diags
}

func exprHasEffect(expr syntax.Expr) bool {
	call, ok := expr.(*syntax.CallExpr)
	if !ok {
		return false
	}
	return call.Callee == "kv_write"
}

func inferSemanticTopKCallType(call *syntax.CallExpr, callable *syntax.CallableDecl, locals map[string]hir.Type, env moduleTypeEnv) (hir.Type, []syntax.Diagnostic) {
	diags := []syntax.Diagnostic{}
	if callable.Kind == syntax.CallableKernel {
		diags = append(diags, diagnosticError(call.Span, "topk is only supported in pipeline bodies"))
	}
	if len(call.Args) != 2 {
		diags = append(diags, diagnosticError(call.Span, "topk expects 2 arguments, got %d", len(call.Args)))
		return hir.Type{}, diags
	}
	scoreType, scoreDiags := inferSemanticExprType(call.Args[0], callable, locals, env)
	diags = append(diags, scoreDiags...)
	if !isRank1Tensor(scoreType) && !isRank2Tensor(scoreType) {
		diags = append(diags, diagnosticError(call.Args[0].ExprSpan(), "topk scores input must be a rank-1 or rank-2 tensor"))
		return hir.Type{}, diags
	}
	k, ok := topKLiteral(call.Args[1])
	if !ok {
		diags = append(diags, diagnosticError(call.Args[1].ExprSpan(), "topk requires a positive integer literal limit"))
		return hir.Type{}, diags
	}
	out := hir.Type{
		Kind: hir.TypeTensor,
		Tensor: &hir.TensorType{
			DType: "i32",
			Shape: []hir.DimExpr{{Name: fmt.Sprintf("%d", k)}},
		},
	}
	if isRank2Tensor(scoreType) {
		out.Tensor.Shape = []hir.DimExpr{{Name: scoreType.Tensor.Shape[0].Name}, {Name: fmt.Sprintf("%d", k)}}
	}
	return out, diags
}

func declName(decl syntax.Decl) string {
	switch d := decl.(type) {
	case *syntax.ParamDecl:
		return d.Name
	case *syntax.CallableDecl:
		return d.Name
	default:
		return ""
	}
}

func sameType(a, b hir.Type) bool {
	if a.Kind != b.Kind {
		return false
	}
	if a.Kind == hir.TypeKVCache {
		return true
	}
	if a.Kind == hir.TypeCandidatePack {
		if a.CandidatePack == nil || b.CandidatePack == nil {
			return a.CandidatePack == b.CandidatePack
		}
		return sameDimShape(a.CandidatePack.Shape, b.CandidatePack.Shape)
	}
	if a.Tensor == nil || b.Tensor == nil {
		return a.Tensor == b.Tensor
	}
	if a.Tensor.DType != b.Tensor.DType {
		return false
	}
	return sameTensorShape(a.Tensor, b.Tensor)
}

func sameTensorShape(a, b *hir.TensorType) bool {
	if a == nil || b == nil {
		return a == b
	}
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i].Name != b.Shape[i].Name {
			return false
		}
	}
	return true
}

func sameDimShape(a, b []hir.DimExpr) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Name != b[i].Name {
			return false
		}
	}
	return true
}

func isRank1Tensor(t hir.Type) bool {
	return t.Kind == hir.TypeTensor && t.Tensor != nil && len(t.Tensor.Shape) == 1
}

func isRank2Tensor(t hir.Type) bool {
	return t.Kind == hir.TypeTensor && t.Tensor != nil && len(t.Tensor.Shape) == 2
}

func isRank3Tensor(t hir.Type) bool {
	return t.Kind == hir.TypeTensor && t.Tensor != nil && len(t.Tensor.Shape) == 3
}

func topKLiteral(expr syntax.Expr) (int, bool) {
	num, ok := expr.(*syntax.NumberExpr)
	if !ok {
		return 0, false
	}
	value := strings.TrimSpace(num.Text)
	if value == "" {
		return 0, false
	}
	n := 0
	for _, r := range value {
		if r < '0' || r > '9' {
			return 0, false
		}
		n = n*10 + int(r-'0')
	}
	return n, n > 0
}

func cloneDims(in []hir.DimExpr) []hir.DimExpr {
	if len(in) == 0 {
		return nil
	}
	out := make([]hir.DimExpr, len(in))
	copy(out, in)
	return out
}

func primaryCallableResultType(callable *syntax.CallableDecl) hir.Type {
	fields := callableResultFields(callable)
	if len(fields) == 0 {
		return hir.Type{}
	}
	return lowerType(fields[0].Type)
}

func typeString(t hir.Type) string {
	switch t.Kind {
	case hir.TypeKVCache:
		return "kv_cache"
	case hir.TypeTensor:
		if t.Tensor == nil {
			return "tensor<?>"
		}
		if len(t.Tensor.Shape) == 0 {
			return t.Tensor.DType
		}
		dims := make([]string, 0, len(t.Tensor.Shape))
		for _, dim := range t.Tensor.Shape {
			dims = append(dims, dim.Name)
		}
		return fmt.Sprintf("%s[%s]", t.Tensor.DType, strings.Join(dims, ", "))
	case hir.TypeCandidatePack:
		if t.CandidatePack == nil {
			return "candidate_pack<?>"
		}
		dims := make([]string, 0, len(t.CandidatePack.Shape))
		for _, dim := range t.CandidatePack.Shape {
			dims = append(dims, dim.Name)
		}
		return fmt.Sprintf("candidate_pack[%s]", strings.Join(dims, ", "))
	default:
		return "<invalid>"
	}
}

func diagnosticError(span syntax.Span, format string, args ...any) syntax.Diagnostic {
	return syntax.Diagnostic{
		Severity: syntax.SeverityError,
		Message:  fmt.Sprintf(format, args...),
		Span:     span,
	}
}

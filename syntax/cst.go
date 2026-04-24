package syntax

import (
	"fmt"
	"strconv"
	"strings"

	gotreesitter "github.com/odvcencio/gotreesitter"
)

type cstLowerer struct {
	src   []byte
	lang  *gotreesitter.Language
	diags []Diagnostic
}

func parseWithTreeSitter(moduleName string, src []byte) (*File, []Diagnostic) {
	tree, lang, err := ParseTree(src)
	file := &File{
		ModuleName: moduleName,
		Source:     append([]byte(nil), src...),
	}
	if err != nil {
		file.Diagnostics = []Diagnostic{{
			Severity: SeverityError,
			Message:  err.Error(),
			Span:     Span{Start: 0, End: 0, Line: 1, Column: 1},
		}}
		return file, file.Diagnostics
	}

	root := tree.RootNode()
	l := &cstLowerer{src: append([]byte(nil), src...), lang: lang}
	if root.HasError() {
		if !l.collectParseErrors(root) {
			l.diags = append(l.diags, Diagnostic{
				Severity: SeverityError,
				Message:  "invalid Manta syntax",
				Span:     spanOf(root),
			})
		}
		file.Diagnostics = append(file.Diagnostics, l.diags...)
		return file, file.Diagnostics
	}

	for i := 0; i < root.NamedChildCount(); i++ {
		decl := l.lowerDecl(root.NamedChild(i))
		if decl != nil {
			file.Decls = append(file.Decls, decl)
		}
	}
	file.Diagnostics = append(file.Diagnostics, l.diags...)
	return file, file.Diagnostics
}

func (l *cstLowerer) lowerDecl(n *gotreesitter.Node) Decl {
	switch nodeType(n, l.lang) {
	case "param_declaration":
		return l.lowerParamDecl(n)
	case "kernel_declaration":
		return l.lowerCallableDecl(n, CallableKernel)
	case "pipeline_declaration":
		return l.lowerCallableDecl(n, CallablePipeline)
	case "comment":
		return nil
	default:
		l.errorf(n, "expected declaration, found %q", nodeType(n, l.lang))
		return nil
	}
}

func (l *cstLowerer) lowerParamDecl(n *gotreesitter.Node) Decl {
	nameNode := childByField(n, l.lang, "name")
	typeNode := childByField(n, l.lang, "type")
	bindingNode := childByField(n, l.lang, "binding")
	if nameNode == nil || typeNode == nil || bindingNode == nil {
		l.errorf(n, "malformed param declaration")
		return nil
	}
	typ, ok := l.lowerType(typeNode)
	if !ok {
		return nil
	}
	weight := firstNamedChildOfType(bindingNode, l.lang, "string_literal")
	if weight == nil {
		l.errorf(bindingNode, "expected @weight annotation")
		return nil
	}
	return &ParamDecl{
		Name:      nodeText(nameNode, l.src),
		Type:      typ,
		Binding:   unquoteMantaString(nodeText(weight, l.src)),
		Trainable: childByField(n, l.lang, "trainable") != nil,
		Span:      spanOf(n),
	}
}

func (l *cstLowerer) lowerCallableDecl(n *gotreesitter.Node, kind CallableKind) Decl {
	nameNode := childByField(n, l.lang, "name")
	paramsNode := childByField(n, l.lang, "parameters")
	resultNode := childByField(n, l.lang, "result")
	bodyNode := childByField(n, l.lang, "body")
	if nameNode == nil || paramsNode == nil || resultNode == nil || bodyNode == nil {
		l.errorf(n, "malformed %s declaration", kind)
		return nil
	}

	decl := &CallableDecl{
		Kind:   kind,
		Name:   nodeText(nameNode, l.src),
		Params: l.lowerFields(paramsNode),
		Span:   spanOf(n),
	}
	if nodeType(resultNode, l.lang) == "result_list" {
		if kind != CallablePipeline {
			l.errorf(resultNode, "kernel result must be a single type")
			return nil
		}
		decl.Results = l.lowerFields(resultNode)
	} else {
		result, ok := l.lowerType(resultNode)
		if !ok {
			return nil
		}
		decl.Result = result
	}

	for i := 0; i < bodyNode.NamedChildCount(); i++ {
		stmt := l.lowerStmt(bodyNode.NamedChild(i))
		if stmt != nil {
			decl.Body = append(decl.Body, stmt)
		}
	}
	return decl
}

func (l *cstLowerer) lowerFields(parent *gotreesitter.Node) []Field {
	fields := []Field{}
	for i := 0; i < parent.NamedChildCount(); i++ {
		child := parent.NamedChild(i)
		if nodeType(child, l.lang) != "field" {
			continue
		}
		field, ok := l.lowerField(child)
		if ok {
			fields = append(fields, field)
		}
	}
	return fields
}

func (l *cstLowerer) lowerField(n *gotreesitter.Node) (Field, bool) {
	nameNode := childByField(n, l.lang, "name")
	typeNode := childByField(n, l.lang, "type")
	if nameNode == nil || typeNode == nil {
		l.errorf(n, "malformed field")
		return Field{}, false
	}
	typ, ok := l.lowerType(typeNode)
	if !ok {
		return Field{}, false
	}
	return Field{Name: nodeText(nameNode, l.src), Type: typ, Span: spanOf(n)}, true
}

func (l *cstLowerer) lowerType(n *gotreesitter.Node) (TypeRef, bool) {
	nameNode := childByField(n, l.lang, "name")
	if nameNode == nil {
		l.errorf(n, "malformed type reference")
		return TypeRef{}, false
	}
	typ := TypeRef{Name: nodeText(nameNode, l.src), Span: spanOf(n)}
	if shapeNode := childByField(n, l.lang, "shape"); shapeNode != nil {
		for i := 0; i < shapeNode.NamedChildCount(); i++ {
			dimNode := shapeNode.NamedChild(i)
			if nodeType(dimNode, l.lang) != "dimension" {
				continue
			}
			typ.Shape = append(typ.Shape, DimExpr{Text: nodeText(dimNode, l.src), Span: spanOf(dimNode)})
		}
	}
	return typ, true
}

func (l *cstLowerer) lowerStmt(n *gotreesitter.Node) Stmt {
	switch nodeType(n, l.lang) {
	case "let_statement":
		nameNode := childByField(n, l.lang, "name")
		valueNode := childByField(n, l.lang, "value")
		if nameNode == nil || valueNode == nil {
			l.errorf(n, "malformed let statement")
			return nil
		}
		expr := l.lowerExpr(valueNode)
		if expr == nil {
			return nil
		}
		return &LetStmt{Name: nodeText(nameNode, l.src), Expr: expr, Span: spanOf(n)}
	case "return_statement":
		valuesNode := childByField(n, l.lang, "values")
		if valuesNode == nil {
			l.errorf(n, "malformed return statement")
			return nil
		}
		exprs := l.lowerExprList(valuesNode)
		if len(exprs) == 0 {
			return nil
		}
		stmt := &ReturnStmt{Span: spanOf(n)}
		if len(exprs) == 1 {
			stmt.Expr = exprs[0]
		} else {
			stmt.Exprs = exprs
		}
		return stmt
	case "expression_statement":
		exprNode := childByField(n, l.lang, "expression")
		if exprNode == nil {
			l.errorf(n, "malformed expression statement")
			return nil
		}
		expr := l.lowerExpr(exprNode)
		if expr == nil {
			return nil
		}
		return &ExprStmt{Expr: expr, Span: spanOf(n)}
	case "comment":
		return nil
	default:
		l.errorf(n, "expected statement, found %q", nodeType(n, l.lang))
		return nil
	}
}

func (l *cstLowerer) lowerExprList(n *gotreesitter.Node) []Expr {
	exprs := []Expr{}
	for i := 0; i < n.NamedChildCount(); i++ {
		expr := l.lowerExpr(n.NamedChild(i))
		if expr != nil {
			exprs = append(exprs, expr)
		}
	}
	return exprs
}

func (l *cstLowerer) lowerExpr(n *gotreesitter.Node) Expr {
	switch nodeType(n, l.lang) {
	case "identifier":
		return &IdentExpr{Name: nodeText(n, l.src), Span: spanOf(n)}
	case "number":
		return &NumberExpr{Text: nodeText(n, l.src), Span: spanOf(n)}
	case "string_literal":
		return &StringExpr{Value: unquoteMantaString(nodeText(n, l.src)), Span: spanOf(n)}
	case "call_expression":
		return l.lowerCallExpr(n, false)
	case "intrinsic_call_expression":
		return l.lowerCallExpr(n, true)
	case "binary_expression":
		left := l.lowerExpr(childByField(n, l.lang, "left"))
		right := l.lowerExpr(childByField(n, l.lang, "right"))
		op := nodeText(childByField(n, l.lang, "operator"), l.src)
		if left == nil || right == nil || op == "" {
			l.errorf(n, "malformed binary expression")
			return nil
		}
		return &BinaryExpr{Op: op, Left: left, Right: right, Span: spanOf(n)}
	case "unary_expression":
		operand := l.lowerExpr(childByField(n, l.lang, "operand"))
		op := nodeText(childByField(n, l.lang, "operator"), l.src)
		if operand == nil || op != "-" {
			l.errorf(n, "malformed unary expression")
			return nil
		}
		zero := &NumberExpr{Text: "0", Span: spanOf(n)}
		return &BinaryExpr{Op: "-", Left: zero, Right: operand, Span: spanOf(n)}
	case "parenthesized_expression":
		return l.lowerExpr(childByField(n, l.lang, "expression"))
	default:
		if n == nil {
			return nil
		}
		l.errorf(n, "expected expression, found %q", nodeType(n, l.lang))
		return nil
	}
}

func (l *cstLowerer) lowerCallExpr(n *gotreesitter.Node, intrinsic bool) Expr {
	calleeNode := childByField(n, l.lang, "callee")
	argsNode := childByField(n, l.lang, "arguments")
	if calleeNode == nil || argsNode == nil {
		l.errorf(n, "malformed call expression")
		return nil
	}
	call := &CallExpr{
		Callee:    nodeText(calleeNode, l.src),
		Intrinsic: intrinsic,
		Span:      spanOf(n),
	}
	for i := 0; i < argsNode.NamedChildCount(); i++ {
		arg := l.lowerExpr(argsNode.NamedChild(i))
		if arg != nil {
			call.Args = append(call.Args, arg)
		}
	}
	return call
}

func (l *cstLowerer) collectParseErrors(n *gotreesitter.Node) bool {
	if n == nil {
		return false
	}
	if n.IsError() || n.IsMissing() {
		l.diags = append(l.diags, Diagnostic{
			Severity: SeverityError,
			Message:  fmt.Sprintf("invalid Manta syntax near %q", strings.TrimSpace(nodeText(n, l.src))),
			Span:     spanOf(n),
		})
		return true
	}
	found := false
	for i := 0; i < n.ChildCount(); i++ {
		if l.collectParseErrors(n.Child(i)) {
			found = true
		}
	}
	return found
}

func (l *cstLowerer) errorf(n *gotreesitter.Node, format string, args ...any) {
	l.diags = append(l.diags, Diagnostic{
		Severity: SeverityError,
		Message:  fmt.Sprintf(format, args...),
		Span:     spanOf(n),
	})
}

func childByField(n *gotreesitter.Node, lang *gotreesitter.Language, name string) *gotreesitter.Node {
	if n == nil {
		return nil
	}
	return n.ChildByFieldName(name, lang)
}

func firstNamedChildOfType(n *gotreesitter.Node, lang *gotreesitter.Language, typ string) *gotreesitter.Node {
	if n == nil {
		return nil
	}
	for i := 0; i < n.NamedChildCount(); i++ {
		child := n.NamedChild(i)
		if nodeType(child, lang) == typ {
			return child
		}
	}
	return nil
}

func nodeType(n *gotreesitter.Node, lang *gotreesitter.Language) string {
	if n == nil {
		return ""
	}
	return n.Type(lang)
}

func nodeText(n *gotreesitter.Node, src []byte) string {
	if n == nil {
		return ""
	}
	return n.Text(src)
}

func spanOf(n *gotreesitter.Node) Span {
	if n == nil {
		return Span{Line: 1, Column: 1}
	}
	start := n.StartPoint()
	return Span{
		Start:  int(n.StartByte()),
		End:    int(n.EndByte()),
		Line:   int(start.Row) + 1,
		Column: int(start.Column) + 1,
	}
}

func unquoteMantaString(raw string) string {
	value, err := strconv.Unquote(raw)
	if err == nil {
		return value
	}
	return strings.Trim(raw, `"`)
}

package syntax

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

type tokenKind int

const (
	tokInvalid tokenKind = iota
	tokEOF
	tokIdent
	tokNumber
	tokString
	tokParam
	tokKernel
	tokPipeline
	tokLet
	tokReturn
	tokAt
	tokColon
	tokComma
	tokLParen
	tokRParen
	tokLBracket
	tokRBracket
	tokLBrace
	tokRBrace
	tokArrow
	tokAssign
	tokPlus
	tokMinus
	tokStar
	tokSlash
)

type token struct {
	kind tokenKind
	text string
	span Span
}

type lexer struct {
	src    []byte
	pos    int
	line   int
	column int
}

func newLexer(src []byte) *lexer {
	return &lexer{src: src, line: 1, column: 1}
}

func (l *lexer) next() token {
	l.skipSpaceAndComments()
	startPos := l.pos
	startLine := l.line
	startCol := l.column

	if l.pos >= len(l.src) {
		return token{kind: tokEOF, span: Span{Start: l.pos, End: l.pos, Line: l.line, Column: l.column}}
	}

	r, size := utf8.DecodeRune(l.src[l.pos:])
	switch {
	case isIdentStart(r):
		l.bump(size)
		for l.pos < len(l.src) {
			r, size = utf8.DecodeRune(l.src[l.pos:])
			if !isIdentPart(r) {
				break
			}
			l.bump(size)
		}
		text := string(l.src[startPos:l.pos])
		return token{kind: keywordKind(text), text: text, span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case unicode.IsDigit(r):
		l.bump(size)
		for l.pos < len(l.src) {
			r, size = utf8.DecodeRune(l.src[l.pos:])
			if !(unicode.IsDigit(r) || r == '.') {
				break
			}
			l.bump(size)
		}
		text := string(l.src[startPos:l.pos])
		return token{kind: tokNumber, text: text, span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '"':
		l.bump(size)
		for l.pos < len(l.src) {
			r, size = utf8.DecodeRune(l.src[l.pos:])
			if r == '"' {
				text := string(l.src[startPos+1 : l.pos])
				l.bump(size)
				return token{kind: tokString, text: text, span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
			}
			l.bump(size)
		}
		return token{kind: tokInvalid, text: "unterminated string", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '@':
		l.bump(size)
		return token{kind: tokAt, text: "@", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == ':':
		l.bump(size)
		return token{kind: tokColon, text: ":", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == ',':
		l.bump(size)
		return token{kind: tokComma, text: ",", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '(':
		l.bump(size)
		return token{kind: tokLParen, text: "(", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == ')':
		l.bump(size)
		return token{kind: tokRParen, text: ")", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '[':
		l.bump(size)
		return token{kind: tokLBracket, text: "[", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == ']':
		l.bump(size)
		return token{kind: tokRBracket, text: "]", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '{':
		l.bump(size)
		return token{kind: tokLBrace, text: "{", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '}':
		l.bump(size)
		return token{kind: tokRBrace, text: "}", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '=':
		l.bump(size)
		return token{kind: tokAssign, text: "=", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '+':
		l.bump(size)
		return token{kind: tokPlus, text: "+", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '*':
		l.bump(size)
		return token{kind: tokStar, text: "*", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '/':
		l.bump(size)
		return token{kind: tokSlash, text: "/", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	case r == '-':
		l.bump(size)
		if l.pos < len(l.src) {
			r, size = utf8.DecodeRune(l.src[l.pos:])
			if r == '>' {
				l.bump(size)
				return token{kind: tokArrow, text: "->", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
			}
		}
		return token{kind: tokMinus, text: "-", span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	default:
		l.bump(size)
		return token{kind: tokInvalid, text: string(r), span: Span{Start: startPos, End: l.pos, Line: startLine, Column: startCol}}
	}
}

func (l *lexer) skipSpaceAndComments() {
	for l.pos < len(l.src) {
		r, size := utf8.DecodeRune(l.src[l.pos:])
		if unicode.IsSpace(r) {
			l.bump(size)
			continue
		}
		if r == '/' && l.pos+1 < len(l.src) && l.src[l.pos+1] == '/' {
			l.bump(1)
			l.bump(1)
			for l.pos < len(l.src) {
				r, size = utf8.DecodeRune(l.src[l.pos:])
				l.bump(size)
				if r == '\n' {
					break
				}
			}
			continue
		}
		return
	}
}

func (l *lexer) bump(size int) {
	text := l.src[l.pos : l.pos+size]
	for _, b := range text {
		if b == '\n' {
			l.line++
			l.column = 1
		} else {
			l.column++
		}
	}
	l.pos += size
}

func keywordKind(text string) tokenKind {
	switch text {
	case "param":
		return tokParam
	case "kernel":
		return tokKernel
	case "pipeline":
		return tokPipeline
	case "let":
		return tokLet
	case "return":
		return tokReturn
	default:
		return tokIdent
	}
}

func isIdentStart(r rune) bool {
	return r == '_' || unicode.IsLetter(r)
}

func isIdentPart(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}

type parser struct {
	moduleName string
	src        []byte
	lx         *lexer
	cur        token
	peek       token
	diags      []Diagnostic
}

// Parse parses Manta source into a tiny v0 AST.
func Parse(moduleName string, src []byte) (*File, []Diagnostic) {
	p := &parser{
		moduleName: moduleName,
		src:        append([]byte(nil), src...),
		lx:         newLexer(src),
	}
	p.cur = p.lx.next()
	p.peek = p.lx.next()
	file := &File{
		ModuleName: moduleName,
		Source:     append([]byte(nil), src...),
	}
	for p.cur.kind != tokEOF && p.cur.kind != tokInvalid {
		decl := p.parseDecl()
		if decl != nil {
			file.Decls = append(file.Decls, decl)
			continue
		}
		p.advance()
	}
	if p.cur.kind == tokInvalid {
		p.errorf(p.cur.span, "invalid token %q", p.cur.text)
	}
	file.Diagnostics = append(file.Diagnostics, p.diags...)
	return file, file.Diagnostics
}

func (p *parser) parseDecl() Decl {
	switch p.cur.kind {
	case tokParam:
		return p.parseParamDecl()
	case tokKernel:
		return p.parseCallableDecl(CallableKernel)
	case tokPipeline:
		return p.parseCallableDecl(CallablePipeline)
	default:
		p.errorf(p.cur.span, "expected declaration, found %q", p.cur.text)
		return nil
	}
}

func (p *parser) parseParamDecl() Decl {
	start := p.cur.span
	p.advance()
	nameTok, ok := p.expect(tokIdent, "param name")
	if !ok {
		return nil
	}
	if _, ok := p.expect(tokColon, "':' after param name"); !ok {
		return nil
	}
	typ, ok := p.parseType()
	if !ok {
		return nil
	}
	if _, ok := p.expect(tokAt, "'@' annotation"); !ok {
		return nil
	}
	ann, ok := p.expect(tokIdent, "annotation name")
	if !ok {
		return nil
	}
	if ann.text != "weight" {
		p.errorf(ann.span, "expected @weight annotation")
		return nil
	}
	if _, ok := p.expect(tokLParen, "'(' after @weight"); !ok {
		return nil
	}
	binding, ok := p.expect(tokString, "weight binding string")
	if !ok {
		return nil
	}
	if _, ok := p.expect(tokRParen, "')' after @weight binding"); !ok {
		return nil
	}
	trainable := false
	if p.cur.kind == tokAt {
		p.advance()
		ann, ok := p.expect(tokIdent, "annotation name")
		if !ok {
			return nil
		}
		if ann.text != "trainable" {
			p.errorf(ann.span, "unsupported param annotation @%s", ann.text)
			return nil
		}
		trainable = true
	}
	return &ParamDecl{
		Name:      nameTok.text,
		Type:      typ,
		Binding:   binding.text,
		Trainable: trainable,
		Span:      mergeSpans(start, binding.span),
	}
}

func (p *parser) parseCallableDecl(kind CallableKind) Decl {
	start := p.cur.span
	p.advance()
	nameTok, ok := p.expect(tokIdent, "callable name")
	if !ok {
		return nil
	}
	if _, ok := p.expect(tokLParen, "'(' after callable name"); !ok {
		return nil
	}
	params := p.parseFields()
	if _, ok := p.expect(tokRParen, "')' after parameter list"); !ok {
		return nil
	}
	if _, ok := p.expect(tokArrow, "'->' after parameter list"); !ok {
		return nil
	}
	result, results, ok := p.parseCallableResult(kind)
	if !ok {
		return nil
	}
	if _, ok := p.expect(tokLBrace, "'{' to start callable body"); !ok {
		return nil
	}
	var body []Stmt
	for p.cur.kind != tokRBrace && p.cur.kind != tokEOF {
		stmt := p.parseStmt()
		if stmt != nil {
			body = append(body, stmt)
			continue
		}
		p.advance()
	}
	endTok, ok := p.expect(tokRBrace, "'}' to end callable body")
	if !ok {
		return nil
	}
	return &CallableDecl{
		Kind:    kind,
		Name:    nameTok.text,
		Params:  params,
		Result:  result,
		Results: results,
		Body:    body,
		Span:    mergeSpans(start, endTok.span),
	}
}

func (p *parser) parseFields() []Field {
	var out []Field
	if p.cur.kind == tokRParen {
		return out
	}
	for {
		nameTok, ok := p.expect(tokIdent, "parameter name")
		if !ok {
			return out
		}
		if _, ok := p.expect(tokColon, "':' after parameter name"); !ok {
			return out
		}
		typ, ok := p.parseType()
		if !ok {
			return out
		}
		out = append(out, Field{Name: nameTok.text, Type: typ, Span: mergeSpans(nameTok.span, typ.Span)})
		if p.cur.kind != tokComma {
			return out
		}
		p.advance()
	}
}

func (p *parser) parseType() (TypeRef, bool) {
	nameTok, ok := p.expect(tokIdent, "type name")
	if !ok {
		return TypeRef{}, false
	}
	typ := TypeRef{Name: nameTok.text, Span: nameTok.span}
	if p.cur.kind != tokLBracket {
		return typ, true
	}
	p.advance()
	for p.cur.kind != tokRBracket && p.cur.kind != tokEOF {
		switch p.cur.kind {
		case tokIdent, tokNumber:
			dimTok := p.cur
			typ.Shape = append(typ.Shape, DimExpr{Text: dimTok.text, Span: dimTok.span})
			typ.Span = mergeSpans(typ.Span, dimTok.span)
			p.advance()
		default:
			p.errorf(p.cur.span, "expected shape dimension, found %q", p.cur.text)
			return TypeRef{}, false
		}
		if p.cur.kind == tokComma {
			p.advance()
			continue
		}
	}
	endTok, ok := p.expect(tokRBracket, "']' after type shape")
	if !ok {
		return TypeRef{}, false
	}
	typ.Span = mergeSpans(typ.Span, endTok.span)
	return typ, true
}

func (p *parser) parseCallableResult(kind CallableKind) (TypeRef, []Field, bool) {
	if kind == CallablePipeline && p.cur.kind == tokLParen {
		start := p.cur.span
		p.advance()
		fields := p.parseFields()
		endTok, ok := p.expect(tokRParen, "')' after result list")
		if !ok {
			return TypeRef{}, nil, false
		}
		for i := range fields {
			fields[i].Span = mergeSpans(fields[i].Span, endTok.span)
			if fields[i].Name == "" {
				p.errorf(start, "pipeline result names are required")
				return TypeRef{}, nil, false
			}
		}
		return TypeRef{}, fields, true
	}
	result, ok := p.parseType()
	if !ok {
		return TypeRef{}, nil, false
	}
	return result, nil, true
}

func (p *parser) parseStmt() Stmt {
	switch p.cur.kind {
	case tokLet:
		start := p.cur.span
		p.advance()
		nameTok, ok := p.expect(tokIdent, "local name")
		if !ok {
			return nil
		}
		if _, ok := p.expect(tokAssign, "'=' after local name"); !ok {
			return nil
		}
		expr, ok := p.parseExpr()
		if !ok {
			return nil
		}
		return &LetStmt{Name: nameTok.text, Expr: expr, Span: mergeSpans(start, expr.ExprSpan())}
	case tokReturn:
		start := p.cur.span
		p.advance()
		exprs, end, ok := p.parseExprList()
		if !ok {
			return nil
		}
		stmt := &ReturnStmt{Span: mergeSpans(start, end)}
		if len(exprs) == 1 {
			stmt.Expr = exprs[0]
		} else {
			stmt.Exprs = exprs
		}
		return stmt
	default:
		expr, ok := p.parseExpr()
		if !ok {
			return nil
		}
		return &ExprStmt{Expr: expr, Span: expr.ExprSpan()}
	}
}

func (p *parser) parseExprList() ([]Expr, Span, bool) {
	first, ok := p.parseExpr()
	if !ok {
		return nil, Span{}, false
	}
	exprs := []Expr{first}
	end := first.ExprSpan()
	for p.cur.kind == tokComma {
		p.advance()
		expr, ok := p.parseExpr()
		if !ok {
			return nil, Span{}, false
		}
		exprs = append(exprs, expr)
		end = expr.ExprSpan()
	}
	return exprs, end, true
}

func (p *parser) parseExpr() (Expr, bool) {
	return p.parseBinaryExpr(0)
}

func (p *parser) parseBinaryExpr(minPrec int) (Expr, bool) {
	left, ok := p.parsePrimary()
	if !ok {
		return nil, false
	}
	for {
		op, prec, ok := binaryInfo(p.cur.kind)
		if !ok || prec < minPrec {
			return left, true
		}
		opTok := p.cur
		p.advance()
		right, ok := p.parseBinaryExpr(prec + 1)
		if !ok {
			return nil, false
		}
		left = &BinaryExpr{
			Op:    op,
			Left:  left,
			Right: right,
			Span:  mergeSpans(left.ExprSpan(), mergeSpans(opTok.span, right.ExprSpan())),
		}
	}
}

func (p *parser) parsePrimary() (Expr, bool) {
	switch p.cur.kind {
	case tokIdent:
		tok := p.cur
		p.advance()
		if p.cur.kind == tokLParen {
			return p.finishCall(tok.text, false, tok.span)
		}
		return &IdentExpr{Name: tok.text, Span: tok.span}, true
	case tokAt:
		start := p.cur.span
		p.advance()
		nameTok, ok := p.expect(tokIdent, "intrinsic name")
		if !ok {
			return nil, false
		}
		if p.cur.kind != tokLParen {
			p.errorf(nameTok.span, "expected '(' after intrinsic name")
			return nil, false
		}
		return p.finishCall(nameTok.text, true, mergeSpans(start, nameTok.span))
	case tokNumber:
		tok := p.cur
		p.advance()
		return &NumberExpr{Text: tok.text, Span: tok.span}, true
	case tokString:
		tok := p.cur
		p.advance()
		return &StringExpr{Value: tok.text, Span: tok.span}, true
	case tokLParen:
		p.advance()
		expr, ok := p.parseExpr()
		if !ok {
			return nil, false
		}
		if _, ok := p.expect(tokRParen, "')' after grouped expression"); !ok {
			return nil, false
		}
		return expr, true
	case tokMinus:
		opTok := p.cur
		p.advance()
		right, ok := p.parsePrimary()
		if !ok {
			return nil, false
		}
		zero := &NumberExpr{Text: "0", Span: opTok.span}
		return &BinaryExpr{Op: "-", Left: zero, Right: right, Span: mergeSpans(opTok.span, right.ExprSpan())}, true
	default:
		p.errorf(p.cur.span, "expected expression, found %q", p.cur.text)
		return nil, false
	}
}

func (p *parser) finishCall(name string, intrinsic bool, start Span) (Expr, bool) {
	if _, ok := p.expect(tokLParen, "'(' after call name"); !ok {
		return nil, false
	}
	var args []Expr
	for p.cur.kind != tokRParen && p.cur.kind != tokEOF {
		arg, ok := p.parseExpr()
		if !ok {
			return nil, false
		}
		args = append(args, arg)
		if p.cur.kind != tokComma {
			break
		}
		p.advance()
	}
	endTok, ok := p.expect(tokRParen, "')' after call arguments")
	if !ok {
		return nil, false
	}
	return &CallExpr{
		Callee:    name,
		Intrinsic: intrinsic,
		Args:      args,
		Span:      mergeSpans(start, endTok.span),
	}, true
}

func (p *parser) expect(kind tokenKind, context string) (token, bool) {
	if p.cur.kind != kind {
		p.errorf(p.cur.span, "expected %s, found %q", context, p.cur.text)
		return token{}, false
	}
	tok := p.cur
	p.advance()
	return tok, true
}

func (p *parser) advance() {
	p.cur = p.peek
	p.peek = p.lx.next()
}

func (p *parser) errorf(span Span, format string, args ...any) {
	p.diags = append(p.diags, Diagnostic{
		Severity: SeverityError,
		Message:  fmt.Sprintf(format, args...),
		Span:     span,
	})
}

func binaryInfo(kind tokenKind) (string, int, bool) {
	switch kind {
	case tokPlus:
		return "+", 1, true
	case tokMinus:
		return "-", 1, true
	case tokStar:
		return "*", 2, true
	case tokSlash:
		return "/", 2, true
	default:
		return "", 0, false
	}
}

func mergeSpans(a, b Span) Span {
	line := a.Line
	if line == 0 {
		line = b.Line
	}
	col := a.Column
	if col == 0 {
		col = b.Column
	}
	start := a.Start
	if start == 0 && a.End == 0 {
		start = b.Start
	}
	end := b.End
	if end == 0 {
		end = a.End
	}
	return Span{Start: start, End: end, Line: line, Column: col}
}

// DebugString renders diagnostics for tests and callers.
func DebugString(diags []Diagnostic) string {
	var parts []string
	for _, d := range diags {
		parts = append(parts, fmt.Sprintf("%s:%d:%d: %s", d.Severity, d.Span.Line, d.Span.Column, d.Message))
	}
	return strings.Join(parts, "\n")
}

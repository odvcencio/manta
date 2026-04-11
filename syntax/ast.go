package syntax

// File is the parsed Barracuda source module.
type File struct {
	ModuleName  string
	Source      []byte
	Decls       []Decl
	Diagnostics []Diagnostic
}

// Decl is a top-level declaration.
type Decl interface {
	declNode()
	DeclSpan() Span
}

// ParamDecl declares an externally bound model parameter.
type ParamDecl struct {
	Name      string
	Type      TypeRef
	Binding   string
	Trainable bool
	Span      Span
}

func (d *ParamDecl) declNode()      {}
func (d *ParamDecl) DeclSpan() Span { return d.Span }

// CallableKind identifies a top-level callable declaration.
type CallableKind string

const (
	CallableKernel   CallableKind = "kernel"
	CallablePipeline CallableKind = "pipeline"
)

// CallableDecl is a kernel or pipeline declaration.
type CallableDecl struct {
	Kind    CallableKind
	Name    string
	Params  []Field
	Result  TypeRef
	Results []Field
	Body    []Stmt
	Span    Span
}

func (d *CallableDecl) declNode()      {}
func (d *CallableDecl) DeclSpan() Span { return d.Span }

// Field is a named parameter or result binding.
type Field struct {
	Name string
	Type TypeRef
	Span Span
}

// TypeRef is a parsed type reference.
type TypeRef struct {
	Name  string
	Shape []DimExpr
	Span  Span
}

// DimExpr is a symbolic or literal dimension expression in v0.
type DimExpr struct {
	Text string
	Span Span
}

// Stmt is a statement in a callable body.
type Stmt interface {
	stmtNode()
	StmtSpan() Span
}

// LetStmt binds a local name.
type LetStmt struct {
	Name string
	Expr Expr
	Span Span
}

func (s *LetStmt) stmtNode()      {}
func (s *LetStmt) StmtSpan() Span { return s.Span }

// ReturnStmt returns a value from the callable.
type ReturnStmt struct {
	Expr  Expr
	Exprs []Expr
	Span  Span
}

func (s *ReturnStmt) stmtNode()      {}
func (s *ReturnStmt) StmtSpan() Span { return s.Span }

// ExprStmt evaluates an expression for effect.
type ExprStmt struct {
	Expr Expr
	Span Span
}

func (s *ExprStmt) stmtNode()      {}
func (s *ExprStmt) StmtSpan() Span { return s.Span }

// Expr is an expression node.
type Expr interface {
	exprNode()
	ExprSpan() Span
}

// IdentExpr references a named value.
type IdentExpr struct {
	Name string
	Span Span
}

func (e *IdentExpr) exprNode()      {}
func (e *IdentExpr) ExprSpan() Span { return e.Span }

// NumberExpr is a numeric literal.
type NumberExpr struct {
	Text string
	Span Span
}

func (e *NumberExpr) exprNode()      {}
func (e *NumberExpr) ExprSpan() Span { return e.Span }

// StringExpr is a string literal.
type StringExpr struct {
	Value string
	Span  Span
}

func (e *StringExpr) exprNode()      {}
func (e *StringExpr) ExprSpan() Span { return e.Span }

// CallExpr is a regular or intrinsic call.
type CallExpr struct {
	Callee    string
	Intrinsic bool
	Args      []Expr
	Span      Span
}

func (e *CallExpr) exprNode()      {}
func (e *CallExpr) ExprSpan() Span { return e.Span }

// BinaryExpr is a binary operator expression.
type BinaryExpr struct {
	Op    string
	Left  Expr
	Right Expr
	Span  Span
}

func (e *BinaryExpr) exprNode()      {}
func (e *BinaryExpr) ExprSpan() Span { return e.Span }

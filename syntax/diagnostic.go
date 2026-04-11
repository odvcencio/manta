package syntax

// Span identifies a byte range in source.
type Span struct {
	Start  int
	End    int
	Line   int
	Column int
}

// Severity classifies a diagnostic.
type Severity string

const (
	SeverityError   Severity = "error"
	SeverityWarning Severity = "warning"
)

// Diagnostic records a compiler message tied to source.
type Diagnostic struct {
	Severity Severity
	Message  string
	Span     Span
}

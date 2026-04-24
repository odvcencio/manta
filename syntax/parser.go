package syntax

import (
	"fmt"
	"strings"
)

// Parse parses Manta source into the compiler-facing AST.
func Parse(moduleName string, src []byte) (*File, []Diagnostic) {
	return parseWithTreeSitter(moduleName, src)
}

// DebugString renders diagnostics for tests and callers.
func DebugString(diags []Diagnostic) string {
	var parts []string
	for _, d := range diags {
		parts = append(parts, fmt.Sprintf("%s:%d:%d: %s", d.Severity, d.Span.Line, d.Span.Column, d.Message))
	}
	return strings.Join(parts, "\n")
}

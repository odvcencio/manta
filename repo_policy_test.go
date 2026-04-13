package manta_test

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestNoTrackedSpecOrPlanDocs(t *testing.T) {
	cmd := exec.Command("git", "ls-files")
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("git ls-files unavailable: %v", err)
	}
	var bad []string
	for _, raw := range strings.Split(string(out), "\n") {
		path := strings.TrimSpace(raw)
		if path == "" {
			continue
		}
		if isSpecOrPlanDoc(path) {
			bad = append(bad, path)
		}
	}
	if len(bad) > 0 {
		t.Fatalf("spec/plan docs must not be tracked:\n%s", strings.Join(bad, "\n"))
	}
}

func isSpecOrPlanDoc(path string) bool {
	path = strings.ToLower(filepath.ToSlash(path))
	if strings.HasPrefix(path, "specs/") ||
		strings.HasPrefix(path, "plans/") ||
		strings.HasPrefix(path, "docs/specs/") ||
		strings.HasPrefix(path, "docs/plans/") {
		return true
	}
	base := filepath.Base(path)
	switch base {
	case "spec.md", "specs.md", "specification.md", "specifications.md", "plan.md", "plans.md":
		return true
	}
	return strings.HasSuffix(path, ".spec.md") || strings.HasSuffix(path, ".plan.md")
}

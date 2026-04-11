package barruntime

import (
	"path/filepath"
	"strings"
)

func defaultManifestPath(artifactPath, suffix string) string {
	ext := filepath.Ext(artifactPath)
	if ext == "" {
		return artifactPath + suffix
	}
	if ext == suffix {
		return strings.TrimSuffix(artifactPath, ext) + ".sealed" + suffix
	}
	return strings.TrimSuffix(artifactPath, ext) + suffix
}

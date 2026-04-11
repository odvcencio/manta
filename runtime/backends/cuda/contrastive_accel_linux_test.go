//go:build linux && cgo

package cuda

import (
	"math"
	"strings"
	"testing"

	"github.com/odvcencio/manta/runtime/backend"
)

func TestCUDAContrastiveAcceleratorMatchesHostInfoNCE(t *testing.T) {
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		t.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query := backend.NewTensorF32([]int{3, 4}, []float32{
		0.25, -0.5, 0.75, 1.25,
		-0.25, 0.8, 0.3, -0.6,
		0.9, 0.1, -0.4, 0.2,
	})
	positive := backend.NewTensorF32([]int{3, 4}, []float32{
		-0.75, 0.5, 0.125, 1.5,
		0.15, 0.9, -0.25, -0.35,
		0.8, -0.2, -0.55, 0.45,
	})
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}

	got, err := accelAny.RunInfoNCE(query, positive, cfg)
	if err != nil {
		t.Fatalf("run infonce: %v", err)
	}
	wantQ, wantP, wantLoss, wantScore := hostInfoNCEGrad(query.F32, positive.F32, 3, 4, cfg.Temperature)
	assertTensorClose(t, got.QueryGrads, []int{3, 4}, wantQ)
	assertTensorClose(t, got.PositiveGrads, []int{3, 4}, wantP)
	assertCloseF32(t, got.LossSum, wantLoss, 0.0001)
	assertCloseF32(t, got.ScoreSum, wantScore, 0.0001)

	stats := accelAny.Stats()
	if stats.RunCalls != 1 {
		t.Fatalf("run calls = %d, want 1", stats.RunCalls)
	}
	if stats.UploadedBytes <= 0 {
		t.Fatalf("uploaded bytes = %d, want positive", stats.UploadedBytes)
	}
	if stats.DownloadedBytes <= 0 {
		t.Fatalf("downloaded bytes = %d, want positive", stats.DownloadedBytes)
	}
	if stats.RunNanos <= 0 {
		t.Fatalf("run nanos = %d, want positive", stats.RunNanos)
	}
}

func TestCUDAContrastiveKernelsUse64BitPairIndexing(t *testing.T) {
	for name, source := range map[string]string{
		"scores": contrastiveScoresKernelSource,
		"grad":   contrastiveGradKernelSource,
	} {
		if !strings.Contains(source, "long long idx = (long long)blockIdx.x") {
			t.Fatalf("%s kernel does not use 64-bit global index", name)
		}
	}
	if !strings.Contains(contrastiveGradKernelSource, "(long long)rows * (long long)rows * (long long)width") {
		t.Fatal("grad kernel does not compute total grad elements in 64-bit")
	}
}

func BenchmarkCUDAContrastiveAccelerator256x64(b *testing.B) {
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		b.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		b.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query, positive := syntheticContrastiveTensors(256, 64)
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := accelAny.RunInfoNCE(query, positive, cfg); err != nil {
			b.Fatalf("run infonce: %v", err)
		}
	}
}

func BenchmarkHostContrastiveInfoNCE256x64(b *testing.B) {
	query, positive := syntheticContrastiveTensors(256, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hostInfoNCEGrad(query.F32, positive.F32, 256, 64, 0.05)
	}
}

func syntheticContrastiveTensors(rows, width int) (*backend.Tensor, *backend.Tensor) {
	query := make([]float32, rows*width)
	positive := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		for col := 0; col < width; col++ {
			idx := row*width + col
			value := float32(((row+1)*(col+3))%23) / 23
			query[idx] = value - 0.5
			positive[idx] = value*0.75 + float32((row+col)%7)/17 - 0.35
		}
	}
	return backend.NewTensorF32([]int{rows, width}, query), backend.NewTensorF32([]int{rows, width}, positive)
}

func hostInfoNCEGrad(query, positive []float32, rows, width int, temperature float32) ([]float32, []float32, float32, float32) {
	if temperature <= 0 {
		temperature = 0.05
	}
	queryNorms := tensorRowNorms(query, rows, width)
	positiveNorms := tensorRowNorms(positive, rows, width)
	queryGrads := make([]float32, rows*width)
	positiveGrads := make([]float32, rows*width)
	totalLoss := float32(0)
	totalScore := float32(0)
	rowScores := make([]float32, rows)
	probs := make([]float32, rows)
	for row := 0; row < rows; row++ {
		for col := 0; col < rows; col++ {
			score := hostCosineScore(query[row*width:(row+1)*width], positive[col*width:(col+1)*width], queryNorms[row], positiveNorms[col])
			rowScores[col] = score
			totalScore += score
		}
		totalLoss += hostInfoNCEProbs(rowScores, row, temperature, probs)
		for col, prob := range probs {
			target := float32(0)
			if row == col {
				target = 1
			}
			scale := (prob - target) / temperature
			hostAccumulateCosineGrad(
				query[row*width:(row+1)*width],
				positive[col*width:(col+1)*width],
				queryNorms[row],
				positiveNorms[col],
				rowScores[col],
				scale,
				queryGrads[row*width:(row+1)*width],
				positiveGrads[col*width:(col+1)*width],
			)
		}
	}
	return queryGrads, positiveGrads, totalLoss, totalScore
}

func hostCosineScore(left, right []float32, leftNorm, rightNorm float32) float32 {
	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	dot := float32(0)
	for i := range left {
		dot += left[i] * right[i]
	}
	return dot / (leftNorm * rightNorm)
}

func hostAccumulateCosineGrad(left, right []float32, leftNorm, rightNorm, score, scale float32, gradLeft, gradRight []float32) {
	if leftNorm == 0 || rightNorm == 0 {
		return
	}
	denom := leftNorm * rightNorm
	leftScale := score / (leftNorm * leftNorm)
	rightScale := score / (rightNorm * rightNorm)
	for i := range left {
		gradLeft[i] += scale * (right[i]/denom - left[i]*leftScale)
		gradRight[i] += scale * (left[i]/denom - right[i]*rightScale)
	}
}

func hostInfoNCEProbs(scores []float32, target int, temperature float32, probs []float32) float32 {
	maxLogit := scores[0] / temperature
	for _, score := range scores[1:] {
		logit := score / temperature
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	sum := float32(0)
	for i, score := range scores {
		value := float32(math.Exp(float64(score/temperature - maxLogit)))
		probs[i] = value
		sum += value
	}
	for i := range probs {
		probs[i] /= sum
	}
	prob := probs[target]
	if prob < 1e-12 {
		prob = 1e-12
	}
	return -float32(math.Log(float64(prob)))
}

func assertCloseF32(t *testing.T, got, want, tol float32) {
	t.Helper()
	diff := got - want
	if diff < -tol || diff > tol {
		t.Fatalf("got %f, want %f", got, want)
	}
}

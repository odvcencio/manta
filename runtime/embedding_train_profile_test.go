package barruntime

import (
	"path/filepath"
	"testing"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

func TestDefaultEmbeddingTrainProfilePath(t *testing.T) {
	got := DefaultEmbeddingTrainProfilePath("/tmp/tiny_train_embed_q8.barr")
	if want := "/tmp/tiny_train_embed_q8.train-profile.mll"; got != want {
		t.Fatalf("training profile path = %q, want %q", got, want)
	}
}

func TestEmbeddingTrainProfileRoundTrip(t *testing.T) {
	want := EmbeddingTrainProfile{
		Version:           EmbeddingTrainProfileVersion,
		Step:              7,
		ForwardBackend:    "cuda",
		OptimizerBackend:  "cuda",
		ActivationBackend: "cuda",
		ForwardResidency: EmbeddingForwardResidencyStats{
			BindSkips: 3,
			MatMul: backend.MatMulAcceleratorStats{
				BindCalls:          8,
				UploadedBytes:      512,
				QuantizePasses:     5,
				QuantizedBytes:     320,
				BindNanos:          1000,
				QuantizeNanos:      500,
				BoundMatrices:      2,
				RunCalls:           13,
				BoundLeftCalls:     4,
				BoundRightCalls:    6,
				RunUploadedBytes:   2048,
				RunDownloadedBytes: 1536,
				RunNanos:           7000,
			},
		},
		Optimizer: backend.OptimizerAcceleratorStats{
			UpdateCalls:     4,
			SyncCalls:       2,
			UploadedBytes:   1024,
			DownloadedBytes: 512,
			UpdateNanos:     100,
			SyncNanos:       40,
			ResidentParams:  3,
		},
		Activation: backend.ActivationAcceleratorStats{
			BindCalls:              4,
			GELUBackwardCalls:      3,
			SoftmaxBackwardCalls:   2,
			LayerNormBackwardCalls: 1,
			UploadedBytes:          768,
			DownloadedBytes:        256,
			RunNanos:               80,
			BoundTensors:           2,
		},
	}
	path := filepath.Join(t.TempDir(), "tiny.train-profile.mll")
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write training profile: %v", err)
	}
	got, err := ReadEmbeddingTrainProfileFile(path)
	if err != nil {
		t.Fatalf("read training profile: %v", err)
	}
	if got.Version != want.Version {
		t.Fatalf("version = %q, want %q", got.Version, want.Version)
	}
	if got.Step != want.Step {
		t.Fatalf("step = %d, want %d", got.Step, want.Step)
	}
	if got.ForwardResidency.BindSkips != want.ForwardResidency.BindSkips {
		t.Fatalf("bind skips = %d, want %d", got.ForwardResidency.BindSkips, want.ForwardResidency.BindSkips)
	}
	if got.ForwardResidency.MatMul.BindCalls != want.ForwardResidency.MatMul.BindCalls {
		t.Fatalf("bind calls = %d, want %d", got.ForwardResidency.MatMul.BindCalls, want.ForwardResidency.MatMul.BindCalls)
	}
	if got.ForwardResidency.MatMul.QuantizePasses != want.ForwardResidency.MatMul.QuantizePasses {
		t.Fatalf("quantize passes = %d, want %d", got.ForwardResidency.MatMul.QuantizePasses, want.ForwardResidency.MatMul.QuantizePasses)
	}
	if got.ForwardResidency.MatMul.RunCalls != want.ForwardResidency.MatMul.RunCalls {
		t.Fatalf("run calls = %d, want %d", got.ForwardResidency.MatMul.RunCalls, want.ForwardResidency.MatMul.RunCalls)
	}
	if got.ForwardResidency.MatMul.RunUploadedBytes != want.ForwardResidency.MatMul.RunUploadedBytes {
		t.Fatalf("run uploaded bytes = %d, want %d", got.ForwardResidency.MatMul.RunUploadedBytes, want.ForwardResidency.MatMul.RunUploadedBytes)
	}
	if got.Optimizer.UpdateCalls != want.Optimizer.UpdateCalls {
		t.Fatalf("optimizer update calls = %d, want %d", got.Optimizer.UpdateCalls, want.Optimizer.UpdateCalls)
	}
	if got.Activation.GELUBackwardCalls != want.Activation.GELUBackwardCalls {
		t.Fatalf("activation gelu calls = %d, want %d", got.Activation.GELUBackwardCalls, want.Activation.GELUBackwardCalls)
	}
	if got.Activation.BindCalls != want.Activation.BindCalls {
		t.Fatalf("activation bind calls = %d, want %d", got.Activation.BindCalls, want.Activation.BindCalls)
	}
}

func TestEmbeddingTrainerFitCapturesProfileDelta(t *testing.T) {
	trainer := newTinyTrainableAttentionEmbeddingTrainer(t, 0.05)
	if trainer.forwardMatMul != nil {
		trainer.forwardMatMul.Close()
	}
	fake := &countingMatMulAccelerator{}
	trainer.forwardMatMul = fake
	trainer.forwardBackend = barr.BackendCUDA

	summary, err := trainer.FitContrastive(tinyEmbeddingContrastiveDataset(), tinyEmbeddingContrastiveDataset(), EmbeddingTrainRunConfig{
		Epochs:      2,
		BatchSize:   2,
		Shuffle:     false,
		Seed:        1,
		RestoreBest: true,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.StartProfile.Version != EmbeddingTrainProfileVersion {
		t.Fatalf("start profile version = %q, want %q", summary.StartProfile.Version, EmbeddingTrainProfileVersion)
	}
	if summary.EndProfile.Version != EmbeddingTrainProfileVersion {
		t.Fatalf("end profile version = %q, want %q", summary.EndProfile.Version, EmbeddingTrainProfileVersion)
	}
	if summary.DeltaProfile.Step != summary.StepsCompleted {
		t.Fatalf("delta profile step = %d, want %d", summary.DeltaProfile.Step, summary.StepsCompleted)
	}
	if summary.DeltaProfile.ForwardResidency.MatMul.BindCalls <= 0 {
		t.Fatalf("delta bind calls = %d, want positive", summary.DeltaProfile.ForwardResidency.MatMul.BindCalls)
	}
	if summary.EndProfile.ForwardResidency.MatMul.BindCalls < summary.StartProfile.ForwardResidency.MatMul.BindCalls {
		t.Fatalf("end profile bind calls = %d, want at least start count %d", summary.EndProfile.ForwardResidency.MatMul.BindCalls, summary.StartProfile.ForwardResidency.MatMul.BindCalls)
	}
	if summary.EndProfile.Optimizer.UpdateCalls < summary.StartProfile.Optimizer.UpdateCalls {
		t.Fatalf("end optimizer update calls = %d, want at least start count %d", summary.EndProfile.Optimizer.UpdateCalls, summary.StartProfile.Optimizer.UpdateCalls)
	}
	if summary.EndProfile.Activation.SoftmaxBackwardCalls < summary.StartProfile.Activation.SoftmaxBackwardCalls {
		t.Fatalf("end activation softmax calls = %d, want at least start count %d", summary.EndProfile.Activation.SoftmaxBackwardCalls, summary.StartProfile.Activation.SoftmaxBackwardCalls)
	}
	if summary.EndProfile.Activation.BindCalls < summary.StartProfile.Activation.BindCalls {
		t.Fatalf("end activation bind calls = %d, want at least start count %d", summary.EndProfile.Activation.BindCalls, summary.StartProfile.Activation.BindCalls)
	}
	if summary.DeltaProfile.Optimizer.UpdateCalls != summary.EndProfile.Optimizer.UpdateCalls-summary.StartProfile.Optimizer.UpdateCalls {
		t.Fatalf("optimizer delta update calls = %d, want %d", summary.DeltaProfile.Optimizer.UpdateCalls, summary.EndProfile.Optimizer.UpdateCalls-summary.StartProfile.Optimizer.UpdateCalls)
	}
	if summary.DeltaProfile.Activation.SoftmaxBackwardCalls != summary.EndProfile.Activation.SoftmaxBackwardCalls-summary.StartProfile.Activation.SoftmaxBackwardCalls {
		t.Fatalf("activation delta softmax calls = %d, want %d", summary.DeltaProfile.Activation.SoftmaxBackwardCalls, summary.EndProfile.Activation.SoftmaxBackwardCalls-summary.StartProfile.Activation.SoftmaxBackwardCalls)
	}
	if summary.DeltaProfile.Activation.BindCalls != summary.EndProfile.Activation.BindCalls-summary.StartProfile.Activation.BindCalls {
		t.Fatalf("activation delta bind calls = %d, want %d", summary.DeltaProfile.Activation.BindCalls, summary.EndProfile.Activation.BindCalls-summary.StartProfile.Activation.BindCalls)
	}
	if fake.bindCalls <= 0 {
		t.Fatalf("fake bind calls = %d, want positive", fake.bindCalls)
	}
}

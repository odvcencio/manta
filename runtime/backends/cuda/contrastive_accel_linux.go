//go:build linux && cgo

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#include <cuda.h>
*/
import "C"

import (
	"fmt"
	"math"
	"time"

	"github.com/odvcencio/barracuda/artifact/barr"
	"github.com/odvcencio/barracuda/runtime/backend"
)

const contrastiveScoresKernelSource = `
extern "C" __global__ void barr_contrastive_scores(
    const float* query,
    const float* positive,
    const float* query_norms,
    const float* positive_norms,
    float* scores,
    int rows,
    int width
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)rows * (long long)rows;
    if (idx >= total) {
        return;
    }
    int row = (int)(idx / rows);
    int col = (int)(idx - (long long)row * rows);
    float qn = query_norms[row];
    float pn = positive_norms[col];
    if (qn == 0.0f || pn == 0.0f) {
        scores[idx] = 0.0f;
        return;
    }
    const float* q = query + row * width;
    const float* p = positive + col * width;
    float dot = 0.0f;
    for (int k = 0; k < width; ++k) {
        dot += q[k] * p[k];
    }
    scores[idx] = dot / (qn * pn);
}
`

const infoNCEScalesKernelSource = `
extern "C" __global__ void barr_infonce_scales(
    const float* scores,
    float* scales,
    float* row_loss,
    float* row_score,
    int rows,
    float temperature
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * rows;
    float temp = temperature > 0.0f ? temperature : 0.05f;
    float max_logit = scores[base] / temp;
    float score_sum = 0.0f;
    for (int col = 0; col < rows; ++col) {
        float score = scores[base + col];
        score_sum += score;
        float logit = score / temp;
        if (logit > max_logit) {
            max_logit = logit;
        }
    }
    float denom = 0.0f;
    for (int col = 0; col < rows; ++col) {
        denom += expf(scores[base + col] / temp - max_logit);
    }
    float target_prob = 0.0f;
    for (int col = 0; col < rows; ++col) {
        float prob = 0.0f;
        if (denom != 0.0f) {
            prob = expf(scores[base + col] / temp - max_logit) / denom;
        }
        if (col == row) {
            target_prob = prob;
            scales[base + col] = (prob - 1.0f) / temp;
        } else {
            scales[base + col] = prob / temp;
        }
    }
    if (target_prob < 1e-12f) {
        target_prob = 1e-12f;
    }
    row_loss[row] = -logf(target_prob);
    row_score[row] = score_sum;
}
`

const contrastiveGradKernelSource = `
extern "C" __global__ void barr_contrastive_grad(
    const float* query,
    const float* positive,
    const float* query_norms,
    const float* positive_norms,
    const float* scores,
    const float* scales,
    float* query_grads,
    float* positive_grads,
    int rows,
    int width
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)rows * (long long)rows * (long long)width;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % width);
    long long pair = idx / width;
    int row = (int)(pair / rows);
    int col = (int)(pair - (long long)row * rows);
    float qn = query_norms[row];
    float pn = positive_norms[col];
    if (qn == 0.0f || pn == 0.0f) {
        return;
    }
    int qBase = row * width;
    int pBase = col * width;
    float score = scores[pair];
    float scale = scales[pair];
    float denom = qn * pn;
    float qScale = score / (qn * qn);
    float pScale = score / (pn * pn);
    float qGrad = scale * (positive[pBase + k] / denom - query[qBase + k] * qScale);
    float pGrad = scale * (query[qBase + k] / denom - positive[pBase + k] * pScale);
    atomicAdd(&query_grads[qBase + k], qGrad);
    atomicAdd(&positive_grads[pBase + k], pGrad);
}
`

type contrastiveAccelerator struct {
	device      *deviceRuntime
	scoreKernel *auxKernel
	scaleKernel *auxKernel
	gradKernel  *auxKernel
	stats       backend.ContrastiveAcceleratorStats
}

func init() {
	backend.RegisterContrastiveAccelerator(barr.BackendCUDA, NewContrastiveAccelerator)
}

func NewContrastiveAccelerator() (backend.ContrastiveAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	scoreKernel, err := device.compileAuxKernel(contrastiveScoresKernelSource, "barr_contrastive_scores")
	if err != nil {
		device.close()
		return nil, err
	}
	scaleKernel, err := device.compileAuxKernel(infoNCEScalesKernelSource, "barr_infonce_scales")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.close()
		return nil, err
	}
	gradKernel, err := device.compileAuxKernel(contrastiveGradKernelSource, "barr_contrastive_grad")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.destroyAuxKernel(scaleKernel)
		device.close()
		return nil, err
	}
	return &contrastiveAccelerator{
		device:      device,
		scoreKernel: scoreKernel,
		scaleKernel: scaleKernel,
		gradKernel:  gradKernel,
	}, nil
}

func (a *contrastiveAccelerator) Backend() barr.BackendKind {
	return barr.BackendCUDA
}

func (a *contrastiveAccelerator) RunInfoNCE(query, positive *backend.Tensor, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	if a == nil || a.device == nil || a.scoreKernel == nil || a.scaleKernel == nil || a.gradKernel == nil {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda contrastive accelerator is not initialized")
	}
	rows, width, err := contrastiveShape(query, positive)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if rows < 2 {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce requires at least two rows")
	}
	temperature := cfg.Temperature
	if temperature <= 0 {
		temperature = 0.05
	}
	start := time.Now()
	queryNorms := tensorRowNorms(query.F32, rows, width)
	positiveNorms := tensorRowNorms(positive.F32, rows, width)
	queryBuf, err := a.device.uploadFloat32(query.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryBuf)
	positiveBuf, err := a.device.uploadFloat32(positive.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(positiveBuf)
	queryNormBuf, err := a.device.uploadFloat32(queryNorms)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryNormBuf)
	positiveNormBuf, err := a.device.uploadFloat32(positiveNorms)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(positiveNormBuf)
	a.stats.UploadedBytes += int64((len(query.F32) + len(positive.F32) + len(queryNorms) + len(positiveNorms)) * 4)

	pairCount := rows * rows
	scoresBuf, err := a.device.allocFloat32(pairCount)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(scoresBuf)
	scalesBuf, err := a.device.allocFloat32(pairCount)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(scalesBuf)
	rowLossBuf, err := a.device.allocFloat32(rows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowLossBuf)
	rowScoreBuf, err := a.device.allocFloat32(rows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowScoreBuf)

	zeroGrads := make([]float32, rows*width)
	queryGradBuf, err := a.device.uploadFloat32(zeroGrads)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryGradBuf)
	positiveGradBuf, err := a.device.uploadFloat32(zeroGrads)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(positiveGradBuf)
	a.stats.UploadedBytes += int64(len(zeroGrads) * 8)

	block := uint(128)
	scoreGrid := uint((pairCount + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveScores(a.scoreKernel, scoreGrid, block, queryBuf, positiveBuf, queryNormBuf, positiveNormBuf, scoresBuf, rows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	rowGrid := uint((rows + int(block) - 1) / int(block))
	if err := a.device.launchAuxInfoNCEScales(a.scaleKernel, rowGrid, block, scoresBuf, scalesBuf, rowLossBuf, rowScoreBuf, rows, temperature); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	gradElements := pairCount * width
	gradGrid := uint((gradElements + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveGrad(a.gradKernel, gradGrid, block, queryBuf, positiveBuf, queryNormBuf, positiveNormBuf, scoresBuf, scalesBuf, queryGradBuf, positiveGradBuf, rows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	queryGrads := make([]float32, rows*width)
	positiveGrads := make([]float32, rows*width)
	rowLoss := make([]float32, rows)
	rowScore := make([]float32, rows)
	if err := a.device.downloadFloat32(queryGrads, queryGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(positiveGrads, positiveGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowLoss, rowLossBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowScore, rowScoreBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	a.stats.DownloadedBytes += int64((len(queryGrads) + len(positiveGrads) + len(rowLoss) + len(rowScore)) * 4)
	a.stats.RunCalls++
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32([]int{rows, width}, queryGrads),
		PositiveGrads: backend.NewTensorF32([]int{rows, width}, positiveGrads),
		LossSum:       sumFloat32(rowLoss),
		ScoreSum:      sumFloat32(rowScore),
	}, nil
}

func (a *contrastiveAccelerator) Stats() backend.ContrastiveAcceleratorStats {
	if a == nil {
		return backend.ContrastiveAcceleratorStats{}
	}
	return a.stats
}

func (a *contrastiveAccelerator) Close() {
	if a == nil || a.device == nil {
		return
	}
	a.device.destroyAuxKernel(a.scoreKernel)
	a.device.destroyAuxKernel(a.scaleKernel)
	a.device.destroyAuxKernel(a.gradKernel)
	a.scoreKernel = nil
	a.scaleKernel = nil
	a.gradKernel = nil
	a.device.close()
	a.device = nil
}

func contrastiveShape(query, positive *backend.Tensor) (rows, width int, err error) {
	if query == nil || positive == nil {
		return 0, 0, fmt.Errorf("cuda contrastive tensors are required")
	}
	if query.Rank() != 2 || positive.Rank() != 2 {
		return 0, 0, fmt.Errorf("cuda contrastive tensors must be rank-2")
	}
	if query.Shape[0] != positive.Shape[0] || query.Shape[1] != positive.Shape[1] {
		return 0, 0, fmt.Errorf("cuda contrastive tensor shape mismatch %v vs %v", query.Shape, positive.Shape)
	}
	if len(query.F32) != len(positive.F32) || len(query.F32) != query.Shape[0]*query.Shape[1] {
		return 0, 0, fmt.Errorf("cuda contrastive tensor data does not match shape")
	}
	return query.Shape[0], query.Shape[1], nil
}

func tensorRowNorms(data []float32, rows, width int) []float32 {
	out := make([]float32, rows)
	for row := 0; row < rows; row++ {
		base := row * width
		sum := float32(0)
		for col := 0; col < width; col++ {
			value := data[base+col]
			sum += value * value
		}
		out[row] = float32(math.Sqrt(float64(sum)))
	}
	return out
}

func sumFloat32(values []float32) float32 {
	sum := float32(0)
	for _, value := range values {
		sum += value
	}
	return sum
}

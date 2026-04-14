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

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
)

const contrastiveScoresKernelSource = `
extern "C" __global__ void manta_contrastive_scores(
    const float* query,
    const float* candidates,
    const float* query_norms,
    const float* candidate_norms,
    float* scores,
    int query_rows,
    int candidate_rows,
    int width
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)query_rows * (long long)candidate_rows;
    if (idx >= total) {
        return;
    }
    int row = (int)(idx / candidate_rows);
    int col = (int)(idx - (long long)row * candidate_rows);
    float qn = query_norms[row];
    float pn = candidate_norms[col];
    if (qn == 0.0f || pn == 0.0f) {
        scores[idx] = 0.0f;
        return;
    }
    const float* q = query + row * width;
    const float* p = candidates + col * width;
    float dot = 0.0f;
    for (int k = 0; k < width; ++k) {
        dot += q[k] * p[k];
    }
    scores[idx] = dot / (qn * pn);
}
`

const infoNCEScalesKernelSource = `
extern "C" __global__ void manta_infonce_scales(
    const float* scores,
    const float* target_indexes,
    float* scales,
    float* row_loss,
    float* row_score,
    int query_rows,
    int candidate_rows,
    float temperature
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= query_rows) {
        return;
    }
    int base = row * candidate_rows;
    int target_index = (int)target_indexes[row];
    if (target_index < 0 || target_index >= candidate_rows) {
        target_index = 0;
    }
    float temp = temperature > 0.0f ? temperature : 0.05f;
    float max_logit = scores[base] / temp;
    float score_sum = 0.0f;
    for (int col = 0; col < candidate_rows; ++col) {
        float score = scores[base + col];
        score_sum += score;
        float logit = score / temp;
        if (logit > max_logit) {
            max_logit = logit;
        }
    }
    float denom = 0.0f;
    for (int col = 0; col < candidate_rows; ++col) {
        denom += expf(scores[base + col] / temp - max_logit);
    }
    float target_prob = 0.0f;
    for (int col = 0; col < candidate_rows; ++col) {
        float prob = 0.0f;
        if (denom != 0.0f) {
            prob = expf(scores[base + col] / temp - max_logit) / denom;
        }
        if (col == target_index) {
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
extern "C" __global__ void manta_contrastive_grad(
    const float* query,
    const float* candidates,
    const float* query_norms,
    const float* candidate_norms,
    const float* scores,
    const float* scales,
    float* query_grads,
    float* candidate_grads,
    int query_rows,
    int candidate_rows,
    int width
) {
    long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long total = (long long)query_rows * (long long)candidate_rows * (long long)width;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % width);
    long long pair = idx / width;
    int row = (int)(pair / candidate_rows);
    int col = (int)(pair - (long long)row * candidate_rows);
    float qn = query_norms[row];
    float pn = candidate_norms[col];
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
    float qGrad = scale * (candidates[pBase + k] / denom - query[qBase + k] * qScale);
    float pGrad = scale * (query[qBase + k] / denom - candidates[pBase + k] * pScale);
    atomicAdd(&query_grads[qBase + k], qGrad);
    atomicAdd(&candidate_grads[pBase + k], pGrad);
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
	backend.RegisterContrastiveAccelerator(mantaartifact.BackendCUDA, NewContrastiveAccelerator)
}

func NewContrastiveAccelerator() (backend.ContrastiveAccelerator, error) {
	device, err := newDeviceRuntime()
	if err != nil {
		return nil, err
	}
	if device == nil {
		return nil, nil
	}
	scoreKernel, err := device.compileAuxKernel(contrastiveScoresKernelSource, "manta_contrastive_scores")
	if err != nil {
		device.close()
		return nil, err
	}
	scaleKernel, err := device.compileAuxKernel(infoNCEScalesKernelSource, "manta_infonce_scales")
	if err != nil {
		device.destroyAuxKernel(scoreKernel)
		device.close()
		return nil, err
	}
	gradKernel, err := device.compileAuxKernel(contrastiveGradKernelSource, "manta_contrastive_grad")
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

func (a *contrastiveAccelerator) Backend() mantaartifact.BackendKind {
	return mantaartifact.BackendCUDA
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
	targetIndexes := make([]int, rows)
	for i := range targetIndexes {
		targetIndexes[i] = i
	}
	return a.runInfoNCE(query, positive, targetIndexes, cfg, rows, rows, width)
}

func (a *contrastiveAccelerator) RunInfoNCEWithTargets(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig) (backend.ContrastiveGradResult, error) {
	if a == nil || a.device == nil || a.scoreKernel == nil || a.scaleKernel == nil || a.gradKernel == nil {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda contrastive accelerator is not initialized")
	}
	queryRows, candidateRows, width, err := contrastiveRectShape(query, candidates)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if queryRows < 1 {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce requires at least one query row")
	}
	if candidateRows < 2 {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce requires at least two candidate rows")
	}
	if len(targetIndexes) != queryRows {
		return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce target indexes length %d does not match query rows %d", len(targetIndexes), queryRows)
	}
	for row, target := range targetIndexes {
		if target < 0 || target >= candidateRows {
			return backend.ContrastiveGradResult{}, fmt.Errorf("cuda infonce target index %d for row %d is outside %d candidates", target, row, candidateRows)
		}
	}
	return a.runInfoNCE(query, candidates, targetIndexes, cfg, queryRows, candidateRows, width)
}

func (a *contrastiveAccelerator) runInfoNCE(query, candidates *backend.Tensor, targetIndexes []int, cfg backend.ContrastiveLossConfig, queryRows, candidateRows, width int) (backend.ContrastiveGradResult, error) {
	temperature := cfg.Temperature
	if temperature <= 0 {
		temperature = 0.05
	}
	start := time.Now()
	queryNorms := tensorRowNorms(query.F32, queryRows, width)
	candidateNorms := tensorRowNorms(candidates.F32, candidateRows, width)
	queryBuf, err := a.device.uploadFloat32(query.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryBuf)
	candidateBuf, err := a.device.uploadFloat32(candidates.F32)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateBuf)
	queryNormBuf, err := a.device.uploadFloat32(queryNorms)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryNormBuf)
	candidateNormBuf, err := a.device.uploadFloat32(candidateNorms)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateNormBuf)
	targetData := make([]float32, len(targetIndexes))
	for i, target := range targetIndexes {
		targetData[i] = float32(target)
	}
	targetBuf, err := a.device.uploadFloat32(targetData)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(targetBuf)
	a.stats.UploadedBytes += int64((len(query.F32) + len(candidates.F32) + len(queryNorms) + len(candidateNorms) + len(targetData)) * 4)

	pairCount := queryRows * candidateRows
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
	rowLossBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowLossBuf)
	rowScoreBuf, err := a.device.allocFloat32(queryRows)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(rowScoreBuf)

	queryZeroGrads := make([]float32, queryRows*width)
	queryGradBuf, err := a.device.uploadFloat32(queryZeroGrads)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(queryGradBuf)
	candidateZeroGrads := make([]float32, candidateRows*width)
	candidateGradBuf, err := a.device.uploadFloat32(candidateZeroGrads)
	if err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	defer a.device.freeBuffer(candidateGradBuf)
	a.stats.UploadedBytes += int64((len(queryZeroGrads) + len(candidateZeroGrads)) * 4)

	block := uint(128)
	scoreGrid := uint((pairCount + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveScores(a.scoreKernel, scoreGrid, block, queryBuf, candidateBuf, queryNormBuf, candidateNormBuf, scoresBuf, queryRows, candidateRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	rowGrid := uint((queryRows + int(block) - 1) / int(block))
	if err := a.device.launchAuxInfoNCEScales(a.scaleKernel, rowGrid, block, scoresBuf, targetBuf, scalesBuf, rowLossBuf, rowScoreBuf, queryRows, candidateRows, temperature); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	gradElements := pairCount * width
	gradGrid := uint((gradElements + int(block) - 1) / int(block))
	if err := a.device.launchAuxContrastiveGrad(a.gradKernel, gradGrid, block, queryBuf, candidateBuf, queryNormBuf, candidateNormBuf, scoresBuf, scalesBuf, queryGradBuf, candidateGradBuf, queryRows, candidateRows, width); err != nil {
		return backend.ContrastiveGradResult{}, err
	}

	queryGrads := make([]float32, queryRows*width)
	candidateGrads := make([]float32, candidateRows*width)
	rowLoss := make([]float32, queryRows)
	rowScore := make([]float32, queryRows)
	if err := a.device.downloadFloat32(queryGrads, queryGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(candidateGrads, candidateGradBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowLoss, rowLossBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	if err := a.device.downloadFloat32(rowScore, rowScoreBuf); err != nil {
		return backend.ContrastiveGradResult{}, err
	}
	a.stats.DownloadedBytes += int64((len(queryGrads) + len(candidateGrads) + len(rowLoss) + len(rowScore)) * 4)
	a.stats.RunCalls++
	a.stats.RunNanos += time.Since(start).Nanoseconds()
	return backend.ContrastiveGradResult{
		QueryGrads:    backend.NewTensorF32([]int{queryRows, width}, queryGrads),
		PositiveGrads: backend.NewTensorF32([]int{candidateRows, width}, candidateGrads),
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

func contrastiveRectShape(query, candidates *backend.Tensor) (queryRows, candidateRows, width int, err error) {
	if query == nil || candidates == nil {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensors are required")
	}
	if query.Rank() != 2 || candidates.Rank() != 2 {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensors must be rank-2")
	}
	if query.Shape[1] != candidates.Shape[1] {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensor width mismatch %v vs %v", query.Shape, candidates.Shape)
	}
	if len(query.F32) != query.Shape[0]*query.Shape[1] || len(candidates.F32) != candidates.Shape[0]*candidates.Shape[1] {
		return 0, 0, 0, fmt.Errorf("cuda contrastive tensor data does not match shape")
	}
	return query.Shape[0], candidates.Shape[0], query.Shape[1], nil
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

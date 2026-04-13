//go:build js && wasm

package webgpu

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"syscall/js"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/runtime/backend"
	turboquant "github.com/odvcencio/turboquant"
)

type deviceRuntime struct {
	device    js.Value
	queue     js.Value
	pipelines map[mantaartifact.StepKind]js.Value
}

func newDeviceRuntime(ctx context.Context) (*deviceRuntime, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	navigator := js.Global().Get("navigator")
	if !navigator.Truthy() {
		return nil, nil
	}
	gpu := navigator.Get("gpu")
	if !gpu.Truthy() {
		return nil, nil
	}
	adapter, err := awaitPromise(ctx, gpu.Call("requestAdapter"))
	if err != nil {
		return nil, fmt.Errorf("request WebGPU adapter: %w", err)
	}
	if !adapter.Truthy() {
		return nil, nil
	}
	device, err := awaitPromise(ctx, adapter.Call("requestDevice"))
	if err != nil {
		return nil, fmt.Errorf("request WebGPU device: %w", err)
	}
	if !device.Truthy() {
		return nil, nil
	}
	return &deviceRuntime{
		device:    device,
		queue:     device.Get("queue"),
		pipelines: map[mantaartifact.StepKind]js.Value{},
	}, nil
}

func (d *deviceRuntime) dispatchStep(ctx context.Context, step mantaartifact.Step, inputs []*backend.Tensor) (backend.StepDispatchResult, bool, error) {
	if d == nil || !d.device.Truthy() {
		return backend.StepDispatchResult{}, false, nil
	}
	switch step.Kind {
	case mantaartifact.StepConv2D:
		out, meta, ok, err := d.runConv2D(ctx, step, inputs)
		return stepResult(out, meta), ok, err
	case mantaartifact.StepConv2DTrans:
		out, meta, ok, err := d.runConv2DTranspose(ctx, step, inputs)
		return stepResult(out, meta), ok, err
	case mantaartifact.StepGDN:
		out, meta, ok, err := d.runGDN(ctx, step, inputs, false)
		return stepResult(out, meta), ok, err
	case mantaartifact.StepIGDN:
		out, meta, ok, err := d.runGDN(ctx, step, inputs, true)
		return stepResult(out, meta), ok, err
	case mantaartifact.StepTurboQDecode:
		out, meta, ok, err := d.runTurboQuantDecode(ctx, step, inputs)
		return stepResult(out, meta), ok, err
	default:
		return backend.StepDispatchResult{}, false, nil
	}
}

func stepResult(out []*backend.Tensor, meta map[string]any) backend.StepDispatchResult {
	if len(out) == 0 {
		return backend.StepDispatchResult{}
	}
	var entry, sourceHash string
	if raw, ok := meta["launch_entry"].(string); ok {
		entry = raw
	}
	if raw, ok := meta["wgsl_source_hash"].(string); ok {
		sourceHash = raw
	}
	return backend.StepDispatchResult{
		Outputs:      out,
		VariantEntry: entry,
		SourceHash:   sourceHash,
		Metadata:     meta,
	}
}

func (d *deviceRuntime) runConv2D(ctx context.Context, step mantaartifact.Step, inputs []*backend.Tensor) ([]*backend.Tensor, map[string]any, bool, error) {
	if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
		return nil, nil, false, nil
	}
	input, weight := inputs[0], inputs[1]
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(input.F32) != input.Elements() || len(weight.F32) != weight.Elements() {
		return nil, nil, false, nil
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	outC, inPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	groups := webgpuAttrInt(step.Attributes, "groups", 1)
	if groups <= 0 || inPerGroup*groups != inC || outC%groups != 0 {
		return nil, nil, false, nil
	}
	strideH := webgpuAttrInt(step.Attributes, "stride_h", webgpuAttrInt(step.Attributes, "stride", 1))
	strideW := webgpuAttrInt(step.Attributes, "stride_w", webgpuAttrInt(step.Attributes, "stride", 1))
	padH := webgpuAttrInt(step.Attributes, "pad_h", webgpuAttrInt(step.Attributes, "padding", 0))
	padW := webgpuAttrInt(step.Attributes, "pad_w", webgpuAttrInt(step.Attributes, "padding", 0))
	dilH := webgpuAttrInt(step.Attributes, "dilation_h", webgpuAttrInt(step.Attributes, "dilation", 1))
	dilW := webgpuAttrInt(step.Attributes, "dilation_w", webgpuAttrInt(step.Attributes, "dilation", 1))
	if strideH <= 0 || strideW <= 0 || dilH <= 0 || dilW <= 0 {
		return nil, nil, false, nil
	}
	outH := (inH+2*padH-dilH*(kH-1)-1)/strideH + 1
	outW := (inW+2*padW-dilW*(kW-1)-1)/strideW + 1
	if outH <= 0 || outW <= 0 {
		return nil, nil, false, nil
	}
	hasBias := len(inputs) == 3 && inputs[2] != nil
	biasData := []float32{0}
	if hasBias {
		if len(inputs[2].F32) < outC {
			return nil, nil, false, nil
		}
		biasData = inputs[2].F32
	}
	cfg := uint32Bytes(
		uint32(n), uint32(inC), uint32(inH), uint32(inW),
		uint32(outC), uint32(outH), uint32(outW),
		uint32(inPerGroup), uint32(outC/groups), uint32(kH), uint32(kW), uint32(groups),
		uint32(strideH), uint32(strideW), uint32(int32(padH)), uint32(int32(padW)),
		uint32(dilH), uint32(dilW), boolU32(hasBias),
	)
	kernel, _ := BuiltinForStep(mantaartifact.StepConv2D)
	outputBytes, meta, err := d.dispatch(ctx, kernel, []js.Value{
		d.floatStorageBuffer(input.F32),
		d.floatStorageBuffer(weight.F32),
		d.floatStorageBuffer(biasData),
		d.outputBuffer(outC * outH * outW * n * 4),
		d.uniformBuffer(cfg),
	}, 3, outC*outH*outW*n*4, ceilDiv(outW, 8), ceilDiv(outH, 8), n*outC)
	if err != nil {
		return nil, nil, true, err
	}
	out := tensorForWebGPUDType(input.DType, []int{n, outC, outH, outW}, bytesToFloat32s(outputBytes, n*outC*outH*outW))
	return []*backend.Tensor{out}, meta, true, nil
}

func (d *deviceRuntime) runConv2DTranspose(ctx context.Context, step mantaartifact.Step, inputs []*backend.Tensor) ([]*backend.Tensor, map[string]any, bool, error) {
	if len(inputs) < 2 || len(inputs) > 3 || inputs[0] == nil || inputs[1] == nil {
		return nil, nil, false, nil
	}
	input, weight := inputs[0], inputs[1]
	if len(input.Shape) != 4 || len(weight.Shape) != 4 || len(input.F32) != input.Elements() || len(weight.F32) != weight.Elements() {
		return nil, nil, false, nil
	}
	n, inC, inH, inW := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	weightInC, outPerGroup, kH, kW := weight.Shape[0], weight.Shape[1], weight.Shape[2], weight.Shape[3]
	if weightInC != inC {
		return nil, nil, false, nil
	}
	groups := webgpuAttrInt(step.Attributes, "groups", 1)
	if groups <= 0 || inC%groups != 0 {
		return nil, nil, false, nil
	}
	strideH := webgpuAttrInt(step.Attributes, "stride_h", webgpuAttrInt(step.Attributes, "stride", 1))
	strideW := webgpuAttrInt(step.Attributes, "stride_w", webgpuAttrInt(step.Attributes, "stride", 1))
	padH := webgpuAttrInt(step.Attributes, "pad_h", webgpuAttrInt(step.Attributes, "padding", 0))
	padW := webgpuAttrInt(step.Attributes, "pad_w", webgpuAttrInt(step.Attributes, "padding", 0))
	dilH := webgpuAttrInt(step.Attributes, "dilation_h", webgpuAttrInt(step.Attributes, "dilation", 1))
	dilW := webgpuAttrInt(step.Attributes, "dilation_w", webgpuAttrInt(step.Attributes, "dilation", 1))
	outPadH := webgpuAttrInt(step.Attributes, "output_padding_h", webgpuAttrInt(step.Attributes, "output_padding", 0))
	outPadW := webgpuAttrInt(step.Attributes, "output_padding_w", webgpuAttrInt(step.Attributes, "output_padding", 0))
	if strideH <= 0 || strideW <= 0 || dilH <= 0 || dilW <= 0 {
		return nil, nil, false, nil
	}
	outC := outPerGroup * groups
	outH := (inH-1)*strideH - 2*padH + dilH*(kH-1) + outPadH + 1
	outW := (inW-1)*strideW - 2*padW + dilW*(kW-1) + outPadW + 1
	if outH <= 0 || outW <= 0 {
		return nil, nil, false, nil
	}
	hasBias := len(inputs) == 3 && inputs[2] != nil
	biasData := []float32{0}
	if hasBias {
		if len(inputs[2].F32) < outC {
			return nil, nil, false, nil
		}
		biasData = inputs[2].F32
	}
	cfg := uint32Bytes(
		uint32(n), uint32(inC), uint32(inH), uint32(inW),
		uint32(outC), uint32(outH), uint32(outW),
		uint32(outPerGroup), uint32(inC/groups), uint32(kH), uint32(kW), uint32(groups),
		uint32(strideH), uint32(strideW), uint32(int32(padH)), uint32(int32(padW)),
		uint32(dilH), uint32(dilW), boolU32(hasBias),
	)
	kernel, _ := BuiltinForStep(mantaartifact.StepConv2DTrans)
	outputBytes, meta, err := d.dispatch(ctx, kernel, []js.Value{
		d.floatStorageBuffer(input.F32),
		d.floatStorageBuffer(weight.F32),
		d.floatStorageBuffer(biasData),
		d.outputBuffer(outC * outH * outW * n * 4),
		d.uniformBuffer(cfg),
	}, 3, outC*outH*outW*n*4, ceilDiv(outW, 8), ceilDiv(outH, 8), n*outC)
	if err != nil {
		return nil, nil, true, err
	}
	out := tensorForWebGPUDType(input.DType, []int{n, outC, outH, outW}, bytesToFloat32s(outputBytes, n*outC*outH*outW))
	return []*backend.Tensor{out}, meta, true, nil
}

func (d *deviceRuntime) runGDN(ctx context.Context, step mantaartifact.Step, inputs []*backend.Tensor, inverse bool) ([]*backend.Tensor, map[string]any, bool, error) {
	if len(inputs) < 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return nil, nil, false, nil
	}
	input, beta, gamma := inputs[0], inputs[1], inputs[2]
	if len(input.Shape) != 4 || len(input.F32) != input.Elements() {
		return nil, nil, false, nil
	}
	n, channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	if len(beta.F32) < channels || len(gamma.F32) < channels*channels {
		return nil, nil, false, nil
	}
	cfg := uint32Bytes(uint32(input.Elements()), uint32(n), uint32(channels), uint32(height), uint32(width))
	kind := mantaartifact.StepGDN
	if inverse {
		kind = mantaartifact.StepIGDN
	}
	kernel, _ := BuiltinForStep(kind)
	outputBytes, meta, err := d.dispatch(ctx, kernel, []js.Value{
		d.floatStorageBuffer(input.F32),
		d.floatStorageBuffer(beta.F32),
		d.floatStorageBuffer(gamma.F32),
		d.outputBuffer(input.Elements() * 4),
		d.uniformBuffer(cfg),
	}, 3, input.Elements()*4, ceilDiv(input.Elements(), 64), 1, 1)
	if err != nil {
		return nil, nil, true, err
	}
	out := tensorForWebGPUDType(input.DType, input.Shape, bytesToFloat32s(outputBytes, input.Elements()))
	return []*backend.Tensor{out}, meta, true, nil
}

func (d *deviceRuntime) runTurboQuantDecode(ctx context.Context, step mantaartifact.Step, inputs []*backend.Tensor) ([]*backend.Tensor, map[string]any, bool, error) {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return nil, nil, false, nil
	}
	coords, norms := inputs[0], inputs[1]
	if len(coords.Shape) != 4 || len(norms.Shape) != 3 || len(coords.F32) != coords.Elements() || len(norms.F32) != norms.Elements() {
		return nil, nil, false, nil
	}
	n, channels, height, width := coords.Shape[0], coords.Shape[1], coords.Shape[2], coords.Shape[3]
	if channels > 256 || norms.Shape[0] != n || norms.Shape[1] != height || norms.Shape[2] != width {
		return nil, nil, false, nil
	}
	bits := webgpuAttrInt(step.Attributes, "bits", webgpuBitsForQTensor(coords))
	if bits != 2 && bits != 4 && bits != 8 {
		return nil, nil, false, nil
	}
	seed := webgpuAttrInt64(step.Attributes, "seed", 0x4d697261)
	spec := turboquant.NewHadamardWithSeed(channels, bits, seed).Spec()
	if spec.RotationKind != "hadamard" || len(spec.Perm) != channels || len(spec.Signs1) != channels || len(spec.Signs2) != channels {
		return nil, nil, false, nil
	}
	cfg := uint32Bytes(
		uint32(n*height*width), uint32(n), uint32(channels), uint32(height), uint32(width),
		uint32(bits), uint32(1<<bits),
	)
	kernel, _ := BuiltinForStep(mantaartifact.StepTurboQDecode)
	outputBytes, meta, err := d.dispatch(ctx, kernel, []js.Value{
		d.floatStorageBuffer(coords.F32),
		d.floatStorageBuffer(norms.F32),
		d.floatStorageBuffer(spec.Centroids),
		d.uint32StorageBuffer(intsToUint32s(spec.Perm)),
		d.floatStorageBuffer(spec.Signs1),
		d.floatStorageBuffer(spec.Signs2),
		d.outputBuffer(coords.Elements() * 4),
		d.uniformBuffer(cfg),
	}, 6, coords.Elements()*4, ceilDiv(n*height*width, 64), 1, 1)
	if err != nil {
		return nil, nil, true, err
	}
	out := backend.NewTensorF16(coords.Shape, bytesToFloat32s(outputBytes, coords.Elements()))
	return []*backend.Tensor{out}, meta, true, nil
}

func (d *deviceRuntime) pipeline(ctx context.Context, kernel BuiltinKernel) (js.Value, error) {
	if pipeline, ok := d.pipelines[kernel.StepKind]; ok {
		return pipeline, nil
	}
	module := d.device.Call("createShaderModule", map[string]any{"code": kernel.Source})
	promise := d.device.Call("createComputePipelineAsync", map[string]any{
		"layout": "auto",
		"compute": map[string]any{
			"module":     module,
			"entryPoint": kernel.Entry,
		},
	})
	pipeline, err := awaitPromise(ctx, promise)
	if err != nil {
		return js.Value{}, fmt.Errorf("create WebGPU pipeline %s: %w", kernel.Entry, err)
	}
	d.pipelines[kernel.StepKind] = pipeline
	return pipeline, nil
}

func (d *deviceRuntime) dispatch(ctx context.Context, kernel BuiltinKernel, buffers []js.Value, outputBinding, outputBytes, workgroupsX, workgroupsY, workgroupsZ int) ([]byte, map[string]any, error) {
	pipeline, err := d.pipeline(ctx, kernel)
	if err != nil {
		return nil, nil, err
	}
	entries := make([]any, len(buffers))
	for binding, buffer := range buffers {
		entries[binding] = map[string]any{
			"binding": binding,
			"resource": map[string]any{
				"buffer": buffer,
			},
		}
	}
	bindGroup := d.device.Call("createBindGroup", map[string]any{
		"layout":  pipeline.Call("getBindGroupLayout", 0),
		"entries": entries,
	})
	encoder := d.device.Call("createCommandEncoder")
	pass := encoder.Call("beginComputePass")
	pass.Call("setPipeline", pipeline)
	pass.Call("setBindGroup", 0, bindGroup)
	pass.Call("dispatchWorkgroups", workgroupsX, workgroupsY, workgroupsZ)
	pass.Call("end")

	readback := d.device.Call("createBuffer", map[string]any{
		"size":  align4(outputBytes),
		"usage": gpuBufferUsage("COPY_DST", "MAP_READ"),
	})
	encoder.Call("copyBufferToBuffer", buffers[outputBinding], 0, readback, 0, outputBytes)
	command := encoder.Call("finish")
	d.queue.Call("submit", []any{command})
	if _, err := awaitPromise(ctx, readback.Call("mapAsync", gpuMapMode("READ"))); err != nil {
		return nil, nil, fmt.Errorf("map WebGPU readback for %s: %w", kernel.Entry, err)
	}
	mapped := readback.Call("getMappedRange")
	u8 := js.Global().Get("Uint8Array").New(mapped)
	out := make([]byte, align4(outputBytes))
	js.CopyBytesToGo(out, u8)
	readback.Call("unmap")
	destroyBuffers(append(buffers, readback))

	meta := kernel.Metadata()
	meta["device_execution"] = true
	meta["execution_mode"] = "webgpu_device"
	meta["fallback_reason"] = ""
	meta["dispatch_workgroups_x"] = workgroupsX
	meta["dispatch_workgroups_y"] = workgroupsY
	meta["dispatch_workgroups_z"] = workgroupsZ
	return out[:outputBytes], meta, nil
}

func (d *deviceRuntime) floatStorageBuffer(values []float32) js.Value {
	return d.bytesBuffer(float32Bytes(values), gpuBufferUsage("COPY_DST", "STORAGE"))
}

func (d *deviceRuntime) uint32StorageBuffer(values []uint32) js.Value {
	return d.bytesBuffer(uint32Bytes(values...), gpuBufferUsage("COPY_DST", "STORAGE"))
}

func (d *deviceRuntime) uniformBuffer(data []byte) js.Value {
	return d.bytesBuffer(data, gpuBufferUsage("COPY_DST", "UNIFORM"))
}

func (d *deviceRuntime) outputBuffer(size int) js.Value {
	return d.device.Call("createBuffer", map[string]any{
		"size":  align4(size),
		"usage": gpuBufferUsage("STORAGE", "COPY_SRC"),
	})
}

func (d *deviceRuntime) bytesBuffer(data []byte, usage int) js.Value {
	buf := d.device.Call("createBuffer", map[string]any{
		"size":  align4(maxInt(len(data), 4)),
		"usage": usage,
	})
	if len(data) > 0 {
		u8 := js.Global().Get("Uint8Array").New(len(data))
		js.CopyBytesToJS(u8, data)
		d.queue.Call("writeBuffer", buf, 0, u8)
	}
	return buf
}

func awaitPromise(ctx context.Context, promise js.Value) (js.Value, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	type result struct {
		value js.Value
		err   error
	}
	done := make(chan result, 1)
	var onResolve, onReject js.Func
	onResolve = js.FuncOf(func(this js.Value, args []js.Value) any {
		value := js.Undefined()
		if len(args) > 0 {
			value = args[0]
		}
		done <- result{value: value}
		return nil
	})
	onReject = js.FuncOf(func(this js.Value, args []js.Value) any {
		message := "promise rejected"
		if len(args) > 0 {
			message = jsErrorString(args[0])
		}
		done <- result{err: fmt.Errorf("%s", message)}
		return nil
	})
	promise.Call("then", onResolve).Call("catch", onReject)
	select {
	case result := <-done:
		onResolve.Release()
		onReject.Release()
		return result.value, result.err
	case <-ctx.Done():
		onResolve.Release()
		onReject.Release()
		return js.Value{}, ctx.Err()
	}
}

func jsErrorString(value js.Value) string {
	if !value.Truthy() {
		return "promise rejected"
	}
	if msg := value.Get("message"); msg.Truthy() {
		return msg.String()
	}
	return value.String()
}

func gpuBufferUsage(names ...string) int {
	usage := js.Global().Get("GPUBufferUsage")
	out := 0
	for _, name := range names {
		out |= usage.Get(name).Int()
	}
	return out
}

func gpuMapMode(name string) int {
	return js.Global().Get("GPUMapMode").Get(name).Int()
}

func destroyBuffers(buffers []js.Value) {
	for _, buffer := range buffers {
		if buffer.Truthy() && buffer.Get("destroy").Truthy() {
			buffer.Call("destroy")
		}
	}
}

func float32Bytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
	}
	return out
}

func uint32Bytes(values ...uint32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], value)
	}
	return out
}

func bytesToFloat32s(data []byte, count int) []float32 {
	out := make([]float32, count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

func intsToUint32s(values []int) []uint32 {
	out := make([]uint32, len(values))
	for i, value := range values {
		out[i] = uint32(value)
	}
	return out
}

func tensorForWebGPUDType(dtype string, shape []int, data []float32) *backend.Tensor {
	switch dtype {
	case "f16":
		return backend.NewTensorF16(shape, data)
	case "q2":
		return backend.NewTensorQ2(shape, data)
	case "q4":
		return backend.NewTensorQ4(shape, data)
	case "q8":
		return backend.NewTensorQ8(shape, data)
	case "q_norm":
		return backend.NewTensorQNorm(shape, data)
	default:
		return backend.NewTensorF32(shape, data)
	}
}

func webgpuBitsForQTensor(t *backend.Tensor) int {
	if t == nil {
		return 0
	}
	switch t.DType {
	case "q2":
		return 2
	case "q4":
		return 4
	case "q8", "q_norm":
		return 8
	default:
		return 0
	}
}

func webgpuAttrInt(attrs map[string]string, key string, fallback int) int {
	if attrs == nil || attrs[key] == "" {
		return fallback
	}
	n, err := strconv.Atoi(attrs[key])
	if err != nil {
		return fallback
	}
	return n
}

func webgpuAttrInt64(attrs map[string]string, key string, fallback int64) int64 {
	if attrs == nil || attrs[key] == "" {
		return fallback
	}
	n, err := strconv.ParseInt(attrs[key], 10, 64)
	if err != nil {
		return fallback
	}
	return n
}

func boolU32(ok bool) uint32 {
	if ok {
		return 1
	}
	return 0
}

func align4(n int) int {
	if n <= 0 {
		return 4
	}
	return (n + 3) &^ 3
}

func ceilDiv(n, d int) int {
	return (n + d - 1) / d
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

package webgpu

import (
	"crypto/sha256"
	"encoding/hex"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

// BuiltinKernel describes a WebGPU WGSL builtin used by Mirage decode steps.
type BuiltinKernel struct {
	StepKind      mantaartifact.StepKind
	Entry         string
	Source        string
	WorkgroupSize [3]int
	Grid          string
}

// MirageDecodeKernels returns the v1 decode-side WGSL kernel surface.
func MirageDecodeKernels() []BuiltinKernel {
	return []BuiltinKernel{
		{StepKind: mantaartifact.StepConv2D, Entry: "manta_webgpu_conv2d_nchw", Source: conv2DWGSL, WorkgroupSize: [3]int{8, 8, 1}, Grid: "2d"},
		{StepKind: mantaartifact.StepConv2DTrans, Entry: "manta_webgpu_conv2d_transpose_nchw", Source: conv2DTransposeWGSL, WorkgroupSize: [3]int{8, 8, 1}, Grid: "2d"},
		{StepKind: mantaartifact.StepGDN, Entry: "manta_webgpu_gdn_nchw", Source: gdnWGSL, WorkgroupSize: [3]int{64, 1, 1}, Grid: "1d"},
		{StepKind: mantaartifact.StepIGDN, Entry: "manta_webgpu_igdn_nchw", Source: igdnWGSL, WorkgroupSize: [3]int{64, 1, 1}, Grid: "1d"},
		{StepKind: mantaartifact.StepTurboQDecode, Entry: "manta_webgpu_turboquant_decode_nchw", Source: turboQuantDecodeWGSL, WorkgroupSize: [3]int{64, 1, 1}, Grid: "1d"},
	}
}

// BuiltinForStep returns the WebGPU builtin kernel for a high-level Manta step.
func BuiltinForStep(kind mantaartifact.StepKind) (BuiltinKernel, bool) {
	for _, kernel := range MirageDecodeKernels() {
		if kernel.StepKind == kind {
			return kernel, true
		}
	}
	return BuiltinKernel{}, false
}

func (k BuiltinKernel) SourceHash() string {
	sum := sha256.Sum256([]byte(k.Source))
	return hex.EncodeToString(sum[:])
}

func (k BuiltinKernel) Metadata() map[string]any {
	return map[string]any{
		"dispatch_mode":                 "backend_native",
		"dispatch_backend":              string(mantaartifact.BackendWebGPU),
		"device_execution":              false,
		"execution_mode":                "wgsl_host_reference",
		"fallback_reason":               "webgpu_device_runtime_not_bound",
		"launch_api":                    "GPUComputePassEncoder.dispatchWorkgroups",
		"launch_entry":                  k.Entry,
		"launch_grid":                   k.Grid,
		"launch_workgroup_size":         k.WorkgroupSize[0],
		"launch_workgroup_size_x":       k.WorkgroupSize[0],
		"launch_workgroup_size_y":       k.WorkgroupSize[1],
		"launch_workgroup_size_z":       k.WorkgroupSize[2],
		"launch_workgroup_memory_bytes": 0,
		"wgsl_source_hash":              k.SourceHash(),
	}
}

const conv2DWGSL = `/* wgsl */
struct Conv2DConfig {
  batches: u32,
  in_channels: u32,
  in_height: u32,
  in_width: u32,
  out_channels: u32,
  out_height: u32,
  out_width: u32,
  in_per_group: u32,
  out_per_group: u32,
  kernel_h: u32,
  kernel_w: u32,
  groups: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: i32,
  pad_w: i32,
  dilation_h: u32,
  dilation_w: u32,
  has_bias: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Conv2DConfig;

fn nchw_offset(n: u32, c: u32, h: u32, w: u32, channels: u32, height: u32, width: u32) -> u32 {
  return ((n * channels + c) * height + h) * width + w;
}

@compute @workgroup_size(8, 8, 1)
fn manta_webgpu_conv2d_nchw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x;
  let oy = gid.y;
  let plane = gid.z;
  if (ox >= cfg.out_width || oy >= cfg.out_height) {
    return;
  }
  let n = plane / cfg.out_channels;
  let oc = plane - n * cfg.out_channels;
  if (n >= cfg.batches) {
    return;
  }
  let group = oc / cfg.out_per_group;
  let ocg = oc - group * cfg.out_per_group;
  var sum = 0.0;
  if (cfg.has_bias != 0u) {
    sum = bias[oc];
  }
  for (var icg = 0u; icg < cfg.in_per_group; icg = icg + 1u) {
    let ic = group * cfg.in_per_group + icg;
    for (var ky = 0u; ky < cfg.kernel_h; ky = ky + 1u) {
      let iy = i32(oy * cfg.stride_h + ky * cfg.dilation_h) - cfg.pad_h;
      if (iy < 0 || iy >= i32(cfg.in_height)) {
        continue;
      }
      for (var kx = 0u; kx < cfg.kernel_w; kx = kx + 1u) {
        let ix = i32(ox * cfg.stride_w + kx * cfg.dilation_w) - cfg.pad_w;
        if (ix < 0 || ix >= i32(cfg.in_width)) {
          continue;
        }
        let in_idx = nchw_offset(n, ic, u32(iy), u32(ix), cfg.in_channels, cfg.in_height, cfg.in_width);
        let w_idx = ((oc * cfg.in_per_group + icg) * cfg.kernel_h + ky) * cfg.kernel_w + kx;
        sum = sum + input[in_idx] * weight[w_idx];
      }
    }
  }
  let out_idx = nchw_offset(n, oc, oy, ox, cfg.out_channels, cfg.out_height, cfg.out_width);
  output[out_idx] = sum;
}
`

const conv2DTransposeWGSL = `/* wgsl */
struct Conv2DTransposeConfig {
  batches: u32,
  in_channels: u32,
  in_height: u32,
  in_width: u32,
  out_channels: u32,
  out_height: u32,
  out_width: u32,
  out_per_group: u32,
  in_per_group: u32,
  kernel_h: u32,
  kernel_w: u32,
  groups: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: i32,
  pad_w: i32,
  dilation_h: u32,
  dilation_w: u32,
  has_bias: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Conv2DTransposeConfig;

fn nchw_offset(n: u32, c: u32, h: u32, w: u32, channels: u32, height: u32, width: u32) -> u32 {
  return ((n * channels + c) * height + h) * width + w;
}

@compute @workgroup_size(8, 8, 1)
fn manta_webgpu_conv2d_transpose_nchw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x;
  let oy = gid.y;
  let plane = gid.z;
  if (ox >= cfg.out_width || oy >= cfg.out_height) {
    return;
  }
  let n = plane / cfg.out_channels;
  let oc = plane - n * cfg.out_channels;
  if (n >= cfg.batches) {
    return;
  }
  let group = oc / cfg.out_per_group;
  let ocg = oc - group * cfg.out_per_group;
  var sum = 0.0;
  if (cfg.has_bias != 0u) {
    sum = bias[oc];
  }
  for (var icg = 0u; icg < cfg.in_per_group; icg = icg + 1u) {
    let ic = group * cfg.in_per_group + icg;
    for (var ky = 0u; ky < cfg.kernel_h; ky = ky + 1u) {
      let raw_y = i32(oy) + cfg.pad_h - i32(ky * cfg.dilation_h);
      if (raw_y < 0 || raw_y % i32(cfg.stride_h) != 0) {
        continue;
      }
      let iy = raw_y / i32(cfg.stride_h);
      if (iy < 0 || iy >= i32(cfg.in_height)) {
        continue;
      }
      for (var kx = 0u; kx < cfg.kernel_w; kx = kx + 1u) {
        let raw_x = i32(ox) + cfg.pad_w - i32(kx * cfg.dilation_w);
        if (raw_x < 0 || raw_x % i32(cfg.stride_w) != 0) {
          continue;
        }
        let ix = raw_x / i32(cfg.stride_w);
        if (ix < 0 || ix >= i32(cfg.in_width)) {
          continue;
        }
        let in_idx = nchw_offset(n, ic, u32(iy), u32(ix), cfg.in_channels, cfg.in_height, cfg.in_width);
        let w_idx = ((ic * cfg.out_per_group + ocg) * cfg.kernel_h + ky) * cfg.kernel_w + kx;
        sum = sum + input[in_idx] * weight[w_idx];
      }
    }
  }
  let out_idx = nchw_offset(n, oc, oy, ox, cfg.out_channels, cfg.out_height, cfg.out_width);
  output[out_idx] = sum;
}
`

const gdnWGSL = `/* wgsl */
struct GDNConfig {
  elements: u32,
  batches: u32,
  channels: u32,
  height: u32,
  width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> beta: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: GDNConfig;

@compute @workgroup_size(64, 1, 1)
fn manta_webgpu_gdn_nchw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= cfg.elements) {
    return;
  }
  let w = idx % cfg.width;
  let h0 = (idx / cfg.width) % cfg.height;
  let c = (idx / (cfg.width * cfg.height)) % cfg.channels;
  let n = idx / (cfg.channels * cfg.height * cfg.width);
  var sum = max(beta[c], 0.000001);
  for (var j = 0u; j < cfg.channels; j = j + 1u) {
    let jidx = ((n * cfg.channels + j) * cfg.height + h0) * cfg.width + w;
    let v = input[jidx];
    sum = sum + gamma[c * cfg.channels + j] * v * v;
  }
  output[idx] = input[idx] / sqrt(max(sum, 0.000000000001));
}
`

const igdnWGSL = `/* wgsl */
struct IGDNConfig {
  elements: u32,
  batches: u32,
  channels: u32,
  height: u32,
  width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> beta: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> cfg: IGDNConfig;

@compute @workgroup_size(64, 1, 1)
fn manta_webgpu_igdn_nchw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= cfg.elements) {
    return;
  }
  let w = idx % cfg.width;
  let h0 = (idx / cfg.width) % cfg.height;
  let c = (idx / (cfg.width * cfg.height)) % cfg.channels;
  let n = idx / (cfg.channels * cfg.height * cfg.width);
  var sum = max(beta[c], 0.000001);
  for (var j = 0u; j < cfg.channels; j = j + 1u) {
    let jidx = ((n * cfg.channels + j) * cfg.height + h0) * cfg.width + w;
    let v = input[jidx];
    sum = sum + gamma[c * cfg.channels + j] * v * v;
  }
  output[idx] = input[idx] * sqrt(max(sum, 0.000000000001));
}
`

const turboQuantDecodeWGSL = `/* wgsl */
struct TurboQDecodeConfig {
  vectors: u32,
  batches: u32,
  channels: u32,
  height: u32,
  width: u32,
  bits: u32,
  levels: u32,
}

@group(0) @binding(0) var<storage, read> coords: array<f32>;
@group(0) @binding(1) var<storage, read> norms: array<f32>;
@group(0) @binding(2) var<storage, read> centroids: array<f32>;
@group(0) @binding(3) var<storage, read> perm: array<u32>;
@group(0) @binding(4) var<storage, read> signs1: array<f32>;
@group(0) @binding(5) var<storage, read> signs2: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<uniform> cfg: TurboQDecodeConfig;

fn nchw_offset(n: u32, c: u32, h: u32, w: u32, channels: u32, height: u32, width: u32) -> u32 {
  return ((n * channels + c) * height + h) * width + w;
}

fn qnorm_decode(raw: f32) -> f32 {
  let q = clamp(round(raw), 0.0, 255.0);
  let t = q / 255.0;
  return exp(-16.0 + t * 32.0);
}

fn fwht_block(values: ptr<function, array<f32, 256>>, start: u32, size: u32) {
  var stride = 1u;
  loop {
    if (stride >= size) {
      break;
    }
    var i = 0u;
    loop {
      if (i >= size) {
        break;
      }
      for (var j = 0u; j < stride; j = j + 1u) {
        let a_idx = start + i + j;
        let b_idx = a_idx + stride;
        let a = (*values)[a_idx];
        let b = (*values)[b_idx];
        (*values)[a_idx] = a + b;
        (*values)[b_idx] = a - b;
      }
      i = i + stride * 2u;
    }
    stride = stride * 2u;
  }
  let scale = inverseSqrt(f32(size));
  for (var k = 0u; k < size; k = k + 1u) {
    (*values)[start + k] = (*values)[start + k] * scale;
  }
}

@compute @workgroup_size(64, 1, 1)
fn manta_webgpu_turboquant_decode_nchw(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vector = gid.x;
  if (vector >= cfg.vectors || cfg.channels > 256u) {
    return;
  }
  let x = vector % cfg.width;
  let y = (vector / cfg.width) % cfg.height;
  let n = vector / (cfg.height * cfg.width);
  var work: array<f32, 256>;
  for (var c = 0u; c < cfg.channels; c = c + 1u) {
    let idx = nchw_offset(n, c, y, x, cfg.channels, cfg.height, cfg.width);
    let q = u32(clamp(round(coords[idx]), 0.0, f32(cfg.levels - 1u)));
    work[c] = centroids[q];
  }
  for (var c = 0u; c < cfg.channels; c = c + 1u) {
    work[c] = work[c] * signs2[c];
  }
  var block_start = 0u;
  loop {
    if (block_start >= cfg.channels) {
      break;
    }
    var remaining = cfg.channels - block_start;
    var size = 1u;
    loop {
      if (size * 2u > remaining) {
        break;
      }
      size = size * 2u;
    }
    fwht_block(&work, block_start, size);
    block_start = block_start + size;
  }
  var shuffled: array<f32, 256>;
  for (var c = 0u; c < cfg.channels; c = c + 1u) {
    shuffled[perm[c]] = work[c] * signs1[c];
  }
  let norm = qnorm_decode(norms[(n * cfg.height + y) * cfg.width + x]);
  for (var c = 0u; c < cfg.channels; c = c + 1u) {
    let idx = nchw_offset(n, c, y, x, cfg.channels, cfg.height, cfg.width);
    output[idx] = shuffled[c] * norm;
  }
}
`

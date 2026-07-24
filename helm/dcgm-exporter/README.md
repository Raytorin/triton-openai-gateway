# Standalone DCGM Exporter Chart

[Project home](../../README.md) / Standalone GPU metrics

**Language:** English | [Русский](README.ru.md)

This wrapper chart deploys only NVIDIA DCGM Exporter. It is useful when Triton is
already running and no cluster-wide exporter is available. It does not deploy,
restart, or modify Triton.

Do not install it when NVIDIA GPU Operator already provides a healthy
`dcgm-exporter`; use that existing Service instead.

## Prerequisites

- NVIDIA driver and Container Toolkit on target nodes.
- NVIDIA device plugin or GPU Operator.
- Helm 3.

The chart does not install host drivers.

## Install On The Triton Node

Find the node:

```bash
kubectl -n TRITON_NAMESPACE get pod TRITON_POD -o wide
```

Pin the exporter to that node:

```bash
helm upgrade --install triton-gpu-metrics ./helm/dcgm-exporter \
  --namespace gpu-metrics \
  --create-namespace \
  --set-string 'dcgm-exporter.nodeSelector.kubernetes\.io/hostname=GPU_NODE'
```

If NVIDIA is not the cluster's default runtime, add:

```text
--set dcgm-exporter.runtimeClassName=nvidia
```

Omit the node selector to run a DaemonSet on all eligible GPU nodes. Prometheus
should scrape every exporter pod separately. A shared ClusterIP may return a
random node and is therefore unsuitable when a test must follow one Triton pod.

## Verify

```bash
kubectl -n gpu-metrics get pods -o wide
kubectl -n gpu-metrics get service
kubectl -n gpu-metrics port-forward \
  service/triton-gpu-metrics-dcgm-exporter 9400:9400

curl -s http://127.0.0.1:9400/metrics \
  | grep DCGM_FI_DEV_GPU_UTIL
```

Common series include:

- `DCGM_FI_DEV_GPU_UTIL`
- `DCGM_FI_DEV_FB_USED`
- `DCGM_FI_DEV_FB_FREE`

Exporter labels identify physical GPUs, MIG instances, Kubernetes pods, and
containers when Kubernetes metric enrichment is available.

## Use An Existing Exporter

Discover a GPU Operator deployment:

```bash
kubectl get daemonset -A | grep -Ei 'dcgm|gpu-operator'
kubectl get service -A | grep -i dcgm
```

Use its Service URL from in-cluster monitoring or port-forward the exporter pod
on the same node as Triton for a local test.

DCGM does not know which vLLM model generated work. Attribute device metrics to
a model only when that GPU or MIG instance is dedicated to the model during the
measurement window.

# Triton OpenAI Gateway Helm Chart

[Project home](../../README.md) / Helm deployment

**Language:** English | [Русский](README.ru.md)

This chart deploys NVIDIA Triton and the OpenAI-compatible gateway in one pod.
It supports S3 model repositories, GPU or MIG resources, model and media
storage, admission controls, Prometheus metrics, DCGM, and Triton OTLP tracing.

## Services

| Port | Name | Purpose |
| --- | --- | --- |
| `8000` | `http` | Raw Triton HTTP and model repository API |
| `8001` | `grpc` | Raw Triton gRPC |
| `8002` | `metrics` | Triton, vLLM, and optional embedded GPU metrics |
| `8080` | `gateway` | OpenAI-compatible API and gateway metrics |

## Prerequisites

- Kubernetes with NVIDIA drivers and device plugin or GPU Operator.
- Helm 3.
- A built Triton OpenAI Gateway image in a registry visible to the cluster.
- A Triton model repository and its credentials.

## Install

Create an environment-specific values file:

```yaml
image:
  repository: registry.example.com/ml/triton-openai-gateway
  tag: "26.06"

triton:
  modelRepository: s3://object-store.example.com/models/triton
  modelControlMode: explicit
  loadModels:
    - chat-model
    - embedding-model
  gpuMetrics:
    mode: external

s3:
  existingSecret: triton-s3-credentials

resources:
  requests:
    cpu: "8"
    memory: 64Gi
    nvidia.com/gpu: "1"
  limits:
    cpu: "8"
    memory: 64Gi
    nvidia.com/gpu: "1"
```

Install or upgrade:

```bash
helm upgrade --install triton-openai-gateway ./helm/triton-gateway \
  --namespace inference \
  --create-namespace \
  --values values.production.yaml
```

[`values.production.example.yaml`](values.production.example.yaml) provides a
complete starting point without real registry, credential, or model names.

The S3 Secret must contain the environment variables required by Triton's S3
repository agent. Keep credentials outside `values.yaml`.

## Rollout And Health Probes

The default rollout strategy is `Recreate`. A rolling update can deadlock when
the current Triton pod owns every GPU requested by its replacement. Set
`updateStrategy.type: RollingUpdate` only when the cluster has enough spare GPU
capacity to run both pods.

Startup, liveness, and readiness probe the gateway `/health` endpoint by
default. This keeps the UI and repository API reachable while an individual
model loads or unloads. To remove the pod from Service endpoints whenever
Triton is not ready, use:

```yaml
readinessProbe:
  path: /ready
  port: gateway
```

The models listed under `triton.loadModels` become repeated
`--load-model=<name>` arguments. Leave the list empty when model lifecycle is
managed only through Triton's repository API.

## GPU And MIG

For a MIG profile, request the resource advertised by your device plugin:

```yaml
resources:
  requests:
    cpu: "10"
    memory: 80Gi
    nvidia.com/mig-3g.40gb: "1"
  limits:
    cpu: "10"
    memory: 80Gi
    nvidia.com/mig-3g.40gb: "1"
```

Add node selectors, tolerations, affinity, or topology spread constraints under
their corresponding top-level values.

## Model Storage

The default model workspace is an `emptyDir` mounted at `/models` and `/tmp`.
Attach an existing PVC when downloaded model data must survive pod replacement:

```yaml
modelStorage:
  persistence:
    enabled: true
    existingClaim: triton-model-workspace
```

Or let the chart create one:

```yaml
modelStorage:
  persistence:
    enabled: true
    size: 200Gi
    storageClassName: fast-rwo
    retain: true
```

Use `ReadWriteMany` storage or one PVC per pod before increasing
`replicaCount`.

By default the watcher follows Triton's temporary directory and creates
`models-active` below it. Set `gateway.watcherModelDir` only when temporary
`folder*` checkouts are materialized elsewhere; set
`gateway.modelsActiveDir` only when active links need a separate path.
`gateway.tmpRoot` remains a compatibility alias.

## Temporary Media Storage

Large videos and PDFs may need more temporary space than the container layer.
Create or attach a dedicated media PVC:

```yaml
mediaPersistence:
  enabled: true
  size: 100Gi
  storageClassName: fast-rwo
  mountPath: /var/lib/triton/media
  useAsTmpDir: true
  retain: true
```

Temporary files are deleted after request processing. This PVC is workspace,
not a media archive.

## Shared Memory

The chart mounts a memory-backed `emptyDir` at `/dev/shm`. This avoids the small
container default that can break Triton Python backend IPC and large
multimodal requests:

```yaml
sharedMemory:
  enabled: true
  sizeLimit: 8Gi
```

Shared-memory usage counts against pod memory. Increase both the pod memory
limit and `sharedMemory.sizeLimit` for highly concurrent large-media workloads.

## Admission And Media Limits

Start conservatively and tune under representative load:

```yaml
gateway:
  admission:
    maxInflight: "256"
    maxQueue: "512"
    chatMaxInflight: "64"
    chatMaxQueue: "256"
    mediaMaxInflight: "4"
    mediaMaxQueue: "16"
  nativeBackend:
    maxMediaBytes: "268435456"
    maxRequestBytes: "805306368"
    maxTotalVideoFrames: "256"
    maxPdfPages: "64"
    preprocessConcurrency: "2"
  vllmMultimodal:
    pdfChunkPages: 2
    videoFps: 1.0
    videoMaxFrames: 32
    videoChunkFrames: 4
```

These limits protect host memory and CPU before requests reach the GPU.

## Logs And Debugging

```yaml
gateway:
  observability:
    logFormat: json
    logLevel: INFO
    debugEnabled: false
    debugLogPayloads: false
    uvicornAccessLog: false
```

Set `debug: true` in one chat request for safe request-chain metadata. Enabling
`debugLogPayloads` may expose prompts, tool results, or user data and is not
recommended in production.

## GPU Metrics

### Embedded DCGM

```yaml
triton:
  gpuMetrics:
    mode: builtin
```

Triton publishes `nv_gpu_*` on port `8002`. Embedded DCGM must run as root; the
chart uses a root security context with privilege escalation disabled and all
Linux capabilities dropped.

### External DCGM Exporter

```yaml
triton:
  gpuMetrics:
    mode: external
```

Use this mode when cluster policy requires non-root Triton. If GPU Operator
already provides DCGM Exporter, scrape its existing Service. Otherwise enable
the vendored dependency:

```yaml
dcgm-exporter:
  enabled: true
  serviceMonitor:
    enabled: false
```

The chart dependency archive is committed, so installation does not require
`helm dependency update`.

## OpenTelemetry

```yaml
gateway:
  observability:
    generationTelemetry: true
    otel:
      enabled: true
      endpoint: http://otel-collector.observability.svc:4318/v1/traces
      sampleRatio: "0.05"
      serviceName: triton-openai-gateway

triton:
  tracing:
    enabled: true
    endpoint: http://otel-collector.observability.svc:4318/v1/traces
    level: TIMESTAMPS
    rate: 0
    count: -1
    serviceName: triton-inference-server
```

The gateway creates the root span and propagates W3C `traceparent` to Triton.
With `rate: 0`, Triton only traces requests selected by the gateway sampler.
Prompts, media, generated text, tool results, and reasoning content are not
exported.

## Validate

```bash
helm lint ./helm/triton-gateway
helm template test ./helm/triton-gateway >/dev/null
```

After installation:

```bash
kubectl -n inference get pods,service,pvc
kubectl -n inference port-forward service/triton-openai-gateway 8080:8080
curl -fsS http://127.0.0.1:8080/ready
```

See [Configuration](../../docs/configuration.md) and
[Operations](../../docs/operations.md) for model files and runtime guidance.

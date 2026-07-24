# Отдельный Chart DCGM Exporter

[Главная проекта](../../README.ru.md) / Отдельные GPU metrics

**Язык:** [English](README.md) | Русский

Этот wrapper chart устанавливает только NVIDIA DCGM Exporter. Он нужен, если
Triton уже работает, а cluster-wide exporter отсутствует. Chart не разворачивает,
не перезапускает и не изменяет Triton.

Не устанавливайте его, если NVIDIA GPU Operator уже предоставляет исправный
`dcgm-exporter`; используйте существующий Service.

## Требования

- NVIDIA driver и Container Toolkit на целевых nodes.
- NVIDIA device plugin либо GPU Operator.
- Helm 3.

Chart не устанавливает driver на host.

## Установка на node Triton

Найдите node:

```bash
kubectl -n TRITON_NAMESPACE get pod TRITON_POD -o wide
```

Закрепите exporter за этим node:

```bash
helm upgrade --install triton-gpu-metrics ./helm/dcgm-exporter \
  --namespace gpu-metrics \
  --create-namespace \
  --set-string 'dcgm-exporter.nodeSelector.kubernetes\.io/hostname=GPU_NODE'
```

Если NVIDIA не является default runtime cluster-а, добавьте:

```text
--set dcgm-exporter.runtimeClassName=nvidia
```

Не задавайте node selector, если DaemonSet должен работать на всех доступных GPU
nodes. Prometheus должен опрашивать каждый exporter pod отдельно. Общий ClusterIP
может направить запрос на случайный node, поэтому он не подходит, когда тест
должен следовать за конкретным Triton pod.

## Проверка

```bash
kubectl -n gpu-metrics get pods -o wide
kubectl -n gpu-metrics get service
kubectl -n gpu-metrics port-forward \
  service/triton-gpu-metrics-dcgm-exporter 9400:9400

curl -s http://127.0.0.1:9400/metrics \
  | grep DCGM_FI_DEV_GPU_UTIL
```

Основные series:

- `DCGM_FI_DEV_GPU_UTIL`
- `DCGM_FI_DEV_FB_USED`
- `DCGM_FI_DEV_FB_FREE`

При наличии Kubernetes enrichment labels определяют physical GPU, MIG instance,
pod и container.

## Использование существующего exporter

Найдите deployment GPU Operator:

```bash
kubectl get daemonset -A | grep -Ei 'dcgm|gpu-operator'
kubectl get service -A | grep -i dcgm
```

Используйте Service URL внутри cluster либо сделайте port-forward exporter pod на
том же node, где работает Triton.

DCGM не знает, какая модель vLLM создала нагрузку. Связывайте device metrics с
моделью только тогда, когда GPU или MIG полностью выделен этой модели на время
измерения.

{{- define "triton-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "triton-gateway.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "triton-gateway.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "triton-gateway.labels" -}}
helm.sh/chart: {{ include "triton-gateway.chart" . }}
{{ include "triton-gateway.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{- define "triton-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "triton-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "triton-gateway.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "triton-gateway.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "triton-gateway.modelClaimName" -}}
{{- default (printf "%s-models" (include "triton-gateway.fullname" .)) .Values.modelStorage.persistence.existingClaim }}
{{- end }}

{{- define "triton-gateway.mediaClaimName" -}}
{{- default (printf "%s-media" (include "triton-gateway.fullname" .)) .Values.mediaPersistence.existingClaim }}
{{- end }}

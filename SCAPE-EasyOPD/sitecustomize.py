"""Project-local startup compatibility shims for SCAPE-EasyOPD jobs.

Loaded through PYTHONPATH for Ray/verl subprocesses. This avoids changing the
frozen /opt runtime package set while handling the current OpenTelemetry version
skew in /opt/scape-easyopd-smoke7.
"""

from __future__ import annotations

try:
    from enum import Enum
    from opentelemetry.semconv._incubating.attributes import otel_attributes

    if not hasattr(otel_attributes, "OtelComponentTypeValues"):
        class OtelComponentTypeValues(Enum):
            PROMETHEUS_HTTP_TEXT_METRIC_EXPORTER = "prometheus_http_text_metric_exporter"

        otel_attributes.OtelComponentTypeValues = OtelComponentTypeValues
except Exception:
    pass

try:
    from opentelemetry.sdk.metrics.export import MetricReader

    _orig_metric_reader_init = MetricReader.__init__

    def _scape_metric_reader_init(self, *args, **kwargs):
        kwargs.pop("otel_component_type", None)
        return _orig_metric_reader_init(self, *args, **kwargs)

    if getattr(MetricReader.__init__, "__name__", "") != "_scape_metric_reader_init":
        MetricReader.__init__ = _scape_metric_reader_init
except Exception:
    pass

try:
    from opentelemetry.sdk.metrics import Meter

    _orig_create_histogram = Meter.create_histogram

    def _scape_create_histogram(self, name, unit="", description="", *args, **kwargs):
        kwargs.pop("explicit_bucket_boundaries_advisory", None)
        return _orig_create_histogram(self, name=name, unit=unit, description=description)

    if getattr(Meter.create_histogram, "__name__", "") != "_scape_create_histogram":
        Meter.create_histogram = _scape_create_histogram
except Exception:
    pass

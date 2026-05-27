"""fbsim-core: domain-agnostic forecasting-benchmark machinery.

A "world" (FreeCiv, pandemic, ...) plugs in by providing: a serializer to the
turn-major time_series schema, a template registry, a report renderer, and an
evaluation WorldAdapter. Core owns the schema, resolver, generator machinery,
scoring, and the eval pipeline.
"""

"""Evaluation: official scorer bridge, aggregation, paired stats, costs."""

from __future__ import annotations

from rsi.evaluation.aggregation import DomainResult, SuiteResult, aggregate_scores
from rsi.evaluation.retention import FourCells
from rsi.evaluation.statistics import StatisticsPlan

__all__ = ["DomainResult", "SuiteResult", "aggregate_scores", "FourCells", "StatisticsPlan"]

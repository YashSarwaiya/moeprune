"""Print the rented-hardware cost estimate for the full sweep.
Usage: PYTHONPATH=src python3 scripts/estimate_costs.py"""
from moeprune.config import ModelSpec, SweepSpec
from moeprune.costs import Assumptions, estimate, format_table

est = estimate(ModelSpec(), SweepSpec(), Assumptions())
print(format_table(est))

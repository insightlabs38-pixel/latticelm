"""Deterministic, verifier-backed synthetic reasoning data."""

from .core import Example, FAMILIES, GENERATOR_VERSION, generate_example, reconstruct
from .dataset import MILESTONES, build_canonical, verify_manifest

__all__ = ["Example", "FAMILIES", "GENERATOR_VERSION", "MILESTONES",
           "generate_example", "reconstruct", "build_canonical", "verify_manifest"]

"""Compatibility shim: the learned ReadHead lives in the library (jul.llada_head).

Kept so `import llada_head` in train_entry.py / the SageMaker tarballs keeps working.
"""
from jul.llada_head import ReadHead, _ordinal_target, head_loss, load_head  # noqa: F401

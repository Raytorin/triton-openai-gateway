# SPDX-License-Identifier: Apache-2.0
"""Adapt the pinned stock backend's token buckets for the unified vLLM runtime.

Run only against the image copy. Unknown releases and modified sources fail
closed; model/request code and the stock metric families remain unchanged.
"""
import argparse
import hashlib
import os
from pathlib import Path
import stat
import tempfile


ORIGINAL_SHA256 = '5a84f9b261606f154c710d9ef090df134d884ca62430a3ce57a16c6a62359778'
PATCHED_SHA256 = 'fafb1b3afb97c60637be64e2a023cabff37fbabf797e529877514f4111df70df'
ORIGINAL_IMPORTS = (
    b'from vllm.v1.metrics.loggers import StatLoggerBase, build_1_2_5_buckets\n'
    b'from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats\n'
)
REPLACEMENT_IMPORTS = (
    b'from vllm.v1.metrics.loggers import StatLoggerBase\n'
    b'from vllm.v1.metrics.buckets import build_1_2_5_buckets\n'
    b'from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats\n'
)


def patch_backend(backend):
    target = Path(backend)/'utils/metrics.py'
    source = target.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == PATCHED_SHA256:
        return False
    if digest != ORIGINAL_SHA256:
        raise RuntimeError(f'Unreviewed stock metrics SHA256: {digest}')
    if source.count(ORIGINAL_IMPORTS) != 1:
        raise RuntimeError('Stock metrics import context must occur exactly once')
    replacement = source.replace(ORIGINAL_IMPORTS, REPLACEMENT_IMPORTS)
    if hashlib.sha256(replacement).hexdigest() != PATCHED_SHA256:
        raise RuntimeError('Patched stock metrics SHA256 does not match the reviewed result')
    mode = stat.S_IMODE(target.stat().st_mode)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.metrics.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(replacement)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', type=Path, default=Path('/opt/tritonserver/backends/vllm'))
    args = parser.parse_args()
    changed = patch_backend(args.backend)
    print('Stock metrics compatibility: ' + ('patched' if changed else 'already patched'))


if __name__ == '__main__':
    main()

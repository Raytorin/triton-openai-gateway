# SPDX-License-Identifier: Apache-2.0
"""CPU audio codecs and Mistral resampling; no models, weights, GPU or network.

Run with the unified runtime's prepared launch environment so ``ffmpeg`` resolves
to its preserved codec-constrained executable. A generated PCM WAV exercises the
same FFmpeg reader used by the gateway's Transformers ASR pipeline.
"""
from importlib import metadata
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess


def main():
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from mistral_common.tokens.tokenizers.audio import Audio
    from transformers.pipelines.audio_utils import ffmpeg_read

    executable = shutil.which('ffmpeg')
    if executable is None or Path(executable).resolve() != Path('/opt/ffmpeg-safe/bin/ffmpeg').resolve():
        raise RuntimeError(f'Expected preserved codec-constrained ffmpeg on PATH, got {executable}')
    ffmpeg_version = subprocess.check_output([executable, '-version'], text=True, timeout=10).splitlines()[0]
    samples = (0.1 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)).astype(np.float32)
    audio = Audio(samples.copy(), 16000, format='wav')
    audio.resample(24000)
    if audio.sampling_rate != 24000 or audio.audio_array.shape != (2400,):
        raise RuntimeError('Mistral Audio/SoxR did not resample 16 kHz to 24 kHz')
    if not np.isfinite(audio.audio_array).all() or np.max(np.abs(audio.audio_array)) < 0.05:
        raise RuntimeError('Mistral Audio/SoxR lost the input signal')
    buffer = BytesIO()
    sf.write(buffer, samples, 16000, format='WAV', subtype='PCM_16')
    payload = buffer.getvalue()
    decoded, rate = sf.read(BytesIO(payload), dtype='float32')
    if rate != 16000 or decoded.shape != (1600,) or not np.allclose(decoded, samples, atol=4e-5):
        raise RuntimeError('SoundFile PCM WAV round trip failed')
    asr_audio = ffmpeg_read(payload, 24000)
    if asr_audio.shape != (2400,) or not np.isfinite(asr_audio).all():
        raise RuntimeError('Transformers ASR FFmpeg reader produced invalid PCM audio')
    # FFmpeg and SoxR use different boundary filters. Compare signal error
    # over the waveform rather than requiring identical endpoint samples.
    relative_rms_error = float(np.linalg.norm(asr_audio - audio.audio_array)
                               / np.linalg.norm(audio.audio_array))
    if relative_rms_error > 0.01:
        raise RuntimeError('Transformers FFmpeg reader and Mistral SoxR disagree on the generated signal')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU audio smoke unexpectedly initialized CUDA')
    print(json.dumps({
        'versions': {name: metadata.version(name) for name in ('numpy', 'soundfile', 'soxr', 'mistral-common', 'transformers')},
        'modules': {'soundfile': sf.__file__, 'soxr': soxr.__file__},
        'ffmpeg': executable, 'ffmpeg_version': ffmpeg_version,
        'pcm_wav_bytes': len(payload), 'input_samples': len(samples),
        'output_samples': len(asr_audio), 'input_rate': 16000, 'output_rate': 24000,
        'resampler_relative_rms_error': relative_rms_error,
        'cuda_initialized': False,
    }, sort_keys=True))


if __name__ == '__main__':
    main()

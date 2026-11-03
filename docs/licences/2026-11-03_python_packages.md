# Licence scan: Python packages (2026-11-03)

Every package installed in `.venv` on Khang's PC (91), with the licence each one declares in its own files, and
whether the app needs it at run time (`requirements.txt` plus PyTorch and what they pull in: 88) or only training
does (2). Not legal advice: a first pass to show a lawyer what matters.

## Result

**One real risk, and it needs a lawyer:** PyAV (`av` 18.1.0), which faster-whisper imports, ships FFmpeg together
with **libx264 and libx265, which are GPL-2.0-or-later**. FFmpeg itself reports "LGPL version 3 or later", but it
was built with `--enable-libx264 --enable-libx265`, and both libraries are inside the package. Our code never uses
`av` (it decodes audio files; the engine hands Whisper the microphone's samples directly), but `import
faster_whisper` loads it anyway (`faster_whisper/audio.py`, line 15). Shipping GPL code inside a product we sell,
loaded into our process, is exactly what the "commercial use" rule is about.
Ways out, cheapest first, all to decide with Khang and the lawyer:
1. Ship a PyAV built without x264 and x265 (an LGPL-only FFmpeg), or an older PyAV whose wheels didn't include them
   (not checked yet).
2. Don't ship `av` at all: only `faster_whisper.audio` needs it, and only for files; that means a small patch or
   wrapper around faster-whisper's import.
3. Keep it and comply with the GPL for those libraries (the lawyer's call).

**Fine for a product we sell, with conditions (normal for commercial software):**
- LGPL: `pynput` (LGPL-3.0), `soxr` (LGPL-2.1+), libsndfile inside `soundfile` (LGPL-2.1), FFmpeg inside `av` and
  `opencv-contrib-python` (LGPL). Keep them as separate, replaceable libraries (they are: separate packages and
  DLLs), don't change them without publishing the change, and ship their licence texts and a notice. LGPL-3.0 has
  extra wording about "installation information" for consumer devices: a lawyer question, probably not an issue for
  PC software.
- MPL-2.0: `certifi`, `tqdm`. Only changed MPL files would have to be shared; we don't change them.
- NVIDIA: PyTorch's CUDA build bundles cuBLAS, cuDNN and the CUDA runtime (NVIDIA's licence), the same question as
  llama.cpp's CUDA runtime files (DECISIONS.md, 2026-11-03): NVIDIA allows shipping these runtime files with an app
  under conditions; to confirm with the lawyer.

**Everything else (83 packages) is permissive:** MIT, BSD, Apache-2.0, ISC, PSF and similar. `llvmlite` declares
nothing in its metadata, but its own LICENSE file is BSD-2-Clause (plus LLVM's Apache-2.0 with the LLVM exception).

**An old question closed:** Silero VAD is MIT. Its LICENSE file on GitHub says MIT and its README says "Published
under permissive license (MIT)"; the "CC BY-NC 4.0" was only the hover text of a badge whose visible label is MIT
(checked 2026-11-03).

## Also shipped, not Python packages (from DECISIONS.md, not re-checked here)

- llama.cpp b11960: MIT; its CUDA runtime DLLs are NVIDIA's (see above).
- Models: Qwen3.5-4B (Apache 2.0, our clean-up model's base), Whisper large-v3-turbo, MediaPipe's face landmarker.
  Qwen2.5-3B (research licence) and UI-Venus 1.5 (conflicting statements) are being replaced (CLAUDE.md).
- Our trained adapter is built on Claude-written data: the Anthropic terms question is on the lawyer list.

## How this was done, and its limits

- `importlib.metadata` in `.venv` on Khang's PC: each package's `License-Expression`, `License` and licence
  classifiers, and its licence files. Run-time vs training from each package's own dependency list, starting from
  `requirements.txt` (plus torch, torchaudio, torchvision from `scripts/install.ps1`) and `requirements-train.txt`.
- Bundled native libraries were opened for the five packages most likely to carry them (`av`, `torch`,
  `opencv-contrib-python`, `soundfile`, `llvmlite`); FFmpeg's own licence string was read from its DLLs. The other
  packages' binaries were not opened.
- What a package declares is not always the whole story: the lawyer should see this list, not just the result.

## Every package

| Package | Version | Declared licence | Kind | Used by | Note |
| --- | --- | --- | --- | --- | --- |
| absl-py | 2.5.1 | Apache-2.0 | ok | runtime |  |
| accelerate | 1.15.0 | Apache Software License | ok | runtime |  |
| annotated-doc | 0.0.5 | MIT | ok | runtime |  |
| anyio | 4.15.1 | MIT | ok | runtime |  |
| audioread | 3.1.0 | MIT | ok | runtime |  |
| av | 18.1.0 | BSD-3-Clause | risk | runtime | BSD-3 for PyAV; bundles FFmpeg (LGPL-3.0+) **and libx264 / libx265 (GPL-2.0+)**; loaded by faster-whisper |
| certifi | 2026.10.23 | Mozilla Public License 2.0 (MPL 2.0) | conditions | runtime | MPL-2.0 |
| cffi | 2.1.1 | MIT-0 | ok | runtime |  |
| charset-normalizer | 3.5.2 | MIT | ok | runtime |  |
| click | 8.5.0 | BSD-3-Clause | ok | runtime |  |
| cloudpickle | 3.1.2 | BSD License | ok | runtime |  |
| colorama | 0.4.6 | BSD License | ok | runtime |  |
| comtypes | 1.4.17 | MIT | ok | runtime |  |
| contourpy | 1.3.3 | BSD License | ok | runtime |  |
| ctranslate2 | 4.8.3 | MIT | ok | runtime |  |
| cycler | 0.12.1 | BSD License | ok | runtime |  |
| decorator | 5.3.1 | BSD-2-Clause | ok | runtime |  |
| faster-whisper | 1.2.1 | MIT License | ok | runtime |  |
| filelock | 3.32.3 | MIT | ok | runtime |  |
| flatbuffers | 25.12.19 | Apache Software License | ok | runtime |  |
| fonttools | 4.67.0 | MIT | ok | runtime |  |
| fsspec | 2026.7.0 | BSD-3-Clause | ok | runtime |  |
| h11 | 0.16.0 | MIT License | ok | runtime |  |
| hf-xet | 1.7.0 | Apache-2.0 | ok | runtime |  |
| httpcore2 | 2.16.0 | BSD-3-Clause | ok | runtime |  |
| httpx2 | 2.16.0 | BSD-3-Clause | ok | runtime |  |
| huggingface_hub | 2.3.0 | Apache Software License | ok | runtime |  |
| idna | 3.20 | BSD-3-Clause | ok | runtime |  |
| Jinja2 | 3.1.6 | BSD License | ok | runtime |  |
| joblib | 1.6.0 | BSD-3-Clause | ok | runtime |  |
| kiwisolver | 1.5.1 | BSD License | ok | runtime |  |
| lazy-loader | 0.6 | BSD-3-Clause | ok | runtime |  |
| librosa | 0.11.0 | ISC License (ISCL) | ok | runtime |  |
| llvmlite | 0.50.0 | (licence text only: see its LICENSE file) | ok | runtime | no licence in its metadata; its LICENSE file is BSD-2-Clause, plus LLVM's Apache-2.0 with LLVM exception |
| markdown-it-py | 4.2.0 | MIT License | ok | runtime |  |
| MarkupSafe | 3.0.3 | BSD-3-Clause | ok | runtime |  |
| matplotlib | 3.11.2 | Python Software Foundation License | ok | runtime |  |
| mdurl | 0.1.2 | MIT License | ok | runtime |  |
| mediapipe | 1.1.1 | Apache Software License | ok | runtime |  |
| mpmath | 1.3.0 | BSD License | ok | runtime |  |
| msgpack | 1.2.3 | Apache-2.0 | ok | runtime |  |
| mss | 10.2.0 | MIT License | ok | runtime |  |
| narwhals | 2.27.1 | MIT | ok | runtime |  |
| networkx | 3.6.1 | BSD-3-Clause | ok | runtime |  |
| numba | 0.68.0 | BSD License | ok | runtime | BSD-2 |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | ok | runtime | BSD-3; bundles OpenBLAS (BSD) and the gfortran runtime (GPL with the runtime exception) |
| onnxruntime | 1.31.0 | MIT License | ok | runtime |  |
| opencv-contrib-python | 5.0.0.93 | Apache Software License | conditions | runtime | Apache-2.0; bundles an FFmpeg build (LGPL) and other libraries (LICENSE-3RD-PARTY.txt) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | ok | runtime |  |
| pillow | 12.3.0 | MIT-CMU | ok | runtime |  |
| platformdirs | 4.13.5 | MIT | ok | runtime |  |
| pooch | 1.9.0 | BSD-3-Clause | ok | runtime |  |
| protobuf | 7.36.2 | 3-Clause BSD License | ok | runtime |  |
| psutil | 7.2.2 | BSD-3-Clause | ok | runtime |  |
| pycparser | 3.12 | BSD-3-Clause | ok | runtime |  |
| Pygments | 2.21.0 | BSD-2-Clause | ok | runtime |  |
| pynput | 1.8.2 | GNU Lesser General Public License v3 (LGPLv3) | conditions | runtime | LGPL-3.0 |
| pyparsing | 3.3.3 | MIT | ok | runtime |  |
| python-dateutil | 3.0.0 | BSD License; Apache Software License | ok | runtime |  |
| PyYAML | 6.0.3 | MIT License | ok | runtime |  |
| RapidFuzz | 3.14.6 | MIT | ok | runtime |  |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | ok | runtime |  |
| requests | 2.34.2 | Apache Software License | ok | runtime |  |
| rich | 15.0.0 | MIT License | ok | runtime |  |
| safetensors | 0.8.0 | Apache Software License | ok | runtime |  |
| scikit-learn | 1.9.1 | BSD-3-Clause | ok | runtime |  |
| scipy | 1.17.1 | BSD License | ok | runtime | BSD-3; bundles OpenBLAS (BSD) and the gfortran runtime (GPL with the runtime exception) |
| setuptools | 65.5.0 | MIT License | ok | runtime |  |
| shellingham | 1.5.4 | ISC License (ISCL) | ok | runtime |  |
| silero-vad | 6.2.3 | MIT License | ok | runtime | MIT (checked on GitHub 2026-11-03; the "CC BY-NC" was old hover text on the badge); ships its model files |
| six | 1.17.0 | MIT License | ok | runtime |  |
| sounddevice | 0.5.6 | MIT | ok | runtime |  |
| soundfile | 0.14.0 | BSD License | conditions | runtime | BSD-3; bundles libsndfile (LGPL-2.1) |
| soxr | 1.1.0 | LGPL-2.1-or-later | conditions | runtime | LGPL-2.1+ (wraps libsoxr) |
| sympy | 1.14.0 | BSD License | ok | runtime |  |
| threadpoolctl | 3.7.0 | BSD-3-Clause | ok | runtime |  |
| tokenizers | 0.23.3 | Apache Software License | ok | runtime |  |
| torch | 2.11.0+cu128 | BSD-3-Clause | conditions | runtime | BSD-3; the CUDA build bundles NVIDIA cuBLAS, cuDNN and the CUDA runtime (NVIDIA licence) |
| torchvision | 0.26.0+cu128 | BSD | ok | runtime |  |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | conditions | runtime | MPL-2.0 and MIT |
| transformers | 5.20.2 | Apache 2.0 License | ok | runtime |  |
| truststore | 0.10.5 | MIT | ok | runtime |  |
| typer | 0.27.3 | MIT | ok | runtime |  |
| typing_extensions | 4.16.0 | PSF-2.0 | ok | runtime |  |
| tzdata | 2026.5 | Apache-2.0 | ok | runtime |  |
| uiautomation | 2.0.29 | Apache 2.0 | ok | runtime |  |
| urllib3 | 2.8.0 | MIT | ok | runtime |  |
| websockets | 17.3 | BSD-3-Clause | ok | runtime |  |
| jiwer | 4.0.0 | Apache-2.0 | ok | train |  |
| peft | 0.21.2 | Apache Software License | ok | train |  |
| pip | 26.2.2 | MIT | ok | other |  |

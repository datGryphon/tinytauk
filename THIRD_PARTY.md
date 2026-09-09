# Third-party notes

## Tencent Hunyuan AuK

Project: https://github.com/Tencent-Hunyuan/AuK

AuK is released under the MIT license by Tencent.

TinyTAuK's reference-oracle tooling uses the upstream project only as a development oracle. The standalone CPU implementation adapts portions of AuK's inference architecture in:

- `src/tinytauk/generator/modules.py`
- `src/tinytauk/generator/flux2.py`
- `src/tinytauk/vae/bigvgan.py`

Those adaptations derive from Tencent-Hunyuan/AuK commit `d9f30ffe4231dbc90b48cc83a35d310fece0b060`.

Copyright (C) 2026 Tencent. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## BigVGAN / HiFi-GAN decoder lineage

AuK's `BigVGANFlowVAE` decoder carries the following upstream notice, retained for TinyTAuK's decoder-only adaptation:

Copyright (c) 2022 NVIDIA CORPORATION.

The BigVGAN/HiFi-GAN-derived portions are MIT-licensed in the upstream AuK source.

## alias-free-torch activation/resampling lineage

The anti-aliased activation and resampling logic in `src/tinytauk/vae/bigvgan.py` is adapted from the `alias-free-torch` implementation vendored by AuK, originally from:

https://github.com/junjun3518/alias-free-torch

That work is licensed under the Apache License 2.0:

https://www.apache.org/licenses/LICENSE-2.0

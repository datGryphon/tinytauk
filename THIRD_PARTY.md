# Third-party licenses

TinyTAuK adapts code from the projects below.

## Tencent Hunyuan AuK — MIT

Source: https://github.com/Tencent-Hunyuan/AuK

Adapted from commit `d9f30ffe4231dbc90b48cc83a35d310fece0b060` in:

- `src/tinytauk/generator/modules.py`
- `src/tinytauk/generator/flux2.py`
- `src/tinytauk/vae/bigvgan.py`
- `src/tinytauk/vae/encoder.py`

Copyright (C) 2026 Tencent. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BigVGAN / HiFi-GAN — MIT

AuK's VAE is derived from BigVGAN and HiFi-GAN. TinyTAuK retains the
corresponding encoder/decoder lineage in `src/tinytauk/vae/encoder.py` and
`src/tinytauk/vae/bigvgan.py`.

Copyright (c) 2022 NVIDIA CORPORATION.
Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## alias-free-torch — Apache-2.0

Source: https://github.com/junjun3518/alias-free-torch

The anti-aliased activation/resampling implementation in
`src/tinytauk/vae/bigvgan.py` is adapted from alias-free-torch as vendored by
AuK. The Apache License 2.0 is included at `LICENSES/Apache-2.0.txt`.

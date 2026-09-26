# GridTrackNet

The optional five-frame detector, ONNX architecture conversion, and grid decoder
are adapted from https://github.com/VKorpelshoek/GridTrackNet at revision
0764162b73fb64d440fd9e6c363d592965400799. The numeric model checkpoint is downloaded
from that same revision. Upstream training uses RGB input and normalization on
the last (width) axis; the conversion preserves those trained operations.

MIT License

Copyright (c) 2023 Vincent Korpelshoek

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

The publisher's benchmark numbers are not accuracy measurements on the user's
footage. Dataset rights and intended deployment should be reviewed separately.

# Vendored browser dependencies

Inlined into replay pages so they work offline from `file://` with no CDN.

| File | Package | Source | Integrity |
| --- | --- | --- | --- |
| `three-0.159.0.min.js` | three 0.159.0 (MIT, `THREE-LICENSE.txt`) | `https://registry.npmjs.org/three/-/three-0.159.0.tgz`, `package/build/three.min.js` | tarball `sha512-eCmhlLGbBgucuo4VEA9IO3Qpc7dh8Bd4VKzr7WfW4+8hMcIfoAVi1ev0pJYN9PTTsCslbcKgBwr2wNZ1EvLInA==` (matches registry); file SHA-256 `7b1c5d75b28d9de15042e2b374f83566d8c7146697af8fdeb4558b0fb528a585` |

0.159.0 is the last three.js release with a classic (non-module) build. Newer
releases are ES-module only, which browsers refuse to load from `file://`
pages. The build prints a one-line deprecation warning in the console; it is harmless.

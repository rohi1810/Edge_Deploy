# Vendored third-party assets

Everything the demo loads is checked in. Nothing is fetched from a CDN at
runtime, because the page has to work from a USB stick on a machine with no
network — see `demo/schema.py::check_no_remote_subresources`, which fails the
build if a remote `src`/`href`/`url()` ever creeps back in.

## three.js — r147 (0.147.0)

| file | sha256 |
|---|---|
| `assets/vendor/three/three.min.js` | `f34446bf875b5fb0dcd93819ffe1d9e182d46634ee855f5d904c6c4ac7cdbc95` |
| `assets/vendor/three/OrbitControls.js` | `b4c6e53f98538535b11fd6655627d47fb5877b3ad972568bff0a7026c4b6d5c4` |

Source:
- https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js
- https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js
- https://cdn.jsdelivr.net/npm/three@0.147.0/LICENSE (MIT)

### Do not upgrade this without reading the following

**r147 is the last release that ships both a UMD build and a non-module
`OrbitControls`.** From r148 the `examples/js/` tree was deleted and only
`examples/jsm/` (ES modules) remains; r149's `OrbitControls.js` is a 404 at
the path above, which is how this pin was chosen in the first place.

That matters because the demo must run from `file://`. Under an opaque origin
browsers block ES modules *and* `fetch`, so `<script type="module">` and any
loader built on it simply do not execute. Classic `<script>` tags do. Moving
to r148+ therefore means either dropping the double-click-the-folder delivery
mode or adding a bundler — both of which this demo deliberately avoids.

If you do upgrade, the check is not "does it look fine on localhost" (it will)
but **open `demo/index.html` directly from disk with the network off**.

### Loaders deliberately not used

`PLYLoader` and `PCDLoader` use `fetch`/XHR internally and fail on `file://`
for the same reason. The tunnel geometry is therefore pre-converted by
`demo/build_demo.py --only tunnel` into base64 inside a `DEMO.register(...)`
script and decoded to typed arrays in `assets/js/tunnel.js`.

## Fonts — IBM Plex

Not yet vendored (milestone B3). `assets/css/fonts.css` is currently a
placeholder that falls back to `system-ui` / `ui-monospace`. When added they
must ship with `LICENSE-OFL.txt`; OFL-1.1 requires the licence to travel with
the fonts, which matters for a document a university will archive.

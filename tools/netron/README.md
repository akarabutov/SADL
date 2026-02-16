# Netron SADL Module Patch

This patch adds SADL model support to Netron:
- Adds `.sadl` extension to Netron metadata
- Registers `./sadl` model factory for `SADL`
- Adds `source/sadl.js`

## Recommended base (stable)
Netron release tag: `v8.9.0`

## Apply
```bash
git clone https://github.com/lutzroeder/netron.git
cd netron
git checkout v8.9.0
git apply /path/to/netron-sadl.patch   # or git am /path/to/netron-sadl.patch
cd source
python3 -m http.server 8080
```

Open: http://127.0.0.1:8080

Upstream Netron is MIT licensed; applying this patch produces a modified Netron tree; retain upstream LICENSE.

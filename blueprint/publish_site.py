#!/usr/bin/env python3
"""Copy the rendered blueprint (blueprint/_out/site/html-multi) to docs/blueprint, for GitHub Pages.

The renderer records source locations as absolute paths of the machine that built it. This script
rewrites them relative to the repository root, so that the published site does not carry a local
directory and a rebuild on any machine gives the same files up to the build stamp (the time, and
the commit it was built from, in index.html) and the order of the <script> blocks, which the
renderer does not fix. It fails if any absolute local path is left.

usage (from the repository root): python3 blueprint/publish_site.py [--check]
  --check  compare a fresh render with docs/blueprint instead of copying (up to the build stamp and
           the order of <script> blocks)
"""
import os, re, sys, shutil, filecmp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'blueprint', '_out', 'site', 'html-multi')
DST = os.path.join(ROOT, 'docs', 'blueprint')
# absolute local paths, also JSON-escaped (\/) and URL-encoded (%2F); scanned in every file
_DIRS = ['Users', 'home', 'root', 'Volumes', 'mnt', 'var/folders', 'private/tmp', 'private/var']
LOCAL = re.compile('|'.join(
    [re.escape('/' + d + '/') for d in _DIRS] +
    [re.escape('\\/' + d.replace('/', '\\/') + '\\/') for d in _DIRS] +
    [r'(?i:%2F(?:' + '|'.join(re.escape(d.replace('/', '%2F')) for d in _DIRS) + r')%2F)'] +
    [r'[A-Za-z]:\\\\']).encode())
# the build time, this repository's commit (hex only) and its subject; the dependency versions
# (shown as name@hash) are compared
STAMP = re.compile(r'(<span class="bp_build_metadata_value">)\d{4}-\d\d-\d\dT[\d:]+Z(</span>)'
                   r'|(<code class="bp_build_metadata_commit">)[0-9a-f]{7,40}(</code>)'
                   r'|(<span class="bp_build_metadata_subject">)[^<]*(</span>)')
SCRIPT = re.compile(r'<script\b.*?</script>', re.S)
def normal(x):
    x = STAMP.sub(lambda m: ''.join(g for g in m.groups() if g), x)
    return (SCRIPT.sub('<script/>', x), sorted(SCRIPT.findall(x)))
TEXT = ('.html', '.json', '.js', '.mjs', '.ts', '.map', '.css', '.txt', '.svg', '.xml')

def render(dst):
    prefix = ROOT.rstrip('/') + '/'
    forms = [prefix, prefix.replace('/', '\\/')]          # plain and JSON-escaped
    for d, _, fs in os.walk(SRC):
        for f in fs:
            s = os.path.join(d, f); t = os.path.join(dst, os.path.relpath(s, SRC))
            os.makedirs(os.path.dirname(t), exist_ok=True)
            b = open(s, 'rb').read()
            try:
                x = b.decode('utf-8')
                for pf in forms: x = x.replace(pf, '')
                b = x.encode('utf-8')
            except UnicodeDecodeError:
                pass                                          # binary (fonts): scanned, not rewritten
            m = LOCAL.search(b)
            if m:
                sys.exit(f'publish_site: absolute local path left in {os.path.relpath(s, SRC)}: '
                         f'{b[max(0, m.start() - 40):m.start() + 80]!r}')
            open(t, 'wb').write(b)

if not os.path.isdir(SRC):
    sys.exit('publish_site: run `lake exe vbp build` in blueprint/ first')
if '--check' in sys.argv:
    import tempfile
    tmp = tempfile.mkdtemp(); render(tmp)
    bad = []
    for d, _, fs in os.walk(tmp):
        for f in fs:
            a = os.path.join(d, f); rel = os.path.relpath(a, tmp); b = os.path.join(DST, rel)
            if not os.path.exists(b): bad.append(('missing in docs/blueprint', rel)); continue
            if f.endswith(TEXT):
                if normal(open(a, encoding='utf-8').read()) != normal(open(b, encoding='utf-8').read()):
                    bad.append(('differs', rel))
            elif not filecmp.cmp(a, b, shallow=False): bad.append(('differs', rel))
    extra = [os.path.relpath(os.path.join(d, f), DST) for d, _, fs in os.walk(DST) for f in fs
             if not os.path.exists(os.path.join(tmp, os.path.relpath(os.path.join(d, f), DST)))]
    bad += [('only in docs/blueprint', e) for e in extra]
    for k, r in bad: print(f'  {k}: {r}')
    print('publish_site --check:', 'SAME (up to the build stamp and script order)' if not bad else f'{len(bad)} differences')
    sys.exit(1 if bad else 0)
shutil.rmtree(DST, ignore_errors=True); render(DST)
print(f'publish_site: wrote {sum(len(fs) for _, _, fs in os.walk(DST))} files to docs/blueprint (paths relative to the repository root)')

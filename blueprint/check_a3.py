"""Compare the (lean := ...) lists of blueprint/Chapters/*.lean with the paper's Appendix A.3.
The paper's A.3 is read from aoi2026_triviality_osp1_2n.tex (\\leandecl{Module}{name}); full names are resolved
from the module's namespaces. usage: python3 blueprint/check_a3.py aoi2026_triviality_osp1_2n.tex . [expected-mapping.json]"""
import re,sys,glob,json
tex,repo=sys.argv[1],sys.argv[2]
t=open(tex).read(); i=t.index('subsec:formal-binding}'); seg=t[i:i+5200]; seg=seg[:seg.index('\\endgroup')]
def full(mod,name):
    s=open(f'{repo}/InhomogeneousDeformations/{mod}.lean').read(); ns=[]; res=None
    for m in re.finditer(r'^\s*(namespace\s+(\S+)|end\s+(\S+)|(?:@\[[^\]]*\]\s*)?(?:noncomputable\s+|protected\s+|private\s+)*(?:theorem|lemma|def)\s+(\S+))',s,re.M):
        if m.group(2): ns+=m.group(2).split('.')
        elif m.group(3):
            p=m.group(3).split('.')
            if ns[-len(p):]==p: del ns[-len(p):]
        elif m.group(4)==name: res='.'.join(ns+[name])
    return res
items={}
for p in re.split(r'\\textbf\{',seg)[1:]:
    h=p[:40]; k=next((x for x in ('P1','P2','P3','P4','P5') if 'item:'+x in h),'N1')
    items[k]=[full(m.group(1),m.group(2).replace('\\_','_')) for m in re.finditer(r'\\leandecl\{(\w+)\}\{([^}]*)\}',p)]
exp={'item_P1':items['P1'],'thm_main':items['P2']+items['P3'][:4],'item_gR_checks':items['P3'][4:],'prop_recovery':items['P4'],'item_P5':items['P5'],'item_N1':items['N1']}
got={}
for f in glob.glob(f'{repo}/blueprint/OspTrivialityBlueprint/Chapters/*.lean'):
    for m in re.finditer(r':::(\w+) "(\w+)"(?: \(lean := "([^"]*)"\))?',open(f).read()):
        if m.group(3): got[m.group(2)]=[x.strip() for x in m.group(3).split(',')]
ok=True
for k,v in exp.items():
    same=set(got.get(k,[]))==set(v); ok&=same; print(f'{k:16} paper A.3: {len(v)} names   blueprint: {len(got.get(k,[]))}   {"MATCH" if same else "MISMATCH"}')
extra=set(got)-set(exp); ok&=not extra
print('nodes with a lean list but no A.3 item:',sorted(extra)); print('total paper names',sum(len(v) for v in exp.values()),'unique',len({x for v in exp.values() for x in v}))
print('RESULT:','ALL MATCH' if ok else 'MISMATCH'); sys.exit(0 if ok else 1)

import sys,os,re,glob,numpy as np,trimesh,xml.etree.ElementTree as ET
sys.path.insert(0,'/workspace/swx'); from dl2 import parse; import ext
SRC="/workspace/courses/ae-4322/team8-onedrive/Vehicle and System Design Review/FINAL CAD for Review/"
ns='{http://www.solidworks.com/sw2003/schema}'
root=ET.parse('asm/59645811_swXmlContents_COMPINSTANCETREE').getroot()
models={m.get('id'):m for m in root.iter(ns+'swModel')}
cache={}
def part(name):
    if name not in cache:
        out='parts/'+name; 
        if not os.path.isdir(out): ext.ext(SRC+name+'.SLDPRT',out)
        fs=glob.glob(out+'/*Contents_DisplayLists'); best=None
        for f in fs:
            m,_=parse(open(f,'rb').read())
            if m is not None and (best is None or len(m.faces)>len(best.faces)): best=m
        cache[name]=best
    return cache[name]
def M(ref):
    v=np.array(ref.get('swTransform').split(),float); return v
def apply(V,v,trans):
    R=v[[0,1,2,4,5,6,8,9,10]].reshape(3,3); t=v[12:15]
    return (V@R if not trans else V@R.T)+t
def build(cfg_model_id,trans,skip=('TOPEX','Jason')):
    parts=[]
    for ref in models[cfg_model_id].findall(ns+'swReference'):
        if ref.get('swSuppressed')=='YES' or ref.get('swHidden')=='YES': continue
        name=ref.get('swName'); v=M(ref)
        if any(s in name for s in skip): continue
        sub=models[ref.get('swModelRef')]
        if sub.findall(ns+'swReference'):
            for r2 in sub.findall(ns+'swReference'):
                if r2.get('swSuppressed')=='YES' or r2.get('swHidden')=='YES': continue
                p=part(r2.get('swName'))
                if p is None: print('NOMESH',r2.get('swName')); continue
                V=apply(apply(p.vertices,M(r2),trans),v,trans)
                parts.append((name+'/'+r2.get('swName')+'-'+r2.get('swReferenceNumber'),trimesh.Trimesh(V,p.faces,process=False)))
        else:
            p=part(name)
            if p is None: print('NOMESH',name); continue
            parts.append((name+'-'+ref.get('swReferenceNumber'),trimesh.Trimesh(apply(p.vertices,v,trans),p.faces,process=False)))
    return parts

import numpy as np,struct,re,sys,trimesh
P=b'\x0c\x00\x00\x00\x64\x00\x00\x00\x02\x00\x00\x00'
H=b'\x04\x00\x00\x00\x08\x00\x00\x00\x02\x00\x00\x00'
def parse(d,mode='strip'):
    V=[];F=[];base=0;stats=[0,0]
    for m in re.finditer(re.escape(P),d):
        k=m.start(); hit=None
        for K in range(1,4000):
            s=k-4*(K+1)
            if s-12<0: break
            if d[s-12:s]==H and struct.unpack('<I',d[s:s+4])[0]==K:
                hit=(K,struct.unpack(f'<{K}I',d[s+4:k])); break
        if not hit: continue
        K,L=hit; n=struct.unpack('<I',d[k+12:k+16])[0]
        if n==0: continue
        v=np.frombuffer(d[k+16:k+16+12*n],'<f4').reshape(-1,3)
        if sum(L)==n:
            o=0;stats[0]+=1
            for l in L:
                for i in range(l-2):
                    if mode=='strip': t=(o+i,o+i+1,o+i+2) if i%2==0 else (o+i+1,o+i,o+i+2)
                    else: t=(o,o+i+1,o+i+2)
                    F.append((base+t[0],base+t[1],base+t[2]))
                o+=l
        elif n%3==0:
            stats[1]+=1
            F+= [(base+i,base+i+1,base+i+2) for i in range(0,n,3)]
        else: continue
        V.append(v); base+=n
    if not V: return None,stats
    m=trimesh.Trimesh(np.concatenate(V).astype(float),np.array(F),process=True)
    return m,stats
if __name__=='__main__':
    d=open(sys.argv[1],'rb').read()
    for mode in ('strip','fan'):
        m,s=parse(d,mode); m.merge_vertices(digits_vertex=6)
        print(mode,s,len(m.faces),'watertight',m.is_watertight,'winding',m.is_winding_consistent,m.extents.round(4))

import zlib,struct,re,sys,os
def rol(b,n): return ((b<<n)|(b>>(8-n)))&0xff
def ext(path,out):
    d=open(path,'rb').read(); os.makedirs(out,exist_ok=True); res=[]
    for m in re.finditer(re.escape(b'\x14\x00\x06\x00\x08\x00'),d):
        o=m.start()
        cs,us=struct.unpack('<II',d[o+14:o+22]); nl,el=struct.unpack('<HH',d[o+22:o+26])
        nm=d[o+26:o+26+nl]; p=o+26+nl+el
        try: data=zlib.decompress(d[p:p+cs],-15)
        except Exception as e: data=None
        # try name decode via rotations
        best=None
        for k in range(8):
            s=bytes(rol(c,k) for c in nm)
            if all(32<=c<127 for c in s): best=s.decode();break
        res.append((o,best or nm.hex(),cs,us,len(data) if data else -1))
        if data: open(os.path.join(out,f"{o}_{(best or 'x').replace('/','_')}"),'wb').write(data)
    return res
if __name__=="__main__":
    for r in ext(sys.argv[1],sys.argv[2]): print(r)

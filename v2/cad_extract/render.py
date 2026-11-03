import sys,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
def render(groups,out,views=((0,1,2),(0,2,1),(1,2,0))):
    fig,axs=plt.subplots(len(groups),3,figsize=(18,6*len(groups)),squeeze=False)
    for i,(title,ps) in enumerate(groups):
        Ts=[];Cs=[];Ns=[]
        for k,(n,p) in enumerate(ps):
            idx=np.arange(len(p.faces))
            if len(idx)>6000: idx=np.random.default_rng(k).choice(len(idx),6000,replace=False)
            Ts.append(p.vertices[p.faces[idx]]);Ns.append(p.face_normals[idx]);Cs+= [plt.cm.tab20(k%20)]*len(idx)
        T=np.concatenate(Ts);N=np.concatenate(Ns);C=np.array(Cs)
        for j,(a,b,c) in enumerate(views):
            o=np.argsort(T[:,:,c].mean(1)); col=C[o].copy(); col[:,:3]*= (0.4+0.6*np.abs(N[o,c]))[:,None]
            ax=axs[i,j]; ax.add_collection(PolyCollection(T[o][:,:,[a,b]],facecolors=col,edgecolors='none')); ax.autoscale(); ax.set_aspect('equal'); ax.set_title(f'{title} axes {a}{b}')
    plt.tight_layout(); plt.savefig(out,dpi=55)

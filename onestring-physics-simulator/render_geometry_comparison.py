import sys,pickle,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
from onestring_physics import onestring_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render(before,after,output):
    output.mkdir(parents=True,exist_ok=True)
    states=[pickle.loads((p/'state.pkl').read_bytes()) for p in [before,after]]
    for label,field in [('M3D','mesh_3d_initial'),('K3D','mesh_3d_optimized')]:
        fig=plt.figure(figsize=(13,6),layout='constrained')
        v=states[0].target_surface.vertices
        center=(v.max(0)+v.min(0))/2;radius=np.ptp(v,axis=0).max()*.56
        for i,s in enumerate(states):
            ax=fig.add_subplot(1,2,i+1,projection='3d');m=getattr(s,field);q=m.vertices[m.faces]
            colors=plt.cm.viridis((q.mean(1)[:,1]-v[:,1].min())/np.ptp(v[:,1]))
            ax.add_collection3d(Poly3DCollection(q,facecolors=colors,edgecolor='#222',linewidth=.5,alpha=.95))
            for d,fn in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):fn(center[d]-radius,center[d]+radius)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=28,azim=-70)
            ax.set_title(('Before: 013c659 / joint atlas' if i==0 else 'After: fixed BFF + Eq.(1)')+'\n'+label+f' ({len(q)} panels)')
            ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
        fig.savefig(output/(label+'_comparison.png'),dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for i,(s,ax) in enumerate(zip(states,axes)):
        xy=s.tiles_2d_dual_hinge.vertices[:,:4,:2];center=xy.reshape(-1,2).mean(0);xy=xy-center
        v=s.target_surface.vertices
        centers=s.mesh_3d_initial.vertices[s.mesh_3d_initial.faces].mean(1)
        colors=plt.cm.viridis((centers[:,1]-v[:,1].min())/np.ptp(v[:,1]))
        ax.add_collection(PolyCollection(xy,facecolors=colors,edgecolors='#222',linewidth=.5))
        ax.autoscale_view();ax.set_aspect('equal');ax.set_title(('Before' if i==0 else 'After')+' T2D Dual Hinge (top faces)')
    # Same units/scale in both panels; disconnected physical Split pieces are
    # retained at the solver's actual poses, never packed to beautify a figure.
    limits=np.array([ax.get_xlim()+ax.get_ylim() for ax in axes])
    for ax in axes:
        ax.set_xlim(limits[:,0].min(),limits[:,1].max())
        ax.set_ylim(limits[:,2].min(),limits[:,3].max())
    fig.savefig(output/'T2D_comparison.png',dpi=170);plt.close(fig)
    s=states[1];fig=plt.figure(figsize=(8,7),layout='constrained');ax=fig.add_subplot(111,projection='3d')
    v=s.target_surface.vertices;f=s.target_surface.faces
    ax.add_collection3d(Poly3DCollection(v[f],facecolor='#f1a340',edgecolor='none',alpha=.15))
    m=s.mesh_3d_optimized
    ax.add_collection3d(Poly3DCollection(m.vertices[m.faces],facecolor='#25868d',edgecolor='#111',linewidth=.5,alpha=.55))
    center=(v.max(0)+v.min(0))/2;radius=np.ptp(v,axis=0).max()*.56
    for d,fn in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):fn(center[d]-radius,center[d]+radius)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=28,azim=-70)
    ax.set_title('Source S (orange) + final K3D (teal)')
    fig.savefig(output/'source_K3D_overlay.png',dpi=170);plt.close(fig)

if __name__=='__main__':render(*map(Path,sys.argv[1:]))

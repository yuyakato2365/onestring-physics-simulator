"""Alternating corner pairs on the open M2D dual, reported in local corner IDs."""
import json,pickle,sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def write(out):
    with (out/'grid/result.pkl').open('rb') as stream: grid=pickle.load(stream)
    with (out/'remesh3d.pkl').open('rb') as stream: mesh=pickle.load(stream)
    ids=np.delete(np.arange(len(grid['Tg'])),mesh['removed_panels'])
    faces=grid['Tg'][ids]
    if not np.array_equal(mesh['kept_vertices'][mesh['T']],grid['wid'][faces]):
        raise ValueError('M2D and M3D corner correspondence differs')
    owners=defaultdict(list)
    for i,face in enumerate(faces):
        for k in range(3):
            owners[tuple(sorted((int(face[k]),int(face[(k+1)%3]))))].append((i,k))
    graph=[[] for _ in faces]
    for entries in owners.values():
        if len(entries)>2: raise ValueError('Nonmanifold M2D edge')
        if len(entries)==2:
            (a,_),(b,_)=entries
            graph[a].append(b)
            graph[b].append(a)
    color={}
    for i in range(len(faces)):
        if i in color: continue
        queue=[i]
        color[i]=0
        while queue:
            a=queue.pop()
            for b in graph[a]:
                if b not in color:
                    color[b]=1-color[a]
                    queue.append(b)
                elif color[b]==color[a]: raise ValueError('M2D dual is not bipartite')
    hinges=[]
    for entries in owners.values():
        if len(entries)!=2: continue
        (a,i),(b,j)=entries
        ca=(i+color[a])%3
        cb=(j+color[b])%3
        if faces[a,ca]!=faces[b,cb]: raise ValueError('Inconsistent corner pair')
        hinges.append([a,ca,b,cb])
    (out/'hinges.json').write_text(json.dumps(dict(panel_ids=ids.tolist(),count=len(hinges),hinges=hinges)))
    print(len(faces),'panels,',len(hinges),'corner hinges')


if __name__=='__main__': write(Path(sys.argv[1]))

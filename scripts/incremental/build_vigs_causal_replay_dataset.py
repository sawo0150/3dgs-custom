#!/usr/bin/env python3
"""Build fixed-pose COLMAP data plus a causal full-frame arrival schedule."""
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rgb',type=Path,required=True); p.add_argument('--trajectory',type=Path,required=True)
    p.add_argument('--points',type=Path,required=True); p.add_argument('--kf-manifest',type=Path); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iters-per-event',type=int,default=60); p.add_argument('--tail-iters',type=int,default=3000); p.add_argument('--focal',type=float,default=500.)
    a=p.parse_args(); files=sorted([*a.rgb.glob('*.jpg'),*a.rgb.glob('*.png')],key=lambda x:int(x.stem)); traj=np.loadtxt(a.trajectory)
    source_kf_times=traj[:,0].copy()
    if len(files)!=len(traj):
        query=np.asarray([int(x.stem)*1e-9 for x in files]); order=np.argsort(traj[:,0]); traj=traj[order]
        from scipy.spatial.transform import Slerp
        clipped=np.clip(query,traj[0,0],traj[-1,0]); interp=np.zeros((len(files),8)); interp[:,0]=query
        interp[:,1:4]=np.stack([np.interp(clipped,traj[:,0],traj[:,j]) for j in (1,2,3)],axis=1)
        interp[:,4:8]=Slerp(traj[:,0],Rotation.from_quat(traj[:,4:8]))(clipped).as_quat(); traj=interp
    if a.kf_manifest:
        boundaries=np.asarray(sorted({int(x['rgb_idx']) for x in json.loads(a.kf_manifest.read_text())}),dtype=int)
    else:
        image_times=np.asarray([int(x.stem)*1e-9 for x in files])
        right=np.clip(np.searchsorted(image_times,source_kf_times),1,len(image_times)-1)
        left=right-1
        nearest=np.where(abs(image_times[left]-source_kf_times)<=abs(image_times[right]-source_kf_times),left,right)
        boundaries=np.asarray(sorted(set(map(int,nearest))),dtype=int)
    images=a.output/'images'; sparse=a.output/'sparse'/'0'; images.mkdir(parents=True,exist_ok=True); sparse.mkdir(parents=True,exist_ok=True)
    lines=[]; arrivals={}
    for idx,(image,row) in enumerate(zip(files,traj)):
        target=images/image.name
        if not target.exists(): os.symlink(image,target)
        c2w=np.eye(4); c2w[:3,:3]=Rotation.from_quat(row[4:8]).as_matrix(); c2w[:3,3]=row[1:4]; w2c=np.linalg.inv(c2w)
        qx,qy,qz,qw=Rotation.from_matrix(w2c[:3,:3]).as_quat(); t=w2c[:3,3]
        lines.append(f'{idx+1} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {t[0]:.12g} {t[1]:.12g} {t[2]:.12g} 1 {image.name}\n\n')
        if idx%8!=0:
            event=min(int(np.searchsorted(boundaries,idx,side='left')),len(boundaries)-1)
            arrivals[image.name]=1+event*a.iters_per_event
    (sparse/'cameras.txt').write_text(f'1 PINHOLE 1024 1024 {a.focal} {a.focal} 512 512\n')
    (sparse/'images.txt').write_text(''.join(lines))
    pts=np.loadtxt(a.points)
    (sparse/'points3D.txt').write_text(''.join(f'{j} {r[1]} {r[2]} {r[3]} 128 128 128 0\n' for j,r in enumerate(pts)))
    payload={'arrival_iteration_by_name':arrivals,'total_iterations':len(boundaries)*a.iters_per_event+a.tail_iters,
             'iters_per_event':a.iters_per_event,'tail_iters':a.tail_iters,'events':len(boundaries),'all_frames':len(files),
             'train_frames':len(arrivals),'heldout_frames':len(files)-len(arrivals),'heldout_rule':'sorted index modulo 8 equals zero'}
    (a.output/'causal_arrivals.json').write_text(json.dumps(payload,indent=2)); print(json.dumps({k:v for k,v in payload.items() if k!='arrival_iteration_by_name'},indent=2))
if __name__=='__main__': main()

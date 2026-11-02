"""Collision-checked joint-space transfers for the live drum-button demo."""
import time
import copy
import json

import mujoco
import numpy as np
from model import CONFIG


class TransferTrajectory:
    """C1 Hermite trajectory with bounded speed/acceleration and checked blends."""
    def __init__(self, path, velocity, acceleration, valid):
        self.path=np.asarray(path,dtype=float)
        delta=np.diff(self.path,axis=0)
        self.durations=np.maximum(.1,np.maximum(
            np.max(1.5*np.abs(delta)/velocity,axis=1),
            np.max(np.sqrt(6*np.abs(delta)/acceleration),axis=1)))
        slopes=delta/self.durations[:,None]
        tangents=np.zeros_like(self.path)
        for i in range(1,len(self.path)-1):
            before,after=slopes[i-1],slopes[i]
            # Shared tangent carries motion through corners; validate the entire
            # rounded curve below because it can leave the straight-line path.
            tangents[i]=(before+after)/2
        self.blend_scale=1.
        for attempt in range(10):
            self.tangents=tangents*self.blend_scale
            self._coefficients()
            clear=True
            for i,change in enumerate(delta):
                count=max(32,int(np.ceil(np.max(np.abs(change))/.005)))
                for u in np.linspace(0,1,count+1):
                    q=self._sample_segment(i,u)[0]
                    if not valid(q):
                        clear=False
                        break
                if not clear:break
            if clear:break
            self.blend_scale=self.blend_scale*.5 if attempt<8 else 0.
        else:
            raise ValueError('No collision-clear transfer blend')
        # Exact extrema: velocity is quadratic; acceleration is linear.
        peak_v,peak_a=self.peaks()
        scale=max(1.,float(np.max(peak_v/velocity)),
                  float(np.sqrt(np.max(peak_a/acceleration))))
        self.durations*=scale
        self.tangents/=scale
        self._coefficients()
        self.ends=np.cumsum(self.durations)

    def _coefficients(self):
        delta=np.diff(self.path,axis=0)
        first=self.tangents[:-1]*self.durations[:,None]
        last=self.tangents[1:]*self.durations[:,None]
        self.a=-2*delta+first+last
        self.b=3*delta-2*first-last
        self.c=first

    def _sample_segment(self, i, u):
        q=self.path[i]+u*(self.c[i]+u*(self.b[i]+u*self.a[i]))
        v=(self.c[i]+u*(2*self.b[i]+3*u*self.a[i]))/self.durations[i]
        return q,v

    def sample(self, elapsed):
        i=min(int(np.searchsorted(self.ends,elapsed,side='right')),len(self.durations)-1)
        start=0 if i==0 else self.ends[i-1]
        return self._sample_segment(i,float(np.clip((elapsed-start)/self.durations[i],0,1)))

    def peaks(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            root=np.where(np.abs(self.a)>1e-14,-self.b/(3*self.a),0)
        root=np.clip(root,0,1)
        v0=np.abs(self.c/self.durations[:,None])
        v1=np.abs((self.c+2*self.b+3*self.a)/self.durations[:,None])
        vr=np.abs((self.c+2*self.b*root+3*self.a*root**2)/self.durations[:,None])
        acceleration=np.maximum(np.abs(2*self.b),np.abs(2*self.b+6*self.a))/self.durations[:,None]**2
        return np.maximum(np.maximum(v0,v1),vr),acceleration


def clearance_checker(model, controller, start, goal, arm=None):
    clearance=float(json.loads(CONFIG.with_name('demo_motion.json').read_text())['transfer_clearance_m'])
    # Detect near misses on a private model. These margins never change the
    # scene's physical contact geometry or forces.
    model=copy.copy(model)
    model.geom_margin[:]=np.maximum(model.geom_margin,clearance+.004)
    probe=mujoco.MjData(model)
    selected=np.arange(14) if arm is None else np.arange(7) if arm=='left' else np.arange(7,14)
    relevant=lambda contact: arm is None or any((model.geom(g).name or '').startswith(arm+'_') for g in (contact.geom1,contact.geom2))
    endpoint_clearance={}
    for q in (start,goal):
        probe.qpos[controller.q]=q
        mujoco.mj_forward(model,probe)
        if any(c.dist < -1e-6 for c in probe.contact if relevant(c)):
            raise ValueError('Transfer endpoint intersects geometry')
        for contact in probe.contact:
            if not relevant(contact):continue
            pair=tuple(sorted((int(contact.geom1),int(contact.geom2))))
            # A parked arm can already sit closer than the general clearance.
            # Preserve its existing gap instead of making every path impossible.
            endpoint_clearance[pair]=min(endpoint_clearance.get(pair,clearance),max(.0001,float(contact.dist)-.0005))
    def valid(q):
        if np.any(q[selected]<controller.bounds[selected,0]-.001) or np.any(q[selected]>controller.bounds[selected,1]+.001):return False
        probe.qpos[controller.q]=q
        mujoco.mj_forward(model,probe)
        return not any(c.dist < endpoint_clearance.get(tuple(sorted((int(c.geom1),int(c.geom2)))),clearance)
                       for c in probe.contact if relevant(c))
    return valid


def plan_transfer(model, controller, start, goal, cancel=None, timeout=6., arm=None):
    start,goal=np.array(start),np.array(goal)
    valid=clearance_checker(model,controller,start,goal,arm)
    active=np.flatnonzero(np.abs(start-goal)>.005)
    if not len(active):return [start.tolist(),goal.tolist()]
    deadline=time.monotonic()+timeout
    rng=np.random.default_rng(41)
    def edge(a,b):
        count=max(1,int(np.ceil(np.max(np.abs(b-a))/.025)))
        return all(valid(a+(b-a)*i/count) for i in range(1,count+1))
    if not valid(start) or not valid(goal):raise ValueError('Transfer endpoint intersects geometry')
    if edge(start,goal):return [start.tolist(),goal.tolist()]
    trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
    def extend(tree,target):
        nodes,parents=tree
        nearest=int(np.argmin([np.linalg.norm((q-target)[active]) for q in nodes]))
        delta=target-nodes[nearest]
        scale=min(1.,.22/max(np.linalg.norm(delta[active]),1e-12))
        q=nodes[nearest]+scale*delta
        if not edge(nodes[nearest],q):return None,False
        nodes.append(q);parents.append(nearest)
        return len(nodes)-1,scale==1.
    def trace(tree,index):
        path=[]
        while index>=0:
            path.append(tree[0][index]);index=tree[1][index]
        return path[::-1]
    for iteration in range(6000):
        if time.monotonic()>deadline or (cancel is not None and cancel.is_set()):
            raise ValueError('No clear transfer found; choose another drum or refine its pose')
        first=iteration%2;second=1-first
        if rng.random()<.25:
            sample=trees[second][0][0]
        else:
            sample=start.copy()
            sample[active]=rng.uniform(controller.bounds[active,0]+.03,controller.bounds[active,1]-.03)
        index,_=extend(trees[first],sample)
        if index is None:continue
        target=trees[first][0][index]
        for _ in range(80):
            other,reached=extend(trees[second],target)
            if other is None:break
            if reached:
                a,b=trace(trees[first],index),trace(trees[second],other)
                path=a+b[-2::-1] if first==0 else b+a[-2::-1]
                # Shortcut without weakening collision sampling.
                compact=[path[0]];at=0
                while at<len(path)-1:
                    end=len(path)-1
                    while end>at+1 and not edge(path[at],path[end]):end-=1
                    compact.append(path[end]);at=end
                return [q.tolist() for q in compact]
    raise ValueError('Transfer planning exhausted')

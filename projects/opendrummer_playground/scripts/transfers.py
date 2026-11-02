"""Collision-checked joint-space transfers for the live drum-button demo."""
import time
import copy
import json

import mujoco
import numpy as np
from model import CONFIG


def plan_transfer(model, controller, start, goal, cancel=None, timeout=6.):
    start,goal=np.array(start),np.array(goal)
    clearance=float(json.loads(CONFIG.with_name('demo_motion.json').read_text())['transfer_clearance_m'])
    # Detect near misses on a private model. These margins never change the
    # scene's physical contact geometry or forces.
    model=copy.copy(model)
    model.geom_margin[:]=np.maximum(model.geom_margin,clearance+.004)
    probe=mujoco.MjData(model)
    active=np.flatnonzero(np.abs(start-goal)>.005)
    if not len(active):return [start.tolist(),goal.tolist()]
    deadline=time.monotonic()+timeout
    rng=np.random.default_rng(41)
    endpoint_clearance={}
    for q in (start,goal):
        probe.qpos[controller.q]=q
        mujoco.mj_forward(model,probe)
        if any(c.dist < -1e-6 for c in probe.contact):
            raise ValueError('Transfer endpoint intersects geometry')
        for contact in probe.contact:
            pair=tuple(sorted((int(contact.geom1),int(contact.geom2))))
            # A parked arm can already sit closer than the general clearance.
            # Preserve its existing gap instead of making every path impossible.
            endpoint_clearance[pair]=min(endpoint_clearance.get(pair,clearance),max(.0001,float(contact.dist)-.0005))
    def valid(q):
        probe.qpos[controller.q]=q
        mujoco.mj_forward(model,probe)
        return not any(c.dist < endpoint_clearance.get(tuple(sorted((int(c.geom1),int(c.geom2)))),clearance)
                       for c in probe.contact)
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

"""2-D differential drive robot with image inference driving destination selection.

This is a lightweight, explicitly simplified physics simulator, not Gazebo.
It integrates left/right wheel speeds into the robot's (x,y,yaw) pose.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .ai_vision import CLASSES,sample_image,infer

PICKUP=(-2.55,0.0)
BINS={'MEDICINE':(2.30,1.80),'WATER':(2.30,0.0),'FOOD':(2.30,-1.80)}
ORDER=['MEDICINE','WATER','FOOD','MEDICINE']
RADIUS=0.05
SEPARATION=0.20
MAX_V=0.60
MAX_W=1.50
DT=0.05


def wrap(angle):
    return (angle+math.pi)%(2*math.pi)-math.pi


@dataclass
class Frame:
    t:float
    x:float
    y:float
    yaw:float
    v:float
    w:float
    left_wheel_rad_s:float
    right_wheel_rad_s:float
    phase:str
    parcel_index:int
    detected:str
    confidence:float
    carried:bool
    delivered:int
    target_x:float
    target_y:float
    ground_truth:str


def go_to_goal(x,y,yaw,tx,ty):
    dx,dy=tx-x,ty-y
    dist=math.hypot(dx,dy)
    error=wrap(math.atan2(dy,dx)-yaw)
    w=max(-MAX_W,min(MAX_W,2.7*error))
    v=min(MAX_V,1.0*dist)*max(0.0,math.cos(error))**2
    if abs(error)>0.75:
        v=0.0
    return v,w,dist


def simulate(model,seed=1007,max_time=170):
    rng=np.random.default_rng(seed)
    x,y,yaw=-2.55,-1.55,math.pi/2
    t=0.0
    i=0
    delivered=0
    phase='TO_PICKUP'
    pause_until=0
    predicted=''
    confidence=0.0
    scans={}
    completed=[]
    frames=[]
    while t <= max_time:
        carried=phase in ('TO_BIN','UNLOAD')
        if phase == 'TO_PICKUP':
            tx,ty=PICKUP
        elif phase in ('TO_BIN','UNLOAD'):
            tx,ty=BINS[predicted]
        else:
            tx,ty=x,y
        v,w,dist=go_to_goal(x,y,yaw,tx,ty) if phase in ('TO_PICKUP','TO_BIN') else (0.,0.,0.)
        if phase=='TO_PICKUP' and dist<0.12:
            label=ORDER[i]
            img=sample_image(label,rng)
            predicted,confidence,_=infer(model,img)
            scans[i]=img
            phase='INSPECT'
            pause_until=t+1.8
            v=w=0.
            tx,ty=PICKUP
        elif phase=='INSPECT' and t>=pause_until:
            phase='TO_BIN'
        elif phase=='TO_BIN' and dist<0.14:
            phase='UNLOAD'
            pause_until=t+1.4
            v=w=0.
        elif phase=='UNLOAD' and t>=pause_until:
            completed.append({'parcel':i+1,'actual':ORDER[i],
                              'predicted':predicted,'correct':predicted==ORDER[i],
                              'time_s':round(t,2)})
            delivered+=1
            i+=1
            if i==len(ORDER):
                phase='COMPLETED'
                v=w=0.
            else:
                phase='TO_PICKUP'
                predicted=''
                confidence=0.
        # Differential-drive wheel angular velocities and integration.
        wl=(v-0.5*SEPARATION*w)/RADIUS
        wr=(v+0.5*SEPARATION*w)/RADIUS
        v_actual=0.5*RADIUS*(wr+wl)
        w_actual=RADIUS*(wr-wl)/SEPARATION
        x+=v_actual*math.cos(yaw+0.5*w_actual*DT)*DT
        y+=v_actual*math.sin(yaw+0.5*w_actual*DT)*DT
        yaw=wrap(yaw+w_actual*DT)
        ground_truth=ORDER[min(i,len(ORDER)-1)]
        frames.append(Frame(round(t,3),x,y,yaw,round(v_actual,6),round(w_actual,6),
                            round(wl,6),round(wr,6),phase,i+1,predicted,
                            confidence,carried,delivered,tx,ty,ground_truth))
        if phase=='COMPLETED' and t>5:
            for _ in range(int(3/DT)):
                t+=DT
                frames.append(Frame(round(t,3),x,y,yaw,0,0,0,0,'COMPLETED',i,
                                    predicted,confidence,False,delivered,tx,ty,ground_truth))
            break
        t+=DT
    results={
        'mission_complete':phase=='COMPLETED',
        'items_total':len(ORDER),
        'items_delivered':delivered,
        'items_correctly_sorted':sum(p['correct'] for p in completed),
        'duration_s':round(frames[-1].t,2),
        'mean_abs_angular_command':round(float(np.mean([abs(f.w) for f in frames])),4),
        'max_linear_velocity_m_s':max(abs(f.v) for f in frames),
        'max_angular_velocity_rad_s':max(abs(f.w) for f in frames),
        'wheel_radius_m':RADIUS,'wheel_separation_m':SEPARATION,
        'deliveries':completed,
        'limitations':['2-D kinematics only','Synthetic image sensor','Ideal wheel contact/no slippage',
                       'Virtual grasp at pickup; no manipulator','No Gazebo run claimed'],
    }
    return frames,scans,results


def frame_to_dict(frame):
    return asdict(frame)

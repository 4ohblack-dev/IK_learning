import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "SO-ARM100/Simulation/SO100/so100.urdf"
ARM_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll"
]
#入力：RPY角、出力：回転行列(3,3)
#固定姿勢を計算
def rpy_matrix(rpy):
    r,p,y = rpy
    cr , sr = np.cos(r), np.sin(r)
    cp , sp = np.cos(p), np.sin(p)
    cy , sy = np.cos(y), np.sin(y)

    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz@Ry@Rx

#入力：回転軸(3,)、角度float　、出力：回転行列(3,3)
#関節の回転を計算
def axis_rotation(axis,theta):
    a = np.asarray(axis,dtype=float)
    a /= np.linalg.norm(a)
    x,y,z = a
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (np.eye(3)*np.cos(theta) + (1-np.cos(theta)*np.outer(a,a))+np.sin(theta)*K)

def as_xyz(s):
    return np.fromstring(s,sep=" ",dtype=float)
#入力：URDFのファイルパス、出力：関節情報のリスト
#FK用の情報を取得
def load_joints(path=URDF):
    root = ET.parse(path).getroot()
    by_name = {j.attrib["name"]: j for j in root.findall("joint")}
    joints = []
    last_child = "base"
    for name in ARM_NAMES:
        j = by_name[name]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        if parent != last_child:
            raise ValueError(f"joint chain is broken at {name}")
        origin = j.find("origin")
        xyz = as_xyz(origin.attrib["xyz"])
        rpy = as_xyz(origin.attrib["rpy"])
        axis = as_xyz(j.find("axis").attrib["xyz"])
        limit = j.find("limit")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])

        if xyz.size != 3 or rpy.size != 3 or axis.size != 3:
            raise ValueError(f"invalid URDF vector at {name}")
        if np.linalg.norm(axis) == 0:
            raise ValueError(f"zero rotation axis at {name}")

        joints.append((name,xyz,rpy,axis,lower,upper))
        last_child = child
    return joints

JOINTS = load_joints()
Q_MIN = np.array([j[4] for j in JOINTS])
Q_MAX = np.array([j[5] for j in JOINTS])

#入力：関節角度(5,)、TCP位置(3,)　、　出力：同時変換行列(4,4)
def fk(q,tcp_xyz=(0.0,0.0,0.0)):
    q = np.array(q,dtype=float)
    if q.shape != (5,) or not np.all(np.isfinite(q)):
        raise ValueError("q must contain five finite radians")
    T = np.eye(4)
    for angle, (_, xyz, rpy, axis, _, _) in zip(q, JOINTS):
        fixed = np.eye(4)
        fixed[:3,:3] = rpy_matrix(rpy)
        fixed[:3,3] = xyz
        motion = np.eye(4)
        motion[:3,:3] = axis_rotation(axis,angle)
        T = T @ fixed @ motion

    tcp = np.eye(4)
    tcp[:3,3] = np.asarray(tcp_xyz,dtype=float)
    return T@tcp

def position(q,tcp_xyz=(0.0,0.0,0.0)):
    return fk(q,tcp_xyz)[:3,3]

def jacobian(q,tcp_xyz=(0.0,0.0,0.0),h=1e-6):
    q = np.asarray(q,dtype=float)
    if h <= 0:
        raise ValueError("h must be positive")
    J = np.zeros(3,5)
    for i in range(5):
        delta = np.zeros(5)
        delta[i] = h
        J[:,i] = (position(q+delta,tcp_xyz) - position(q-delta,tcp_xyz))/(2*h)
    return J

#target:目標位置、seed:IKの初期関節角度、tol:位置誤差の許容値、max_iter:最大反復回数、damping:DLSの減衰係数、max_step:1回の香辛料のノルム上限
def solve_position_IK(target, seed, tcp_xyz=(0., 0., 0.),
                      tol=0.002, max_iter=300,
                      damping=0.04, max_step=0.10):
    target = np.asarray(target,dtype=float)
    seed = np.asarray(seed,dtype=float)
    if target.shape != (3,) or seed.shape != (5,):
        raise ValueError("target must be 3D and seed must be 5D")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(seed)):
        raise ValueError("nonfinite input")
    if np.any(seed < Q_MIN) or np.any(seed > Q_MAX):
        raise ValueError("seed is outside joint limits")

    q = seed.copy()
    if max_iter<1:
        raise ValueError("max_iter must be >= 1")
    
    for k in range(max_iter):
        p = fk(q,tcp_xyz)[:3,3]
        e = target - p
        err = np.linalg.norm(e)
        if err < tol:
            return {"success": True, "q": q, "error": err,"iterations": k}
        
        J = jacobian(q,tcp_xyz)
        A = J @ J.T + damping**2*np.eye(3)
        dq = J.T @ np.linalg.solve(A,e) #Ax = eを満たすxを求める
        norm_dq = np.linalg.norm(dq)
        if norm_dq > max_step:
            dq *=max_step / norm_dq

        accepted = False

        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = np.clip(q + scale*dq,Q_MIN,Q_MAX)
            trial_err = np.linalg.norm(target - fk(trial,tcp_xyz)[:3,3])
            if trial_err < err:
                q = trial
                accepted = True
                break
        if not accepted:
            break

    err = np.linalg.norm(target - fk(q,tcp_xyz)[:3,3])
    return {"success": False, "q": q, "error": err,
            "iterations": k+1}


def main():
    q = np.asarray([0.0,1.5,-1.2,0.0,0.2])
    goal = fk(q)[:3,3]
    seed = np.array([0.0, 1.75, -1.57, -0.5, 0.0])
    result = solve_position_IK(goal,seed)
    print("goal[m]:", goal)
    print("result:", result)
    print("measured from FK:", fk(result["q"])[:3, 3])


if __name__ == "__main__":
    main()
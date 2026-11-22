import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class IKResult(TypedDict):
    success: bool
    q: NDArray[np.float64]
    error: float
    iterations: int

ROOT = Path(__file__).resolve().parent
URDF = ROOT / "models/so100_image_zero.urdf"
KINEMATICS_MODEL_ID = "so100_image_zero_v1"
ARM_NAMES =("shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll")
    
#入力：RPY角、出力：回転行列(3,3)
#固定姿勢を計算
def rpy_matrix(rpy):
    r,p,y = rpy
    cr , sr = np.cos(r), np.sin(r)
    cp , sp = np.cos(p), np.sin(p)
    cy , sy = np.cos(y), np.sin(y)

    rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return rz@ry@rx

#入力：回転軸(3,)、角度float　、出力：回転行列(3,3)
#関節の回転を計算
def axis_rotation(axis,theta):
    a = np.asarray(axis,dtype=float)
    a /= np.linalg.norm(a)
    x,y,z = a
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (np.eye(3)*np.cos(theta) + (1-np.cos(theta))*np.outer(a,a)+np.sin(theta)*K)

def as_xyz(s):
    values = np.array([float(value) for value in s.split()])
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise  ValueError(f"invalid xyz vector: {s}")
    return values

@dataclass
class Joint:
    name: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float

class ArmKinematics:
    def __init__(self,urdf_path=URDF,joint_names=ARM_NAMES,base_link="base_link",tcp_xyz=None):
        self.urdf_path = Path(urdf_path)
        self.joint_names = tuple(joint_names)
        self.base_link = base_link
        if not self.joint_names:
            raise ValueError("joint_names must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names): #setで重複する要素が除外される
            raise ValueError("joint_names must not contain duplicates")
        self.joints = self.load_joints()
        # Default to the URDF's fixed jaw tip. Explicit offsets remain relative
        # to the last actuated link, for callers needing a different tool.
        self.tcp_transform = self.load_tcp_transform() if tcp_xyz is None else np.eye(4)
        if tcp_xyz is not None:
            self.tcp_transform[:3, 3] = self.validate_vector(tcp_xyz,3,"tcp_xyz")
        self.tcp_xyz = self.tcp_transform[:3, 3].copy()
        self.dof = len(self.joints)
        self.q_min = np.array([joint.lower for joint in self.joints])
        self.q_max = np.array([joint.upper for joint in self.joints])
        self.fixed_tarnsforms = []
        for joint in self.joints:
            fixed = np.eye(4)
            fixed[:3,:3] = rpy_matrix(joint.rpy)
            fixed[:3,3] = joint.xyz
            self.fixed_tarnsforms.append(fixed)

    @staticmethod
    def validate_vector(value,size,name):
        vector = np.array(value,dtype=float,copy=True)
        if vector.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values") 

        return vector

    def validate_q(self,q):
        return self.validate_vector(q,self.dof,"q")

    @staticmethod
    def joint_link(joint: ET.Element, tag: str) -> str:
        element = joint.find(tag)
        if element is None:
            raise ValueError(f"missing {tag} at joint: {joint.get('name')}")
        link = element.get("link")
        if not link:
            raise ValueError(f"missing {tag} link at joint: {joint.get('name')}")
        return link

    def load_tcp_transform(self):
        """Resolve tool_tip through fixed joints from the last arm link."""
        root = ET.parse(self.urdf_path).getroot()
        joints = root.findall("joint")
        last_joint = next(j for j in joints if j.get("name") == self.joint_names[-1])
        arm_tip = self.joint_link(last_joint, "child")
        by_child = {self.joint_link(j, "child"): j for j in joints}
        link = "tool_tip"
        transform = np.eye(4)
        visited = set()
        while link != arm_tip:
            if link in visited or link not in by_child:
                raise ValueError("tool_tip must be connected to the last arm link by fixed joints")
            visited.add(link)
            joint = by_child[link]
            if joint.get("type") != "fixed":
                raise ValueError("tool_tip path must contain only fixed joints")
            origin = joint.find("origin")
            fixed = np.eye(4)
            if origin is not None:
                fixed[:3, 3] = as_xyz(origin.get("xyz", "0 0 0"))
                fixed[:3, :3] = rpy_matrix(as_xyz(origin.get("rpy", "0 0 0")))
            transform = fixed @ transform
            link = self.joint_link(joint, "parent")
        return transform
    
    #入力：URDFのファイルパス、出力：関節情報のリスト
    #FK用の情報を取得
    def load_joints(self):
        root = ET.parse(self.urdf_path).getroot()
        by_name = {j.attrib["name"]: j for j in root.findall("joint")}
        joints = []
        last_child = self.base_link

        for name in self.joint_names:
            if name not in by_name:
                raise ValueError(f"joint not found in URDF: {name}")
            
            j = by_name[name]
            if j.get("type") != "revolute":
                raise ValueError( f"joint must be revolute: {name}")
            
            parent_element = j.find("parent")
            child_element = j.find("child")
            if parent_element is None or child_element is None:
                raise ValueError(f"missing parent or child at joint: {name}")

            parent = parent_element.attrib["link"]
            child = child_element.attrib["link"]
            if parent != last_child:
                raise ValueError(f"joint chain is broken at {name}")
            origin = j.find("origin")
            if origin is None:
                xyz = np.zeros(3)
                rpy = np.zeros(3)
            else:
                xyz = as_xyz(origin.get("xyz", "0 0 0"))
                rpy = as_xyz(origin.get("rpy", "0 0 0"))
            axis_element = j.find("axis")
            if axis_element is None:
                axis = np.array([1.0,0.0,0.0])
            else:
                axis = as_xyz(axis_element.get("xyz","1 0 0"))
            
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                raise ValueError(f"zero rotation axis at joint: {name}")
            axis /= axis_norm
            limit = j.find("limit")
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise ValueError(f"missing joint limits at joint:{name}")
            
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])

            if (not np.isfinite(lower) or not np.isfinite(upper) or lower > upper):
                raise ValueError(f"invalid joint limits at joint: {name}")
            joints.append(Joint(
                    name=name,
                    xyz=xyz,
                    rpy=rpy,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                ))
            last_child = child
        return joints

    #入力：関節角度(5,)、TCP位置(3,)　、　出力：同時変換行列(4,4)
    def fk(self,q):
        q = self.validate_q(q)
        transform = np.eye(4)
        for angle, joint, fixed in zip(q,self.joints,self.fixed_tarnsforms,):
            motion = np.eye(4)
            motion[:3, :3] = axis_rotation(joint.axis, angle)
            transform = transform @ fixed @ motion

        tcp = np.eye(4)
        tcp[:3, :3] = self.tcp_transform[:3, :3]
        tcp[:3, 3] = self.tcp_xyz

        return transform @ tcp

    def position(self,q):
        return self.fk(q)[:3,3]

    def jacobian(self,q,h=1e-6):
        q = self.validate_q(q)
        if h <= 0:
            raise ValueError("h must be positive")
        transform = np.eye(4)
        origins, axes = [], []
        for angle, joint, fixed in zip(q, self.joints, self.fixed_tarnsforms):
            transform = transform @ fixed
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ joint.axis)
            motion = np.eye(4)
            motion[:3, :3] = axis_rotation(joint.axis, angle)
            transform = transform @ motion
        tcp = transform[:3, 3] + transform[:3, :3] @ self.tcp_xyz
        return np.column_stack([np.cross(axis, tcp - origin)
                                for axis, origin in zip(axes, origins)])

#target:目標位置、seed:IKの初期関節角度、tol:位置誤差の許容値、max_iter:最大反復回数、damping:DLSの減衰係数、max_step:1回の香辛料のノルム上限
    def solve_position_IK(self,target, seed,tol=0.002, max_iter=300,damping=0.04, max_step=0.10) -> IKResult:
        target = self.validate_vector(target,3,"target")
        q = self.validate_vector(seed,self.dof,"seed")
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            raise ValueError("seed is outside joint limits")

        #isinstance:ある変数（オブジェクト）が、指定した型に属しているかを判定する
        if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer))
            or max_iter < 1):
            raise ValueError("max_iter must be a positive integer")

        for name, value in (("tol", tol),("damping", damping),("max_step", max_step),):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        
        iterations = 0

        for k in range(max_iter):
            p = self.position(q)
            e = target - p
            err = float(np.linalg.norm(e))
            if err < tol:
                return {"success": True, "q": q, "error": err,"iterations": k}

            J = self.jacobian(q)
            # A clipped unconstrained step can stall at a joint limit even
            # when the other joints can reach the target. Re-solve with
            # outward-moving joints at their limits excluded from this step.
            free = np.ones(self.dof, dtype=bool)
            dq = np.zeros(self.dof)
            for _ in range(self.dof + 1):
                active_J = J[:, free]
                A = active_J @ active_J.T + damping**2*np.eye(3)
                dq.fill(0.0)
                dq[free] = active_J.T @ np.linalg.solve(A, e)
                blocked = free & (
                    ((q <= self.q_min + 1e-10) & (dq < 0)) |
                    ((q >= self.q_max - 1e-10) & (dq > 0))
                )
                if not np.any(blocked):
                    break
                free[blocked] = False
            norm_dq = np.linalg.norm(dq)
            if norm_dq > max_step:
                dq *=max_step / norm_dq

            accepted = False
            iterations = k+1

            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = np.clip(q + scale*dq,self.q_min,self.q_max)
                trial_err = np.linalg.norm(target - self.fk(trial)[:3,3])
                if trial_err < err:
                    q = trial
                    accepted = True
                    break
            if not accepted:
                break

        err = float(np.linalg.norm(target - self.position(q)))
        return {"success": bool(err < tol), "q": q, "error": err,"iterations": iterations}


def main():
    arm = ArmKinematics(
        urdf_path=URDF,
        base_link="base_link",
        tcp_xyz=(0.0, 0.0, 0.0),
    )

    q = np.array([0.0, 1.5, -1.2, 0.0, 0.2])
    goal = arm.position(q)

    seed = np.array([0.0, 1.70, -1.57, -0.5, 0.0])

    result = arm.solve_position_IK(
        target=goal,
        seed=seed,
    )

    print("goal [m]:", goal)
    print("result:", result)
    print("position from FK [m]:", arm.position(result["q"]))


if __name__ == "__main__":
    main()

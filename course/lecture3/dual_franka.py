import polyscope as ps
import polyscope.imgui as psim
import torch
from robofab.robot import load_franka_robot
from robofab.render import draw_robot
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytorch_kinematics as pk

device = 'cpu'

def visualize_robot(franka_chain, q):
    franka_mesh = draw_robot("franka", franka_chain, q)
    T = np.eye(4)
    T[:3, 3] = np.array([1.0, 0, 0])
    T[:3, :3] = R.from_euler("XYZ", [0, 0, np.pi], degrees=False).as_matrix()
    franka_mesh = franka_mesh.apply_transform(T)
    ps.register_surface_mesh("franka2", franka_mesh.vertices, franka_mesh.faces, color=(1, 1, 1, 1))

def inverse(ik, ee_frame, ee_offset):

    goal_frame = ee_frame.clone()
    goal_frame[:, :3, 3] += ee_offset
    rot = R.from_matrix(ee_frame[:, :3, :3].cpu().numpy()).as_euler("xyz", degrees=False)
    goal = pk.Transform3d(pos = goal_frame[:, :3, 3], rot = rot)
    # solve IK
    sol = ik.solve(goal)
    if sol.converged:
        q = sol.solutions[0, 0, :]
    else:
        q = ik.initial_config.clone()
    q = torch.hstack([q, torch.tensor([0.04, 0.04])])
    return q

def move_l():
    global q, ee_frame, ee_offset, franka_chain, ik
    lb, ub = -0.1, 0.1
    names = ["x", "y", "z"]
    for dim in range(ee_offset.shape[0]):
        changed, ee_offset[dim] = psim.SliderFloat(names[dim], ee_offset[dim].item(), lb, ub)
        if changed:
            q = inverse(ik, ee_frame, ee_offset)
            visualize_robot(franka_chain, q)

if __name__ == "__main__":

    # Load a robot from file
    franka_chain = load_franka_robot().to(device=device)
    tcp_chain = pk.SerialChain(franka_chain, "fr3_hand_tcp").to(device=device)
    lim = torch.tensor(tcp_chain.get_joint_limits(), device=device)
    ik = pk.PseudoInverseIK(tcp_chain, max_iterations=30, num_retries=1,
                            joint_limits=lim.T,
                            early_stopping_any_converged=True,
                            early_stopping_no_improvement="all",
                            debug=False,
                            lr=0.2)

    # init joint angles
    q = torch.tensor([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0., 0.04, 0.04], dtype=torch.float32)
    ee_frame = tcp_chain.forward_kinematics(q[:7]).get_matrix()
    ee_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    ik.initial_config = q[:7]

    # init rendering system
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")

    q = inverse(ik, ee_frame, ee_offset)
    visualize_robot(franka_chain, q)

    # infinite loop to show the robot meshes
    ps.set_user_callback(move_l)
    ps.show()




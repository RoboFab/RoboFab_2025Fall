import numpy as np
import scipy.spatial.transform
import trimesh
from scipy.spatial.transform import Rotation as R
import polyscope as ps
import polyscope.imgui as psim
from trimesh import Trimesh
from robofab import ROBOFAB_DATA_DIR
from robofab.render import draw_frames

def compute_robot_frames(joint_xyz,
                         joint_rpy,
                         joint_axis,
                         joint_angles):
    frames = [np.eye(4)]
    for joint_id in range(joint_angles.shape[0]):
        rot = R.from_rotvec(joint_axis[joint_id] * joint_angles[joint_id], degrees=False)
        T_rot = np.eye(4)
        T_rot[:3, :3] = rot.as_matrix()
        T_joint = np.eye(4)
        T_joint[:3, :3] = R.from_euler("XYZ", joint_rpy[joint_id]).as_matrix()
        T_joint[:3, 3] = joint_xyz[joint_id]
        T = T_joint @ T_rot
        frame = frames[-1] @ T
        frames.append(frame)
    return frames

def draw_mesh(link_meshes, frames):
    for link_id, link_mesh in enumerate(link_meshes):
        mesh = Trimesh(link_mesh.vertices, link_mesh.faces)
        mesh.apply_transform(frames[link_id])
        ps.register_surface_mesh(f"link_{link_id}", mesh.vertices, mesh.faces)

def user_interface_func():
    global joint_angles, joint_rpy, joint_axis, joint_xyz, joint_ubs, joint_lbs, link_meshes
    for joint_id in range(joint_angles.shape[0]):
        changed, joint_angles[joint_id] = psim.SliderFloat(label=f"Joint_{joint_id}",
                                                           v=joint_angles[joint_id],
                                                           v_min=joint_lbs[joint_id],
                                                           v_max=joint_ubs[joint_id])
        if changed:
            frames = compute_robot_frames(joint_xyz, joint_rpy, joint_axis, joint_angles)
            draw_frames("robot", frames)
            draw_mesh(link_meshes, frames)

if __name__ == "__main__":
    joint_xyz = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float64)
    joint_rpy = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    joint_axis = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float64)
    joint_angles = np.array([0, 0],dtype=np.float64)
    joint_lbs = np.array([-np.pi / 2, -np.pi / 2], dtype=np.float64)
    joint_ubs = np.array([np.pi / 2, np.pi / 2], dtype=np.float64)

    link_names = ["base_link.obj", "link1.obj", "link2.obj"]
    link_meshes = []
    for name in link_names:
        file_path = ROBOFAB_DATA_DIR + f"/robot/simple_robot/meshes/visual/{name}"
        link_meshes.append(trimesh.load(file_path))

    # init rendering system
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")

    # draw frames
    frames = compute_robot_frames(joint_xyz, joint_rpy, joint_axis, joint_angles)
    draw_frames("robot", frames)
    draw_mesh(link_meshes, frames)

    # infinite loop to show the robot meshes
    ps.set_user_callback(user_interface_func)
    ps.show()
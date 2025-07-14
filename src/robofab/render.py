import numpy as np
import polyscope as ps
import pytorch_kinematics as pk
import torch
from trimesh import Trimesh, Scene

def draw_robot(name: str, robot_chain: pk.SerialChain, joint_angle: torch.tensor, **args):
    draw_part = args.get("draw_part", False)
    links = robot_chain.get_links()
    ret = robot_chain.forward_kinematics(joint_angle)
    scene = Scene()
    link_id = 0
    for link in links:
        for visual in link.visuals:
            if visual.geom_param is not None:
                mesh = visual.geom_param[1]
                new_mesh = Trimesh(mesh.vertices, mesh.faces)
                T = ret[link.name].get_matrix().numpy().reshape(4, 4)
                new_mesh.apply_transform(T)
                scene.add_geometry(new_mesh)
                if draw_part:
                    ps.register_surface_mesh(f"{link.name}", vertices=new_mesh.vertices, faces=new_mesh.faces)
                link_id = link_id + 1

    scene_mesh = scene.to_mesh()
    if not draw_part:
        ps.register_surface_mesh(f"{name}", vertices=scene_mesh.vertices, faces=scene_mesh.faces, color=(1, 1, 1, 1))
    return scene_mesh

def draw_frames(name: str, frames: list[np.ndarray]):
    o = []
    x, y, z = [], [], []
    for frame in frames:
        x.append(frame[:3, 0])
        y.append(frame[:3, 1])
        z.append(frame[:3, 2])
        o.append(frame[:3, 3])
    o = np.vstack(o)
    x = np.vstack(x)
    y = np.vstack(y)
    z = np.vstack(z)
    frame = ps.register_point_cloud(f"{name}_frame", o, color=(0, 0, 0, 1))
    frame.set_radius(0.01, relative=False)
    frame.add_vector_quantity("x", x, enabled=True, color=(1, 0, 0, 1), radius=0.01, length=0.05)
    frame.add_vector_quantity("y", y, enabled=True, color=(0, 1, 0, 1), radius=0.01, length=0.05)
    frame.add_vector_quantity("z", z, enabled=True, color=(0, 0, 1, 1), radius=0.01, length=0.05)

def draw_link_frames(name: str, robot_chain: pk.SerialChain, joint_angle: torch.tensor):
    links = robot_chain.get_links()
    ret = robot_chain.forward_kinematics(joint_angle)

    frames = []
    for link in links:
        for visual in link.visuals:
            if visual.geom_param is not None:
                T = ret[link.name].get_matrix().numpy().reshape(4, 4)
                frames.append(T)

    draw_frames(name, frames)


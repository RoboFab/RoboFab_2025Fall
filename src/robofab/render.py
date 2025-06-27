import polyscope as ps
import pytorch_kinematics as pk
import torch
from trimesh import Trimesh, Scene

def draw_robot(name: str, robot_chain: pk.SerialChain, joint_angle: torch.tensor):
    links = robot_chain.get_links()
    ret = robot_chain.forward_kinematics(joint_angle)
    scene = Scene()
    for link in links:
        for visual in link.visuals:
            if visual.geom_param is not None:
                mesh = visual.geom_param[1]
                new_mesh = Trimesh(mesh.vertices, mesh.faces)
                T = ret[link.name].get_matrix().numpy().reshape(4, 4)
                new_mesh.apply_transform(T)
                scene.add_geometry(new_mesh)

    scene_mesh = scene.to_mesh()
    ps.register_surface_mesh(f"{name}", vertices=scene_mesh.vertices, faces=scene_mesh.faces, color=(1, 1, 1, 1))
    return scene_mesh
import pytorch_kinematics as pk
import trimesh
from robofab import ROBOFAB_DATA_DIR
def load_franka_robot(robot_id = 0, base_link_name = "base_link", ee_link_name = "fr3_hand_tcp"):
    urdf = ROBOFAB_DATA_DIR + f"/robot/franka/fr3_franka_{robot_id}.urdf"
    chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())

    for link in chain.get_links():
        for visual in link.visuals:
            if visual.geom_param is not None:
                mesh_file = ROBOFAB_DATA_DIR + f"/robot/franka/{visual.geom_param[0]}"
                mesh = trimesh.load_mesh(mesh_file)
                visual.geom_param = (visual.geom_param[0], mesh)

    return chain

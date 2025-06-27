import polyscope as ps
import polyscope.imgui as psim
import torch
from robofab.robot import load_franka_robot
from robofab.render import draw_robot

# Set robot's joint angles
def user_interface_func():
    global q, franka_chain
    lb, ub = franka_chain.get_joint_limits()
    for joint_id in range(franka_chain.n_joints):
        changed, q[joint_id] = psim.SliderFloat(f"Joint {joint_id}", q[joint_id].item(), lb[joint_id], ub[joint_id])
        if changed:
            draw_robot("franka", franka_chain, q)

# Load a robot from file
franka_chain = load_franka_robot()

# init joint angles
q = torch.tensor([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0., 0.04, 0.04], dtype=torch.float32)

# init rendering system
ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")

# visualize robot
draw_robot("franka", franka_chain, q)

# infinite loop to show the robot meshes
ps.set_user_callback(user_interface_func)
ps.show()




import polyscope as ps
import polyscope.imgui as psim
import torch
from robofab.robot import load_simple_robot
from robofab.render import draw_robot, draw_link_frames

# Set robot's joint angles
def user_interface_func():
    global q, simple_robot
    lb, ub = simple_robot.get_joint_limits()
    for joint_id in range(simple_robot.n_joints):
        changed, q[joint_id] = psim.SliderFloat(f"Joint {joint_id}", q[joint_id].item(), lb[joint_id], ub[joint_id])
        if changed:
            draw_robot("simple_robot", simple_robot, q, draw_part=True)
            draw_link_frames("simple_robot", simple_robot, q)

# Load a robot from file
simple_robot = load_simple_robot()

# init joint angles
q = torch.tensor([0.0, 0.0], dtype=torch.float32)

# init rendering system
ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")

# visualize robot
draw_robot("simple_robot", simple_robot, q, draw_part = True)
draw_link_frames("simple_robot", simple_robot, q)

# infinite loop to show the robot meshes
ps.set_user_callback(user_interface_func)
ps.show()
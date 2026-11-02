import rclpy
from mf_planner_diag import Collector

rclpy.init()
node = Collector()
for planner, name in (("orca", "block"), ("orca", "leave"), ("pedvo", "block"), ("pedvo", "leave"), ("sfm", "block")):
    err = node.set_planner(planner)
    print("set", planner, err or "ok", flush=True)
    episode = node.record(planner, 5, name, 0)
    print("---", planner, name, flush=True)
    for index, sample in enumerate(episode["samples"]):
        if index % 10:
            continue
        _cmd, robot, _yaw, ped, vel = sample
        speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
        print(f"t={index / 10:.1f} ped=({ped[0]:.2f},{ped[1]:.2f}) robot=({robot[0]:.2f},{robot[1]:.2f}) v={speed:.2f}", flush=True)
node.destroy_node()
rclpy.shutdown()

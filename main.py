import taichi as ti

from mpm_solver import MPMSolver

ti.init(arch=ti.gpu)

# 坐标系统：      +y
#               |
#               |
#               |
#               |
#               |_______________ +x
#              /
#             /
#            /
#           /
#          /
#        +z
grid_num = 128
surface_grid_num = 80
quality = 1
particle_num = 30000

write_ply = 1

mpm_solver = MPMSolver(particle_num, surface_grid_num=surface_grid_num, grid_num=grid_num)
# # 将三角面片信息给碰撞检测算法，并初始化。
# # 将流体表面所用到的marching cube初始化
mpm_solver.init_surface()

mpm_solver.add_cube(ti.Vector([0.35, 0.5, 0.35]), 0.23, particle_num, 0)

frame_id = 0

while frame_id < 500:
    # while frame_id < 500:
    frame_id += 1
    print(frame_id)
    mpm_solver.run(frame_id, write_ply)
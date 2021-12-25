import taichi as ti
import numpy as np


@ti.data_oriented
class FluidSurface:
    def __init__(self,
                 grid_num,
                 particle_type,
                 radius):
        self.grid_num = grid_num
        self.particle_type = particle_type
        self.radius = radius
        self.dx = 1 / (grid_num - 1)
        self.inv_dx = 1 / self.dx
        self.sign_distance_field = ti.field(ti.f32, shape=(self.grid_num,) * 3)
        self.gradient = ti.Vector.field(3, ti.f32, shape=(self.grid_num,) * 3)
        self.divergence = ti.field(ti.f32, shape=(self.grid_num,) * 3)
        self.laplacian = ti.field(ti.f32, shape=(self.grid_num,) * 3)
        # 绘制用
        # self.color_list = ti.Vector.field(3, ti.f32, shape=self.grid_num ** 3)
        # self.node_position = ti.Vector.field(3, ti.f32, shape=self.grid_num ** 3)
        self.node_position_field = ti.Vector.field(3, ti.f32, shape=(self.grid_num,) * 3)
        self._edge_table = np.array([
            0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
            0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
            0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
            0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
            0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
            0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
            0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
            0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
            0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
            0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
            0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
            0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
            0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
            0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
            0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
            0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
            0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
            0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
            0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
            0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
            0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
            0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
            0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
            0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
            0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
            0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
            0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
            0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
            0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
            0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
            0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
            0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0])
        self._triangle_table = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                                         [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
                                         [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
                                         [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                                         [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
                                         [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
                                         [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
                                         [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
                                         [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
                                         [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
                                         [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                                         [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
                                         [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
                                         [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
                                         [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
                                         [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
                                         [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
                                         [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                                         [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
                                         [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
                                         [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
                                         [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
                                         [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
                                         [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
                                         [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
                                         [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
                                         [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
                                         [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
                                         [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
                                         [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
                                         [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                                         [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
                                         [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
                                         [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
                                         [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
                                         [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
                                         [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
                                         [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
                                         [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
                                         [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
                                         [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                                         [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
                                         [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
                                         [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
                                         [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
                                         [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
                                         [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
                                         [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
                                         [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
                                         [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
                                         [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
                                         [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
                                         [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
                                         [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
                                         [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
                                         [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
                                         [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
                                         [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
                                         [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
                                         [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
                                         [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
                                         [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
                                         [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
                                         [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
                                         [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
                                         [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
                                         [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
                                         [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
                                         [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
                                         [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
                                         [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
                                         [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
                                         [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
                                         [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
                                         [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
                                         [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
                                         [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
                                         [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
                                         [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
                                         [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
                                         [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
                                         [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
                                         [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
                                         [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
                                         [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
                                         [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
                                         [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
                                         [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
                                         [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
                                         [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
                                         [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                                         [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
                                         [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
                                         [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
                                         [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
                                         [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
                                         [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
                                         [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
                                         [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
                                         [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
                                         [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
                                         [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
                                         [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
                                         [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
                                         [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
                                         [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
                                         [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
                                         [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
                                         [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
                                         [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
                                         [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
                                         [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
                                         [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
                                         [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
                                         [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
                                         [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]], np.int32)
        self.edge_table = ti.field(ti.i32)
        self.triangle_table = ti.field(ti.i32)

        self.explicit_triangles = ti.Vector.field(3, ti.f32, shape=self.grid_num ** 3)
        self.create_triangle_num = ti.field(ti.i32, shape=())
        self.discrete_num = 3

        self.surface_particle_num = ti.field(ti.i32, shape=())
        self.surface_particles = ti.Struct.field({
            "position": ti.types.vector(3, ti.f32),
        }, shape=80000)

    def init_field(self):
        ti.root.dense(ti.i, 256).place(self.edge_table)
        ti.root.dense(ti.ij, self._triangle_table.shape).place(self.triangle_table)

    def numpy_to_field(self):
        self.edge_table.from_numpy(self._edge_table)
        self.triangle_table.from_numpy(self._triangle_table)

    # 每帧开始，将构建的表面粒子删除
    @ti.kernel
    def init_surface_particles(self):
        for i in range(self.surface_particle_num[None]):
            self.surface_particles[i].position = ti.Vector([0.0, 0.0, 0.0])
        self.surface_particle_num[None] = 0

    @ti.func
    def create_particle(self, pos):
        n = ti.atomic_add(self.surface_particle_num[None], 1)
        self.surface_particles[n].position = pos

    @ti.func
    def discrete_triangle(self, A, B, C):
        ab = B - A
        ac = C - A
        for i, j in ti.ndrange(self.discrete_num, self.discrete_num):
            if i + j < self.discrete_num + 1:
                pos = A + ((i / self.discrete_num) * ab + (j / self.discrete_num) * ac)
                self.create_particle(pos)

    # 初始化level set
    # union每个粒子的球形level set，求出每个网格顶点的SDF
    @ti.kernel
    def create_level_set(self, position: ti.template(), material: ti.template(), create_particle_num: int):
        for I in ti.grouped(self.sign_distance_field):
            node_pos = I * self.dx
            self.node_position_field[I] = node_pos
            min_dis = 10.0
            for p in range(create_particle_num):
                if material[p] == self.particle_type:
                    distance = (position[p] - node_pos).norm() - self.radius
                    if distance < min_dis:
                        min_dis = distance
            self.sign_distance_field[I] = min_dis

    # 传入连接当前边的两个顶点，以及两个顶点上的值，找到三角形顶点的位置
    @ti.func
    def get_point_position(self, position1, position2, val1, val2):
        offset = -val1 / (val2 - val1)
        result = position1 + offset * (position2 - position1)

        if abs(val1) < 0.00001:
            result = position1
        if abs(val2) < 0.00001:
            result = position2
        if abs(val1 - val2) < 0.00001:
            result = position1
        return result

    @ti.func
    def edge_position(self, edge, i, j, k):
        result = ti.Vector([0.0, 0.0, 0.0])
        if edge == 0:
            result = self.get_point_position(self.node_position_field[i, j, k],
                                             self.node_position_field[i + 1, j, k],
                                             self.sign_distance_field[i, j, k],
                                             self.sign_distance_field[i + 1, j, k])
        if edge == 1:
            result = self.get_point_position(self.node_position_field[i + 1, j, k],
                                             self.node_position_field[i + 1, j, k + 1],
                                             self.sign_distance_field[i + 1, j, k],
                                             self.sign_distance_field[i + 1, j, k + 1])
        if edge == 2:
            result = self.get_point_position(self.node_position_field[i + 1, j, k + 1],
                                             self.node_position_field[i, j, k + 1],
                                             self.sign_distance_field[i + 1, j, k + 1],
                                             self.sign_distance_field[i, j, k + 1])
        if edge == 3:
            result = self.get_point_position(self.node_position_field[i, j, k + 1],
                                             self.node_position_field[i, j, k],
                                             self.sign_distance_field[i, j, k + 1],
                                             self.sign_distance_field[i, j, k])
        if edge == 4:
            result = self.get_point_position(self.node_position_field[i, j + 1, k],
                                             self.node_position_field[i + 1, j + 1, k],
                                             self.sign_distance_field[i, j + 1, k],
                                             self.sign_distance_field[i + 1, j + 1, k])
        if edge == 5:
            result = self.get_point_position(self.node_position_field[i + 1, j + 1, k],
                                             self.node_position_field[i + 1, j + 1, k + 1],
                                             self.sign_distance_field[i + 1, j + 1, k],
                                             self.sign_distance_field[i + 1, j + 1, k + 1])
        if edge == 6:
            result = self.get_point_position(self.node_position_field[i + 1, j + 1, k + 1],
                                             self.node_position_field[i, j + 1, k + 1],
                                             self.sign_distance_field[i + 1, j + 1, k + 1],
                                             self.sign_distance_field[i, j + 1, k + 1])
        if edge == 7:
            result = self.get_point_position(self.node_position_field[i, j + 1, k + 1],
                                             self.node_position_field[i, j + 1, k],
                                             self.sign_distance_field[i, j + 1, k + 1],
                                             self.sign_distance_field[i, j + 1, k])
        if edge == 8:
            result = self.get_point_position(self.node_position_field[i, j, k],
                                             self.node_position_field[i, j + 1, k],
                                             self.sign_distance_field[i, j, k],
                                             self.sign_distance_field[i, j + 1, k])
        if edge == 9:
            result = self.get_point_position(self.node_position_field[i + 1, j, k],
                                             self.node_position_field[i + 1, j + 1, k],
                                             self.sign_distance_field[i + 1, j, k],
                                             self.sign_distance_field[i + 1, j + 1, k])
        if edge == 10:
            result = self.get_point_position(self.node_position_field[i + 1, j, k + 1],
                                             self.node_position_field[i + 1, j + 1, k + 1],
                                             self.sign_distance_field[i + 1, j, k + 1],
                                             self.sign_distance_field[i + 1, j + 1, k + 1])
        if edge == 11:
            result = self.get_point_position(self.node_position_field[i, j, k + 1],
                                             self.node_position_field[i, j + 1, k + 1],
                                             self.sign_distance_field[i, j, k + 1],
                                             self.sign_distance_field[i, j + 1, k + 1])
        return result

    # 将隐式Level Set转化为显示Marching Cube
    @ti.kernel
    def implicit_to_explicit(self):
        self.create_triangle_num[None] = 0
        for i, j, k in ti.ndrange(self.grid_num - 1, self.grid_num - 1, self.grid_num - 1):
            id = 0
            if self.sign_distance_field[i, j, k] < 0:
                id |= 1
            if self.sign_distance_field[i + 1, j, k] < 0:
                id |= 2
            if self.sign_distance_field[i + 1, j, k + 1] < 0:
                id |= 4
            if self.sign_distance_field[i, j, k + 1] < 0:
                id |= 8
            if self.sign_distance_field[i, j + 1, k] < 0:
                id |= 16
            if self.sign_distance_field[i + 1, j + 1, k] < 0:
                id |= 32
            if self.sign_distance_field[i + 1, j + 1, k + 1] < 0:
                id |= 64
            if self.sign_distance_field[i, j + 1, k + 1] < 0:
                id |= 128
            for t in range(4):
                if self.triangle_table[id, t * 3] != -1:
                    n = ti.atomic_add(self.create_triangle_num[None], 1)
                    self.explicit_triangles[n * 3] = self.edge_position(self.triangle_table[id, t * 3], i, j, k)
                    self.explicit_triangles[n * 3 + 1] = self.edge_position(self.triangle_table[id, t * 3 + 1], i, j, k)
                    self.explicit_triangles[n * 3 + 2] = self.edge_position(self.triangle_table[id, t * 3 + 2], i, j, k)

    # 将Marching Cube得到的三角形离散为表面粒子
    @ti.kernel
    def discrete_triangles(self):
        self.surface_particle_num[None] = 0
        for n in range(self.create_triangle_num[None]):
            self.discrete_triangle(
                self.explicit_triangles[n * 3],
                self.explicit_triangles[n * 3 + 1],
                self.explicit_triangles[n * 3 + 2])

    # 计算梯度算子（法线）
    @ti.kernel
    def calculate_gradient(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j, k = I
            u, v, w = .0, .0, .0
            # 判断边界条件
            if i == 0:
                u = (self.sign_distance_field[i + 1, j, k] - self.sign_distance_field[i, j, k]) * 0.5 * self.inv_dx
            elif i == self.grid_num - 1:
                u = (self.sign_distance_field[i, j, k] - self.sign_distance_field[i - 1, j, k]) * 0.5 * self.inv_dx
            else:
                u = (self.sign_distance_field[i + 1, j, k] - self.sign_distance_field[i - 1, j, k]) * 0.5 * self.inv_dx

            if j == 0:
                v = (self.sign_distance_field[i, j + 1, k] - self.sign_distance_field[i, j, k]) * 0.5 * self.inv_dx
            elif j == self.grid_num - 1:
                v = (self.sign_distance_field[i, j, k] - self.sign_distance_field[i, j - 1, k]) * 0.5 * self.inv_dx
            else:
                v = (self.sign_distance_field[i, j + 1, k] - self.sign_distance_field[i, j - 1, k]) * 0.5 * self.inv_dx

            if k == 0:
                w = (self.sign_distance_field[i, j, k + 1] - self.sign_distance_field[i, j, k]) * 0.5 * self.inv_dx
            elif k == self.grid_num - 1:
                w = (self.sign_distance_field[i, j, k] - self.sign_distance_field[i, j, k - 1]) * 0.5 * self.inv_dx
            else:
                w = (self.sign_distance_field[i, j, k + 1] - self.sign_distance_field[i, j, k - 1]) * 0.5 * self.inv_dx
            self.gradient[I] = ti.Vector([u, v, w]).normalized()

    # 计算拉普拉斯算子（曲率）
    @ti.kernel
    def calculate_laplacian(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j, k = I
            u, v, w = .0, .0, .0
            if i == 0:
                u = (self.sign_distance_field[i + 1, j, k] - self.sign_distance_field[
                    i, j, k]) * self.inv_dx * self.inv_dx
            elif i == self.grid_num - 1:
                u = (-self.sign_distance_field[i, j, k] + self.sign_distance_field[
                    i - 1, j, k]) * self.inv_dx * self.inv_dx
            else:
                u = (self.sign_distance_field[i + 1, j, k] - 2 * self.sign_distance_field[i, j, k] +
                     self.sign_distance_field[i - 1, j, k]) * self.inv_dx * self.inv_dx

            if j == 0:
                v = (self.sign_distance_field[i, j + 1, k] - self.sign_distance_field[
                    i, j, k]) * self.inv_dx * self.inv_dx
            elif j == self.grid_num - 1:
                v = (-self.sign_distance_field[i, j, k] + self.sign_distance_field[
                    i, j - 1, k]) * self.inv_dx * self.inv_dx
            else:
                v = (self.sign_distance_field[i, j + 1, k] - 2 * self.sign_distance_field[i, j, k] +
                     self.sign_distance_field[i, j - 1, k]) * self.inv_dx * self.inv_dx

            if k == 0:
                w = (self.sign_distance_field[i, j, k + 1] - self.sign_distance_field[
                    i, j, k]) * self.inv_dx * self.inv_dx
            elif k == self.grid_num - 1:
                w = (-self.sign_distance_field[i, j, k] + self.sign_distance_field[
                    i, j, k - 1]) * self.inv_dx * self.inv_dx
            else:
                w = (self.sign_distance_field[i, j, k + 1] - 2 * self.sign_distance_field[i, j, k] +
                     self.sign_distance_field[i, j, k - 1]) * self.inv_dx * self.inv_dx
            self.laplacian[I] = u + v + w

    # 三次线性插值函数
    @ti.func
    def linear_interpolation_sdf(self, pos: ti.template()):
        base = (pos * self.inv_dx).cast(int)

        fx = pos * self.inv_dx - base.cast(float)
        w = [(1 - fx) * self.dx, fx * self.dx]
        result = 0.0
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            weight = w[i][0] * w[j][1] * w[k][2] * self.inv_dx * self.inv_dx * self.inv_dx
            offset = [i, j, k]
            result += self.sign_distance_field[base + offset] * weight
        return result

    @ti.func
    def linear_interpolation_normal(self, pos: ti.template()):
        base = (pos * self.inv_dx).cast(int)
        fx = pos * self.inv_dx - base.cast(float)
        w = [(1 - fx) * self.dx, fx * self.dx]
        result = ti.Vector([0.0, 0.0, 0.0])
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            weight = w[i][0] * w[j][1] * w[k][2] * self.inv_dx * self.inv_dx * self.inv_dx
            offset = [i, j, k]
            result += self.gradient[base + offset] * weight
        return result

    @ti.func
    def linear_interpolation_curvature(self, pos: ti.template()):
        base = (pos * self.inv_dx).cast(int)

        fx = pos * self.inv_dx - base.cast(float)
        w = [(1 - fx) * self.dx, fx * self.dx]
        result = 0.0
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            weight = w[i][0] * w[j][1] * w[k][2] * self.inv_dx * self.inv_dx * self.inv_dx
            offset = [i, j, k]
            result += self.laplacian[base + offset] * weight
        return result

import taichi as ti

from fluid_surface import FluidSurface


@ti.data_oriented
class MPMSolver:
    material_water = 0
    material_solid = 1
    material_snow = 2

    mark_solid = 1
    mark_contact = 2
    mark_fluid = 3
    mark_air = 0

    def __init__(self,
                 max_particle_num,
                 grid_num,
                 surface_grid_num
                 ):
        self.surface_grid_num = surface_grid_num
        self.max_particle_num = max_particle_num
        self.grid_num = grid_num
        self.dx = 1 / self.grid_num
        self.surface_dx = 1 / self.surface_grid_num
        self.inv_dx = float(self.grid_num)
        self.dt = 1e-4
        self.steps = 32
        self.p_vol = (self.dx * 0.5) ** 2
        self.bound = 3
        self.E = 1000
        self.nu = 0.2
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        self.particles = ti.Struct.field({
            "position": ti.types.vector(3, ti.f32),
            "velocity": ti.types.vector(3, ti.f32),
            "F": ti.types.matrix(3, 3, ti.f32),
            "C": ti.types.matrix(3, 3, ti.f32),
            "Jp": ti.f32,
            "mass": ti.f32,
            "material": ti.i32,
            "color": ti.types.vector(3, ti.f32)
        }, shape=self.max_particle_num)

        self.node = ti.Struct.field({
            "node_m": ti.f32,
            "node_v": ti.types.vector(3, ti.f32),
            "tension": ti.types.vector(3, ti.f32),
        }, shape=(self.grid_num,) * 3)

        self.neighbour = (3,) * 3
        self.create_particle_num = ti.field(ti.i32, shape=())

        self.fluid_surface_solver = FluidSurface(grid_num=self.surface_grid_num, particle_type=self.material_water,
                                                 radius=self.surface_dx * 0.8)
        self.tension_coefficient = 0.07

    # 初始化碰撞检测类的顶点信息和顶点坐标信息
    def init_surface(self):
        # 将numpy数组转为field
        self.fluid_surface_solver.init_field()
        # 初始化之后的赋值
        self.fluid_surface_solver.numpy_to_field()

    @ti.kernel
    def reset_node(self):
        for I in ti.grouped(self.node):
            self.node[I].node_v = ti.zero(self.node[I].node_v)
            self.node[I].node_m = 0

    # 将网格节点的表面张力映射给流体粒子。
    @ti.kernel
    def add_tension_to_particle(self):
        for p in range(self.create_particle_num[None]):
            base = (self.particles[p].position * self.inv_dx - 0.5).cast(int)
            fx = self.particles[p].position * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                weight = 1.0
                for i in ti.static(range(3)):
                    weight *= w[offset[i]][i]
                tension = self.node[base + offset].tension
                self.particles[p].velocity += weight * tension

    @ti.kernel
    def P2G(self):
        for p in range(self.create_particle_num[None]):
            Xp = self.particles[p].position / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.particles[p].F = (ti.Matrix.identity(float, 3) + self.dt * self.particles[p].C) @ self.particles[p].F

            h = ti.exp(10 * (1.0 - self.particles[p].Jp))  # Hardening coefficient: snow gets harder when compressed
            if self.particles[p].material == self.material_solid:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.particles[p].material == self.material_water:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.particles[p].F)
            J = 1.0
            for d in ti.static(range(3)):
                new_sig = sig[d, d]
                if self.particles[p].material == self.material_snow:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                  1 + 4.5e-3)  # Plasticity
                self.particles[p].Jp *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.particles[p].material == self.material_water:
                new_F = ti.Matrix.identity(float, 3)
                new_F[0, 0] = J
                self.particles[p].F = new_F
            elif self.particles[p].material == self.material_snow:
                self.particles[
                    p].F = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (self.particles[p].F - U @ V.transpose()) @ self.particles[p].F.transpose(
            ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4) * stress / self.dx ** 2
            affine = stress + self.particles[p].mass * self.particles[p].C

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(3)):
                    weight *= w[offset[i]][i]
                self.node[base + offset].node_v += weight * (
                        self.particles[p].mass * self.particles[p].velocity + affine @ dpos)
                self.node[base + offset].node_m += weight * self.particles[p].mass

    @ti.kernel
    def grid_operator(self):
        for I in ti.grouped(self.node):
            if self.node[I].node_m > 0:
                self.node[I].node_v /= self.node[I].node_m
            self.node[I].node_v += self.dt * ti.Vector([0.0, -9.8, 0.0])
            cond = I < self.bound and self.node[I].node_v < 0 or I > self.grid_num - self.bound and self.node[
                I].node_v > 0
            self.node[I].node_v = 0 if cond else self.node[I].node_v

    @ti.kernel
    def G2P(self):
        for p in range(self.create_particle_num[None]):
            Xp = self.particles[p].position / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.zero(self.particles[p].velocity)
            new_C = ti.zero(self.particles[p].C)
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(3)):
                    weight *= w[offset[i]][i]
                g_v = self.node[base + offset].node_v
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx ** 2

            self.particles[p].velocity = new_v
            self.particles[p].position += self.dt * self.particles[p].velocity
            self.particles[p].C = new_C

    # 根据插值函数求出每个表面粒子处的表面张力带来的速度，然后映射到网格节点
    @ti.kernel
    def add_tension(self):
        for I in ti.grouped(self.node):
            self.node[I].tension = [0.0, 0.0, 0.0]
        for p in range(self.fluid_surface_solver.surface_particle_num[None]):
            Xp = self.fluid_surface_solver.surface_particles.position[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            mark_base = int(self.fluid_surface_solver.surface_particles.position[p] / self.surface_dx)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            normal = self.fluid_surface_solver.linear_interpolation_normal(
                self.fluid_surface_solver.surface_particles.position[p])
            curvature = self.fluid_surface_solver.linear_interpolation_curvature(
                self.fluid_surface_solver.surface_particles.position[p])
            tension = normal * curvature * self.tension_coefficient * self.dt
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                weight = 1.0
                for i in ti.static(range(3)):
                    weight *= w[offset[i]][i]
                self.node[base + offset].tension -= weight * tension

    @ti.kernel
    def add_cube(self, position: ti.template(), length: float, particle_num: int, material: int):
        rho = 1
        if material == self.material_solid:
            rho = 1
        for i in range(self.create_particle_num[None], self.create_particle_num[None] + particle_num):
            n = ti.atomic_add(self.create_particle_num[None], 1)
            self.particles[n].position = ti.Vector([ti.random() for i in range(3)]) * ti.Vector(
                [length, length, length]) + ti.Vector([position[0], position[1], position[2]])
            self.particles[n].material = material
            self.particles[n].velocity = ti.Vector([0.0, 0.0, 0.0])
            self.particles[n].F = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.particles[n].mass = self.p_vol * rho
            self.particles[n].Jp = 1
            self.particles[n].color = [1.0, 0.0, 0.0]

    def run(self, frame, write_ply):
        for s in range(32):
            self.fluid_surface_solver.init_surface_particles()
            self.fluid_surface_solver.create_level_set(self.particles.position, self.particles.material,
                                                       self.create_particle_num[None])
            self.fluid_surface_solver.calculate_gradient()
            self.fluid_surface_solver.calculate_laplacian()
            self.fluid_surface_solver.implicit_to_explicit()
            self.fluid_surface_solver.discrete_triangles()
            self.add_tension()
            self.reset_node()
            self.add_tension_to_particle()
            self.P2G()
            self.grid_operator()
            self.G2P()
        if write_ply:
            pos = self.particles.position.to_numpy()
            writer = ti.PLYWriter(num_vertices=len(pos))
            writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
            writer.export_frame(frame, 'D:\\homework' + '\\water.ply')

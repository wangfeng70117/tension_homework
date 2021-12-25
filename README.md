# 太极图形课S1-大作业-在MPM基础上添加流体表面张力

## 作业来源

流体在运动过程中很多显著的特征都是由表面张力的作用引起的，尤其是在微观的环境中表面张力的效应更加明显，比如水银落在地上保持球形，雨后露珠在植物叶子上的悬挂，有趣的红酒挂杯等现象等。很多论文都会在流体模拟时添加表面张力效果提高模拟的真实性。

近两年新出了两篇关于MPM添加表面张力的论文，和我的研究方向非常契合，简单的复现了一下，但是并不是完全按照论文里的思路写的。

参考资料：

[1]. An Implicit Updated Lagrangian Formulation for Liquids with Large Surface Energy. 

[2]. A Momentum-Conserving Implicit Material Point Method for Surface Energies with Spatial Gradients.

[3]. A Continuum Method for Modeling Surface Tension

[4]. Fluid Engine Development 

## 运行方式

#### 运行环境：

Windows10, Taichi 0.8.5, python 3.7.3

#### 运行：

运行main.py即可。

## 效果展示
![Image](https://github.com/wangfeng70117/tension_homework/blob/main/tension_result.gif?raw=true)


因为构建Level Set十分消耗性能，所以每帧导出的ply文件，然后用houdini渲染了一下。

## 整体结构

```
-README.md
-main.py
-mpm_solver.py
-fluid_surface.py
-tension_result.gif
```

## 实现细节：

### 整体流程

1. 初始化流体粒子。

2. 初始化表面粒子。

3. 创建SDF。

4. 计算SDF的梯度场、散度场。

5. 将隐式表面转化为显式Marching Cube表面。

6. 将通过Marching Cube得到的三角形离散为表面点。

7. 通过插值函数，在离散化得到的表面粒子处计算表面粒子处的表面张力。

8. 将计算得到的表面张力映射到MPM背景网格。

9. 将网格上的表面张力映射给流体最外层粒子。

   （最外层粒子和表面粒子并不是同一种，表面粒子只根据其位置计算张力，并不直接参与到MPM系统，而最外层粒子是MPM流体最外层粒子。）

### 理论的简要介绍

1. Sign Distance field（SDF）

   在我眼中SDF和Level Set是差不多的东西，用来构建隐式表面，所谓的隐式曲面就是根据某个表达式来确定一个曲面，比如，圆形的隐式曲面就是：![c1](https://user-images.githubusercontent.com/46859758/147384020-445dad04-71c8-46e3-8395-72dc8deee71f.png)



   那么空间中任意一点(x, y)的SDF的值就是:![c2](https://user-images.githubusercontent.com/46859758/147384022-92dce984-578f-40cf-a73a-6ec22fe12105.png)




如下图所示：
![sdf1](https://user-images.githubusercontent.com/46859758/147383663-a71532ce-2fe8-4aa2-bbf3-0c024f911be8.png)



这样，空间中所有SDF为0的点就构成了一个曲面，这种曲面就是隐式曲面。

那么给定表面上任意一点，这一点的法向量为
$$
n=\frac{\nabla f(x)}{|\nabla f(x)|}
$$
为什么法线方向可以这样表示呢？因为梯度方向就是标量场数值变化最快的方向，SDF中每一点的值都是距离表面的最小值，从物理意义上理解，距离最小就是垂直于表面，恰好是梯度方向就是法线方向。

那么任意一点的曲率，只需要对SDF求拉普拉斯算子就可以了。

隐式曲面也是可以合并的，如下图：
![sdf2](https://user-images.githubusercontent.com/46859758/147383670-8c4cc9ad-4afb-487b-90ed-a1bf71526ced.png)



2. Marching Cube

   Marching Cube是一种构建显式曲面的方法，我们将空间分割为一个个细小的网格单元，然后每个网格节点上都有网格节点的SDF的值（当然也可以用别的场，比如速度场），如果网格节点上的标量值大于等值面的值则标记的“1”,如果小于等值面的值就标记为“0”。

   如下图：
![mc1](https://user-images.githubusercontent.com/46859758/147383673-d98f341b-be2c-4d27-8baf-ef8aeb2bf886.png)
![mc2](https://user-images.githubusercontent.com/46859758/147383675-082e7177-4cd4-4e06-a4fb-3d2e63e7b8f1.png)


黑点处的标记为“1”，然后就根据网格中黑点处的标记，绘制显示表面。

（更加详细的讲解在我的作业分享直播课上有讲，直播地址：[太极图形课S1第08讲：弹性物体仿真 01_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1eY411x7mK?p=8)）

3. CSF表面张力计算模型

   根据CSF表面张力模型，流体表面上任意一点的表面张力计算公式为：
   $$
   tension= c·n·\kappa \\c为张力系数，n为法向量方向，\kappa 为曲率
   $$
   

### 代码细节

本作业代码主要分为MPM基础框架部分、流体表面构建部分以及表面张力计算部分。



main.py只作为程序入口，以及进行程序的初始化。

fluid_surface包括构建流体的显示曲面、隐式曲面以及两种曲面的相互转化。



首先是隐式曲面的构建：

```
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
```

因为不同的SDF是可以union的，所以我对每个流体粒子构建了球形Level Set，然后结合这些球形Level Set得到了一个流体的整体Level Set。

这是一种比较耗费性能的方法，但是我还没有找到别的办法去构建Level Set（论文里也是使用这种方式构建的流体Level Set）。



隐式曲面转化为显式曲面：

```
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
```

这一部分即Marching Cube算法的执行算法。遍历每个网格的每个网格节点，构建Marching Cube三角形表面。

这个方法执行完成后会得到很多的Marching Cube三角形。



计算SDF的梯度、散度：

```
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
```

因为使用的是中心差分求解，所以在边界处进行了简单的处理。



mpm_solver.py主要是进行MPM框架的流体仿真。



表面张力计算：

```
    # 根据插值函数求出每个表面粒子处的表面张力带来的速度，然后映射到网格节点
    @ti.kernel
    def add_tension(self):
        for I in ti.grouped(self.node):
            self.node[I].tension = [0.0, 0.0, 0.0]
        for p in range(self.fluid_surface_solver.surface_particle_num[None]):
            Xp = self.fluid_surface_solver.surface_particles.position[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
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
```

这一部分是表面张力的计算，通过差值函数，得到每个表面粒子处的法向量以及曲率，然后求解得到表面张力，映射到MPM背景网格。



```
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
```

这一部分是将MPM背景网格上记录的表面张力映射给流体的表面粒子。

（流体的表面粒子和离散化得到的表面粒子不是一个东西）

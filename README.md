# 太极图形课S1-大作业-在MPM基础上添加流体表面张力

## 作业来源

流体在运动过程中很多显著的特征都是由表面张力的作用引起的，尤其是在微观的环境中表面张力的效应更加明显，比如水银落在地上保持球形，雨后露珠在植物叶子上的悬挂，有趣的红酒挂杯等现象等。很多论文都会在流体模拟时添加表面张力效果提高模拟的真实性。

参考资料：

[1]. An Implicit Updated Lagrangian Formulation for Liquids with Large Surface Energy. 

[2]. A Momentum-Conserving Implicit Material Point Method for Surface Energies with Spatial Gradients.

[3]. A Continuum Method for Modeling Surface Tension

## 运行方式

#### 运行环境：

Windows10, Taichi 0.8.5, python 3.7.3

#### 运行：

运行main.py即可。

## 效果展示
[!Image]
https://github.com/wangfeng70117/tension_homework/blob/main/tension_result.gif
因为构建Level Set十分消耗性能，所以每帧导出的ply文件，然后用houdini渲染了一下。

## 整体结构

```
-README.md
-main.py
-mpm_solver.py
-fluid_surface.py
```

## 实现细节：

### 整体流程

1. 

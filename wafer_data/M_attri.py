import torch
import numpy as np

# jupyter引用py模块只能引用常量和类，所以这里写成类的形式

class Att():
    def __init__(self):
        # 属性向量对应的属性名，按顺序
        self.att_name = [
            # 形状相关属性
            'Solid_circle',  # 实心圆形
            'Dense',  # 构成缺陷的点是否密集，也就是每个缺陷点的四周是否也都是缺陷点
            'Cluster_without_center',  # 集群但不在中心
            'Localized_cluster',  # 局部集群
            'Thin_line_shape',  # 细线形状
            'Symmetric_to_rotation',  # 旋转后对称
            'located_near_an_edge',  # 边缘
            'Annular_shaped',  # 环形
            'Multiple_appearance',  # 多重外观
            'No_defect',  # 无缺陷模式
            'Over_90%_defective',  # 90%以上缺陷
            'Random_patterns',  # 随机缺陷模式
            # 原因相关属性
            'Chemical_mechanical_polishing',  # 化学机械抛光
            'Particles',  # 粒子
            'Human errors',  # 人为错误
            'Deposition',  # 沉积
            'Rapid_thermal_annealing',  # 快速热退火
            'Photo_lithography',  # 拍照
            'Lay-wise_misalignment',  # 横向错位
            'Uneven cleaning'  # 清洁不均匀
        ]
        self.single_defect_att = {
            'Center':torch.tensor([1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0], dtype=torch.float32),
            'Donut':torch.tensor([0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0], dtype=torch.float32),
            'Edge_loc':torch.tensor([0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1], dtype=torch.float32),
            'Loc':torch.tensor([0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,1], dtype=torch.float32),
            'Edge_ring':torch.tensor([0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0], dtype=torch.float32),
            'Scratch':torch.tensor([0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,], dtype=torch.float32),
            'Random':torch.tensor([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,], dtype=torch.float32),
            'Nearfull':torch.tensor([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,], dtype=torch.float32),
            'Normal':torch.tensor([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
        }

    # 手打多缺陷的属性太过麻烦，通过定义函数来解决
    def define_mul_defect_att(self, dlist):  # 参数是缺陷类型名称的一个列表，缺陷名称如single_defect_att中所示
        att =torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32)
        for d in dlist:
            i = 0
            for ele in self.single_defect_att[d]:
                if ele == 1:
                    att[i] = 1
                i = i+1
        return att
    
    def compute_mul_defect_att(self):

        self.two_defect_att = {
            'C+EL':self.define_mul_defect_att(['Center','Edge_loc']),
            'C+ER':self.define_mul_defect_att(['Center','Edge_ring']),
            'C+L':self.define_mul_defect_att(['Center','Loc']),
            'C+S':self.define_mul_defect_att(['Center','Scratch']),
            'D+EL':self.define_mul_defect_att(['Donut','Edge_loc']),
            'D+ER':self.define_mul_defect_att(['Donut','Edge_ring']),
            'D+L':self.define_mul_defect_att(['Donut','Loc']),
            'D+S':self.define_mul_defect_att(['Donut','Scratch']),
            'EL+L':self.define_mul_defect_att(['Edge_loc','Loc']),
            'EL+S':self.define_mul_defect_att(['Edge_loc','Scratch']),
            'ER+L':self.define_mul_defect_att(['Edge_ring','Loc']),
            'ER+S':self.define_mul_defect_att(['Edge_ring','Scratch']),
            'L+S':self.define_mul_defect_att(['Loc','Scratch'])
        }

        self.three_defect_att = {
            'C+EL+L':self.define_mul_defect_att(['Center','Edge_loc','Loc']),
            'C+EL+S':self.define_mul_defect_att(['Center','Edge_loc','Scratch']),
            'C+ER+L':self.define_mul_defect_att(['Center','Edge_ring','Loc']),
            'C+ER+S':self.define_mul_defect_att(['Center','Edge_ring','Scratch']),
            'C+L+S':self.define_mul_defect_att(['Center','Loc','Scratch']),
            'D+EL+L':self.define_mul_defect_att(['Donut','Edge_loc','Loc']),
            'D+EL+S':self.define_mul_defect_att(['Donut','Edge_loc','Scratch']),
            'D+ER+L':self.define_mul_defect_att(['Donut','Edge_ring','Loc']),
            'D+ER+S':self.define_mul_defect_att(['Donut','Edge_ring','Scratch']),
            'D+L+S':self.define_mul_defect_att(['Donut','Loc','Scratch']),
            'EL+L+S':self.define_mul_defect_att(['Edge_loc','Loc','Scratch']),
            'ER+L+S':self.define_mul_defect_att(['Edge_ring','Loc','Scratch'])
        }

        self.four_defect_att = {
            'C+EL+L+S':self.define_mul_defect_att(['Center','Loc','Edge_loc','Scratch']),
            'C+ER+L+S':self.define_mul_defect_att(['Center','Loc','Edge_ring','Scratch']),
            'D+EL+L+S':self.define_mul_defect_att(['Donut','Loc','Edge_loc','Scratch']),
            'D+ER+L+S':self.define_mul_defect_att(['Donut','Loc','Edge_ring','Scratch'])
        }

        self.mul_defect_att = { **self.two_defect_att, **self.three_defect_att, **self.four_defect_att}
        self.total_defect_att = {**self.single_defect_att, **self.two_defect_att, **self.three_defect_att, **self.four_defect_att}
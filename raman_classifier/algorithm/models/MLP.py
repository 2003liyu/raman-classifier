from typing import Tuple

import torch.nn as nn
import torch

from .BaseNet import BaseNet


class MLP(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        self.__n_inputs:int = kwargs["n_inputs"]
        self.__n_outputs:int = kwargs["n_outputs"]
        self.__hidden_sizes:Tuple[int] = kwargs["hidden_sizes"]
        self.model = None

        layers = []
        prev_size = self.__n_inputs

        for hidden_size in self.__hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.__n_outputs))
        self.model = nn.Sequential(*layers)
        self.init_weights()

    def __bool__(self)->bool:
        return (self.__n_inputs > 0 and self.__n_outputs > 0)

    @property
    def n_inputs(self)->int:
        return self.__n_inputs

    @property
    def n_outputs(self)->int:
        return self.__n_outputs

    @property
    def hidden_sizes(self)->Tuple[int]:
        return self.__hidden_sizes

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.model(x)

    @property
    def title(self)->str:
        return f"{self.base_name}{(self.n_inputs, *self.hidden_sizes, self.n_outputs)}"

#-------------------蒸馏+增强-------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# from .BaseNet import BaseNet
#
#
# # -------------------------------------------------------------
# # 1. Teacher MLP (结构比学生更宽，用于指导)
# # -------------------------------------------------------------
# class MLPTeacher(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         n_in = kwargs["n_inputs"]
#         n_out = kwargs["n_outputs"]
#         # 老师的隐藏层宽度是学生的 2 倍
#         hs = [h * 2 for h in kwargs["hidden_sizes"]]
#
#         layers = []
#         prev = n_in
#         for h in hs:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.ReLU())
#             prev = h
#         self.feature = nn.Sequential(*layers)
#         self.classifier = nn.Linear(prev, n_out)
#         self.init_weights()
#
#     def forward(self, x):
#         feats = self.feature(x)
#         return self.classifier(feats)
#
#
# # -------------------------------------------------------------
# # 2. Student MLP (带增强和蒸馏)
# # -------------------------------------------------------------
# class MLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.__n_inputs = kwargs["n_inputs"]
#         self.__n_outputs = kwargs["n_outputs"]
#         self.__hidden_sizes = kwargs["hidden_sizes"]
#
#         self.kd_temperature = 4.0
#         self.kd_alpha = 0.7
#
#         # 1. 构建学生网络特征提取层
#         layers = []
#         prev = self.__n_inputs
#         for h in self.__hidden_sizes:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.3))
#             prev = h
#         self.feature = nn.Sequential(*layers)
#
#         # 2. 分类器
#         self.classifier = nn.Linear(prev, self.__n_outputs)
#
#         # 3. 初始化老师网络 (不更新梯度)
#         self.teacher = MLPTeacher(**kwargs)
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#
#         self.init_weights()
#
#     def forward(self, x, labels=None, return_teacher=False, return_feats=False):
#         # ---------------------------------------------------------
#         # Train 模式：执行数据增强和准备蒸馏
#         # ---------------------------------------------------------
#         if self.training:
#             x_orig = x
#             device = x.device
#             batch_size = x.size(0)
#
#             # --- A. 类内数据增强 (Mixup) ---
#             lam = torch.distributions.Beta(2.0, 2.0).sample().to(device)
#             index = torch.arange(batch_size).to(device)
#
#             if labels is not None:
#                 for i in range(batch_size):
#                     # 寻找同类样本进行混合
#                     same_label_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
#                     if len(same_label_indices) > 1:
#                         shift = torch.randint(1, len(same_label_indices), (1,)).item()
#                         curr_pos = (same_label_indices == i).nonzero(as_tuple=True)[0].item()
#                         index[i] = same_label_indices[(curr_pos + shift) % len(same_label_indices)]
#
#             x_aug = lam * x + (1 - lam) * x[index]
#
#             # --- B. 前向传播 ---
#             x_all = torch.cat([x_orig, x_aug], dim=0)
#             feats_all = self.feature(x_all)
#             logits_all = self.classifier(feats_all)
#
#             logits_orig, logits_aug = torch.chunk(logits_all, 2, dim=0)
#             feats_orig, _ = torch.chunk(feats_all, 2, dim=0)
#
#             # --- C. 教师模型预测 (蒸馏目标) ---
#             with torch.no_grad():
#                 t_logits = self.teacher(x_orig)
#
#             # 返回 6 个参数，适配 NormalClassifier.py 的 compute_dual_path_loss
#             return logits_orig, logits_aug, feats_orig, t_logits, lam, index
#
#         # ---------------------------------------------------------
#         # Eval 模式：保持简单输出
#         # ---------------------------------------------------------
#         feats = self.feature(x)
#         logits = self.classifier(feats)
#
#         if return_teacher:
#             with torch.no_grad():
#                 t_logits = self.teacher(x)
#             return logits, feats, t_logits
#
#         if return_feats:
#             return logits, feats
#
#         return logits
#
#     @property
#     def n_inputs(self) -> int:
#         return self.__n_inputs
#
#     @property
#     def n_outputs(self) -> int:
#         return self.__n_outputs
#
#     @property
#     def title(self):
#         return f"MLP_Enhanced({self.__hidden_sizes})"

#-------------------蒸馏----------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# from .BaseNet import BaseNet
#
#
# # -------------------------------------------------------------
# # 1. Teacher MLP (结构比学生更宽，用于指导)
# # -------------------------------------------------------------
# class MLPTeacher(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         n_in = kwargs["n_inputs"]
#         n_out = kwargs["n_outputs"]
#         # 老师的隐藏层宽度是学生的 2 倍
#         hs = [h * 2 for h in kwargs["hidden_sizes"]]
#
#         layers = []
#         prev = n_in
#         for h in hs:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.ReLU())
#             prev = h
#         self.feature = nn.Sequential(*layers)
#         self.classifier = nn.Linear(prev, n_out)
#         self.init_weights()
#
#     def forward(self, x):
#         feats = self.feature(x)
#         return self.classifier(feats)
#
#
# # -------------------------------------------------------------
# # 2. Student MLP
# # -------------------------------------------------------------
# class MLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.__n_inputs = kwargs["n_inputs"]
#         self.__n_outputs = kwargs["n_outputs"]
#         self.__hidden_sizes = kwargs["hidden_sizes"]
#
#         self.kd_temperature = 4.0
#         self.kd_alpha = 0.7
#
#         # 1. 构建学生网络特征提取层
#         layers = []
#         prev = self.__n_inputs
#         for h in self.__hidden_sizes:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.3))
#             prev = h
#         self.feature = nn.Sequential(*layers)
#
#         # 2. 分类器
#         self.classifier = nn.Linear(prev, self.__n_outputs)
#
#         # 3. 初始化老师网络 (不更新梯度)
#         self.teacher = MLPTeacher(**kwargs)
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#
#         self.init_weights()
#
#     def forward(self, x, labels=None, return_teacher=False, return_feats=False):
#         # ---------------------------------------------------------
#         # Train 模式：执行蒸馏
#         # ---------------------------------------------------------
#         if self.training:
#             x_orig = x
#             device = x.device
#             batch_size = x.size(0)
#
#             lam = torch.tensor(1.0, device=device)  # 比例固定为1.0
#             index = torch.arange(batch_size, device=device)  # 索引指向自己
#             x_aug = x  # 增强数据等于原始数据
#
#             # --- B. 前向传播 ---
#             x_all = torch.cat([x_orig, x_aug], dim=0)
#             feats_all = self.feature(x_all)
#             logits_all = self.classifier(feats_all)
#
#             logits_orig, logits_aug = torch.chunk(logits_all, 2, dim=0)
#             feats_orig, _ = torch.chunk(feats_all, 2, dim=0)
#
#             # --- C. 教师模型预测 (蒸馏目标) ---
#             with torch.no_grad():
#                 t_logits = self.teacher(x_orig)
#
#             # 返回 6 个参数，适配 NormalClassifier.py 的 compute_dual_path_loss
#             return logits_orig, logits_aug, feats_orig, t_logits, lam, index
#
#         # ---------------------------------------------------------
#         # Eval 模式：保持简单输出
#         # ---------------------------------------------------------
#         feats = self.feature(x)
#         logits = self.classifier(feats)
#
#         if return_teacher:
#             with torch.no_grad():
#                 t_logits = self.teacher(x)
#             return logits, feats, t_logits
#
#         if return_feats:
#             return logits, feats
#
#         return logits
#
#     @property
#     def n_inputs(self) -> int:
#         return self.__n_inputs
#
#     @property
#     def n_outputs(self) -> int:
#         return self.__n_outputs
#
#     @property
#     def title(self):
#         return f"MLP_Enhanced({self.__hidden_sizes})"

#------------------增强-------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# from .BaseNet import BaseNet
#
#
# # -------------------------------------------------------------
# # 1. Student MLP (只做增强，不做蒸馏)
# # -------------------------------------------------------------
# class MLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.__n_inputs = kwargs["n_inputs"]
#         self.__n_outputs = kwargs["n_outputs"]
#         self.__hidden_sizes = kwargs["hidden_sizes"]
#
#         # 必须定义这两个属性，即使不蒸馏，loss_helper 也需要读取它们
#         self.kd_temperature = 4.0
#         self.kd_alpha = 0.0  # 设为 0，意味着完全不计算蒸馏损失权重
#
#         # 1. 构建网络特征提取层
#         layers = []
#         prev = self.__n_inputs
#         for h in self.__hidden_sizes:
#             layers.append(nn.Linear(prev, h))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.3))
#             prev = h
#         self.feature = nn.Sequential(*layers)
#
#         # 2. 分类器
#         self.classifier = nn.Linear(prev, self.__n_outputs)
#
#         # 注意：这里不再初始化 self.teacher，节省内存
#
#         self.init_weights()
#
#     def forward(self, x, labels=None, return_teacher=False, return_feats=False):
#         # ---------------------------------------------------------
#         # Train 模式：执行数据增强，跳过蒸馏
#         # ---------------------------------------------------------
#         if self.training:
#             x_orig = x
#             device = x.device
#             batch_size = x.size(0)
#
#             # === A. 数据增强逻辑 (Mixup) ===
#             # 从 Beta 分布采样混合比例
#             lam = torch.distributions.Beta(2.0, 2.0).sample().to(device)
#             index = torch.arange(batch_size).to(device)
#
#             if labels is not None:
#                 for i in range(batch_size):
#                     # 寻找同类样本索引
#                     same_label_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
#                     if len(same_label_indices) > 1:
#                         shift = torch.randint(1, len(same_label_indices), (1,)).item()
#                         curr_pos = (same_label_indices == i).nonzero(as_tuple=True)[0].item()
#                         index[i] = same_label_indices[(curr_pos + shift) % len(same_label_indices)]
#
#             # 生成增强数据
#             x_aug = lam * x + (1 - lam) * x[index]
#
#             # === B. 前向传播 ===
#             x_all = torch.cat([x_orig, x_aug], dim=0)
#             feats_all = self.feature(x_all)
#             logits_all = self.classifier(feats_all)
#
#             logits_orig, logits_aug = torch.chunk(logits_all, 2, dim=0)
#             feats_orig, _ = torch.chunk(feats_all, 2, dim=0)
#
#             # === C. 跳过教师网络 ===
#             # 伪造一个 t_logits，使其等于 logits_orig 的副本
#             # 这样 KL 散度计算结果为 0，且不消耗任何计算资源
#             t_logits = logits_orig.detach()
#
#             return logits_orig, logits_aug, feats_orig, t_logits, lam, index
#
#         # ---------------------------------------------------------
#         # Eval 模式
#         # ---------------------------------------------------------
#         feats = self.feature(x)
#         logits = self.classifier(feats)
#
#         if return_teacher:
#             # Eval 时如果请求 teacher，由于没有 teacher，返回自身的副本
#             return logits, feats, logits.detach()
#
#         if return_feats:
#             return logits, feats
#
#         return logits
#
#     @property
#     def n_inputs(self) -> int:
#         return self.__n_inputs
#
#     @property
#     def n_outputs(self) -> int:
#         return self.__n_outputs
#
#     @property
#     def title(self):
#         return f"MLP_Mixup_Only({self.__hidden_sizes})"

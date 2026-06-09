# import torch
# import torch.nn as nn
# from .BaseNet import BaseNet
#
#
# class SpatialGatingUnit(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.proj = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)
#
#     def forward(self, x):
#         # x: (B, L, dim)
#         u, v = x.chunk(2, dim=-1)  # (B, L, dim/2)
#
#         v = self.norm(v)
#         v = v.transpose(1, 2)  # (B, dim/2, L)
#         v = self.proj(v)
#         v = v.transpose(1, 2)  # (B, L, dim/2)
#
#         return u * v  # (B, L, dim/2)
#
#
# class gMLPLayer(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.sgu = SpatialGatingUnit(hidden_dim // 2)
#         self.fc2 = nn.Linear(hidden_dim // 2, dim)
#
#     def forward(self, x):
#         h = self.fc1(x)  # (B,L,H)
#         h = self.act(h)
#
#         h = self.sgu(h)  # (B,L,H/2)
#
#         h = self.fc2(h)  # (B,L,dim)
#         return h + x  # 残差
#
#
# class gMLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#         self.hidden_sizes = kwargs["hidden_sizes"]
#
#         layers = []
#         dim = self.n_inputs
#
#         # 每一层都是 gMLPLayer(dim, hidden_dim)
#         for h in self.hidden_sizes:
#             layers.append(gMLPLayer(dim, h))
#
#         self.feature = nn.Sequential(*layers)
#         self.classifier = nn.Linear(dim, self.n_outputs)
#
#         self.init_weights()
#
#     def forward(self, x):
#         # ---- 输入维度自动处理 ----
#         if x.dim() == 1:  # (C,)
#             x = x.unsqueeze(0).unsqueeze(1)  # (1,1,C)
#         elif x.dim() == 2:  # (B,C)
#             x = x.unsqueeze(1)  # (B,1,C)
#
#         # (B,L,C)
#         x = self.feature(x)  # (B,L,C)
#
#         x = x.mean(1)  # → (B,C)
#
#         x = self.classifier(x)
#         return x
#
#     @property
#     def title(self):
#         return f"gMLP({self.n_inputs},{self.hidden_sizes},{self.n_outputs})"

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .BaseNet import BaseNet
#
#
# # -------------------------------------------------------------
# # 1. Prototype Layer —— 小样本增强
# # -------------------------------------------------------------
# class ProtoLayer(nn.Module):
#     def __init__(self, feature_dim, n_classes):
#         super().__init__()
#         self.prototypes = nn.Parameter(torch.randn(n_classes, feature_dim))
#         nn.init.xavier_uniform_(self.prototypes)
#
#     def forward(self, feats):
#         # feats: (B, C)
#         dist = torch.cdist(feats, self.prototypes)  # (B, n_classes)
#         return -dist  # 负距离作为 logits
#
#
# # -------------------------------------------------------------
# # 2. Spatial Gating Unit —— gMLP 特征门控
# # -------------------------------------------------------------
# class SpatialGatingUnit(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.proj = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)  # depthwise conv
#
#     def forward(self, x):
#         # x: (B, L, dim*2)  → 这里 dim 已经是 hidden_dim//2
#         u, v = x.chunk(2, dim=-1)
#
#         v = self.norm(v)
#         v = v.transpose(1, 2)  # (B, dim/2, L)
#         v = self.proj(v)
#         v = v.transpose(1, 2)  # (B, L, dim/2)
#
#         return u * v
#
#
# # -------------------------------------------------------------
# # 3. gMLP Block（支持 Dropout）
# # -------------------------------------------------------------
# class gMLPLayer(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout1=0.3, dropout2=0.2):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout1)
#
#         # hidden_dim 一半进入 SGU
#         self.sgu = SpatialGatingUnit(hidden_dim // 2)
#         self.dropout2 = nn.Dropout(dropout2)
#
#         self.fc2 = nn.Linear(hidden_dim // 2, dim)
#
#     def forward(self, x):
#         # x: (B, L, C)
#         h = self.fc1(x)
#         h = self.act(h)
#         h = self.dropout1(h)
#
#         h = self.sgu(h)
#         h = self.dropout2(h)
#
#         h = self.fc2(h)
#         return h + x  # 残差结构
#
#
# # -------------------------------------------------------------
# # 4. Teacher gMLP（宽度 ×2）
# # -------------------------------------------------------------
# class gMLPTeacher(BaseNet):
#     def __init__(self, **kwargs):
#         # Teacher 也是 BaseNet，必须接收 base_name
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#
#         hidden_sizes = kwargs["hidden_sizes"]
#         wide_hidden = [h * 2 for h in hidden_sizes]
#
#         layers = []
#         dim = self.n_inputs
#         for h in wide_hidden:
#             layers.append(gMLPLayer(dim, h, dropout1=0.1, dropout2=0.1))
#         self.feature = nn.Sequential(*layers)
#
#         self.classifier = nn.Linear(dim, self.n_outputs)
#
#         self.init_weights()
#
#     def forward(self, x):
#         # reshape 保持与 Student 完全一致
#         if x.dim() == 1:
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:
#             x = x.unsqueeze(1)
#
#         feats = self.feature(x)
#         feats = feats.mean(1)  # 全局平均池化
#         return self.classifier(feats)
#
#
# # -------------------------------------------------------------
# # 5. Student gMLP（Dropout + ProtoLayer + 自蒸馏）
# # -------------------------------------------------------------
# class gMLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#
#         # -------------------------------------------------
#         # hidden_sizes 支持：
#         #   [512,256,256]
#         #   [512,256,256, {dropout1:..., kd_alpha:...}]
#         # -------------------------------------------------
#         raw_hs = kwargs["hidden_sizes"]
#
#         if isinstance(raw_hs[-1], dict):
#             self.extra_cfg = raw_hs[-1]
#             self.hidden_sizes = raw_hs[:-1]
#         else:
#             self.extra_cfg = {}
#             self.hidden_sizes = raw_hs
#
#         # dropout 配置
#         dropout1 = self.extra_cfg.get("dropout1", 0.3)
#         dropout2 = self.extra_cfg.get("dropout2", 0.2)
#
#         # KD / Proto 参数
#         self.proto_lr_ratio = self.extra_cfg.get("proto_lr_ratio", 0.5)
#         self.kd_alpha = self.extra_cfg.get("kd_alpha", 0.7)
#         self.kd_temperature = self.extra_cfg.get("kd_temperature", 4)
#
#         # -------------------------------------------------
#         # backbone
#         # -------------------------------------------------
#         layers = []
#         dim = self.n_inputs
#         for h in self.hidden_sizes:
#             layers.append(gMLPLayer(dim, h, dropout1, dropout2))
#         self.feature = nn.Sequential(*layers)
#
#         # 3 个头：分类（线性）+ 原型分类（ProtoLayer）+ Teacher（冻结）
#         self.classifier = nn.Linear(dim, self.n_outputs)
#         self.proto = ProtoLayer(dim, self.n_outputs)
#
#         # -------------------------------------------------
#         # 实例化 Teacher 时传入 base_name
#         # -------------------------------------------------
#         self.teacher = gMLPTeacher(
#             n_inputs=self.n_inputs,
#             n_outputs=self.n_outputs,
#             hidden_sizes=self.hidden_sizes,
#             base_name="gMLP_Teacher"
#         )
#
#         # 冻结 Teacher 参数
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#
#         self.init_weights()
#
#     # ---------------------------------------------------------
#     # forward 支持 return_feats / return_teacher
#     # ---------------------------------------------------------
#     def forward(self, x, return_feats=False, return_teacher=False):
#         # reshape 与 Teacher 完全一致
#         if x.dim() == 1:  # (C,)
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:  # (B,C)
#             x = x.unsqueeze(1)
#
#         feats = self.feature(x)
#         feats = feats.mean(1)  # (B, C)
#
#         cls_logits = self.classifier(feats)
#         proto_logits = self.proto(feats)
#
#         logits = cls_logits + proto_logits
#
#         # teacher 路径（KD 用）
#         if return_teacher:
#             with torch.no_grad():
#                 t_logits = self.teacher(x)
#             return logits, feats, t_logits
#
#         if return_feats:
#             return logits, feats
#
#         return logits
#     #---------------------------------------------------------
#
#     @property
#     def title(self):
#         return f"gMLP_Full({self.hidden_sizes},{self.extra_cfg})"

#----------------------蒸馏+数据增强----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseNet import BaseNet


# -------------------------------------------------------------
# 1. Prototype Layer —— 小样本增强
# -------------------------------------------------------------
class ProtoLayer(nn.Module):
    def __init__(self, feature_dim, n_classes, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(n_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, feats):
        dist = torch.cdist(feats, self.prototypes)  # (B, n_classes)
        return -dist / self.temperature  # 温度缩放


# -------------------------------------------------------------
# 2. Spatial Gating Unit —— gMLP 特征门控
# -------------------------------------------------------------
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)  # depthwise conv

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.transpose(1, 2)  # (B, dim/2, L)
        v = self.proj(v)
        v = v.transpose(1, 2)  # (B, L, dim/2)
        return u * v


# -------------------------------------------------------------
# 3. gMLP Block
# -------------------------------------------------------------
class gMLPLayer(nn.Module):
    def __init__(self, dim, hidden_dim, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout1)
        self.sgu = SpatialGatingUnit(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(hidden_dim // 2, dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout1(h)
        h = self.sgu(h)
        h = self.dropout2(h)
        h = self.fc2(h)
        return h + x  # 残差结构


# -------------------------------------------------------------
# 4. Teacher gMLP（宽度 ×2）
# -------------------------------------------------------------
class gMLPTeacher(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = kwargs["n_inputs"]
        self.n_outputs = kwargs["n_outputs"]
        hidden_sizes = kwargs["hidden_sizes"]
        wide_hidden = [h * 2 for h in hidden_sizes]

        layers = []
        dim = self.n_inputs
        for h in wide_hidden:
            layers.append(gMLPLayer(dim, h, dropout1=0.1, dropout2=0.1))
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(dim, self.n_outputs)
        self.init_weights()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.feature(x)
        feats = feats.mean(1)
        return self.classifier(feats)


# -------------------------------------------------------------
# 5. Student gMLP（类内数据增强版）
# -------------------------------------------------------------
class gMLP(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = kwargs["n_inputs"]
        self.n_outputs = kwargs["n_outputs"]

        raw_hs = kwargs["hidden_sizes"]
        if isinstance(raw_hs[-1], dict):
            self.extra_cfg = raw_hs[-1]
            self.hidden_sizes = raw_hs[:-1]
        else:
            self.extra_cfg = {}
            self.hidden_sizes = raw_hs

        dropout1 = self.extra_cfg.get("dropout1", 0.3)
        dropout2 = self.extra_cfg.get("dropout2", 0.2)
        self.kd_alpha = self.extra_cfg.get("kd_alpha", 0.7)
        self.kd_temperature = self.extra_cfg.get("kd_temperature", 4)

        layers = []
        dim = self.n_inputs
        for h in self.hidden_sizes:
            layers.append(gMLPLayer(dim, h, dropout1, dropout2))
        self.feature = nn.Sequential(*layers)

        self.classifier = nn.Linear(dim, self.n_outputs)
        self.proto = ProtoLayer(dim, self.n_outputs, temperature=0.1)

        # ✅【新增】DeepLIFT输入映射（1198 → PCA维度）
        self.input_proj = nn.Linear(1198, self.n_inputs)

        self.teacher = gMLPTeacher(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            hidden_sizes=self.hidden_sizes,
            base_name="gMLP_Teacher"
        )
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.init_weights()

    # def forward(self, x, labels=None, return_teacher=False, return_feats=False):
    def forward(self, x, labels=None, return_teacher=False, return_feats=False, use_raw=False):
        # ✅【新增】DeepLIFT路径（1198 → PCA）
        if use_raw:
            x = self.input_proj(x)

        # reshape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # ---------------------------------------------------------
        # Train 模式（核心修改：类内增强逻辑）
        # ---------------------------------------------------------
        if self.training:
            x_orig = x
            device = x.device
            batch_size = x.size(0)

            # 使用 Beta(2.0, 2.0) 使得混合比例更倾向于均衡，强化特征学习
            lam = torch.distributions.Beta(2.0, 2.0).sample().to(device)

            # 初始化索引，默认指向自己
            index = torch.arange(batch_size).to(device)

            # 执行类内寻找配对
            if labels is not None:
                for i in range(batch_size):
                    # 找到 Batch 内与当前样本 i 标签相同的所有索引
                    same_label_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]

                    if len(same_label_indices) > 1:
                        # 随机选一个同类的“兄弟”样本（排除自己）
                        shift = torch.randint(1, len(same_label_indices), (1,)).item()
                        curr_pos = (same_label_indices == i).nonzero(as_tuple=True)[0].item()
                        index[i] = same_label_indices[(curr_pos + shift) % len(same_label_indices)]

            # 执行同类 Mixup
            x_aug = lam * x + (1 - lam) * x[index]

            x_all = torch.cat([x_orig, x_aug], dim=0)

            feats_all = self.feature(x_all).mean(1)
            logits_all = self.classifier(feats_all) + self.proto(feats_all)

            logits_orig, logits_aug = torch.chunk(logits_all, 2, dim=0)
            feats_orig, _ = torch.chunk(feats_all, 2, dim=0)

            # teacher 仅对原始路径进行蒸馏指导
            with torch.no_grad():
                t_logits = self.teacher(x_orig)

            return logits_orig, logits_aug, feats_orig, t_logits, lam, index

        # ---------------------------------------------------------
        # Eval 模式（简单推理）
        # ---------------------------------------------------------
        feats = self.feature(x).mean(1)
        logits = self.classifier(feats) + self.proto(feats)

        if return_teacher:
            with torch.no_grad():
                t_logits = self.teacher(x)
            return logits, feats, t_logits

        if return_feats:
            return logits, feats

        return logits

    @property
    def title(self):
        return f"gMLP_Full({self.hidden_sizes},{self.extra_cfg})"


#--------------------------蒸馏-------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .BaseNet import BaseNet
#
#
# # -------------------------------------------------------------
# # 1. Prototype Layer —— 小样本增强
# # -------------------------------------------------------------
# class ProtoLayer(nn.Module):
#     def __init__(self, feature_dim, n_classes, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.prototypes = nn.Parameter(torch.randn(n_classes, feature_dim))
#         nn.init.xavier_uniform_(self.prototypes)
#
#     def forward(self, feats):
#         dist = torch.cdist(feats, self.prototypes)  # (B, n_classes)
#         return -dist / self.temperature  # 温度缩放
#
#
# # -------------------------------------------------------------
# # 2. Spatial Gating Unit —— gMLP 特征门控
# # -------------------------------------------------------------
# class SpatialGatingUnit(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.proj = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)  # depthwise conv
#
#     def forward(self, x):
#         u, v = x.chunk(2, dim=-1)
#         v = self.norm(v)
#         v = v.transpose(1, 2)  # (B, dim/2, L)
#         v = self.proj(v)
#         v = v.transpose(1, 2)  # (B, L, dim/2)
#         return u * v
#
#
# # -------------------------------------------------------------
# # 3. gMLP Block
# # -------------------------------------------------------------
# class gMLPLayer(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout1=0.3, dropout2=0.2):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout1)
#         self.sgu = SpatialGatingUnit(hidden_dim // 2)
#         self.dropout2 = nn.Dropout(dropout2)
#         self.fc2 = nn.Linear(hidden_dim // 2, dim)
#
#     def forward(self, x):
#         h = self.fc1(x)
#         h = self.act(h)
#         h = self.dropout1(h)
#         h = self.sgu(h)
#         h = self.dropout2(h)
#         h = self.fc2(h)
#         return h + x  # 残差结构
#
#
# # -------------------------------------------------------------
# # 4. Teacher gMLP（宽度 ×2）
# # -------------------------------------------------------------
# class gMLPTeacher(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#         hidden_sizes = kwargs["hidden_sizes"]
#         wide_hidden = [h * 2 for h in hidden_sizes]
#
#         layers = []
#         dim = self.n_inputs
#         for h in wide_hidden:
#             layers.append(gMLPLayer(dim, h, dropout1=0.1, dropout2=0.1))
#         self.feature = nn.Sequential(*layers)
#         self.classifier = nn.Linear(dim, self.n_outputs)
#         self.init_weights()
#
#     def forward(self, x):
#         if x.dim() == 1:
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:
#             x = x.unsqueeze(1)
#         feats = self.feature(x)
#         feats = feats.mean(1)
#         return self.classifier(feats)
#
#
# # -------------------------------------------------------------
# # 5. Student gMLP
# # -------------------------------------------------------------
# class gMLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#
#         raw_hs = kwargs["hidden_sizes"]
#         if isinstance(raw_hs[-1], dict):
#             self.extra_cfg = raw_hs[-1]
#             self.hidden_sizes = raw_hs[:-1]
#         else:
#             self.extra_cfg = {}
#             self.hidden_sizes = raw_hs
#
#         dropout1 = self.extra_cfg.get("dropout1", 0.3)
#         dropout2 = self.extra_cfg.get("dropout2", 0.2)
#         self.kd_alpha = self.extra_cfg.get("kd_alpha", 0.7)
#         self.kd_temperature = self.extra_cfg.get("kd_temperature", 4)
#
#         layers = []
#         dim = self.n_inputs
#         for h in self.hidden_sizes:
#             layers.append(gMLPLayer(dim, h, dropout1, dropout2))
#         self.feature = nn.Sequential(*layers)
#
#         self.classifier = nn.Linear(dim, self.n_outputs)
#         self.proto = ProtoLayer(dim, self.n_outputs, temperature=0.1)
#
#         self.teacher = gMLPTeacher(
#             n_inputs=self.n_inputs,
#             n_outputs=self.n_outputs,
#             hidden_sizes=self.hidden_sizes,
#             base_name="gMLP_Teacher"
#         )
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#
#         self.init_weights()
#
#     def forward(self, x, labels=None, return_teacher=False, return_feats=False):
#         # reshape
#         if x.dim() == 1:
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:
#             x = x.unsqueeze(1)
#
#         # ---------------------------------------------------------
#         # Train 模式
#         # ---------------------------------------------------------
#         if self.training:
#             x_orig = x
#             device = x.device
#             batch_size = x.size(0)
#
#             lam = torch.tensor(1.0, device=device)  # 混合比例为 1.0 (全保留原样)
#             index = torch.arange(batch_size, device=device)  # 索引指向自己
#             x_aug = x  # 增强数据 = 原始数据
#
#             x_all = torch.cat([x_orig, x_aug], dim=0)
#             feats_all = self.feature(x_all).mean(1)
#             logits_all = self.classifier(feats_all) + self.proto(feats_all)
#
#             logits_orig, logits_aug = torch.chunk(logits_all, 2, dim=0)
#             feats_orig, _ = torch.chunk(feats_all, 2, dim=0)
#
#             # teacher 仅对原始路径进行蒸馏指导
#             with torch.no_grad():
#                 t_logits = self.teacher(x_orig)
#
#             return logits_orig, logits_aug, feats_orig, t_logits, lam, index
#
#         # ---------------------------------------------------------
#         # Eval 模式（简单推理）
#         # ---------------------------------------------------------
#         feats = self.feature(x).mean(1)
#         logits = self.classifier(feats) + self.proto(feats)
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
#     def title(self):
#         return f"gMLP_Full({self.hidden_sizes},{self.extra_cfg})"

#---------------------数据增强-------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .BaseNet import BaseNet
#
#
# # -------------------------------------------------------------
# # 1. Prototype Layer —— 小样本增强
# # -------------------------------------------------------------
# class ProtoLayer(nn.Module):
#     def __init__(self, feature_dim, n_classes, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.prototypes = nn.Parameter(torch.randn(n_classes, feature_dim))
#         nn.init.xavier_uniform_(self.prototypes)
#
#     def forward(self, feats):
#         dist = torch.cdist(feats, self.prototypes)  # (B, n_classes)
#         return -dist / self.temperature  # 温度缩放
#
#
# # -------------------------------------------------------------
# # 2. Spatial Gating Unit —— gMLP 特征门控
# # -------------------------------------------------------------
# class SpatialGatingUnit(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.proj = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)  # depthwise conv
#
#     def forward(self, x):
#         u, v = x.chunk(2, dim=-1)
#         v = self.norm(v)
#         v = v.transpose(1, 2)  # (B, dim/2, L)
#         v = self.proj(v)
#         v = v.transpose(1, 2)  # (B, L, dim/2)
#         return u * v
#
#
# # -------------------------------------------------------------
# # 3. gMLP Block
# # -------------------------------------------------------------
# class gMLPLayer(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout1=0.3, dropout2=0.2):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout1)
#         self.sgu = SpatialGatingUnit(hidden_dim // 2)
#         self.dropout2 = nn.Dropout(dropout2)
#         self.fc2 = nn.Linear(hidden_dim // 2, dim)
#
#     def forward(self, x):
#         h = self.fc1(x)
#         h = self.act(h)
#         h = self.dropout1(h)
#         h = self.sgu(h)
#         h = self.dropout2(h)
#         h = self.fc2(h)
#         return h + x  # 残差结构
#
#
# # -------------------------------------------------------------
# # 4. Teacher gMLP（宽度 ×2）
# # -------------------------------------------------------------
# class gMLPTeacher(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#         hidden_sizes = kwargs["hidden_sizes"]
#         wide_hidden = [h * 2 for h in hidden_sizes]
#
#         layers = []
#         dim = self.n_inputs
#         for h in wide_hidden:
#             layers.append(gMLPLayer(dim, h, dropout1=0.1, dropout2=0.1))
#         self.feature = nn.Sequential(*layers)
#         self.classifier = nn.Linear(dim, self.n_outputs)
#         self.init_weights()
#
#     def forward(self, x):
#         if x.dim() == 1:
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:
#             x = x.unsqueeze(1)
#         feats = self.feature(x)
#         feats = feats.mean(1)
#         return self.classifier(feats)
#
#
# # -------------------------------------------------------------
# # 5. Student gMLP（类内数据增强版）
# # -------------------------------------------------------------
# class gMLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#
#         raw_hs = kwargs["hidden_sizes"]
#         if isinstance(raw_hs[-1], dict):
#             self.extra_cfg = raw_hs[-1]
#             self.hidden_sizes = raw_hs[:-1]
#         else:
#             self.extra_cfg = {}
#             self.hidden_sizes = raw_hs
#
#         dropout1 = self.extra_cfg.get("dropout1", 0.3)
#         dropout2 = self.extra_cfg.get("dropout2", 0.2)
#         self.kd_alpha = self.extra_cfg.get("kd_alpha", 0.7)
#         self.kd_temperature = self.extra_cfg.get("kd_temperature", 4)
#
#         layers = []
#         dim = self.n_inputs
#         for h in self.hidden_sizes:
#             layers.append(gMLPLayer(dim, h, dropout1, dropout2))
#         self.feature = nn.Sequential(*layers)
#
#         self.classifier = nn.Linear(dim, self.n_outputs)
#         self.proto = ProtoLayer(dim, self.n_outputs, temperature=0.1)
#
#         self.teacher = gMLPTeacher(
#             n_inputs=self.n_inputs,
#             n_outputs=self.n_outputs,
#             hidden_sizes=self.hidden_sizes,
#             base_name="gMLP_Teacher"
#         )
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#
#         self.init_weights()
#
#     def forward(self, x, labels=None, return_teacher=False, return_feats=False):
#         # reshape
#         if x.dim() == 1:
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:
#             x = x.unsqueeze(1)
#
#         # ---------------------------------------------------------
#         # Train 模式
#         # ---------------------------------------------------------
#         if self.training:
#             x_orig = x
#             device = x.device
#             batch_size = x.size(0)
#
#             # 使用 Beta(2.0, 2.0) 使得混合比例更倾向于均衡，强化特征学习
#             lam = torch.distributions.Beta(2.0, 2.0).sample().to(device)
#
#             # 初始化索引，默认指向自己
#             index = torch.arange(batch_size).to(device)
#
#             # 执行类内寻找配对
#             if labels is not None:
#                 for i in range(batch_size):
#                     # 找到 Batch 内与当前样本 i 标签相同的所有索引
#                     same_label_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
#
#                     if len(same_label_indices) > 1:
#                         # 随机选一个同类的“兄弟”样本（排除自己）
#                         shift = torch.randint(1, len(same_label_indices), (1,)).item()
#                         curr_pos = (same_label_indices == i).nonzero(as_tuple=True)[0].item()
#                         index[i] = same_label_indices[(curr_pos + shift) % len(same_label_indices)]
#
#             # 执行同类 Mixup
#             x_aug = lam * x + (1 - lam) * x[index]
#
#             x_all = torch.cat([x_orig, x_aug], dim=0)
#
#             feats_all = self.feature(x_all).mean(1)
#             logits_all = self.classifier(feats_all) + self.proto(feats_all)
#
#             logits_orig, logits_aug = torch.chunk(logits_all, 2, dim=0)
#             feats_orig, _ = torch.chunk(feats_all, 2, dim=0)
#
#             t_logits = logits_orig.detach()
#             return logits_orig, logits_aug, feats_orig, t_logits, lam, index
#
#         # ---------------------------------------------------------
#         # Eval 模式（简单推理）
#         # ---------------------------------------------------------
#         feats = self.feature(x).mean(1)
#         logits = self.classifier(feats) + self.proto(feats)
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
#     def title(self):
#         return f"gMLP_Full({self.hidden_sizes},{self.extra_cfg})"

#-----------------------不蒸馏不增强--------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .BaseNet import BaseNet
#
#
# # -------------------------------------------------------------
# # 1. Prototype Layer (保留结构，不做修改)
# # -------------------------------------------------------------
# class ProtoLayer(nn.Module):
#     def __init__(self, feature_dim, n_classes, temperature=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.prototypes = nn.Parameter(torch.randn(n_classes, feature_dim))
#         nn.init.xavier_uniform_(self.prototypes)
#
#     def forward(self, feats):
#         dist = torch.cdist(feats, self.prototypes)  # (B, n_classes)
#         return -dist / self.temperature
#
#
# # -------------------------------------------------------------
# # 2. Spatial Gating Unit
# # -------------------------------------------------------------
# class SpatialGatingUnit(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.proj = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)
#
#     def forward(self, x):
#         u, v = x.chunk(2, dim=-1)
#         v = self.norm(v)
#         v = v.transpose(1, 2)
#         v = self.proj(v)
#         v = v.transpose(1, 2)
#         return u * v
#
#
# # -------------------------------------------------------------
# # 3. gMLP Block
# # -------------------------------------------------------------
# class gMLPLayer(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout1=0.3, dropout2=0.2):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, hidden_dim)
#         self.act = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout1)
#         self.sgu = SpatialGatingUnit(hidden_dim // 2)
#         self.dropout2 = nn.Dropout(dropout2)
#         self.fc2 = nn.Linear(hidden_dim // 2, dim)
#
#     def forward(self, x):
#         h = self.fc1(x)
#         h = self.act(h)
#         h = self.dropout1(h)
#         h = self.sgu(h)
#         h = self.dropout2(h)
#         h = self.fc2(h)
#         return h + x
#
#
# # -------------------------------------------------------------
# # 4. Teacher gMLP
# # -------------------------------------------------------------
# class gMLPTeacher(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#         hidden_sizes = kwargs["hidden_sizes"]
#         wide_hidden = [h * 2 for h in hidden_sizes]
#
#         layers = []
#         dim = self.n_inputs
#         for h in wide_hidden:
#             layers.append(gMLPLayer(dim, h, dropout1=0.1, dropout2=0.1))
#         self.feature = nn.Sequential(*layers)
#         self.classifier = nn.Linear(dim, self.n_outputs)
#         self.init_weights()
#
#     def forward(self, x):
#         if x.dim() == 1:
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:
#             x = x.unsqueeze(1)
#         feats = self.feature(x)
#         feats = feats.mean(1)
#         return self.classifier(feats)
#
#
# # -------------------------------------------------------------
# # 5. Student gMLP (已修改为：无蒸馏、无增强)
# # -------------------------------------------------------------
# class gMLP(BaseNet):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_inputs = kwargs["n_inputs"]
#         self.n_outputs = kwargs["n_outputs"]
#
#         raw_hs = kwargs["hidden_sizes"]
#         if isinstance(raw_hs[-1], dict):
#             self.extra_cfg = raw_hs[-1]
#             self.hidden_sizes = raw_hs[:-1]
#         else:
#             self.extra_cfg = {}
#             self.hidden_sizes = raw_hs
#
#         dropout1 = self.extra_cfg.get("dropout1", 0.3)
#         dropout2 = self.extra_cfg.get("dropout2", 0.2)
#         self.kd_alpha = self.extra_cfg.get("kd_alpha", 0.7)
#         self.kd_temperature = self.extra_cfg.get("kd_temperature", 4)
#
#         layers = []
#         dim = self.n_inputs
#         for h in self.hidden_sizes:
#             layers.append(gMLPLayer(dim, h, dropout1, dropout2))
#         self.feature = nn.Sequential(*layers)
#
#         self.classifier = nn.Linear(dim, self.n_outputs)
#         self.proto = ProtoLayer(dim, self.n_outputs, temperature=0.1)
#
#         # ✅ DeepLIFT专用输入映射（1198 → 34）
#         # ================================
#         self.input_proj = nn.Linear(1198, self.n_inputs)
#
#         # 即使不用，保留Teacher初始化以防止 __init__ 报错
#         self.teacher = gMLPTeacher(
#             n_inputs=self.n_inputs,
#             n_outputs=self.n_outputs,
#             hidden_sizes=self.hidden_sizes,
#             base_name="gMLP_Teacher"
#         )
#         for p in self.teacher.parameters():
#             p.requires_grad = False
#
#         self.init_weights()
#
#     # def forward(self, x, labels=None, return_teacher=False, return_feats=False):
#     def forward(self, x, labels=None, return_teacher=False, return_feats=False, use_raw=False):
#         # ✅ DeepLIFT路径（绕开PCA）
#         # ================================
#         if use_raw:
#             # x 是 (B, 1198)
#             x = self.input_proj(x)
#
#
#         # reshape
#         if x.dim() == 1:
#             x = x.unsqueeze(0).unsqueeze(1)
#         elif x.dim() == 2:
#             x = x.unsqueeze(1)
#
#         # ---------------------------------------------------------
#         # Train 模式 (修改区域)
#         # ---------------------------------------------------------
#         if self.training:
#             x_orig = x
#             device = x.device
#             batch_size = x.size(0)
#
#             # =========== [修改1：关闭数据增强] ===========
#             # 原逻辑：Beta分布采样
#             # lam = torch.distributions.Beta(2.0, 2.0).sample().to(device)
#             # 新逻辑：强制为 1.0 (只看原始数据)
#             lam = torch.tensor(1.0, device=device)
#
#             # 原逻辑：寻找同类样本索引
#             # ... (循环逻辑已省略) ...
#             # 新逻辑：索引指向自己
#             index = torch.arange(batch_size, device=device)
#
#             # 原逻辑：Mixup
#             # x_aug = lam * x + (1 - lam) * x[index]
#             # 新逻辑：增强数据 = 原始数据
#             x_aug = x
#             # ===========================================
#
#             # 为了兼容 loss_helper，保持拼接形式（虽然 x_orig 和 x_aug 一样）
#             x_all = torch.cat([x_orig, x_aug], dim=0)
#
#             feats_all = self.feature(x_all).mean(1)
#             # 这里的 + self.proto 属于结构部分，一般保留
#             logits_all = self.classifier(feats_all) + self.proto(feats_all)
#
#             logits_orig, logits_aug = torch.chunk(logits_all, 2, dim=0)
#             feats_orig, _ = torch.chunk(feats_all, 2, dim=0)
#
#             # =========== [修改2：关闭蒸馏] ===========
#             # 原逻辑：运行 Teacher 模型
#             # with torch.no_grad():
#             #     t_logits = self.teacher(x_orig)
#
#             # 新逻辑：伪造 t_logits，使其等于学生输出。
#             # 这样 KL(Student, Student) = 0，蒸馏 Loss 消失。
#             t_logits = logits_orig.detach()
#             # ========================================
#
#             return logits_orig, logits_aug, feats_orig, t_logits, lam, index
#
#         # ---------------------------------------------------------
#         # Eval 模式
#         # ---------------------------------------------------------
#         feats = self.feature(x).mean(1)
#         logits = self.classifier(feats) + self.proto(feats)
#
#         if return_teacher:
#             # 如果你在 Eval 也不想跑 teacher，可以同样伪造
#             # with torch.no_grad():
#             #    t_logits = self.teacher(x)
#             t_logits = logits.detach()  # 简单伪造
#             return logits, feats, t_logits
#
#         if return_feats:
#             return logits, feats
#
#         return logits
#
#     @property
#     def title(self):
#         return f"gMLP_Simple({self.hidden_sizes})"


import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2 import utils as pointnet2_utils  # 假设已安装/导入pointnet2_ops


# ==============================================================================
# 辅助函数 (假设已存在)
# ==============================================================================
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# ==============================================================================
# 1. 核心改进：可学习的邻域选择器
# ==============================================================================
class OptimalNeighborhoodSelector(nn.Module):
    """
    通过一个小型MLP学习为邻域中的点打分，以替代传统的KNN。
    它从一个较大的候选邻域中，根据几何和特征的相对关系，选择出最重要的一组邻居。
    """

    def __init__(self, feature_channels, k_candidate=64, k_final=16):
        super().__init__()
        self.k_candidate = k_candidate
        self.k_final = k_final

        # 输入维度 = 相对坐标(3) + 相对特征(C)
        mlp_in_channels = 3 + feature_channels

        # 一个小型MLP网络，用于给每个候选邻居点打分
        self.scoring_mlp = nn.Sequential(
            nn.Linear(mlp_in_channels, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        print(f"Initialized OptimalNeighborhoodSelector: k_candidate={k_candidate}, k_final={k_final}")

    def forward(self, xyz, features):
        """
        xyz: 点云坐标 [B, N, 3]
        features: 点云特征 [B, N, C]
        返回:
            final_indices: 最终选择的邻居的索引 [B, N, k_final]
        """
        B, N, C_feat = features.shape
        device = xyz.device

        with torch.no_grad():
            # 步骤 1: 使用标准KNN找到一个较大的候选邻域，以减少计算量
            dists = torch.cdist(xyz, xyz)  # [B, N, N]
            _, candidate_idx = dists.topk(self.k_candidate + 1, largest=False, sorted=False)
            candidate_idx = candidate_idx[:, :, 1:]  # [B, N, k_candidate]

        # 收集候选邻居的数据
        neighbors_xyz = index_points(xyz, candidate_idx)  # [B, N, k_candidate, 3]
        neighbors_feat = index_points(features, candidate_idx)  # [B, N, k_candidate, C_feat]

        # 扩展中心点以进行减法操作
        center_xyz = xyz.unsqueeze(2).expand_as(neighbors_xyz)
        center_feat = features.unsqueeze(2).expand_as(neighbors_feat)

        # 步骤 2: 计算相对信息（坐标和特征）
        relative_xyz = neighbors_xyz - center_xyz
        relative_feat = neighbors_feat - center_feat

        # 将相对信息拼接，作为打分网络的输入
        mlp_input = torch.cat([relative_xyz, relative_feat], dim=-1)  # [B, N, k_candidate, 3+C_feat]

        # 步骤 3: 使用MLP为每个候选邻居打分
        scores = self.scoring_mlp(mlp_input)  # [B, N, k_candidate, 1]

        # 步骤 4: 根据分数选择最高的k_final个邻居
        # topk的输入需要是 [B, N, k_candidate]
        _, topk_indices_in_candidate = torch.topk(scores.squeeze(-1), self.k_final, dim=2)  # [B, N, k_final]

        # 步骤 5: 从候选邻居索引中收集最终的邻居索引
        final_indices = torch.gather(candidate_idx, 2, topk_indices_in_candidate)

        return final_indices


# ==============================================================================
# 2. 改进的重要性计算函数
# ==============================================================================
def calculate_robust_importance(xyz, feat, neighborhood_indices, alpha=0.5, distance_metric='l2'):
    """
    在一个给定的邻域内，计算每个点的结构重要性。
    - 曲率: 使用协方差矩阵的特征值计算，更鲁棒。
    - 特征变化: 可选择使用L2（欧氏）或L1（曼哈顿）距离。
    """
    B, N, C = feat.shape
    k = neighborhood_indices.shape[-1]

    # --- 检查邻居有效性 ---
    # `index_points` 会在索引越界时报错，这是一个隐式的检查
    assert neighborhood_indices.max() < N and neighborhood_indices.min() >= 0, "邻居索引越界！"

    # 收集邻居数据
    neighbors_xyz = index_points(xyz, neighborhood_indices)  # [B, N, k, 3]
    neighbors_feat = index_points(feat, neighborhood_indices)  # [B, N, k, C]
    center_xyz = xyz.unsqueeze(2)
    center_feat = feat.unsqueeze(2)

    # --- 1. 几何曲率评估 (基于特征值) ---
    # 这是对点云局部表面弯曲程度的经典且稳健的度量
    diff_xyz = neighbors_xyz - center_xyz
    # 计算每个点邻域的协方差矩阵 [B, N, 3, 3]
    cov_matrix = torch.matmul(diff_xyz.transpose(-1, -2), diff_xyz) / k

    # 使用SVD计算奇异值 (比eig更稳定)，奇异值的平方即为特征值
    # torch.linalg.svdvals 返回降序的奇异值
    singular_values = torch.linalg.svdvals(cov_matrix)  # [B, N, 3]

    # 特征值 lambdas (升序)
    lambda_2, lambda_1, lambda_0 = singular_values[..., 0] ** 2, singular_values[..., 1] ** 2, singular_values[
        ..., 2] ** 2

    # 曲率定义为最小特征值占总特征值和的比例
    # $\sigma_c(p) = \frac{\lambda_0}{\lambda_0 + \lambda_1 + \lambda_2}$
    sum_eigenvalues = lambda_0 + lambda_1 + lambda_2
    # 添加一个极小值epsilon防止除以零
    curvature = lambda_0 / (sum_eigenvalues + 1e-8)  # [B, N]

    # --- 2. 特征距离计算 ---
    diff_feat = neighbors_feat - center_feat
    if distance_metric == 'l2':
        # 欧氏距离 (L2 Norm)
        feat_dist = torch.norm(diff_feat, dim=-1)
    elif distance_metric == 'l1':
        # 曼哈顿距离 (L1 Norm)
        feat_dist = torch.sum(torch.abs(diff_feat), dim=-1)
    else:
        raise ValueError(f"不支持的距离度量: {distance_metric}")

    feat_variation = feat_dist.mean(dim=-1)  # [B, N]

    # --- 3. 组合最终的重要性分数 ---
    # 将曲率和特征变化进行加权求和。alpha是超参数。
    # 可以在应用前对两者进行归一化，以平衡它们的尺度
    curvature_norm = (curvature - curvature.mean(dim=1, keepdim=True)) / (curvature.std(dim=1, keepdim=True) + 1e-8)
    feat_variation_norm = (feat_variation - feat_variation.mean(dim=1, keepdim=True)) / (
                feat_variation.std(dim=1, keepdim=True) + 1e-8)

    importance = curvature_norm + alpha * feat_variation_norm
    return importance


# ==============================================================================
# 3. 封装后的智能下采样模块
# ==============================================================================
class IntelligentDownsampler(nn.Module):
    """
    封装了完整的“智能”下采样逻辑。
    结合了基于学习的重要性采样和最远点采样(FPS)来保证覆盖率和特征感知。
    """

    def __init__(self, feature_channels, k_candidate=64, k_final=16, alpha=0.5, distance_metric='l2'):
        super().__init__()
        self.neighborhood_selector = OptimalNeighborhoodSelector(feature_channels, k_candidate, k_final)
        self.num_samples = 0
        self.curvature_ratio = 0.0
        self.alpha = alpha
        self.distance_metric = distance_metric

    def forward(self, xyz, features, num_samples, curvature_ratio=0.5):
        """
        xyz: [B, N, 3]
        features: [B, N, C]
        num_samples: 目标采样点数
        curvature_ratio: 重要性采样点所占的比例
        返回:
            sampled_xyz: [B, S, 3] 采样后的坐标
            merged_indices: [B, S] 采样后的索引
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # 步骤 1: 使用可学习的选择器获取最佳邻域
        learned_indices = self.neighborhood_selector(xyz, features)

        # 步骤 2: 基于该邻域计算鲁棒的重要性分数
        importance = calculate_robust_importance(
            xyz, features, learned_indices, self.alpha, self.distance_metric
        )

        # 步骤 3: 结合重要性采样和FPS
        num_curv = int(num_samples * curvature_ratio)
        num_fps = num_samples - num_curv

        # 从高重要性分数中选择点
        _, curv_indices = torch.topk(importance, k=num_curv, dim=1)

        # 对剩余的点使用FPS，以保证覆盖度
        # 首先需要排除已经被选中的点
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, curv_indices, False)

        # 创建一个xyz的副本，用于FPS计算，其中不包含已选中的点
        masked_xyz = xyz.clone()
        masked_xyz[~mask] = masked_xyz.max() + 1  # 将已选点移到很远的地方

        fps_indices = pointnet2_utils.furthest_point_sample(masked_xyz, num_fps)

        # 合并两组索引
        merged_indices = torch.cat([curv_indices, fps_indices], dim=1)
        sampled_xyz = index_points(xyz, merged_indices)

        return sampled_xyz, merged_indices


# ==============================================================================
# 4. 修正后的主编码器 (PCT_encoder)
# ==============================================================================
class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        # ... (cross_transformer, Learned3DPositionalEmbedding等模块定义保持不变) ...
        # cross_transformer 和 Learned3DPositionalEmbedding 的定义请从原代码中复制过来

        # 初始化新的智能下采样模块
        # 注意: 这里的 `feature_channels` 必须与输入特征的通道数匹配
        # 第一次采样时，特征是64维，来自于conv2
        self.intelligent_sampler = IntelligentDownsampler(feature_channels=64, k_final=16, distance_metric='l1')

        # 其他层定义...
        self.channel = channel
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.learned_pe = Learned3DPositionalEmbedding(in_dim=3, embed_dim=64)
        self.relu = nn.GELU()
        # ... (sa1, sa1_1, sa2, etc. from original code) ...
        # 请将原代码中的所有 sa* transformer层, ps* 层, conv_out* 层等复制到此处
        # 为保持示例简洁，此处省略了这些层的定义
        # self.sa1 = cross_transformer(channel, channel)
        # ...

    def forward(self, points):
        # points: [B, 6, N]
        batch_size, _, N = points.size()

        # 初始特征提取
        xyz = points[:, :3, :].transpose(1, 2).contiguous()  # [B, N, 3]
        x = self.relu(self.conv1(points))
        x0 = self.conv2(x)  # [B, 64, N]

        # 添加可学习的位置编码
        pos_embed = self.learned_pe(xyz)  # [B, N, 64]
        x0 = x0 + pos_embed.permute(0, 2, 1)

        # --- 第1阶段下采样 (N -> N/4) ---
        # **修正点**: 使用新的智能下采样模块，并正确使用其返回的索引
        # `x0.transpose(1,2)` 是采样器需要的特征格式 [B, N, C]
        xyz_s1, idx_s1 = self.intelligent_sampler(
            xyz, x0.transpose(1, 2).contiguous(), N // 4, curvature_ratio=0.5
        )
        x_g1 = pointnet2_utils.gather_operation(x0, idx_s1)  # [B, 64, N/4]

        # Transformer 特征提取
        # x1 = self.sa1(x_g1, x0).contiguous() + x_g1
        # x1 = torch.cat([x_g1, x1], dim=1)
        # x1 = self.sa1_1(x1, x1).contiguous() # [B, 128, N/4]

        # --- 第2阶段下采样 (N/4 -> N/8) ---
        # **修正点**: 后续采样在已经采样过的点集上进行 (xyz_s1)
        # 这里我们使用标准的FPS，因为它更快，且关键特征已在第一步提取
        idx_s2 = pointnet2_utils.furthest_point_sample(xyz_s1, N // 8)
        xyz_s2 = index_points(xyz_s1, idx_s2)
        # x_g2 = pointnet2_utils.gather_operation(x1, idx_s2) # [B, 128, N/8]

        # Transformer 特征提取
        # x2 = self.sa2(x_g2, x1).contiguous() + x_g2
        # ... (后续网络结构) ...

        # ...
        # 此处省略了完整的网络结构，重点展示采样部分的修改
        # return x_g, fine

        # 这是一个示例返回值，您需要填充完整的网络逻辑
        # 假设x3是最后一层特征
        # x_g = F.adaptive_max_pool1d(x3, 1)
        # fine = ...
        # return x_g, fine
        pass  # Placeholder for the rest of the network
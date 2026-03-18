import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_fpsample
from point_ops.pointnet2_ops import pointnet2_utils as p2u
from pytorch3d.ops import ball_query
class PointSelector(nn.Module):
    def __init__(self, feature_dim=128, walk_length=6, k=16, temperature=0.1):
        super(PointSelector, self).__init__()
        self.walk_length = walk_length
        self.k = k
        self.temperature = temperature

        self.beta = nn.Linear(feature_dim, feature_dim)
        self.gamma = nn.Linear(feature_dim, feature_dim)

    def forward(self, feats, coords, start_indices):
        """
        feats:      [B, N, C]  - 点的特征
        coords:     [B, N, 3]  - 点的坐标
        start_indices: [B, M]  - 采样起点索引 (e.g. from Neighbour Similarity Sampling)

        Return:
            walks: [B, M, L, C] - 每个采样点对应的walk特征序列
        """
        B, N, C = feats.shape
        M = start_indices.shape[1]
        walks = torch.zeros((B, M, self.walk_length, C), dtype=feats.dtype, device=feats.device)

        for b in range(B):
            for m in range(M):
                idx = start_indices[b, m]
                walk_feats = [feats[b, idx]]
                current_idx = idx

                for l in range(1, self.walk_length):
                    # 1. KNN邻居索引（可替换为KD-Tree/Open3D加速）
                    dist = torch.norm(coords[b] - coords[b, current_idx], dim=-1)  # [N]
                    knn_idx = dist.topk(self.k + 1, largest=False).indices[1:]    # exclude self

                    # 2. attention logits 计算
                    f_i = feats[b, current_idx]                      # [C]
                    f_neighbors = feats[b, knn_idx]                 # [k, C]

                    q = self.beta(f_i)                              # [C]
                    k = self.gamma(f_neighbors)                    # [k, C]
                    attn_logits = torch.matmul(q, k.T) / np.sqrt(C)  # [k]

                    # 3. Gumbel Softmax采样
                    attn = F.gumbel_softmax(attn_logits, tau=self.temperature, hard=True)  # [k]
                    next_idx = (attn.unsqueeze(-1) * knn_idx.unsqueeze(-1).float()).sum(dim=0).long()

                    current_idx = next_idx
                    walk_feats.append(feats[b, current_idx])

                walks[b, m] = torch.stack(walk_feats, dim=0)

        return walks  # [B, M, walk_length, C]

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=128):
        super(SimpleEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, xyz):
        return self.mlp(xyz)  # [B, N, C]


def neighbour_similarity_sampling(points, num_samples=1024, k=16):
    B, N, C = points.shape

    #features = points[:,:,3:6]
    _,fps_idx = torch_fpsample.sample(points, num_samples)  # [B, S]
    #fps_idx.to(device)
    sampled_points = p2u.gather_operation(points, fps_idx)

    # ball query
    grouped_feats = ball_query(sampled_points,points,radius=0.5,K=16)  # [B, S, K, C]
    sim_scores = torch.einsum('bskc,bsjc->bsk', grouped_feats, grouped_feats)  # cosine sim
    sim_scores = sim_scores / (grouped_feats.norm(dim=-1) * grouped_feats.norm(dim=-1)).sum(-1)  # [B, S]

    max_sim_idx = sim_scores.argmax(dim=-1)  # [B, S]
    sampled_points = p2u.gather_operation(points, max_sim_idx)
    #sampled_features = index_points(features, fps_idx)
    return sampled_points

class RouteTransformer(nn.Module):
    def __init__(self, feature_dim=128, k=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, curr_walks, prev_walks):
        """
        curr_walks: [B, M, L, C]
        prev_walks: [B, M, L, C]
        """
        B, M, L, C = curr_walks.shape
        q = curr_walks.view(B*M, L, C)
        k = prev_walks.view(B*M, L, C)
        v = k
        out, _ = self.cross_attn(q, k, v)
        return self.mlp(out).view(B, M, L, C)

class DisplacementPredictor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

    def forward(self, walks):
        """
        walks: [B, M, L, C]
        return: [B, M, 3]
        """
        central = walks[:, :, 0, :]  # start point feature
        rel_feats = walks[:, :, 1:, :] - central.unsqueeze(2)  # [B, M, L-1, C]
        concat_feat = torch.cat([rel_feats.mean(dim=2), central], dim=-1)  # [B, M, 2C]
        return self.mlp(concat_feat)


class WalkFormer(nn.Module):
    def __init__(self, feature_dim=128, walk_len=6, k=16, num_steps=5):
        super().__init__()
        self.encoder = SimpleEncoder(3, feature_dim)
        self.selector = PointSelector(feature_dim, walk_len, k)
        self.route_transformer = RouteTransformer(feature_dim)
        self.predictor = DisplacementPredictor(feature_dim)
        self.num_steps = num_steps
        self.walk_len = walk_len
        self.k = k

    def forward(self, xyz):
        B, N, _ = xyz.shape
        feats = self.encoder(xyz)  # [B, N, C]
        positions = xyz.clone()

        for t in range(self.num_steps):
            sampled_xyz, sampled_feats, sampled_idx = neighbour_similarity_sampling(positions, feats)
            walks = self.selector(feats, positions, sampled_idx)
            if t == 0:
                prev_walks = walks
            walks = self.route_transformer(walks, prev_walks)
            disp = self.predictor(walks)
            positions[:, sampled_idx] += disp
            prev_walks = walks  # update history

        return positions



class GeoHyperEncoding(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.fuse = nn.Sequential(
            nn.Linear(out_dim + 6 + 6, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, coords, neighbors):
        """
        coords:     [B, N, 3]
        neighbors:  [B, N, k, 3]
        """
        B, N, k, _ = neighbors.shape
        # R1: |p - v|
        r1 = torch.abs(coords.unsqueeze(2) - neighbors)  # [B, N, k, 3]
        r1_mean = r1.mean(dim=2)                         # [B, N, 3]

        # R2: edge deviation
        edges = coords.unsqueeze(2) - neighbors          # [B, N, k, 3]
        mean_edge = edges.mean(dim=2, keepdim=True)
        r2 = torch.abs(mean_edge - edges).mean(dim=2)    # [B, N, 3]

        hpe = self.project(coords)                       # [B, N, D]
        fused = self.fuse(torch.cat([hpe, r1_mean, r2], dim=-1))
        return fused  # [B, N, D]


class GeoScaleFormer(nn.Module):
    def __init__(self, feat_dim=128, walk_len=6, k=16):
        super().__init__()
        self.encoder = SimpleEncoder(3, feat_dim)
        self.ghe = GeoHyperEncoding(out_dim=feat_dim)
        self.walkformer = GuidedWalkFormer(feat_dim, walk_len, k)
        self.refiner = GeoCorrectionRefiner(feat_dim)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        feats = self.encoder(xyz)

        # Grouping: build local regions (simulate with KNN)
        knn_idx = knn(xyz, k=16)  # [B, N, k]
        neighbors = index_points(xyz, knn_idx)  # [B, N, k, 3]
        pos_embed = self.ghe(xyz, neighbors)

        fused_feats = feats + pos_embed

        # Coarse: Farthest point sampling for guidance
        start_idx = farthest_point_sample(xyz, 512)  # [B, M]
        coarse_disp, selected_idx = self.walkformer(fused_feats, xyz, start_idx)
        coarse_points = index_points(xyz, selected_idx) + coarse_disp  # [B, M, 3]

        # Mid+Fine: Combine original input + coarse results
        all_points = torch.cat([xyz, coarse_points], dim=1)
        all_feats = self.encoder(all_points)
        fine_points = self.refiner(all_feats)  # [B, N+M, 3]

        return {
            "coarse": coarse_points,
            "fine": fine_points
        }

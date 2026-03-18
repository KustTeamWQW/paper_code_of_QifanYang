from torch import nn, einsum
import torch
from einops import rearrange, repeat
from point_ops.pointnet2_ops import pointnet2_utils
from loss_pcc import chamfer_loss_sqrt, chamfer_loss, density_cd
import numpy as np
from torch.nn import functional as F


# from pytorch3d.ops.knn import knn_gather, knn_points


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel, channel)
        self.sa1_1 = cross_transformer(channel * 2, channel * 2)
        self.sa2 = cross_transformer((channel) * 2, channel * 2)
        self.sa2_1 = cross_transformer((channel) * 4, channel * 4)
        self.sa3 = cross_transformer((channel) * 4, channel * 4)
        self.sa3_1 = cross_transformer((channel) * 8, channel * 8)
        # self.sa4 = cross_transformer((channel)*8, channel*16)
        self.relu = nn.GELU()

        self.sa0_d = cross_transformer(channel * 8, channel * 8)
        self.sa1_d = cross_transformer(channel * 8, channel * 8)
        self.sa2_d = cross_transformer(channel * 8, channel * 8)

        self.conv_out = nn.Conv1d(64, 6, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel * 4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)

    def forward(self, points):
        points = points.transpose(2, 1).contiguous()
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = pointnet2_utils.gather_operation(x0, idx_0)
        # x_g0 = gather_points(x0, idx_0)
        # points = gather_points(points, idx_0)
        points = pointnet2_utils.gather_operation(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1, x1).contiguous()
        # GDP
        idx_1 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        # x_g1 = gather_points(x1, idx_1)
        x_g1 = pointnet2_utils.gather_operation(x1, idx_1)
        # points = gather_points(points, idx_1)
        points = pointnet2_utils.gather_operation(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = pointnet2_utils.furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        # x_g2 = gather_points(x2, idx_2)
        x_g2 = pointnet2_utils.gather_operation(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3, x3).contiguous()
        # x4 = self.sa4(x3,x3).contiguous()
        # seed generator
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size, self.channel * 4, N // 8)

        # x_g = F.adaptive_max_pool1d(x4,1).view(batch_size, -1).unsqueeze(-1)
        # x_g = x_g.max(dim=-1, keepdim=False)[0]
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return x_g, fine


class PCT_refine(nn.Module):
    def __init__(self, channel=128, ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(6, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel * 2, 512)
        self.sa2 = cross_transformer(512, 512)
        self.sa3 = cross_transformer(512, channel * ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 6, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel * 1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)

    def forward(self, x, coarse, feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio)

        y_up = y.repeat(1, 1, self.ratio)
        y_cat = torch.cat([y3, y_up], dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)

        return x, y3



class PCCNet(nn.Module):
    def __init__(self, kmax=20, code_dim=512):
        super().__init__()
        self.k = kmax
        self.code_dim = code_dim
        self.enc1 = PCT_encoder()
        self.refine = PCT_refine(ratio=8)
        self.refine1 = PCT_refine(ratio=1)
        self.num_coarse = 1024

    def forward(self, x):
        # p_input, glob_feat = self.enc(x)
        glob_feat, p_input = self.enc1(x)
        x = x.permute(0, 2, 1)
        new_x = torch.cat([p_input, x], dim=2)
        new_xx = new_x
        new_x = pointnet2_utils.gather_operation(new_x, pointnet2_utils.furthest_point_sample(
            new_x.transpose(1, 2).contiguous(), 512))
        fine, feat_fine = self.refine(None, new_x, glob_feat)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, glob_feat)
        coarse = new_xx.transpose(1, 2).contiguous()
        fine2 = fine.transpose(1, 2).contiguous()
        fine1 = fine.transpose(1, 2).contiguous()
        return coarse, fine1, fine2



def validate(model, loader, epoch, args, device, rand_save=False):
    print("Validating ...")
    model.eval()
    num_iters = len(loader)

    with torch.no_grad():
        cdt_coarse, cdp_coarse, cdt_fine, cdp_fine, d_fine, d_coarse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i, data in enumerate(loader):
            # data
            xyz = data[0][:, :, :6].to(device).float()  # partial: [B 2048, 6] include normals
            # model
            coarse, fine, finer = model(xyz)
            # coarse, fine = coarse[:, :, :3], fine[:, :, :3]
            # losses
            gt_xyz = data[1][:, :, :3].to(device).float()  # partial: [B 16348, 6]
            if args.tr_loss == 'dcd':
                d_fine += density_cd(fine[:, :, :3], gt_xyz).item()  # inputs shd be BNC; cd_p
                d_coarse += density_cd(coarse[:, :, :3], gt_xyz).item()

            cdp_fine += chamfer_loss_sqrt(fine[:, :, :3], gt_xyz).item()  # inputs shd be BNC; cd_p
            cdp_coarse += chamfer_loss_sqrt(coarse[:, :, :3], gt_xyz).item()
            cdt_fine += chamfer_loss(fine[:, :, :3], gt_xyz).item()  # cd_t
            cdt_coarse += chamfer_loss(coarse[:, :, :3], gt_xyz).item()

            if rand_save and args.max_epoch % 10 == 0 and i in [0, 7, 15]:
                # if finer is not None:
                #     np.savez(str(args.file_dir) + '/rand_outs.npz', gt_pnts=gt_xyz.data.cpu().numpy(),
                #                                                     final_pnts=finer.data.cpu().numpy(),
                #                                                     fine_pnts=fine.data.cpu().numpy(),
                #                                                     coarse_pnts=coarse.data.cpu().numpy(),
                #                                                     als_pnts=xyz.data.cpu().numpy()[:, :, :3])
                # else:
                np.savez(str(args.file_dir) + f'/rand_outs{i}.npz', gt_pnts=gt_xyz.data.cpu().numpy(),
                         final_pnts=fine.data.cpu().numpy(),
                         coarse_pnts=coarse.data.cpu().numpy(),
                         als_pnts=xyz.data.cpu().numpy()[:, :, :3])

    return {'fine_d': d_fine / num_iters, 'fine_p': cdp_fine / num_iters, 'coarse_p': cdp_coarse / num_iters,
            'coarse_d': d_coarse / num_iters, 'fine_t': cdt_fine / num_iters, 'coarse_t': cdt_coarse / num_iters}

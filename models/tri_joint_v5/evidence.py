import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tri_calib.blocks import BasicBlock


class EarlyTriModalEvidence(nn.Module):
    def __init__(self, feat_ch: int = 256, proj_ch: int = 128, window_size: int = 5):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.feat_ch = feat_ch
        self.proj_ch = proj_ch
        self.window_size = window_size
        self.window_radius = window_size // 2
        self.num_offsets = window_size * window_size

        self.cam_proj = nn.Conv2d(feat_ch, proj_ch, kernel_size=1, bias=False)
        self.lid_proj = nn.Conv2d(feat_ch, proj_ch, kernel_size=1, bias=False)
        self.rad_proj = nn.Conv2d(feat_ch, proj_ch, kernel_size=1, bias=False)

        self.query_net = nn.Sequential(
            nn.Conv2d(proj_ch * 3 + 4, proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_ch, proj_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(proj_ch * 2, feat_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            BasicBlock(feat_ch, feat_ch),
        )

        self.modality_embedding = nn.Parameter(torch.zeros(3, proj_ch))
        self.offset_embedding = nn.Parameter(torch.zeros(self.num_offsets, proj_ch))
        self.token_gate = nn.Sequential(
            nn.Conv3d(proj_ch * 2 + 2, proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(proj_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(proj_ch, 1, kernel_size=1, bias=True),
        )
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.offset_embedding, mean=0.0, std=0.02)

    def _unfold_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        bsz, ch, h, w = feat.shape
        unfolded = F.unfold(feat, kernel_size=self.window_size, padding=self.window_radius)
        return unfolded.view(bsz, ch, self.num_offsets, h, w)

    def _unfold_mask(self, mask: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = mask.shape
        unfolded = F.unfold(mask, kernel_size=self.window_size, padding=self.window_radius)
        return unfolded.view(bsz, 1, self.num_offsets, h, w)

    def _context_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(mask, kernel_size=self.window_size, stride=1, padding=self.window_radius)

    def _add_embeddings(self, tokens: torch.Tensor, modality_index: int) -> torch.Tensor:
        mod = self.modality_embedding[modality_index].view(1, self.proj_ch, 1, 1, 1)
        off = self.offset_embedding.t().view(1, self.proj_ch, self.num_offsets, 1, 1)
        return tokens + mod + off

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = (valid > 0).to(values.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (values * valid).sum(dim=1, keepdim=True) / denom

    def forward(
        self,
        cam_feat: torch.Tensor,
        lid_feat: torch.Tensor,
        rad_feat: torch.Tensor,
        lid_mask: torch.Tensor,
        rad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p_cam = self.cam_proj(cam_feat)
        p_lid = self.lid_proj(lid_feat)
        p_rad = self.rad_proj(rad_feat)

        lid_context = self._context_mask(lid_mask)
        rad_context = self._context_mask(rad_mask)

        query = self.query_net(torch.cat([
            p_cam,
            p_lid * lid_context,
            p_rad * rad_context,
            lid_context,
            rad_context,
            lid_mask,
            rad_mask,
        ], dim=1))
        q = F.normalize(query, dim=1)

        cam_tokens = self._add_embeddings(self._unfold_tokens(p_cam), modality_index=0)
        lid_tokens = self._add_embeddings(self._unfold_tokens(p_lid), modality_index=1)
        rad_tokens = self._add_embeddings(self._unfold_tokens(p_rad), modality_index=2)

        cam_mask = torch.ones_like(self._unfold_mask(lid_mask))
        lid_token_mask = self._unfold_mask(lid_mask)
        rad_token_mask = self._unfold_mask(rad_mask)
        lid_window_mask = lid_token_mask.max(dim=2, keepdim=True)[0].expand(-1, -1, self.num_offsets, -1, -1)
        rad_window_mask = rad_token_mask.max(dim=2, keepdim=True)[0].expand(-1, -1, self.num_offsets, -1, -1)

        tokens = torch.cat([cam_tokens, lid_tokens, rad_tokens], dim=2)
        token_mask = torch.cat([cam_mask, lid_window_mask, rad_window_mask], dim=2)
        token_valid = torch.cat([cam_mask, lid_token_mask, rad_token_mask], dim=2)
        token_values = F.normalize(tokens, dim=1)

        scores = (q.unsqueeze(2) * token_values).sum(dim=1) / math.sqrt(self.proj_ch)
        gate_input = torch.cat([
            q.unsqueeze(2).expand(-1, -1, tokens.shape[2], -1, -1),
            token_values,
            token_valid,
            token_mask,
        ], dim=1)
        gate_bias = self.token_gate(gate_input).squeeze(1)
        scores = scores + gate_bias

        lid_scores = scores[:, self.num_offsets:2 * self.num_offsets]
        rad_scores = scores[:, 2 * self.num_offsets:3 * self.num_offsets]
        lid_valid = lid_token_mask.squeeze(1)
        rad_valid = rad_token_mask.squeeze(1)
        lid_attn_hint = self._masked_mean(lid_scores, lid_valid)
        rad_attn_hint = self._masked_mean(rad_scores, rad_valid)

        scores = scores.masked_fill(token_mask.squeeze(1) <= 0, -1e4)
        attn = torch.softmax(scores, dim=1)
        attended = (token_values * attn.unsqueeze(1)).sum(dim=2)

        lid_attn = attn[:, self.num_offsets:2 * self.num_offsets]
        rad_attn = attn[:, 2 * self.num_offsets:3 * self.num_offsets]
        lid_attn_mean = self._masked_mean(lid_attn, lid_valid)
        rad_attn_mean = self._masked_mean(rad_attn, rad_valid)

        match_cl = torch.cat([lid_attn_hint, lid_attn_mean], dim=1)
        match_cr = torch.cat([rad_attn_hint, rad_attn_mean], dim=1)
        match_lr = torch.cat([lid_attn_hint, lid_attn_mean, rad_attn_hint, rad_attn_mean], dim=1)

        evidence = self.out_proj(torch.cat([attended, query], dim=1))
        return {
            "J_t": evidence,
            "query": query,
            "attended": attended,
            "attn": attn,
            "p_cam": p_cam,
            "p_lid": p_lid,
            "p_rad": p_rad,
            "match_cl": match_cl,
            "match_cr": match_cr,
            "match_lr": match_lr,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwisePoseHead(nn.Module):
    """
    Minimal pose head:
      input vector -> hidden -> (translation, quaternion)
    """

    def __init__(self, in_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_t = nn.Linear(hidden_dim, 3)
        self.fc_q = nn.Linear(hidden_dim, 4)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        t = self.fc_t(x)
        q = F.normalize(self.fc_q(x), dim=1)
        return t, q


class TriPairwiseHeads(nn.Module):
    """
    Three independent pairwise heads for:
      CL, CR, LR
    """

    def __init__(self, pair_feature_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.head_cl = PairwisePoseHead(pair_feature_dim, hidden_dim, dropout)
        self.head_cr = PairwisePoseHead(pair_feature_dim, hidden_dim, dropout)
        self.head_lr = PairwisePoseHead(pair_feature_dim, hidden_dim, dropout)

    def forward(self, feat_cl, feat_cr, feat_lr):
        t_cl, q_cl = self.head_cl(feat_cl)
        t_cr, q_cr = self.head_cr(feat_cr)
        t_lr, q_lr = self.head_lr(feat_lr)
        return {
            "T_CL_t": t_cl,
            "T_CL_q": q_cl,
            "T_CR_t": t_cr,
            "T_CR_q": q_cr,
            "T_LR_t": t_lr,
            "T_LR_q": q_lr,
        }


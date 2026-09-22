import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.attn import Attention
from lib.models.layers.CA import ChannelAggregationFFN
import torch.utils.checkpoint as cp
from functools import partial
from lib.models.layers.moe_adapter import moe_adapter
from lib.models.layers.pde_adapter import PDEAdapter
# from lib.models.layers.LocalAtt import PDEAdapter
# from lib.models.layers.alt_layer import GatedConvRefiner
# from lib.models.layers.xiaorong1 import PDEA1
# from lib.models.layers.xiaorong2 import PDEA2
# from lib.models.layers.xiaorong3 import PDEA3
# from lib.models.layers.pinn_adapter import PDEAdapter
def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


# class CEBlock(nn.Module):
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         #self.adapter1=Mlp(in_features=dim,hidden_features=8,act_layer=act_layer,drop=drop)
#         #self.adapter2=Mlp(in_features=dim,hidden_features=8,act_layer=act_layer,drop=drop)
#         self.keep_ratio_search = keep_ratio_search
#
#     def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
#         x_attn, attn = self.attn(self.norm1(x), mask, True)
#         x = x + self.drop_path(x_attn)
#         lens_t = global_index_template.shape[1]
#
#         removed_index_search = None
#         keep_ratio_search= 3
#         if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
#             keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
#             x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
#         identy=self.norm2(x)
#         x = x + self.drop_path(self.mlp(identy))
#
#
#         return x, global_index_template, global_index_search, removed_index_search, attn
class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,scale=1,use_pde=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.adapter1=Mlp(in_features=dim,hidden_features=8,act_layer=act_layer,drop=drop)
        # self.adapter2=moe_adapter(in_features=dim,hidden_features=8,out_features=dim)
        self.keep_ratio_search = keep_ratio_search
        self.scale=scale
        self.use_pde = use_pde
        if self.use_pde:
            self.pde_adapter = PDEAdapter(dim=dim)

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        # x_attn1=self.adapter1(x_attn)
        # x_attn=x_attn+self.adapter1(x_attn)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        keep_ratio_search= 3
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_pde:
            pde,_=self.pde_adapter(x, template_len=64)
            x = x + pde
            # x = x+self.pde_adapter(x, template_len=64)


        return x, global_index_template, global_index_search, removed_index_search, attn

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Injector(nn.Module):
    def __init__(self, dim, hide_dim=8, num_heads=6, attn_drop=0., qkv_bias=False,
                 drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, cffn_ratio=0.25,
                 drop_path=0., ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(hide_dim)
        self.feat_norm = norm_layer(hide_dim)
        self.attn = Cross_Attention(hide_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        # self.r = hide_dim
        # self.lora_A_q = nn.Parameter(torch.zeros((self.r, dim)))
        # self.lora_A_kv = nn.Parameter(torch.zeros((self.r, dim)))
        # self.lora_B_q = nn.Parameter(torch.zeros((dim, self.r)))
        # nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.lora_A_kv, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B_q)

        self.q_down = nn.Linear(dim, hide_dim)
        self.q_up = nn.Linear(hide_dim, dim)
        self.kv_down = nn.Linear(dim, hide_dim)

    def forward(self, query, feat):

        def _inner_forward(query, feat):
            # 2024.7.5
            # zc_split = feat[:, :336, :]
            # zc_select1, zc_select2, zc_select3 = zc_split[:, :H_z * W_z * 4, :], zc_split[:,H_z * W_z * 4:H_z * W_z * 4 + H_z * W_z,:] \
            #                                      , zc_split[:, H_z * W_z * 4 + H_z * W_z:, :]
            # zx_split2 = query[:, :64, :]
            # zx_split2 = torch.cat([zx_split2, zc_select2], dim=2)
            # zx_split2 = self.fusion(zx_split2)
            # zx_split1 = self.attn(self.query_norm(zx_split2), self.feat_norm(zc_select1))
            # zx_split3 = self.attn(self.query_norm(zx_split2), self.feat_norm(zc_select3))
            # zx = zx_split1 + zx_split3 + zx_split2
            #
            # xc_split = feat[:, 336:, :]
            # xc_select1, xc_select2, xc_select3 = xc_split[:, :H_x * W_x * 4, :], xc_split[:,H_x * W_x * 4:H_x * W_x * 4 + H_x * W_x,:] \
            #                                     , xc_split[:, H_x * W_x * 4 + H_x * W_x:, :]
            # xx_split2 = query[:, 64:, :]
            # xx_split2 = torch.cat([xx_split2, xc_select2], dim=2)
            # xx_split2 = self.fusion(xx_split2)
            # xx_split1 = self.attn(self.query_norm(xx_split2), self.feat_norm(xc_select1))
            # xx_split3 = self.attn(self.query_norm(xx_split2), self.feat_norm(xc_select3))
            # xx = xx_split1 + xx_split3 + xx_split2
            # attn = torch.cat([zx, xx], dim=1)

            # q = query @ self.lora_A_q.T
            # kv = feat @ self.lora_A_kv.T
            # attn = self.attn(self.query_norm(q),
            #                  self.feat_norm(kv))
            # attn = attn @ self.lora_B_q.T

            q = self.q_down(query)
            kv = self.kv_down(feat)

            attn = self.attn(self.query_norm(q),
                             self.feat_norm(kv))
            attn = self.q_up(attn)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query
class Cross_Attention(nn.Module):
    def __init__(self, dim,  num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = dim ** -0.5
        # self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, feat, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = query.shape
        q = query
        k = feat
        v = feat

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'), )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C
        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x
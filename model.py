import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from einops.layers.torch import Rearrange
import math

# Define Multi-Head Self Attention (MSA) Module
# Define Multi-Head Self Attention (MSA) Module
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        # Compute Q, K, V and reshape to (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if return_attention:
            return x_out, attn
        return x_out

# Define MLP Module
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Define Transformer Block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Define Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, num_classes=512,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic Depth Decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m.bias, nn.Parameter):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


class DinoHead(nn.Module):
    def __init__(self, emb = 768, dropout = 0.2 ,norm_last_layer=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb),  # Added normalization
            nn.Linear(emb, emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb, emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb, emb),
        )
        # Final layer with weight normalization
        self.final = nn.utils.weight_norm(nn.Linear(emb, emb, bias=False))
        self.final.weight_g.data.fill_(1)
        if norm_last_layer:
            self.final.weight_g.requires_grad = False
    
    def forward(self,x):
        x = self.net(x)
        x = F.normalize(x, dim=-1, p=2, eps=1e-8)
        x = self.final(x)

        return x


class DINO(nn.Module):
    def __init__(self, in_channels=3, img_size=128, patch_size=8, embed_dim=256, depth=12, num_heads=8,
                 mlp_ratio=4., num_classes=256, dropout=0.2, norm_last_layer=True):
        super().__init__()

        # Vision Transformer backbone
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=0.1,  # Stochastic depth rate
            norm_layer=nn.LayerNorm
        )

        # DINO head
        self.head = DinoHead(
            emb=embed_dim,
            dropout=dropout,
            norm_last_layer=norm_last_layer
        )

    def forward(self, x, return_attention=False):
        n_crops = len(x)
        x = torch.cat(x, dim=0)
        
        # Forward pass through Vision Transformer
        x = self.backbone(x)

        # Forward pass through DINO head
        x = self.head(x)
        
        x_chunk = x.chunk(n_crops)
        return x_chunk
    
    def get_last_selfattention(self, x):
        """
        Returns the full self-attention weights from the last transformer block's attention module.
        Returns the complete attention matrix with shape [B, num_heads, N, N]
        """
        B = x.size(0)
        # Patch embedding and prepare tokens
        x = self.backbone.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        last_attn = None
        # Process through all blocks; capture attention from the last block
        for i, block in enumerate(self.backbone.blocks):
            if i == len(self.backbone.blocks) - 1:
                # For the last block, call the attention module with return_attention=True
                x_norm = block.norm1(x)
                attn_out, attn = block.attn(x_norm, return_attention=True)
                x = x + block.drop_path(attn_out)
                x = x + block.drop_path(block.mlp(block.norm2(x)))
                last_attn = attn  # shape: [B, num_heads, N, N]
            else:
                x = block(x)
        
        x = self.backbone.norm(x)
        # Return the full attention matrix
        return last_attn  # shape: [B, num_heads, N, N]


class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))  # Center buffer for teacher logits

    def forward(self, student_outputs, teacher_outputs):
        # Ensure center is on the correct device
        self.center = self.center.to(teacher_outputs[0].device)

        # Save raw teacher logits for the center update
        teacher_logits = teacher_outputs
        
        # Scale teacher outputs with the center and temperature
        teacher_outputs = [(t - self.center) / self.teacher_temp for t in teacher_logits]
        # Scale student outputs
        student_outputs = [s / self.student_temp for s in student_outputs]

        # Compute probabilities
        teacher_probs = [F.softmax(t, dim=-1).detach() for t in teacher_outputs]
        student_log_probs = [F.log_softmax(s, dim=-1) for s in student_outputs]

        total_loss = 0
        n_loss_terms = 0
        for t_ix, t_prob in enumerate(teacher_probs):
            for s_ix, s_log_prob in enumerate(student_log_probs):
                if t_ix == s_ix:
                    continue
                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center using raw teacher logits
        self.update_center(teacher_logits)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_logits):
        batch_center = torch.cat(teacher_logits).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


# Unit Test
class TestViT(unittest.TestCase):
    def test_vit_forward(self):
        model = DINO(in_channels=3, p=8, dim=128, depth=2, latent_dim=64)
        x = torch.randn(4, 3, 256, 256)  # Batch of 4 RGB images (128x128)
        output = model(x)
        
        # Ensure output shape is correct
        self.assertEqual(output.shape, (4, 64))  # (batch_size, latent_dim)

        # Check if cls_token is trainable
        self.assertTrue(model.cls_token.requires_grad)

        # Check cls_token shape
        self.assertEqual(model.cls_token.shape, (1, 1, 128))

        print("Test passed successfully!")


if __name__ == "__main__":
    unittest.main()

import torch
import torch.nn as nn
import types
import torch.utils.checkpoint as checkpoint

# -------------------------
# 1) SLA-Adapter (Time-Aware 3DConv Adapter)
# -------------------------
class SLAAdapter3D(nn.Module):
    def __init__(self, d_model, reduction_dim=256, frames=300, spatial_side=14):
        super().__init__()
        self.reduction_dim = reduction_dim
        self.frames = frames
        self.spatial_side = spatial_side 
        self.down_proj = nn.Linear(d_model, reduction_dim)
        self.conv3d = nn.Conv3d(reduction_dim, reduction_dim, kernel_size=3, padding=1, groups=reduction_dim, bias=False)
        self.act = nn.GELU() 
        self.up_proj = nn.Linear(reduction_dim, d_model)

    def forward(self, x):
        # 1. Handle OpenAI CLIP's (Sequence, Batch, Dim) format
        seq_first = False
        if x.shape[0] == (self.spatial_side ** 2) + 1:
            seq_first = True
            x = x.permute(1, 0, 2)  # Flip to (Batch*Time, Sequence, Dim)
            
        B_times_T, L, D = x.shape
        T = self.frames
        B = B_times_T // T
        
        # 2. Separate CLS token and Spatial Patches
        cls_token = x[:, :1, :]        # (B*T, 1, D)
        patch_tokens = x[:, 1:, :]     # (B*T, H*W, D)
        
        # 3. Down-project and reshape for 3D spatio-temporal convolution
        h = self.down_proj(patch_tokens)
        H = W = self.spatial_side
        
        # Reshape to (B, T, H, W, reduction_dim) then permute for Conv3d (B, C, T, H, W)
        h = h.view(B, T, H, W, self.reduction_dim)
        h = h.permute(0, 4, 1, 2, 3).contiguous() 
        
        # 4. Apply Depth-wise 3D Convolution & Activation
        h = self.act(self.conv3d(h))
        
        # 5. Reshape back and Up-project
        h = h.permute(0, 2, 3, 4, 1).contiguous()
        h = h.view(B_times_T, H * W, self.reduction_dim)
        
        adapter_out_patches = self.up_proj(h)
        adapter_out_cls = torch.zeros_like(cls_token)
        
        adapter_out = torch.cat([adapter_out_cls, adapter_out_patches], dim=1)
        
        # 6. Apply Adapter Residual
        out = x + adapter_out
        
        # 7. Restore OpenAI's Sequence-first format before passing back to the ViT Block
        if seq_first:
            out = out.permute(1, 0, 2)
            
        return out


# -------------------------
# 2) ViT Block Patching Logic
# -------------------------
def insert_sla_adapters_into_clip(clip_model, reduction_dim=256, layers_to_insert=range(2,12), frames=4, spatial_side=14):
    """
    Hooks into the CLIP model's transformer blocks to strategically place the 
    SLA-Adapter modules before the MHSA and MLP layers.
    """
    # Locate blocks depending on the specific CLIP/ViT implementation
    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "transformer"):
        blocks = clip_model.visual.transformer.resblocks
        d_model = clip_model.visual.transformer.width
    else:
        raise RuntimeError("Could not locate visual transformer blocks. Adjust block targeting based on your CLIP repo.")

    # Helper to find standard block attributes
    def get_components(block):
        n1 = getattr(block, "ln_1", None)
        attn = getattr(block, "attn", None)
        n2 = getattr(block, "ln_2", None)
        mlp = getattr(block, "mlp", None)
        return n1, attn, n2, mlp

    for idx in layers_to_insert:
        block = blocks[idx]
        norm1, attn, norm2, mlp = get_components(block)
        
        # Register the adapters as submodules on the block
        block.sla_adapter_attn = SLAAdapter3D(d_model, reduction_dim, frames, spatial_side)
        block.sla_adapter_mlp = SLAAdapter3D(d_model, reduction_dim, frames, spatial_side)

        # Define the custom forward pass matching Figure 3
        def custom_forward(self, x, *args, **kwargs):
            def _forward_computation(x_input):
                # --- MHSA Pathway ---
                # 1. Adapter placed BEFORE the MHSA layer
                x_adapted_attn = self.sla_adapter_attn(x_input)
                
                # 2. Standard Pre-Norm and MHSA
                # Standard OpenAI attributes: ln_1, attn, ln_2, mlp
                h_attn = self.ln_1(x_adapted_attn) if hasattr(self, "ln_1") else x_adapted_attn
                
                # MultiheadAttention call
                attn_out, _ = self.attn(h_attn, h_attn, h_attn, need_weights=False)
                
                # 3. Outer Residual Connection
                x2 = x_adapted_attn + attn_out
                
                # --- MLP Pathway ---
                # 4. Adapter placed BEFORE the MLP block
                x_adapted_mlp = self.sla_adapter_mlp(x2)
                
                # 5. Standard Pre-Norm and MLP
                h_mlp = self.ln_2(x_adapted_mlp) if hasattr(self, "ln_2") else x_adapted_mlp
                mlp_out = self.mlp(h_mlp)
                
                # 6. Outer Residual Connection
                x4 = x_adapted_mlp + mlp_out
                
                return x4
            
            # Apply gradient checkpointing during training to save VRAM
            if self.training and torch.is_grad_enabled():
                # use_reentrant=False is the modern PyTorch standard
                return checkpoint.checkpoint(_forward_computation, x, use_reentrant=False)
            else:
                return _forward_computation(x)

        # Bind the custom forward method to this specific block instance
        block.forward = types.MethodType(custom_forward, block)

    return clip_model


# -------------------------
# 3) Model Wrapper
# -------------------------
class CLIP_SLA_Wrapper(nn.Module):
    """
    Wrapper to load a CLIP model, freeze the appropriate backbone weights, 
    and facilitate passing batched video frames.
    """
    def __init__(self, clip_model, reduction_dim=256, adapter_layers=range(2,12), frames=4, spatial_side=14, freeze_clip=True):
        super().__init__()
        self.clip = clip_model.float()
        
        # Insert the custom adapters into the specified ViT blocks
        self.clip = insert_sla_adapters_into_clip(
            self.clip, reduction_dim, adapter_layers, frames, spatial_side
        )
        
        # Parameter freezing strategy: freeze backbone, train only adapters
        if freeze_clip:
            for name, param in self.clip.named_parameters():
                if "sla_adapter" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x):
        """
        x: Videos of shape (B, T, C, H, W)
        Returns: Per-frame CLS token embeddings of shape (B, T, D)
        """
        B, T, C, H, W = x.shape
        
        # Flatten temporal dimension into batch dimension for CLIP processing
        x = x.view(B * T, C, H, W)
        
        # # Only track gradients for trainable parameters (the adapters)
        # with torch.set_grad_enabled(any(p.requires_grad for p in self.parameters())):
        # Only track gradients if globally enabled AND there are trainable parameters #added
        with torch.set_grad_enabled(torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())):
            # Obtain the sequence features before final pooling
            # Note: This relies on how your specific CLIP extracts unpooled tokens.
            # You may need to adjust this hook depending on if you use OpenAI CLIP or OpenCLIP.
            try:
                vis_out = self.clip.visual(x)
            except Exception:
                vis_out = self.clip.encode_image(x)

        # Handle the two common output shapes:
        #  - (B*T, D): pooled features
        #  - (B*T, L, D): token features (e.g., [CLS] + patches)
        if vis_out.dim() == 2:
            D = vis_out.shape[1]
            vis_out = vis_out.view(B, T, D)
            return vis_out

        BxT, L, D = vis_out.shape
        vis_out = vis_out.view(B, T, L, D)
        
        # Extract the CLS token for each frame
        cls_tokens = vis_out[:, :, 0, :]  
        
        return cls_tokens # Shape: (B, T, D)

class TemporalConvNetwork(nn.Module):
    """
    TCN module commonly used in CSLR (e.g., VAC [38]).
    It processes the sequence of spatial features to capture local 
    temporal correlations. Typically consists of 1D convolutions 
    and temporal pooling.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Standard CSLR TCN block: Conv1D -> BN -> ReLU -> MaxPool1D
        # This reduces the temporal sequence length by a factor of 4 (2 pooling layers)
        # which is standard for CSLR to reduce CTC decoding length and capture broader context.
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=False),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        """
        x: (B, T, D)
        Returns: (B, T', D') where T' = T // 4
        """
        # Conv1d expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1).contiguous()
        x = self.tcn(x)
        # Permute back to (Batch, Length, Channels) for the LSTM
        x = x.permute(0, 2, 1).contiguous()
        return x

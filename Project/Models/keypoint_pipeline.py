# Pose GraphNet (adapted from your provided gcn.py & rep.net)
# We include GraphNet class that was in your code.
# -------------------------
# We'll re-use class GraphNet and GCN_Block from the provided code, but place them here (cleaned)
# For brevity, copy the GCN_Block and GraphNet classes with minimal modifications:
import torch
import torch.nn as nn
import numpy as np
try:
    from torch_geometric.nn import GCNConv, GraphConv  # ensure torch_geometric installed for training
except Exception:  # pragma: no cover - optional dependency
    GCNConv = None
    GraphConv = None

class ConvBn(nn.Module):
    def __init__(self, deploy, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBn, self).__init__()
        self.deploy = deploy
        if deploy:
            self.conv_reparam = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, bias=True)
        else:
            self.convbn = nn.Sequential()
            self.convbn.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                                     bias=False))
            self.convbn.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        if self.deploy:
            return self.conv_reparam(x)
        else:
            return self.convbn(x)

    def _fuse_bn_tensor(self):
        kernel = self.convbn.conv.weight
        running_mean = self.convbn.bn.running_mean
        running_var = self.convbn.bn.running_var
        gamma = self.convbn.bn.weight
        beta = self.convbn.bn.bias
        eps = self.convbn.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvBn(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding,deploy=deploy)
            self.rbr_1x1 = ConvBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                  stride=stride, padding=0, deploy=deploy)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        
        # If the branch is a ConvBn wrapper (our custom class)
        if isinstance(branch, ConvBn):
            kernel = branch.convbn.conv.weight
            running_mean = branch.convbn.bn.running_mean
            running_var = branch.convbn.bn.running_var
            gamma = branch.convbn.bn.weight
            beta = branch.convbn.bn.bias
            eps = branch.convbn.bn.eps
        # If the branch is just a BatchNorm2d (the identity branch)
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                # Create identity kernel: shape (out_c, in_c, 1, 1)
                kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 0, 0] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            return 0, 0

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        # 1. Fuse the Dense (3x3 or 1x1) branch
        kernel_dense, bias_dense = self._fuse_bn_tensor(self.rbr_dense)
        
        # 2. Fuse the 1x1 branch
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        
        # 3. Fuse the identity branch (BatchNorm)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        
        # Padding logic: If dense is 3x3, we must pad the 1x1 and ID kernels to 3x3
        if self.kernel_size == 3:
            return (kernel_dense + 
                    self._pad_1x1_to_3x3_tensor(kernel_1x1) + 
                    self._pad_1x1_to_3x3_tensor(kernel_id)), \
                   (bias_dense + bias_1x1 + bias_id)
        else:
            return (kernel_dense + kernel_1x1 + kernel_id), (bias_dense + bias_1x1 + bias_id)

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu(), bias.detach().cpu()
    
def convert_to_deploy(model):
    """
    Recursively finds all RepBlocks in the model and converts them 
    to inference-optimized single-conv layers.
    """
    for name, module in model.named_children():
        if isinstance(module, RepBlock):
            # 1. Get the fused kernel and bias
            kernel, bias = module.get_equivalent_kernel_bias()
            
            # 2. Create the new reparameterized convolution
            # Note: module.rbr_dense.convbn.conv contains the stride/padding info
            new_conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=kernel.shape[0],
                kernel_size=module.kernel_size,
                stride=module.rbr_dense.convbn.conv.stride,
                padding=module.rbr_dense.convbn.conv.padding,
                bias=True
            )
            
            # 3. Load the fused weights
            new_conv.weight.data = kernel
            new_conv.bias.data = bias
            
            # 4. Replace the multi-branch module with the single conv + ReLU
            # RepVGG is Conv + ReLU, so we wrap them in a Sequential
            setattr(model, name, nn.Sequential(new_conv, nn.ReLU()))
            print(f"Reparameterized layer: {name}")
        else:
            # Recurse into sub-modules (like GCN_Block or GraphNet)
            convert_to_deploy(module)

# Usage before saving:
# model.eval()
# convert_to_deploy(model)
# torch.save(model.state_dict(), "slt_model_deployed.pth")

def build_graph_star(num_joints=105, video_len=85):
    edge = []
    # Joint offsets based on your extraction order
    OFFSETS = {'pose': 0, 'left': 33, 'right': 54, 'face': 75}
    
    for t in range(video_len):
        base = t * num_joints
        
        # 1. Star Graph for Left Hand (Hub: Left Wrist)
        hub_l = base + OFFSETS['left']
        for i in range(1, 21):
            edge.append([hub_l, hub_l + i])
            
        # 2. Star Graph for Right Hand (Hub: Right Wrist)
        hub_r = base + OFFSETS['right']
        for i in range(1, 21):
            edge.append([hub_r, hub_r + i])
            
        # 3. Star Graph for Face (Hub: Center point)
        hub_f = base + OFFSETS['face']
        for i in range(1, 30):
            edge.append([hub_f, hub_f + i])

        # 4. Temporal Edges (Crucial for Video)
        if t < video_len - 1:
            for j in range(num_joints):
                edge.append([t * num_joints + j, (t + 1) * num_joints + j])

    edge = np.array(edge)
    # Undirect the graph
    edge = np.concatenate([edge, edge[:, [1, 0]]], axis=0)
    return edge.transpose(1, 0)

class GCN_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3,1), deploy=False):
        super(GCN_Block, self).__init__()
        self.conv = RepBlock(in_channel, out_channel, kernel_size=1, stride=1, padding=0, deploy=deploy)
        self.gcn1 = GCNConv(out_channel, out_channel)
        self.gcn2 = GraphConv(out_channel, out_channel)
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=(1,0)),
            nn.BatchNorm2d(out_channel)
        )
        if in_channel != out_channel:
            self.res = RepBlock(in_channel, out_channel, kernel_size=1, stride=1, deploy=deploy)
        else:
            self.res = lambda x: x

        self.relu = nn.ReLU(inplace=True)

    def _batch_edge_index(self, edge_index, batch_size, nodes_per_graph, device):
        """
        Expand edge_index for batched graphs.

        Input
        edge_index : (2, E)
        batch_size
        nodes_per_graph = T * J

        Output
        edge_index_big : (2, B*E)
        """
        edge_index = edge_index.to(device)

        E = edge_index.shape[1]

        # repeat edges B times
        edge_index = edge_index.repeat(1, batch_size)

        # compute offsets
        offsets = torch.arange(
            batch_size,
            device=device
        ).repeat_interleave(E) * nodes_per_graph

        edge_index = edge_index + offsets.unsqueeze(0)

        return edge_index

    def forward(self, x, edge):
        """
        x shape:
        (B, C, T, J)

        edge shape:
        (2, E)
        """

        edge = edge.to(x.device)

        resx = self.res(x)

        x = self.conv(x)

        B, C, T, J = x.shape
        N = T * J

        # reshape to nodes
        x = x.view(B, C, N).permute(0, 2, 1).contiguous()   # (B, N, C)

        # flatten batch
        x = x.view(B * N, C)                                # (B*N, C)

        # build batched edge index
        edge_big = self._batch_edge_index(edge, B, N, x.device)

        # -------- GCN 1 --------
        x = self.gcn1(x, edge_big)

        # reshape back
        x = x.view(B, N, C).permute(0, 2, 1).contiguous()
        x = x.view(B, C, T, J)

        # -------- Temporal Conv --------
        x = self.temporal(x)

        # flatten again
        x = x.view(B, C, N).permute(0, 2, 1).contiguous()
        x = x.view(B * N, C)

        # -------- GCN 2 --------
        x = self.gcn2(x, edge_big)

        # reshape back
        x = x.view(B, N, C).permute(0, 2, 1).contiguous()
        x = x.view(B, C, T, J)

        return self.relu(x + resx)

class GraphNet(nn.Module):
    def __init__(self, in_channel=256, channels=[256, 256, 512, 512], num_joints=4, video_len=85, deploy=False):
        super(GraphNet, self).__init__()
        self.num_joints = num_joints
        self.video_len = video_len
        self.deploy = deploy
        edge = build_graph_star(num_joints=self.num_joints, video_len=self.video_len)
        self.edge = torch.tensor(edge, dtype=torch.long)
        self.gcn1 = GCN_Block(in_channel, channels[0], kernel_size=(3, 1), deploy=deploy)
        self.gcn2 = GCN_Block(channels[0], channels[1], kernel_size=(3, 1), deploy=deploy)
        self.gcn3 = GCN_Block(channels[1], channels[2], kernel_size=(3, 1), deploy=deploy)
        self.gcn4 = GCN_Block(channels[2], channels[3], kernel_size=(3, 1), deploy=deploy)

    def forward(self, x):
        x = self.gcn1(x, self.edge)
        x = self.gcn2(x, self.edge)
        x = self.gcn3(x, self.edge)
        x = self.gcn4(x, self.edge)
        return x


# -------------------------
# 5) Pose pipeline wrapper: from keypoints -> feature vectors per frame
# -------------------------
class PoseEncoder(nn.Module):
    def __init__(self, in_channel=2, channels=[64,128,256,512], num_joints=105, video_len=85):
        super().__init__()
        self.graphnet = GraphNet(in_channel=in_channel, channels=channels, num_joints=num_joints, video_len=video_len)
        # final pooling to per-frame embedding
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = channels[-1]  # final channel dim (512)
    def forward(self, x):
        """
        x shape expected (B, channel, T, num_joints) per earlier design
        but in the SLRNet they used (B, vis_feature_dim, T, num_joints)
        We'll accept same
        """
        out = self.graphnet(x)  # (B, C, T, J)
        # collapse joint dim with pooling or flattening per frame
        B, C, T, J = out.shape
        # For simplicity, produce per-frame embeddings by averaging joints
        per_frame = out.mean(dim=-1)  # (B, C, T)
        per_frame = per_frame.permute(0,2,1).contiguous()  # (B, T, C)
        return per_frame  # (B, T, C)
import numpy as np
import torch
import torch.nn as nn
from smpl_utils_extend import SMPL

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

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

def point_ball_set(nsample, xyz, new_xyz):
    """
    Input:
        nsample: number of points to sample
        xyz: all points, [B, N, 3]
        new_xyz: anchor points [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    _, sort_idx = torch.sort(sqrdists)
    sort_idx=sort_idx[:,:,:nsample]
    batch_idx=torch.arange(B, dtype=torch.long).to(device).view((B,1,1)).repeat((1,S,nsample))
    centroids_idx=torch.arange(S, dtype=torch.long).to(device).view((1,S,1)).repeat((B,1,nsample))
    return group_idx[batch_idx, centroids_idx, sort_idx]

def AnchorInit(x_min=-0.3, x_max=0.3, x_interval=0.3, y_min=-0.3, y_max=0.3, y_interval=0.3, z_min=-0.3, z_max=2.1, z_interval=0.3):#[z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    Input:
        x,y,z min, max and sample interval
    Return:
        centroids: sampled controids [z_size, y_size, x_size, npoint] => [9,3,3,3]
    """
    x_size=round((x_max-x_min)/x_interval)+1
    y_size=round((y_max-y_min)/y_interval)+1
    z_size=round((z_max-z_min)/z_interval)+1
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    centroids = torch.zeros((z_size, y_size, x_size, 3), dtype=torch.float32).to(device)
    for z_no in range(z_size):
        for y_no in range(y_size):
            for x_no in range(x_size):
                lx=x_min+x_no*x_interval
                ly=y_min+y_no*y_interval
                lz=z_min+z_no*z_interval
                centroids[z_no, y_no, x_no, 0]=lx
                centroids[z_no, y_no, x_no, 1]=ly
                centroids[z_no, y_no, x_no, 2]=lz
    return centroids

def AnchorGrouping(anchors, nsample, xyz, points):
    """
    Input:
        anchors: [B, 9*3*3, 3], npoint=9*3*3
        nsample: number of points to sample
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    _, S, _ = anchors.shape
    idx = point_ball_set(nsample, xyz, anchors)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_anchors=anchors.view(B, S, 1, C).repeat(1,1,nsample,1)
    grouped_xyz_norm = grouped_xyz - grouped_anchors #anchors.view(B, S, 1, C)

    grouped_points = index_points(points, idx)
    new_points = torch.cat([grouped_anchors, grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+C+D]
    return new_points

class AnchorPointNet(nn.Module):
    def __init__(self):
        super(AnchorPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+4+3,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x))) #(Batch, feature, frame_point_number)

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        attn_vec=torch.sum(x*attn_weights, dim=1)
        return attn_vec, attn_weights

class AnchorVoxelNet(nn.Module):
    def __init__(self):
        super(AnchorVoxelNet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), padding=(0,0,0))
        self.cb1 = nn.BatchNorm3d(96)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(5, 1, 1))
        self.cb2 = nn.BatchNorm3d(128)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 1, 1))
        self.cb3 = nn.BatchNorm3d(64)
        self.caf3 = nn.ReLU()

    def forward(self, x):
        batch_size=x.size()[0]
        x=x.permute(0, 4, 1, 2, 3)

        x=self.caf1(self.cb1(self.conv1(x)))
        x=self.caf2(self.cb2(self.conv2(x)))
        x=self.caf3(self.cb3(self.conv3(x)))

        x=x.view(batch_size, 64)
        return x

class AnchorRNN(nn.Module):
    def __init__(self):
        super(AnchorRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn)=self.rnn(x, (h0, c0))
        return a_vec, hn, cn

class AnchorModule(nn.Module):
    def __init__(self):
        super(AnchorModule, self).__init__()
        self.template_point=AnchorInit()
        self.z_size, self.y_size, self.x_size, _=self.template_point.shape
        self.anchor_size=self.z_size*self.y_size*self.x_size
        self.apointnet=AnchorPointNet()
        self.avoxel=AnchorVoxelNet()
        self.arnn=AnchorRNN()

    def forward(self, x, g_loc, h0, c0, batch_size, length_size, feature_size):
        g_loc=g_loc.view(batch_size*length_size, 1, 2).repeat(1,self.anchor_size,1)
        anchors=self.template_point.view(1, self.anchor_size, 3).repeat(batch_size*length_size, 1, 1)
        anchors[:,:,:2]+=g_loc
        grouped_points=AnchorGrouping(anchors, nsample=8, xyz=x[..., :3], points=x[..., 3:])
        grouped_points=grouped_points.view(batch_size*length_size*self.anchor_size, 8, 3+feature_size)
        voxel_points, attn_weights=self.apointnet(grouped_points)
        voxel_points=voxel_points.view(batch_size*length_size, self.z_size, self.y_size, self.x_size, 64)
        voxel_vec=self.avoxel(voxel_points)
        voxel_vec=voxel_vec.view(batch_size, length_size, 64)
        a_vec, hn, cn=self.arnn(voxel_vec, h0, c0)
        return a_vec, attn_weights, hn, cn

class BasePointNet(nn.Module):
    def __init__(self):
        super(BasePointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=6,   out_channels=8,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8,  out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        x = in_mat.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)
        x = torch.cat((in_mat[:,:,:4], x), -1)

        return x

class GlobalPointNet(nn.Module):
    def __init__(self):
        super(GlobalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+4,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        attn_vec=torch.sum(x*attn_weights, dim=1)
        return attn_vec, attn_weights

class GlobalRNN(nn.Module):
    def __init__(self):
        super(GlobalRNN, self).__init__()
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=False)
        self.fc1 = nn.Linear(64, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn)=self.rnn(x, (h0, c0))
        g_loc=self.fc1(g_vec)
        g_loc=self.faf1(g_loc)
        g_loc=self.fc2(g_loc)
        return g_vec, g_loc, hn, cn

class GlobalModule(nn.Module):
    def __init__(self):
        super(GlobalModule, self).__init__()
        self.gpointnet=GlobalPointNet()
        self.grnn=GlobalRNN()

    def forward(self, x, h0, c0,  batch_size, length_size):
        x, attn_weights=self.gpointnet(x)
        x=x.view(batch_size, length_size, 64)
        g_vec, g_loc, hn, cn=self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn

class CombineModule(nn.Module):
    def __init__(self):
        super(CombineModule, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 9*6+3+10+1)

    def forward(self, g_vec, a_vec, batch_size, length_size):
        x=torch.cat((g_vec, a_vec), -1)
        x=self.fc1(x)
        x=self.faf1(x)
        x=self.fc2(x)

        q=x[:,:,:9*6].reshape(batch_size*length_size*9, 6).contiguous()
        tmp_x=nn.functional.normalize(q[:,:3], dim=-1)
        tmp_z=nn.functional.normalize(torch.cross(tmp_x, q[:,3:], dim=-1), dim=-1)
        tmp_y=torch.cross(tmp_z, tmp_x, dim=-1)

        tmp_x=tmp_x.view(batch_size, length_size, 9, 3, 1)
        tmp_y=tmp_y.view(batch_size, length_size, 9, 3, 1)
        tmp_z=tmp_z.view(batch_size, length_size, 9, 3, 1)
        q=torch.cat((tmp_x, tmp_y, tmp_z), -1)

        t=x[:,:,9*6  :9*6+3]
        b=x[:,:,9*6+3:9*6+3+10]
        g=x[:,:,9*6+3+10:]
        return q, t, b, g

class SMPLModule(nn.Module):
    def __init__(self):
        super(SMPLModule, self).__init__()
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.blank_atom=torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32, requires_grad=False, device=self.device)
        self.smpl_model_m=SMPL('m')
        self.smpl_model_f=SMPL('f')

    def forward(self, q, t, b, g): #b: (10,)
        batch_size=q.size()[0]
        length_size=q.size()[1]
        q=q.view(batch_size*length_size, 9, 3, 3)
        t=t.view(batch_size*length_size, 1, 3)
        b=b.view(batch_size*length_size, 10)
        g=g.view(batch_size*length_size)
        q_blank=self.blank_atom.repeat(batch_size*length_size, 1, 1, 1)
        pose=torch.cat((q_blank,
                        q[:,1:3,:,:],
                        q_blank,
                        q[:,3:5,:,:],
                        q_blank.repeat(1,10,1,1),
                        q[:,5:9,:,:],
                        q_blank.repeat(1,4,1,1)), 1)
        rotmat=q[:,0,:,:]

        male=g>0.5
        female=g<0.5
        smpl_vertice=torch.zeros((batch_size*length_size, 6890, 3), dtype=torch.float32, requires_grad=False, device=self.device)
        smpl_skeleton=torch.zeros((batch_size*length_size, 24, 3), dtype=torch.float32, requires_grad=False, device=self.device)
        if male.any().item():
            smpl_vertice[male], smpl_skeleton[male] =self.smpl_model_m(b[male], pose[male], torch.zeros((male.sum().item(), 3), dtype=torch.float32, requires_grad=False, device=self.device))
        if female.any().item():
            smpl_vertice[female], smpl_skeleton[female] =self.smpl_model_f(b[female], pose[female], torch.zeros((female.sum().item(), 3), dtype=torch.float32, requires_grad=False, device=self.device))

        smpl_vertice=torch.transpose(torch.bmm(rotmat, torch.transpose(smpl_vertice, 1, 2)), 1,2)+t
        smpl_skeleton=torch.transpose(torch.bmm(rotmat, torch.transpose(smpl_skeleton, 1, 2)), 1,2)+t
        smpl_vertice=smpl_vertice.view(batch_size, length_size, 6890, 3)
        smpl_skeleton=smpl_skeleton.view(batch_size, length_size, 24, 3)
        return smpl_vertice, smpl_skeleton


class mmWaveModel(nn.Module):
    def __init__(self):
        super(mmWaveModel, self).__init__()
        self.module0=BasePointNet()
        self.module1=GlobalModule()
        self.module2=AnchorModule()
        self.module3=CombineModule()
        self.module4=SMPLModule()

    def forward(self, x, gt_g, h0_g, c0_g, h0_a, c0_a):
        batch_size=x.size()[0]
        length_size=x.size()[1]
        pt_size=x.size()[2]
        in_feature_size=x.size()[3]
        out_feature_size=24+4

        x=x.view(batch_size*length_size, pt_size, in_feature_size)
        x=self.module0(x)

        g_vec, g_loc, global_weights, hn_g, cn_g=self.module1(x, h0_g, c0_g, batch_size, length_size)
        a_vec,        anchor_weights, hn_a, cn_a=self.module2(x, g_loc, h0_a, c0_a, batch_size, length_size, out_feature_size)
        q, t, b, g=self.module3(g_vec, a_vec, batch_size, length_size)

        if gt_g is None:
            g_in=torch.round(g)
        else:
            g_in=gt_g
        v,s=self.module4(q,t,b,g_in)
        return q, t, v, s, g_loc, b, g, global_weights, anchor_weights, hn_g, cn_g, hn_a, cn_a

if __name__=='__main__':
    print('AnchorInit:')
    templates=AnchorInit()
    print('\tOutput:', templates.shape) #(9,3,3,3)
    #print(templates)

    print('AnchorGrouping:')
    templates=templates.view(1, 9*3*3, 3)
    #z=torch.zeros((1,1,3))
    print('\tInput:', templates.shape, 'nsample:', 7, templates.shape, templates.shape)
    points=AnchorGrouping(templates, 7, templates, templates)
    print('\tOutput:', points.shape)
    #print(points)

    print('AnchorPointNet:')
    data=torch.rand((7*13, 50, 24+4+3), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape)
    model=AnchorPointNet()
    x,w=model(data)
    print('\tOutput:', x.shape, w.shape)

    print('AnchorVoxelNet:')
    data=torch.rand((7*13, 9, 3, 3, 64), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape)
    model=AnchorVoxelNet()
    x=model(data)
    print('\tOutput:', x.shape)

    print('AnchorRNN:')
    data=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
    h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape, h0.shape, c0.shape)
    model=AnchorRNN()
    a, hn, cn=model(data, h0, c0)
    print('\tOutput:', a.shape,hn.shape, cn.shape)

    print('AnchorModule:')
    data=torch.rand((7* 13, 50, 24+4), dtype=torch.float32, device='cpu')
    g_loc=torch.full((7, 13, 2), 100.0, dtype=torch.float32, device='cpu')
    h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape, g_loc.shape, h0.shape, c0.shape)
    model=AnchorModule()
    a,w,hn,cn=model(data, g_loc, h0, c0, 7, 13, 24+4)
    print('\tOutput:', a.shape, w.shape, hn.shape, cn.shape)

    print('BasePointNet:')
    data=torch.rand((7*13, 50, 6), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape)
    model=BasePointNet()
    x=model(data)
    print('\tOutput:', x.shape)

    print('GlobalPointNet:')
    data=torch.rand((7*13, 50, 24+4), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape)
    model=GlobalPointNet()
    x,w=model(data)
    print('\tOutput:', x.shape, w.shape)

    print('GlobalRNN:')
    data=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
    h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape, h0.shape, c0.shape)
    model=GlobalRNN()
    g, l, hn, cn=model(data, h0, c0)
    print('\tOutput:', g.shape, l.shape, hn.shape, cn.shape)

    print('GlobalModule:')
    data=torch.rand((7*13, 50, 24+4), dtype=torch.float32, device='cpu')
    h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape, h0.shape, c0.shape)
    model=GlobalModule()
    x,l,w,hn,cn=model(data,h0,c0,7,13)
    print('\tOutput:', x.shape, l.shape, w.shape, hn.shape, cn.shape)

    print('CombineModule:')
    g=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
    a=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
    print('\tInput:', g.shape, a.shape)
    model=CombineModule()
    q,t,b,g=model(g, a, 7, 13)
    print('\tOutput:', q.shape, t.shape, b.shape, g.shape)

    print('SMPLModule:')
    q=torch.rand((7, 13, 9, 3, 3), dtype=torch.float32, device='cpu')
    t=torch.rand((7, 13, 3), dtype=torch.float32, device='cpu')
    b=torch.rand((7, 13, 10), dtype=torch.float32, device='cpu')
    gm=torch.ones((4, 13,1), dtype=torch.float32, device='cpu')
    gf=torch.zeros((3, 13,1), dtype=torch.float32, device='cpu')
    g=torch.cat((gm,gf), 0)
    print('\tInput:', q.shape, t.shape, b.shape, g.shape)
    model=SMPLModule()
    v,s=model(q,t,b,g)
    print('\tOutput:', v.shape, s.shape)

    print('WHOLE:')
    data=torch.rand((7, 13, 50, 6), dtype=torch.float32, device='cpu')
    h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
    print('\tInput:', data.shape, g.shape, h0.shape, c0.shape)
    model=mmWaveModel()
    q, t, v, s, l, b, g, _, _, _, _, _, _=model(data, g, h0, c0, h0, c0)
    print('\tOutput:', q.shape, t.shape, v.shape, s.shape, l.shape, b.shape, g.shape)
    q, t, v, s, l, b, g, _, _, _, _, _, _=model(data, None, h0, c0, h0, c0)
    print('\tOutput:', q.shape, t.shape, v.shape, s.shape, l.shape, b.shape, g.shape)

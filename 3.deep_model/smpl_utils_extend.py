import numpy as np
import torch
import pickle
import os

class SMPL(torch.nn.Module):
    def __init__(self, gender, gpu_ids=[0]):
        super(SMPL, self).__init__()
        if gender not in ['m', 'f']:
            raise ValueError('unconfirmed gender')
        smpl_path = {}
        smpl_path['m'] = './smpl_m.pkl'
        smpl_path['f'] = './smpl_f.pkl'
        with open(smpl_path[gender], 'rb') as f:
            params = pickle.load(f)
        self.J_regressor = torch.from_numpy(
            np.array(params['J_regressor'].todense())
        ).type(torch.float32)
        if 'joint_regressor' in params.keys():
            self.joint_regressor = torch.from_numpy(
                np.array(params['joint_regressor'].T.todense())
            ).type(torch.float32)
        else:
            self.joint_regressor = torch.from_numpy(
                np.array(params['J_regressor'].todense())
            ).type(torch.float32)
        self.weights = torch.from_numpy(params['weights']).type(torch.float32)
        self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float32)
        self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
        self.shapedirs = torch.from_numpy(params['shapedirs']).type(torch.float32)
        self.kintree_table = params['kintree_table']
        self.faces = params['f']

        self.device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
        for name in ['J_regressor', 'joint_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(self.device))

    @staticmethod
    def rodrigues(rvec):
        '''
        Rodrigues's rotation formula that turns axis-angle tenser into roration matrix
        :param rvec: [batch, num_vec, 3]
        :return: [batch, num_vec, 3, 3]
        '''

        batch, num_vec, _ = rvec.shape
        r = rvec.reshape(-1, 1, 3)
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
        m = torch.stack((z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
                         r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
                         -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1).reshape(-1, 3, 3)
        i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) +
                  torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zero(x):
        '''
         Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
        :param x: (batch, 3, 4)
        :return: (batch, 4, 4)
        '''
        ones = torch.tensor([0, 0, 0, 1], dtype=torch.float32).view(1, 4).expand(x.shape[0], -1, -1).to(x.device)
        return torch.cat([x, ones], dim=1)

    @staticmethod
    def pack(x):
        '''
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
        :param x: (batch, _, 4, 1)
        :return: (batch, _, 4, 4)
        '''
        zeros = torch.zeros((x.shape[0], x.shape[1], 4, 3), dtype=torch.float32).to(x.device)
        return torch.cat([zeros, x], dim=-1)

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def forward(self, betas, pose, trans, simplify=False):
        '''
        :param betas: (batch, 10)
        :param pose: (batch, 24, 3) or (batch, 24, 4) or (batch, 24, 3, 3)
        :param trans: (batch, 3)
        :param simplify: if true, pose is not considered
        :return: vertices matrix (batch, 6890, 3) and joint positions (batch, 19, 3) ?
        '''
        extend_flag=False
        if len(betas.shape)==3 and len(trans.shape)==3:
            extend_flag=True
            extend_batch=pose.shape[0]
            extend_length=pose.shape[1]
            betas=betas.view((extend_batch*extend_length, 10))
            trans=trans.view((extend_batch*extend_length, 3))
            if len(pose.shape)==4:
                pose=pose.view((extend_batch*extend_length, 24, pose.shape[-1]))
            elif len(pose.shape)==5:
                pose=pose.view((extend_batch*extend_length, 24, 3, 3))
            else:
                print('SMPL Pose Error!')
                exit()
        else:
            assert len(betas.shape)==2 and len(trans.shape)==2

        batch = betas.shape[0]
        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        parent = {i: id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1])}
        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        if len(pose.shape)==4:
            R_cube_big=pose
        elif pose.shape[-1] == 3:
            R_cube_big = self.rodrigues(pose.reshape(-1, 1, 3)).reshape(batch, -1, 3, 3)
        else:
            R_cube_big = self.quaternion2rotmat(pose)
        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) +
                      torch.zeros((batch, R_cube.shape[1], 3, 3), dtype=torch.float32)).to(self.device)
            lrotmin = (R_cube - I_cube).reshape(batch, -1, 1).squeeze(dim=2)
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))
        results = []
        results.append(
            self.with_zero(torch.cat([R_cube_big[:, 0], J[:, 0, :].reshape(-1, 3, 1)], dim=2)))
        for i in range(1, self.kintree_table.shape[1]):
            res = torch.matmul(results[parent[i]],
                               self.with_zero(torch.cat([R_cube_big[:, i],
                                                         (J[:, i, :] - J[:, parent[i], :]).reshape(-1, 3, 1)], dim=2)))
            results.append(res)
        stacked = torch.stack(results, dim=1)
        zeros = torch.zeros((batch, 24, 1), dtype=torch.float32).to(self.device)
        results = stacked - self.pack(torch.matmul(stacked,
                                                   torch.cat([J, zeros], dim=2).reshape(batch, 24, 4, 1)))
        T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        ones = torch.ones((batch, v_posed.shape[1], 1), dtype=torch.float32).to(self.device)
        rest_shape_h = torch.cat([v_posed, ones], dim=2)
        v = torch.matmul(T, rest_shape_h.reshape(batch, -1, 4, 1))
        v = v.reshape(batch, -1, 4)[..., :3]
        mesh_v = v + trans.reshape(batch, 1, 3)

        joints = torch.tensordot(mesh_v, self.joint_regressor, dims=([1], [1])).transpose(1, 2)

        if extend_flag:
            mesh_v=mesh_v.view(extend_batch, extend_length, 6890, 3)
            joints=joints.view(extend_batch, extend_length, 24, 3)
        return mesh_v, joints

    @staticmethod
    def quaternion2vec(batch_q):
        theta = 2 * torch.acos(batch_q[:, :, -1:])
        vecs = (theta / torch.sin(theta / 2)) * batch_q[:, :, :-1]
        return vecs

    @staticmethod
    def quaternion2rotmat(batch_q):
        '''
        quaternion to rotation matrix
        :param batch_q:  (batch, 24, 4)
        :return:
        '''
        qw, qx, qy, qz = batch_q[:, :, 3], batch_q[:, :, 0], batch_q[:, :, 1], batch_q[:, :, 2]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        dim0 = torch.stack((1.0 - (yy + zz), xy - wz, xz + wy), dim=-1)
        dim1 = torch.stack((xy + wz, 1.0 - (xx + zz), yz - wx), dim=-1)
        dim2 = torch.stack((xz - wy, yz + wx, 1.0 - (xx + yy)), dim=-1)
        m = torch.stack((dim0, dim1, dim2), dim=-2)
        return m

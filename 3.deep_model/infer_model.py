import os
import network as mmwave_model
import data as mmwave_data
import torch
import numpy as np
import torch.nn as nn
import time

V_NO='mmMesh'

class mmwave():
    def __init__(self):
        self.batch_size=32
        self.train_length=64
        self.gpu_id=0
        if torch.cuda.is_available():
            self.device='cuda:%d'%(self.gpu_id)
        else:
            self.device='cpu'
        self.pc_size=128

        self.dataset=mmwave_data.data(self.batch_size, self.train_length, self.pc_size)
        self.model=mmwave_model.mmWaveModel().to(self.device)
        self.name_size=20
        self.act_size=8
        self.test_length_size=self.dataset.test_length_size

    def save_model(self, name):
        torch.save(self.model.state_dict(), './%s/model/'%(V_NO)+name+'.pth')

    def load_model(self, name):
        self.model.load_state_dict(torch.load('./%s/model/'%(V_NO)+name+'.pth', map_location=self.device))

    def infer_all(self):
        self.model.eval()

        q_list=[]
        t_list=[]
        v_list=[]
        s_list=[]
        l_list=[]
        b_list=[]
        g_list=[]
        np_pc=self.dataset.get_test_pc()
        for name_no in range(self.name_size):
            q_name_list=[]
            t_name_list=[]
            v_name_list=[]
            s_name_list=[]
            l_name_list=[]
            b_name_list=[]
            g_name_list=[]
            for act_no in range(self.act_size):
                pc_tensor=torch.tensor([np_pc[name_no, act_no]], dtype=torch.float32, device=self.device)
                h0_g=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
                c0_g=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
                h0_a=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
                c0_a=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
                pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, _, _, _, _, _, _=self.model(pc_tensor, None, h0_g, c0_g, h0_a, c0_a)
                pred_q=pred_q.cpu().detach().numpy()
                pred_t=pred_t.cpu().detach().numpy()
                pred_v=pred_v.cpu().detach().numpy()
                pred_s=pred_s.cpu().detach().numpy()
                pred_l=pred_l.cpu().detach().numpy()
                pred_b=pred_b.cpu().detach().numpy()
                pred_g=pred_g.cpu().detach().numpy()
                q_name_list.append(pred_q)
                t_name_list.append(pred_t)
                v_name_list.append(pred_v)
                s_name_list.append(pred_s)
                l_name_list.append(pred_l)
                b_name_list.append(pred_b)
                g_name_list.append(pred_g)
            q_list.append(q_name_list)
            t_list.append(t_name_list)
            v_list.append(v_name_list)
            s_list.append(s_name_list)
            l_list.append(l_name_list)
            b_list.append(b_name_list)
            g_list.append(g_name_list)
        return np.asarray(q_list), np.asarray(t_list), np.asarray(v_list), np.asarray(s_list), np.asarray(l_list), np.asarray(b_list), np.asarray(g_list), np_pc

if __name__=='__main__':
    m=mmwave()
    m.load_model('batch400000')
    q, t, v, s, l, b, g, pc=m.infer_all()
    with open('./%s/%s.pmat.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, q)
    with open('./%s/%s.trans.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, t)
    with open('./%s/%s.vertices.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, v)
    with open('./%s/%s.skeleton.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, s)
    with open('./%s/%s.delta.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, l)
    with open('./%s/%s.beta.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, b)
    with open('./%s/%s.gender.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, g)
    with open('./%s/%s.test_pc.dat'%(V_NO, V_NO), 'wb') as outfile:
        np.save(outfile, pc)
    m.dataset.close()

import os
import network as mmwave_model
import data as mmwave_data
import torch
import numpy as np
import torch.nn as nn
import time
from smpl_utils_extend import SMPL

V_NO='mmMesh'
os.system('mkdir %s'%(V_NO))
os.system('mkdir %s/model'%(V_NO))

class mmwave():
    def __init__(self):
        self.write_slot=1000
        self.save_slot=1000
        self.batch_size=32
        self.batch_rate=1
        self.train_size=400000
        self.train_length=64
        self.lr=0.001
        self.gpu_id=0
        if torch.cuda.is_available():
            self.device='cuda:%d'%(self.gpu_id)
        else:
            self.device='cpu'
        self.pc_size=128

        self.dataset=mmwave_data.data(self.batch_size, self.train_length, self.pc_size)
        self.model=mmwave_model.mmWaveModel().to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion=nn.L1Loss(reduction='sum')
        self.criterion_gender=self.hinge_loss
        self.cos=nn.CosineSimilarity(-1)
        self.train_eval_size=10
        self.name_size=self.dataset.name_size
        self.act_size=8
        self.vertice_rate=0.001
        self.betas_rate=0.1
        self.test_length_size=self.dataset.test_length_size
        self.test_seg=1
        self.test_length_seg=self.test_length_size//self.test_seg
        root_kp=np.asarray([17,19,16,18,2,5,1,4],dtype=np.int64)
        leaf_kp=np.asarray([19,21,18,20,5,8,4,7],dtype=np.int64)
        self.root_kp=torch.tensor(root_kp, dtype=torch.long, device=self.device)
        self.leaf_kp=torch.tensor(leaf_kp, dtype=torch.long, device=self.device)
        self.male_smpl=SMPL('m')
        self.female_smpl=SMPL('f')

    def hinge_loss(self, x, y):
        return torch.sum(nn.ReLU()(y+(1.0-2*y)*x))

    def angle_loss(self, pred_ske, true_ske):
        batch_size=pred_ske.size()[0]
        length_size=pred_ske.size()[1]
        pred_vec=pred_ske[:,:,self.leaf_kp,:]-pred_ske[:,:,self.root_kp,:]
        true_vec=true_ske[:,:,self.leaf_kp,:]-true_ske[:,:,self.root_kp,:]
        cos_sim=nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
        angle=torch.sum(torch.abs(torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0))/3.14159265358*180.0))
        return angle

    def cal_vs_from_qtbg(self, pquat_tensor, trans_tensor, betas_tensor, gender_tensor, b_size, l_size):
        with torch.no_grad():
            vertice_tensor=torch.zeros((b_size, l_size, 6890, 3), dtype=torch.float32, requires_grad=False, device=self.device)
            ske_tensor=torch.zeros((b_size, l_size, 24, 3), dtype=torch.float32, requires_grad=False, device=self.device)
            wrot_tensor=pquat_tensor[:, :, 0:1, :, :]
            rotmat_tensor=torch.squeeze(wrot_tensor)
            pquat_arr=torch.zeros((b_size, l_size, 24, 3, 3), dtype=torch.float32, requires_grad=False, device=self.device)
            pquat_arr[:, :, :]=torch.eye(3, dtype=torch.float32, requires_grad=False, device=self.device)
            pquat_arr[:,:,[1,2,4,5,16,17,18,19]]=pquat_tensor[:,:,1:]
            male_flag=gender_tensor[:,0,0]>0.5
            female_flag=gender_tensor[:,0,0]<0.5
            if male_flag.any().item():
                vertice_tensor[male_flag], ske_tensor[male_flag]=self.male_smpl(betas_tensor[male_flag], pquat_arr[male_flag], torch.zeros((male_flag.sum().item(), l_size, 3), dtype=torch.float32, requires_grad=False, device=self.device))
            if female_flag.any().item():
                vertice_tensor[female_flag], ske_tensor[female_flag]=self.female_smpl(betas_tensor[female_flag], pquat_arr[female_flag], torch.zeros((female_flag.sum().item(), l_size, 3), dtype=torch.float32, requires_grad=False, device=self.device))

            rotmat_tensor=rotmat_tensor.view(b_size*l_size, 3, 3)
            vertice_tensor=vertice_tensor.view(b_size*l_size, 6890, 3)
            ske_tensor=ske_tensor.view(b_size*l_size, 24, 3)
            trans_tensor=trans_tensor.view(b_size*l_size,1, 3)

            vertice_tensor=torch.transpose(torch.bmm(rotmat_tensor, torch.transpose(vertice_tensor, 1, 2)), 1,2)+trans_tensor
            ske_tensor=torch.transpose(torch.bmm(rotmat_tensor, torch.transpose(ske_tensor, 1, 2)), 1,2)+trans_tensor
            vertice_tensor=vertice_tensor.view(b_size, l_size, 6890, 3)
            ske_tensor=ske_tensor.view(b_size, l_size, 24, 3)
        return vertice_tensor.detach(), ske_tensor.detach()

    def train_once(self):
        self.model.train()

        self.model.zero_grad()

        for i in range(self.batch_rate):
            h0_g=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)
            c0_g=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)
            h0_a=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)
            c0_a=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)

            pc, pquat, trans, betas, gender=self.dataset.next_batch()
            pc_tensor=torch.tensor(pc, dtype=torch.float32, device=self.device)
            pquat_tensor=torch.tensor(pquat, dtype=torch.float32, device=self.device)
            trans_tensor=torch.tensor(trans, dtype=torch.float32, device=self.device)
            betas_tensor=torch.tensor(betas, dtype=torch.float32, device=self.device)
            gender_tensor=torch.tensor(gender, dtype=torch.float32, device=self.device)
            vertice_tensor, ske_tensor=self.cal_vs_from_qtbg(pquat_tensor, trans_tensor, betas_tensor, gender_tensor, self.batch_size, self.train_length)

            pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, _, _, _, _, _, _=self.model(pc_tensor, gender_tensor, h0_g, c0_g, h0_a, c0_a)

            loss=   self.vertice_rate*self.criterion(pred_v, vertice_tensor)+ \
                    self.criterion(pred_s, ske_tensor)+ \
                    self.criterion(pred_l, trans_tensor[...,:2])+ \
                    self.betas_rate*self.criterion(pred_b, betas_tensor)+ \
                    self.criterion_gender(pred_g, gender_tensor)
            loss.backward()

        self.optimizer.step()
        return

    def loss_model(self):
        self.model.eval()
        with torch.no_grad():

            train_pquat_loss=0.0
            train_trans_loss=0.0
            train_vertice_loss=0.0
            train_ske_loss=0.0
            train_loc_loss=0.0
            train_betas_loss=0.0
            train_gender_loss=0.0

            train_angle_report=0.0
            train_trans_report=0.0
            train_vertice_report=0.0
            train_ske_report=0.0
            train_loc_report=0.0
            train_betas_report=0.0

            for _ in range(self.train_eval_size):
                pc, pquat, trans, betas, gender=self.dataset.next_batch()

                pc_tensor=torch.tensor(pc, dtype=torch.float32, device=self.device)
                pquat_tensor=torch.tensor(pquat, dtype=torch.float32, device=self.device)
                trans_tensor=torch.tensor(trans, dtype=torch.float32, device=self.device)
                betas_tensor=torch.tensor(betas, dtype=torch.float32, device=self.device)
                gender_tensor=torch.tensor(gender, dtype=torch.float32, device=self.device)
                vertice_tensor, ske_tensor=self.cal_vs_from_qtbg(pquat_tensor, trans_tensor, betas_tensor, gender_tensor, self.batch_size, self.train_length)
                h0_g=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)
                c0_g=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)
                h0_a=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)
                c0_a=torch.zeros((3, self.batch_size, 64), dtype=torch.float32, device=self.device)

                pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, _, _, _, _, _, _=self.model(pc_tensor, gender_tensor, h0_g, c0_g, h0_a, c0_a)

                train_pquat_loss+=self.criterion(pred_q, pquat_tensor).item()
                train_trans_loss+=self.criterion(pred_t, trans_tensor).item()
                train_vertice_loss+=self.criterion(pred_v, vertice_tensor).item()
                train_ske_loss+=self.criterion(pred_s, ske_tensor).item()
                train_loc_loss+=self.criterion(pred_l, trans_tensor[...,:2]).item()
                train_betas_loss+=self.criterion(pred_b, betas_tensor).item()
                train_gender_loss+=self.criterion_gender(pred_g, gender_tensor).item()

                train_angle_report+=self.angle_loss(pred_s, ske_tensor)
                train_trans_report+=torch.sum(torch.sqrt(torch.sum(torch.square(pred_t-trans_tensor), dim=-1))).item()
                train_vertice_report+=torch.sum(torch.sqrt(torch.sum(torch.square(pred_v-vertice_tensor), dim=-1))).item()
                train_ske_report+=torch.sum(torch.sqrt(torch.sum(torch.square(pred_s-ske_tensor), dim=-1))).item()
                train_loc_report+=torch.sum(torch.sqrt(torch.sum(torch.square(pred_l-trans_tensor[...,:2]), dim=-1))).item()
                train_betas_report+=torch.sum(self.cos(pred_b, betas_tensor)).item()

            train_pquat_loss=                      train_pquat_loss/self.batch_size/self.train_eval_size/self.train_length
            train_trans_loss=                      train_trans_loss/self.batch_size/self.train_eval_size/self.train_length
            train_vertice_loss=self.vertice_rate*train_vertice_loss/self.batch_size/self.train_eval_size/self.train_length
            train_ske_loss=                          train_ske_loss/self.batch_size/self.train_eval_size/self.train_length
            train_loc_loss=                      train_loc_loss/self.batch_size/self.train_eval_size/self.train_length
            train_betas_loss=      self.betas_rate*train_betas_loss/self.batch_size/self.train_eval_size/self.train_length
            train_gender_loss=                    train_gender_loss/self.batch_size/self.train_eval_size/self.train_length

            train_angle_report=                      train_angle_report/self.batch_size/self.train_eval_size/self.train_length/8
            train_trans_report=                  train_trans_report/self.batch_size/self.train_eval_size/self.train_length
            train_vertice_report=              train_vertice_report/self.batch_size/self.train_eval_size/self.train_length/6890
            train_ske_report=                      train_ske_report/self.batch_size/self.train_eval_size/self.train_length/24
            train_loc_report=                  train_loc_report/self.batch_size/self.train_eval_size/self.train_length
            train_betas_report=                  train_betas_report/self.batch_size/self.train_eval_size/self.train_length 

            pquat_loss=0.0
            trans_loss=0.0
            vertice_loss=0.0
            ske_loss=0.0
            loc_loss=0.0
            betas_loss=0.0
            gender_loss=0.0

            angle_report=0.0
            trans_report=0.0
            vertice_report=0.0
            ske_report=0.0
            loc_report=0.0
            betas_report=0.0
            gender_acc=0.0

            np_pc=self.dataset.get_test_pc()
            np_pquat=self.dataset.test_pquat
            np_trans=self.dataset.test_trans

            for name_no in range(self.name_size):
                betas_tensor=torch.tensor(np.expand_dims(self.dataset.betas[name_no:name_no+1], 0), dtype=torch.float32, device=self.device)
                gender_tensor=torch.tensor(np.expand_dims(self.dataset.gender[name_no:name_no+1], 0), dtype=torch.float32, device=self.device)
                for act_no in range(self.act_size):
                    pc_tensor=torch.tensor(          np.expand_dims(np_pc[name_no, act_no, :self.test_length_seg], 0), dtype=torch.float32, device=self.device)
                    pquat_tensor=torch.tensor(    np.expand_dims(np_pquat[name_no, act_no, :self.test_length_seg], 0), dtype=torch.float32, device=self.device)
                    trans_tensor=torch.tensor(    np.expand_dims(np_trans[name_no, act_no, :self.test_length_seg], 0), dtype=torch.float32, device=self.device)
                    vertice_tensor, ske_tensor=self.cal_vs_from_qtbg(pquat_tensor, trans_tensor, betas_tensor.repeat(1,self.test_length_seg,1), gender_tensor.repeat(1,self.test_length_seg,1), 1, self.test_length_seg)
                    h0_g=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
                    c0_g=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
                    h0_a=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
                    c0_a=torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)

                    pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g,  _, _, hn_g, cn_g, hn_a, cn_a=self.model(pc_tensor, None, h0_g, c0_g, h0_a, c0_a)

                    pquat_loss+=  self.criterion(pred_q, pquat_tensor).item()
                    trans_loss+=  self.criterion(pred_t, trans_tensor).item()
                    vertice_loss+=self.criterion(pred_v, vertice_tensor).item()
                    ske_loss+=    self.criterion(pred_s, ske_tensor).item()
                    loc_loss+=  self.criterion(pred_l, trans_tensor[...,:2]).item()
                    betas_loss+=  self.criterion(pred_b, betas_tensor.repeat(1,self.test_length_seg,1)).item()
                    gender_loss+= self.criterion_gender(pred_g, gender_tensor.repeat(1,self.test_length_seg,1)).item()

                    angle_report  +=self.angle_loss(pred_s, ske_tensor)
                    trans_report  +=torch.sum(torch.sqrt(torch.sum(torch.square(pred_t-trans_tensor), dim=-1))).item()
                    vertice_report+=torch.sum(torch.sqrt(torch.sum(torch.square(pred_v-vertice_tensor), dim=-1))).item()
                    ske_report    +=torch.sum(torch.sqrt(torch.sum(torch.square(pred_s-ske_tensor), dim=-1))).item()
                    loc_report  +=torch.sum(torch.sqrt(torch.sum(torch.square(pred_l-trans_tensor[...,:2]), dim=-1))).item()
                    betas_report  +=torch.sum(self.cos(pred_b, betas_tensor)).item()
                    pred_g[pred_g>0.5]=1.0
                    pred_g[pred_g<=0.5]=0.0
                    gender_acc+=(1.0-torch.sum(torch.abs(pred_g-gender_tensor)).item()/self.test_length_seg)/self.test_seg 
                    for seg_no in range(1, self.test_seg):
                        pc_tensor=torch.tensor(          [np_pc[name_no, act_no, self.test_length_seg*seg_no:self.test_length_seg*(seg_no+1)]], dtype=torch.float32, device=self.device)
                        pquat_tensor=torch.tensor(    [np_pquat[name_no, act_no, self.test_length_seg*seg_no:self.test_length_seg*(seg_no+1)]], dtype=torch.float32, device=self.device)
                        trans_tensor=torch.tensor(    [np_trans[name_no, act_no, self.test_length_seg*seg_no:self.test_length_seg*(seg_no+1)]], dtype=torch.float32, device=self.device)
                        vertice_tensor, ske_tensor=self.cal_vs_from_qtbg(pquat_tensor, trans_tensor, betas_tensor.repeat(1,self.test_length_seg,1), gender_tensor.repeat(1,self.test_length_seg,1), 1, self.test_length_seg)

                        pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g,  _, _, hn_g, cn_g, hn_a, cn_a=self.model(pc_tensor, None, hn_g, cn_g, hn_a, cn_a)

                        pquat_loss  +=self.criterion(pred_q, pquat_tensor).item()
                        trans_loss  +=self.criterion(pred_t, trans_tensor).item()
                        vertice_loss+=self.criterion(pred_v, vertice_tensor).item()
                        ske_loss    +=self.criterion(pred_s, ske_tensor).item()
                        loc_loss+=  self.criterion(pred_l, trans_tensor[...,:2]).item()
                        betas_loss  +=self.criterion(pred_b, betas_tensor.repeat(1,self.test_length_seg,1)).item()
                        gender_loss +=self.criterion_gender(pred_g, gender_tensor.repeat(1,self.test_length_seg,1)).item()

                        angle_report  +=self.angle_loss(pred_s, ske_tensor)
                        trans_report  +=torch.sum(torch.sqrt(torch.sum(torch.square(pred_t-trans_tensor), dim=-1))).item()
                        vertice_report+=torch.sum(torch.sqrt(torch.sum(torch.square(pred_v-vertice_tensor), dim=-1))).item()
                        ske_report    +=torch.sum(torch.sqrt(torch.sum(torch.square(pred_s-ske_tensor), dim=-1))).item()
                        loc_report  +=torch.sum(torch.sqrt(torch.sum(torch.square(pred_l-trans_tensor[...,:2]), dim=-1))).item()
                        betas_report  +=torch.sum(self.cos(pred_b, betas_tensor)).item()
                        pred_g[pred_g>0.5]=1.0
                        pred_g[pred_g<=0.5]=0.0
                        gender_acc+=(1.0-torch.sum(torch.abs(pred_g-gender_tensor)).item()/self.test_length_seg)/self.test_seg

            pquat_loss  /=(self.name_size*self.act_size*self.test_length_size)
            trans_loss  /=(self.name_size*self.act_size*self.test_length_size)
            vertice_loss/=(self.name_size*self.act_size*self.test_length_size/self.vertice_rate)
            ske_loss    /=(self.name_size*self.act_size*self.test_length_size)
            loc_loss  /=(self.name_size*self.act_size*self.test_length_size)
            betas_loss  /=(self.name_size*self.act_size*self.test_length_size/self.betas_rate)
            gender_loss /=(self.name_size*self.act_size*self.test_length_size)

            angle_report  /=(self.name_size*self.act_size*self.test_length_size*8)
            trans_report  /=(self.name_size*self.act_size*self.test_length_size)
            vertice_report/=(self.name_size*self.act_size*self.test_length_size*6890)
            ske_report    /=(self.name_size*self.act_size*self.test_length_size*24)
            loc_report  /=(self.name_size*self.act_size*self.test_length_size)
            betas_report  /=(self.name_size*self.act_size*self.test_length_size)
            gender_acc    /=(self.name_size*self.act_size)
        return train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, train_angle_report, train_trans_report, train_vertice_report, train_ske_report, train_loc_report, train_betas_report, \
                     pquat_loss,       trans_loss,       vertice_loss,       ske_loss,       loc_loss,       betas_loss,      gender_loss,       angle_report,        trans_report,       vertice_report,       ske_report,       loc_report,       betas_report, gender_acc

    def train_model(self):
        lossfile=open('./%s/log-loss.txt'%(V_NO), 'w')
        evalfile=open('./%s/log-eval.txt'%(V_NO), 'w')
        train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, train_angle_report, train_trans_report, train_vertice_report, train_ske_report, train_loc_report, train_betas_report, \
              pquat_loss,       trans_loss,       vertice_loss,       ske_loss,        loc_loss,      betas_loss,      gender_loss,       angle_report,       trans_report,       vertice_report,       ske_report,        loc_report,       betas_report, gender_acc =self.loss_model()
        lossfile.write('%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n'%(0, train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, pquat_loss, trans_loss, vertice_loss, ske_loss, loc_loss, betas_loss, gender_loss))
        lossfile.flush()
        evalfile.write('%d %f %f %f %f %f %f %f %f %f %f %f %f %f\n'%(0, train_angle_report, train_trans_report, train_vertice_report, train_ske_report, train_loc_report, train_betas_report, angle_report, trans_report, vertice_report, ske_report, loc_report, betas_report, gender_acc))
        evalfile.flush()
        print('%6d || %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f || %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f'%(0, train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, pquat_loss, trans_loss, vertice_loss, ske_loss, loc_loss, betas_loss, gender_loss))

        begin_time=time.time()
        for i in range(self.train_size):
            self.train_once()
            if (i+1)%self.write_slot==0:
                train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, train_angle_report, train_trans_report, train_vertice_report, train_ske_report, train_loc_report, train_betas_report, \
                      pquat_loss,       trans_loss,       vertice_loss,       ske_loss,        loc_loss,      betas_loss,      gender_loss,       angle_report,       trans_report,       vertice_report,       ske_report,        loc_report,       betas_report, gender_acc =self.loss_model()
                lossfile.write('%d %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n'%(i+1, train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, pquat_loss, trans_loss, vertice_loss, ske_loss, loc_loss, betas_loss, gender_loss))
                lossfile.flush()
                evalfile.write('%d %f %f %f %f %f %f %f %f %f %f %f %f %f\n'%(i+1, train_angle_report, train_trans_report, train_vertice_report, train_ske_report, train_loc_report, train_betas_report, angle_report, trans_report, vertice_report, ske_report, loc_report, betas_report, gender_acc))
                evalfile.flush()
                current_time=time.time()
                diff_time=(current_time-begin_time)/3600
                print('%6d || %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f || %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f || %5.2f / %5.2f'%(i+1, train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, pquat_loss, trans_loss, vertice_loss, ske_loss, loc_loss, betas_loss, gender_loss, diff_time, diff_time/(i+1)*self.train_size))
            if (i+1) < 1000 and (i+1)%100==0:
                current_time=time.time()
                diff_time=(current_time-begin_time)/3600
                train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, train_angle_report, train_trans_report, train_vertice_report, train_ske_report, train_loc_report, train_betas_report, \
                      pquat_loss,       trans_loss,       vertice_loss,       ske_loss,        loc_loss,      betas_loss,      gender_loss,       angle_report,       trans_report,       vertice_report,       ske_report,        loc_report,       betas_report, gender_acc =self.loss_model()
                print('%6d || %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f || %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f | %7.3f || %5.2f / %5.2f'%(i+1, train_pquat_loss, train_trans_loss, train_vertice_loss, train_ske_loss, train_loc_loss, train_betas_loss, train_gender_loss, pquat_loss, trans_loss, vertice_loss, ske_loss, loc_loss, betas_loss, gender_loss, diff_time, diff_time/(i+1)*self.train_size))
            if (i+1)%self.save_slot==0:
                self.save_model('batch%d'%(i+1))

    def save_model(self, name):
        torch.save(self.model.state_dict(), './%s/model/'%(V_NO)+name+'.pth')

    def load_model(self, name):
        self.model.load_state_dict(torch.load('./%s/model/'%(V_NO)+name+'.pth', map_location=self.device))

if __name__=='__main__':
    m=mmwave()
    m.train_model()
    m.dataset.close()

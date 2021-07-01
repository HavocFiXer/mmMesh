import numpy as np
import pickle as pk
import multiprocessing
import time

class data():
    def __init__(self, batch_size, batch_length, pc_size):
        self.batch_size=batch_size
        self.batch_length=batch_length
        self.pc_size=pc_size

        self.pquat_sample_list=[0,1,2,4,5,16,17,18,19]
        self.trans=np.load('./trans.mcl.20p8a.dat').reshape((20, 8, 3000, 3))
        self.pquat=np.load('./pmat.mcl.20p8a.dat')[...,self.pquat_sample_list,:,:].reshape((20, 8, 3000, 9, 3, 3))
        self.betas=np.load('./betas.mcl.20p8a.dat') #(20,10)
        self.gender=np.load('./gender.mcl.20p8a.dat') #(20,1)
        self.pc=pk.load(open('./pc.mcl.20p8a.dat', 'rb'))

        print('load trans:', self.trans.shape)
        print('load pquat:', self.pquat.shape)
        print('load betas:', self.betas.shape)
        print('load gender:', self.gender.shape)
        print('load pc:', len(self.pc), len(self.pc[0]), len(self.pc[0][0]))

        assert self.trans.shape[0]==self.pquat.shape[0]
        assert self.trans.shape[1]==self.pquat.shape[1]
        assert self.trans.shape[2]==self.pquat.shape[2]
        assert self.trans.shape[0]==self.betas.shape[0]
        assert self.trans.shape[0]==self.gender.shape[0]
        assert self.trans.shape[0]==len(self.pc)
        assert self.trans.shape[1]==len(self.pc[0])
        assert self.trans.shape[2]==len(self.pc[0][0])

        self.name_size=self.trans.shape[0] #8
        self.act_size=self.trans.shape[1] #7
        self.total_length_size=self.trans.shape[2] #3000
        self.train_length_size=2400
        self.test_length_size=self.total_length_size-self.train_length_size

        # train
        self.train_trans=self.trans[:,:,:self.train_length_size]
        self.train_pquat=self.pquat[:,:,:self.train_length_size]
        self.train_pc=[[x[:self.train_length_size] for x in y] for y in self.pc]

        # test
        self.test_trans=self.trans[:,:,self.train_length_size:]
        self.test_pquat=self.pquat[:,:,self.train_length_size:]
        self.test_pc=[[x[self.train_length_size:] for x in y] for y in self.pc]

        # multi-processing
        self.next_buffer_size=128
        self.test_buffer_size=2
        self.next_extra_process_size=16
        
        self.q_next_batch=multiprocessing.Manager().Queue()
        self.q_test=multiprocessing.Manager().Queue()
        self.v_flag=multiprocessing.Manager().Value('b', True)

        self.task_test=multiprocessing.Process(target=self.maintain_test_pc)
        self.task_next=multiprocessing.Process(target=self.maintain_next_batch, args=(True,))
        self.task_list=[]
        for i in range(self.next_extra_process_size):
            self.task_list.append(multiprocessing.Process(target=self.maintain_next_batch, args=(False,)))

        self.task_next.start()
        self.task_test.start()
        for i in range(self.next_extra_process_size):
            self.task_list[i].start()

    def close(self):
        self.v_flag.value=False
        print('set to close', self.v_flag)
        time.sleep(10)
        while not self.q_next_batch.empty():
            self.q_next_batch.get()
        while not self.q_test.empty():
            self.q_test.get()
        self.task_next.join()
        self.task_test.join()
        for i in range(self.next_extra_process_size):
            self.task_list[i].join()

    def maintain_test_pc(self):
        while self.v_flag.value:
            if self.q_test.qsize()<self.test_buffer_size:
                self.q_test.put(self.prepare_test_pc())
            else:
                time.sleep(1)
        print('test pc closed')

    def maintain_next_batch(self, print_flag):
        while self.v_flag.value:
            if self.q_next_batch.qsize()<self.next_buffer_size:
                self.q_next_batch.put(self.prepare_next_batch())
            else:
                time.sleep(0.2)
        print('next batch closed')

    def prepare_test_pc(self):
        pc_test_list=[]
        for name_no in range(self.name_size):
            pc_name_list=[]
            for act_no in range(self.act_size):
                pc_tmp_list=[]
                for j in range(self.test_length_size):
                    pc_tmp=np.zeros((self.pc_size, 6), dtype=np.float32)
                    pc_no=self.test_pc[name_no][act_no][j].shape[0]
                    if pc_no<self.pc_size:
                        fill_list=np.random.choice(self.pc_size, size=pc_no, replace=False)
                        fill_set=set(fill_list)
                        pc_tmp[fill_list]=self.test_pc[name_no][act_no][j]
                        dupl_list=[x for x in range(self.pc_size) if x not in fill_set]
                        dupl_pc=np.random.choice(pc_no, size=len(dupl_list), replace=True)
                        pc_tmp[dupl_list]=self.test_pc[name_no][act_no][j][dupl_pc]
                    else:
                        pc_list=np.random.choice(pc_no, size=self.pc_size, replace=False)
                        pc_tmp=self.test_pc[name_no][act_no][j][pc_list]
                    pc_tmp_list.append(pc_tmp)
                pc_name_list.append(pc_tmp_list)
            pc_test_list.append(pc_name_list)
        return np.asarray(pc_test_list)

    def prepare_next_batch(self):
        batch_pc_list=[]
        batch_trans_list=[]
        batch_pquat_list=[]
        batch_betas_list=[]
        batch_gender_list=[]
        length_list=np.random.choice(self.train_length_size-self.batch_length+1, size=self.batch_size, replace=True)
        name_list=np.random.choice(self.name_size, size=self.batch_size, replace=True)
        act_list=np.random.choice(self.act_size, size=self.batch_size, replace=True)
        for i in range(self.batch_size):
            length_begin=length_list[i]
            name_choice=name_list[i]
            act_choice=act_list[i]
            batch_trans_list.append(self.train_trans[name_choice, act_choice, length_begin: length_begin+self.batch_length])
            batch_pquat_list.append(self.train_pquat[name_choice, act_choice, length_begin: length_begin+self.batch_length])
            batch_betas_list.append(np.tile([self.betas[name_choice]], (self.batch_length, 1)))
            batch_gender_list.append(np.tile([self.gender[name_choice]], (self.batch_length, 1)))
            pc_tmp_list=[]
            for j in range(length_begin, length_begin+self.batch_length):
                pc_tmp=np.zeros((self.pc_size, 6), dtype=np.float32)
                pc_no=self.train_pc[name_choice][act_choice][j].shape[0]
                if pc_no<self.pc_size:
                    fill_list=np.random.choice(self.pc_size, size=pc_no, replace=False)
                    fill_set=set(fill_list)
                    pc_tmp[fill_list]=self.train_pc[name_choice][act_choice][j]
                    dupl_list=[x for x in range(self.pc_size) if x not in fill_set]
                    dupl_pc=np.random.choice(pc_no, size=len(dupl_list), replace=True)
                    pc_tmp[dupl_list]=self.train_pc[name_choice][act_choice][j][dupl_pc]
                else:
                    pc_list=np.random.choice(pc_no, size=self.pc_size, replace=False)
                    pc_tmp=self.train_pc[name_choice][act_choice][j][pc_list]
                pc_tmp_list.append(pc_tmp)
            batch_pc_list.append(pc_tmp_list)
        return np.asarray(batch_pc_list), np.asarray(batch_pquat_list), np.asarray(batch_trans_list), np.asarray(batch_betas_list), np.asarray(batch_gender_list)

    def get_test_pc(self):
        while self.q_test.empty():
            time.sleep(0.1)
        return self.q_test.get()

    def next_batch(self):
        while self.q_next_batch.empty():
            time.sleep(0.1)
        return self.q_next_batch.get()

if __name__=='__main__':
    d=data(3, 11, 180)
    print('total:', len(d.pc), len(d.pc[0]), len(d.pc[0][0]), d.pquat.shape, d.trans.shape, d.betas.shape, d.gender.shape)
    print('train:', len(d.train_pc), len(d.train_pc[0]), len(d.train_pc[0][0]), d.train_pquat.shape, d.train_trans.shape)
    print('test:', len(d.test_pc), len(d.test_pc[0]), len(d.test_pc[0][0]), d.test_pquat.shape, d.test_trans.shape)
    p, q, t, b, g = d.next_batch()
    print('Next batch:', p.shape, q.shape, t.shape, b.shape, g.shape)
    p=d.get_test_pc()
    print('Test pc:', p.shape)
    time.sleep(10)
    d.close()

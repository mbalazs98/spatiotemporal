import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import set_seed
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import os

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.build_model()
        self.init_model()

        self.init_pos = []
        if self.config.model_type != 'snn':
            for i in range(len(self.blocks)):
                self.init_pos.append(np.copy(self.blocks[i][0][0].P.cpu().detach().numpy()))


    def optimizers(self):
        ##################################
        #  returns a list of optimizers
        ##################################
        optimizers_return = []

        
        if self.config.optimizer_w == 'adam':
            optimizers_return.append(optim.Adam([{'params':self.weights, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                                    {'params':self.weights_bn, 'lr':self.config.lr_w, 'weight_decay':0}]))
        if self.config.model_type == 'snn_delays':
            if self.config.optimizer_pos == 'adam':
                optimizers_return.append(optim.Adam(self.positions, lr = self.config.lr_pos, weight_decay=0))

        return optimizers_return




    def schedulers(self, optimizers):
        ##################################
        #  returns a list of schedulers
        #  if self.config.scheduler_x is none:  list will be empty
        ##################################
        schedulers_return = []

        if self.config.scheduler_w == 'one_cycle':
            schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[0], max_lr=self.config.max_lr_w,
                                                                            total_steps=self.config.epochs))
        elif self.config.scheduler_w == 'cosine_a':
            schedulers_return.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0],
                                                                                    T_max = self.config.t_max_w))

        if self.config.model_type == 'snn_delays':
            if self.config.scheduler_pos == 'one_cycle':
                schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[1], max_lr=self.config.max_lr_pos,
                                                                            total_steps=self.config.epochs))
            elif self.config.scheduler_pos == 'cosine_a':
                schedulers_return.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1],
                                                                                    T_max = self.config.t_max_pos))

        return schedulers_return



    def calc_loss(self, output, y):

        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)
        CEloss = nn.CrossEntropyLoss()
        loss = CEloss(m, y)
        return loss



    def calc_metric(self, output, y):
        # mean accuracy over batch
        softmax_fn = nn.Softmax(dim=2)
        m = torch.sum(softmax_fn(output), 0)

        return np.mean((torch.max(y,1)[1]==torch.max(m,1)[1]).detach().cpu().numpy())


    def train_model(self, train_loader, valid_loader, test_loader, device):

        #######################################################################################
        #           Main Training Loop for all models
        #
        #
        #
        ##################################    Initializations    #############################

        set_seed(self.config.seed)


        optimizers = self.optimizers()
        schedulers = self.schedulers(optimizers)

        ##################################    Train Loop    ##############################


        loss_epochs = {'train':[], 'valid':[] , 'test':[]}
        metric_epochs = {'train':[], 'valid':[], 'test':[]}
        best_metric_val = 0 #1e6
        best_metric_test = 0 #1e6
        best_loss_val = 1e6

        pre_pos_epoch = self.init_pos.copy()
        pre_pos_5epochs = self.init_pos.copy()
        for epoch in range(self.config.epochs):
            self.train()
            #last element in the tuple corresponds to the collate_fn return
            loss_batch, metric_batch = [], []
            pre_pos = pre_pos_epoch.copy()
            
            for i, (x, y) in enumerate(tqdm(train_loader)):
                # x for shd and ssc is: (batch, time, neurons)
                if self.config.dynamic:
                    self.mask = []
                    for j in range(len(self.blocks)):  
                        self.mask.append(self.blocks[j][0][0].weight.data > 0)
                y = F.one_hot(y, self.config.n_outputs).float()

                x = x.permute(1,0,2).float().to(device)  #(time, batch, neurons)
                y = y.to(device)

                for opt in optimizers: opt.zero_grad()

                output = self.forward(x)
                loss = self.calc_loss(output, y)
                for j in range(len(self.blocks)):
                    loss += self.config.l1_lambda * torch.norm(self.blocks[j][0][0].weight, 1)

                loss.backward()
                if self.config.dynamic:
                    self.grad = []
                    for j in range(len(self.blocks)):  
                        self.grad.append(self.blocks[j][0][0].weight.grad)
                for opt in optimizers: opt.step()

                metric = self.calc_metric(output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                self.reset_model(train=True)

            if self.config.model_type == 'snn_delays':
                pos_logs = {}
                for b in range(len(self.blocks)):
                    pos_logs[f'dpos{b}_epoch'] = np.abs(pre_pos[b] - pre_pos_epoch[b]).mean()
                    pre_pos_epoch[b] = pre_pos[b].copy()

                if epoch%5==0 and epoch>0:
                    for b in range(len(self.blocks)):
                        pos_logs[f'dpos{b}_5epochs'] = np.abs(pre_pos[b] - pre_pos_5epochs[b]).mean()
                        pre_pos_5epochs[b] = pre_pos[b].copy()


            loss_epochs['train'].append(np.mean(loss_batch))
            metric_epochs['train'].append(np.mean(metric_batch))

            for scheduler in schedulers: scheduler.step()
            self.decrease_sig(epoch)



            ##################################    Eval Loop    #########################


            loss_valid, metric_valid = self.eval_model(valid_loader, device)

            loss_epochs['valid'].append(loss_valid)
            metric_epochs['valid'].append(metric_valid)

            if test_loader:
                loss_test, metric_test = self.eval_model(test_loader, device)
            else:
                # could be improved
                loss_test, metric_test = 100, 0

            loss_epochs['test'].append(loss_test)
            metric_epochs['test'].append(metric_test)



            ########################## Logging and Plotting  ##########################

            print(f"=====> Epoch {epoch} : \nLoss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100*metric_epochs['train'][-1]:.2f}%")
            print(f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100*metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100*max(metric_epochs['valid'][-1], best_metric_val):.2f}%")

            if test_loader:
                print(f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100*metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100*max(metric_epochs['test'][-1], best_metric_test):.2f}%")

            if  metric_valid > best_metric_val:#  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                print("# Saving best Metric model...")
                torch.save(self.state_dict(), self.config.save_model_path.replace('REPL', 'Best_ACC'))
                best_metric_val = metric_valid
            
            if  loss_valid < best_loss_val:#  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                print("# Saving best Loss model...")
                torch.save(self.state_dict(), self.config.save_model_path.replace('REPL', 'Best_Loss'))
                best_loss_val = loss_valid
            
            if  metric_test > best_metric_test:#  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                best_metric_test = metric_test



    def eval_model(self, loader, device):
        # the creation of a temporary checkpoint 
        # should be removed as it can lead to errors
        torch.save(self.state_dict(), eventid + '.pt')
        self.eval()
        with torch.no_grad():

            for i in range(len(self.blocks)):
                self.blocks[i][0][0].SIG *= 0
                self.blocks[i][0][0].version = 'max'
                self.blocks[i][0][0].DCK.version = 'max'
            self.round_pos()

            loss_batch, metric_batch = [], []
            for i, (x, y) in enumerate(tqdm(loader)):

                y = F.one_hot(y, self.config.n_outputs).float()

                x = x.permute(1,0,2).float().to(device)
                y = y.to(device)

                output = self.forward(x)
                loss = self.calc_loss(output, y)
                
                metric = self.calc_metric(output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)

                self.reset_model(train=False)

            if self.config.DCLSversion == 'gauss':
                for i in range(len(self.blocks)):
                    self.blocks[i][0][0].version = 'gauss'
                    self.blocks[i][0][0].DCK.version = 'gauss'

            self.load_state_dict(torch.load(eventid + '.pt'), strict=True)
            if os.path.exists(eventid + '.pt'):
                os.remove(eventid + '.pt')
            else:
                print(f"File '{eventid + '.pt'}' does not exist.")

        return np.mean(loss_batch), np.mean(metric_batch)

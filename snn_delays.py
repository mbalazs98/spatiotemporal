import torch
import torch.nn as nn
import torch.nn.functional as F


from layers import Dcls1d, LIF, LI, Dropout_Seq

from model import Model
from utils import set_seed

import numpy as np


class SnnDelays(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
    
    # Try factoring this method
    def build_model(self):

        ########################### Model Description :
        #
        #  self.blocks = (n_layers,  0:weights+bn  |  1: lif+dropout+(synapseFilter) ,  element in sub-block)
        #


        ################################################   First Layer    #######################################################

        self.blocks = [[[Dcls1d(self.config.n_inputs, self.config.n_hidden_neurons, kernel_count=self.config.kernel_count, 
                                dilated_kernel_size = self.config.max_delay, version=self.config.DCLSversion, dynamic=self.config.dynamic, dalean=self.config.dalean)],
                       
                        [Dropout_Seq(self.config.dropout_p)]]]
        
        if self.config.use_batchnorm: self.blocks[0][0].insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))
        self.blocks[0][1].insert(0, LIF(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       detach_reset=self.config.detach_reset))


        ################################################   Hidden Layers    #######################################################

        for i in range(self.config.n_hidden_layers-1):
            self.block = [[Dcls1d(self.config.n_hidden_neurons, self.config.n_hidden_neurons, kernel_count=self.config.kernel_count, 
                                dilated_kernel_size = self.config.max_delay, version=self.config.DCLSversion, dynamic=self.config.dynamic, dalean=self.config.dalean)],
                       
                            [Dropout_Seq(self.config.dropout_p)]]
        
            if self.config.use_batchnorm: self.block[0].insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))
            self.block[1].insert(0, LIF(tau=self.config.init_tau, v_threshold=self.config.v_threshold, 
                                                       detach_reset=self.config.detach_reset))

            self.blocks.append(self.block)


        ################################################   Final Layer    #######################################################


        self.final_block = [[Dcls1d(self.config.n_hidden_neurons, self.config.n_outputs, kernel_count=self.config.kernel_count, 
                                     dilated_kernel_size = self.config.max_delay, version=self.config.DCLSversion, dynamic=self.config.dynamic, dalean=self.config.dalean)]]
        self.final_block.append([LI(tau=self.config.init_tau)])


        self.blocks.append(self.final_block)

        self.model = [l for block in self.blocks for sub_block in block for l in sub_block]
        self.model = nn.Sequential(*self.model)
        #print(self.model)

        self.positions = []
        self.weights = []
        self.weights_bn = []
        for m in self.model.modules():
            if isinstance(m, Dcls1d):
                self.positions.append(m.P)
                self.weights.append(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                self.weights_bn.append(m.weight)
                self.weights_bn.append(m.bias)



    def init_model(self):

        set_seed(self.config.seed)
        self.mask = []

        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.weights ?
                torch.nn.init.kaiming_uniform_(self.blocks[i][0][0].weight, nonlinearity='relu')
                
                if self.config.sparsity_p > 0:
                    with torch.no_grad():
                        is_con = torch.zeros((torch.numel(self.blocks[i][0][0].weight)))
                        #er_sparsity_p = self.config.sparsity_p * (1 - (sum(self.blocks[i][0][0].weight.shape)) / (torch.numel(self.blocks[i][0][0].weight)))
                        #num_connections = int((1-er_sparsity_p) * torch.numel(self.blocks[i][0][0].weight))
                        num_connections = int((1-self.config.sparsity_p) * torch.numel(self.blocks[i][0][0].weight))
                        ind = np.random.choice(np.arange(torch.numel(self.blocks[i][0][0].weight)),size=num_connections,replace=False)
                        is_con[ind] = True
                        is_con = is_con.reshape(self.blocks[i][0][0].weight.shape)
                        self.mask.append(is_con.to(self.blocks[i][0][0].weight.device))
                        self.blocks[i][0][0].weight *= self.mask[i]
                        if self.config.dynamic:
                            self.blocks[i][0][0].weight.data = torch.abs(self.blocks[i][0][0].weight.data)
                                
                        
                        

        if self.config.init_pos_method == 'uniform':
            for i in range(self.config.n_hidden_layers+1):
                # can you replace with self.positions?
                torch.nn.init.uniform_(self.blocks[i][0][0].P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                self.blocks[i][0][0].clamp_parameters()

                if self.config.model_type == 'snn_delays_lr0':
                    self.blocks[i][0][0].P.requires_grad = False

        for i in range(self.config.n_hidden_layers+1):
            # can you replace with self.positions?
            torch.nn.init.constant_(self.blocks[i][0][0].SIG, self.config.sigInit)
            self.blocks[i][0][0].SIG.requires_grad = False


    def reset_model(self, train=True):
        #functional.reset_net(self)

        for i in range(self.config.n_hidden_layers+1):                
            if self.config.sparsity_p > 0:
                with torch.no_grad():
                    self.mask[i] = self.mask[i].to(self.blocks[i][0][0].weight.device)
                    #self.blocks[i][0][0].weight = torch.nn.Parameter(self.blocks[i][0][0].weight * self.mask[i])
                    self.blocks[i][0][0].weight *= self.mask[i]
                    if self.config.dynamic:
                        is_con = self.blocks[i][0][0].weight > 0
                        n_connected = torch.sum(is_con)
                        #er_sparsity_p = self.config.sparsity_p * (1 - (sum(self.blocks[i][0][0].weight.shape)) / (torch.numel(self.blocks[i][0][0].weight)))
                        num_connections = int((1-self.config.sparsity_p) * torch.numel(self.blocks[i][0][0].weight))
                        #num_connections = int((1-er_sparsity_p) * torch.numel(self.blocks[i][0][0].weight))
                        nb_reconnect = num_connections - n_connected
                        if self.config.rigl:
                            self.grad[i][is_con] = float('inf')
                            # Flatten the grad tensor to apply sorting across all dimensions
                            flattened_grad = self.grad[i].view(-1)
                            reconnect_sample_id = torch.sort(flattened_grad, descending=False)[1][:int(nb_reconnect)]
                            if nb_reconnect > 0:
                                with torch.no_grad():
                                    unraveled_indices = torch.unravel_index(reconnect_sample_id, self.grad[i].shape)
                                    self.blocks[i][0][0].weight[unraveled_indices] = 1e-12
                        else:
                            reconnect_candidate_coord = (~is_con).nonzero()
                            reconnect_sample_id = torch.randperm(len(reconnect_candidate_coord))[:int(nb_reconnect)]
                            reconnect_sample_id = torch.randperm(len(reconnect_candidate_coord))[:int(nb_reconnect)]
                            chosen_ind = reconnect_candidate_coord[reconnect_sample_id].T
                            if nb_reconnect > 0:
                                with torch.no_grad():
                                    self.blocks[i][0][0].weight[chosen_ind[0], chosen_ind[1]] = 1e-12    
                        
                        assert int(torch.sum(self.blocks[i][0][0].weight > 0)) == num_connections



        # We use clamp_parameters of the Dcls1d modules
        if train: 
            for block in self.blocks:
                block[0][0].clamp_parameters()






    def decrease_sig(self, epoch):

        # Decreasing to 0.23 instead of 0.5

        alpha = 0
        sig = self.blocks[-1][0][0].SIG[0,0,0,0].detach().cpu().item()
        if self.config.decrease_sig_method == 'exp':
            if epoch < self.config.final_epoch and sig > 0.23:
                if self.config.DCLSversion == 'max':
                    # You have to change this !!
                    alpha = (1/self.config.sigInit)**(1/(self.config.final_epoch))
                elif self.config.DCLSversion == 'gauss':
                    alpha = (0.23/self.config.sigInit)**(1/(self.config.final_epoch))

                for block in self.blocks:
                    block[0][0].SIG *= alpha
                    # No need to clamp after modifying sigma
                    #block[0][0].clamp_parameters()




    def forward(self, x):
        for block_id in range(self.config.n_hidden_layers):
            # x is permuted: (time, batch, neurons) => (batch, neurons, time)  in order to be processed by the convolution
            x = x.permute(1,2,0)
            x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)  # we use padding for the delays kernel

            # we use convolution of delay kernels
            x = self.blocks[block_id][0][0](x)

            # We permute again: (batch, neurons, time) => (time, batch, neurons) in order to be processed by batchnorm or Lif
            x = x.permute(2,0,1)

            if self.config.use_batchnorm:
                # we use x.unsqueeze(3) to respect the expected shape to batchnorm which is (time, batch, channels, length)
                # we do batch norm on the channels since length is the time dimension
                # we use squeeze to get rid of the channels dimension
                x = x.unsqueeze(3)
                shapes = [x.shape[0], x.shape[1]]
                x = x.flatten(0, 1)
                x = self.blocks[block_id][0][1](x)
                shapes.extend(x.shape[1:])
                x= x.view(shapes)
                x = x.squeeze()
            
            
            # we use our spiking neuron filter
            spikes = self.blocks[block_id][1][0](x)
            # we use dropout on generated spikes tensor

            x = self.blocks[block_id][1][1](spikes)
            
            # x is back to shape (time, batch, neurons)
        
        # Finally, we apply same transforms for the output layer
        x = x.permute(1,2,0)
        x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)
        
        # Apply final layer
        out = self.blocks[-1][0][0](x)

        # permute out: (batch, neurons, time) => (time, batch, neurons)  For final spiking neuron filter
        out = out.permute(2,0,1)

        out = self.blocks[-1][1][0](out)


        return out
    

    def round_pos(self):
        with torch.no_grad():
            for i in range(len(self.blocks)):
                self.blocks[i][0][0].P.round_()
                self.blocks[i][0][0].clamp_parameters()

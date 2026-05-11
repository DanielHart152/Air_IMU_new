import torch
import numpy as np
import pypose as pp

import torch.nn as nn
from model.net import ModelBase
from model.cnn import CNNEncoder


class CodeNet(ModelBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        gyro_std = np.pi/180
        if "gyro_std" in conf:
            print(" The gyro std is set to ", conf.gyro_std, " rad/s")
            gyro_std = conf.gyro_std
        self.register_buffer('gyro_std', torch.tensor(gyro_std))

        acc_std = 0.1
        if "acc_std" in conf:
            print(" The acc std is set to ", conf.acc_std, " m/s^2")
            acc_std = conf.acc_std
        self.register_buffer('acc_std', torch.tensor(acc_std))

        ## the encoder have the same correction in one interval 
        self.interval = 9
        self.inter_head = np.floor(self.interval/2.).astype(int)
        self.inter_tail = self.interval - self.inter_head

        self.imu_cnn = CNNEncoder(c_list=[6, 32, 64], k_list=[7, 7], s_list=[3, 3])# acc(3) + gyro(3)
        self.rot_cnn = CNNEncoder(c_list=[3, 16, 32], k_list=[7, 7], s_list=[3, 3])# rot_so3(3) - Lie algebra
        self.d_vel_cnn = CNNEncoder(c_list=[1, 8, 16], k_list=[7, 7], s_list=[3, 3])# vel_z(1)

        self.gru1 = nn.GRU(input_size = 112, hidden_size = 128, num_layers = 1, batch_first = True)# 64+32+16=112
        self.gru2 = nn.GRU(input_size = 128, hidden_size = 256, num_layers = 1, batch_first = True)

        self.accdecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.acccov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

        self.delta_so3_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.delta_so3_cov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

    def encoder(self, imu, rot, d_vel):
        imu_feat = self.imu_cnn(imu.transpose(-1,-2)).transpose(-1,-2)
        rot_feat = self.rot_cnn(rot.transpose(-1,-2)).transpose(-1,-2)
        d_vel_feat = self.d_vel_cnn(d_vel.transpose(-1,-2)).transpose(-1,-2)
        
        x = torch.cat([imu_feat, rot_feat, d_vel_feat], dim=-1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        return x

    def cov_decoder(self, x):
        acc = torch.exp(self.acccov_decoder(x) - 5.)
        delta_so3 = torch.exp(self.delta_so3_cov_decoder(x) - 5.)

        return torch.cat([acc, delta_so3], dim = -1)

    def decoder(self, x):
        acc = self.accdecoder(x) * self.acc_std
        delta_so3 = self.delta_so3_decoder(x) * self.gyro_std

        return torch.cat([acc, delta_so3], dim = -1)

    def _update(self, to_update, feat, frame_len):
        ### Note: This will change the data in the to_update !!!!!!
        def _clip(x,l):
            if x > l:
                return l
            elif x < 0:
                return 0
            else:
                return x

        _feat_range = np.ceil((frame_len-self.inter_head)/self.interval).astype(int) + 1 ## not equivalent to features shape

        for i in range(_feat_range):
            s_p = _clip(i*self.interval-self.inter_head, frame_len)
            e_p = _clip(i*self.interval+self.inter_tail, frame_len)
            idx = _clip(i, feat.shape[1]-1)

            # skip the first padded input
            to_update[:,s_p:e_p,:] += feat[:,idx:idx+1,:]

        return to_update

    def inference(self, data):
        frame_len = data["acc"].shape[1] - self.interval
        imu = torch.cat([data["acc"], data["gyro"]], dim = -1)
        rot = data["rot"].Log().tensor()  # Convert SO3 to so3 Lie algebra (3D)
        d_vel = data["vel"][..., 2:3]
        
        feature = self.encoder(imu, rot, d_vel)[:,1:,:]
        correction = self.decoder(feature)
        zero_signal = torch.zeros_like(data['acc'][:,self.interval:,:])

        # a referenced size 1000
        correction_acc = self._update(zero_signal.clone(), correction[...,:3], frame_len)
        correction_delta_so3 = self._update(zero_signal.clone(), correction[...,3:], frame_len)

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = self._update(torch.zeros_like(correction_acc, device=correction_acc.device),
                                                cov[...,:3], frame_len)
            cov_state['gyro_cov'] = self._update(torch.zeros_like(correction_delta_so3, device=correction_delta_so3.device),
                                                cov[...,3:], frame_len)
        
        return {"cov_state": cov_state, 'correction_acc': correction_acc, 'correction_delta_so3': correction_delta_so3}

    def forward(self, data, init_state):
        inference_state = self.inference(data)

        data['corrected_acc'] = data['acc'][:,self.interval:,:] + inference_state['correction_acc']
        
        # Compose rotation: corrected_rot = gt_rot * exp(delta_so3)
        gt_rot_sliced = data['rot'][:,self.interval:,:]
        delta_so3 = pp.so3(inference_state['correction_delta_so3'])
        data['corrected_rot'] = gt_rot_sliced * delta_so3.Exp()
        
        # Convert corrected rotation to gyro for integration
        data['corrected_gyro'] = data['gyro'][:,self.interval:,:]  # Keep original gyro for now
        data['rot'] = data['corrected_rot']
        data['vel'] = data['vel'][:,self.interval:,:]

        out_state = self.integrate(init_state = init_state, data = data, cov_state = inference_state['cov_state'])

        return {**out_state, 'correction_acc': inference_state['correction_acc'], 'correction_delta_so3': inference_state['correction_delta_so3'], 
                                'corrected_acc': data['corrected_acc'], 'corrected_rot': data['corrected_rot']}


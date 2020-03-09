#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:58:22 2020
@author: abdelpakey
"""
import torch
import torch.nn.functional as F
import numpy as np
x_t =  torch.tensor([1.,2.,3.,56.], requires_grad=True) 
label =torch.tensor([0.,0.,1.,0.])

x2_t = torch.tensor([2.,4.,6.,8.,10.,12.], requires_grad=True)
x_n = np.array([1.,2.,3.,4.,5.,6.])
x2_n = np.array([2.,4.,6.,8.,10.,12.])

loss =torch.dot(x_t, x_t)#torch.sum(torch.max(torch.tensor([0.]),x_t-x_t[label==1]+0))
loss.backward(torch.ones_like(loss))
x_t.grad

sig_n=1/(1+np.exp(-x2_n))

sig_t=torch.sigmoid(x_t)+F.sigmoid(x2_t)

#x_sig.backward(torch.ones_like(x_t))
sig_derv=(sig_t*(1-sig_t))
#sig2_derv=(sig_t*(1-sig_t))
x=5*x_t**2 +x2_t**2
sig_t.backward(torch.ones_like(x_t))
print(x_t.grad)
#y=F.sigmoid(x)

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:06:40 2024

@author: midhu
"""

import torch
print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())

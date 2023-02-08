import random
import numpy as np
import torch

class particle:
    def __init__(self):
        self.position = torch.abs(torch.tensor(np.random.ranf((1,500)), dtype=torch.float32))
        device = "cpu"
        self.position = self.position.to(device)
        self.score = 0
        self.mass = 0
        self.force = 0
        self.history = [self.position]

    def __str__(self):
        return "pos:{}\tscore:{}\tmass:{}\tforce:{}".format(str(self.position), str(self.score), str(self.mass), str(self.force))
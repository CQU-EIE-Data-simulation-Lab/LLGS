import os
import torch
import numpy as np
import torch.nn as nn
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

# 假设输入的点的坐标为（P，3），则输出的编码后的坐标为（P，6*(max-min)）
# 即维度从3维提升到6*(max-min)维
def pos_enc(x, min_deg, max_deg):
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return four_feat
    # print(four_feat)
    # return torch.cat([x] + [four_feat], dim=-1)


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class LTGS(nn.Module):
    def __init__(self, input_size: int, v_size : int, output_size: int = 1, net_width: int = 64):
        super().__init__()
        self.input_size = input_size
        self.v_size = v_size
        self.output_size = output_size
        self.net_width = net_width
        self.factor = 2
        self.active = nn.Sigmoid()
        self.main = nn.Sequential(
            nn.Linear(self.input_size, self.net_width * self.factor),
            self.active,
            nn.Linear(self.net_width * self.factor, self.net_width),
            self.active,
            nn.Linear(self.net_width, self.net_width),
            self.active,
        )
        self.view = nn.Sequential(
            nn.Linear(self.net_width + v_size, self.net_width * self.factor),
            self.active,
            nn.Linear(self.net_width * self.factor, self.net_width),
            self.active,
            nn.Linear(self.net_width, self.net_width),
            self.active,
        )

        self.head = nn.Sequential(
            nn.Linear(self.net_width, self.output_size),
            self.active,
        )

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     nn.init.constant_(self.head[0].weight, 0.01)
    #     nn.init.constant_(self.head[0].bias, 5.0)

    def forward(self, means, shs, R, T):
        shs = shs.view(shs.size(0), -1)
        shs = torch.nn.functional.normalize(shs)
        enc_means = pos_enc(means, 0, 2)
        q = torch.from_numpy(rotmat2qvec(R))
        q = q.unsqueeze(0)
        q = torch.nn.functional.normalize(q)
        t = pos_enc(torch.from_numpy(T), 0, 2)
        t = t.unsqueeze(0)
        viwer = torch.concat([t, q], dim=1)
        viwer = viwer.cuda().to(torch.float32)

        x = torch.concat([enc_means, shs], dim=1)
        # print(x)
        x = self.main(x)
        x = torch.concat([x, viwer.expand(x.size(0), -1)], dim=1)
        # print(x.shape)
        x = self.view(x)
        x = self.head(x)
        # x = torch.exp(-x)
    
        return x


class LightModel:
    def __init__(self, input_size, v_size) -> None:
        self.ligtGS = LTGS(input_size, v_size).cuda()
        self.optimizer = None
        self.lr_scale = 5

    def step(self, means, shs, R, T):
        # for param in list(self.ligtGS.parameters()):
        #     print(f"Shape: {param.shape}, Data Type: {param.dtype}")
        return self.ligtGS(means, shs, R, T)
    
    def train_setting(self, training_args):
        l = [
            {'params': list(self.ligtGS.parameters()),
             'lr': training_args.position_lr_init * self.lr_scale,
             "name": "ligtGS"
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.ligtGS_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.ligtgs_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "ligtgs/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.ligtGS.state_dict(), os.path.join(out_weights_path, 'ligtgs.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "ligtgs"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "ligtgs/iteration_{}/ligtgs.pth".format(loaded_iter))
        self.ligtGS.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "ligtGS":
                lr = self.ligtGS_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr  
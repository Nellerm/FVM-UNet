from models.fvmunet.fvmamba import VSSM
import torch
from torch import nn


class FVMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 use_cfb=True,
                 use_ds=True,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.use_ds = use_ds

        self.fvmunet = VSSM(in_chans=input_channels,
                            num_classes=num_classes,
                            depths=depths,
                            depths_decoder=depths_decoder,
                            drop_path_rate=drop_path_rate,
                            use_cfb=use_cfb,
                            use_ds=use_ds)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # --------------------------
        if self.use_ds:
            logits, gt_pre = self.fvmunet(x)
        else:
            logits = self.fvmunet(x)
        if self.num_classes == 1:
            if self.use_ds:
                return torch.sigmoid(logits), gt_pre
            else:
                return torch.sigmoid(logits)
        else:
            if self.use_ds:
                return logits, gt_pre
            else:
                return logits
        # --------------------------

    def load_from(self):
        if self.load_ckpt_path is not None:
            # 对编码器的加载
            model_dict = self.fvmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.fvmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            # print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            # 对解码器的加载
            model_dict = self.fvmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.fvmunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            # print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
        else:
            print("none pretrain")
import torch
from fvcore.common.config import CfgNode

# models:
from nets.CNN_GCN import CNN_GCN
from nets.EFNet import EFNet
from nets.EFKptsNet import EFKptsNet
from nets.EFKptsSDNet import EFKptsNetSDNet
from nets.EFGCN import EFGCN
from nets.EFGCNTmp import EFGCNTmp
from nets.EFGCNSD import EFGCNSD
import os

def load_freezed_encoder(weights_filename, cfg, model):
        weights_filename = cfg.TRAIN.CHECKPOINT_FILE_PATH
        print("loading file %s.." % weights_filename)
        checkpoint = torch.load(weights_filename)
        # Identify the layer names until 'image_encoder'
        layer_names = []
        for name, _ in model.named_parameters():
            if name.startswith('image_encoder'):   
                layer_names.append(name)
            else:
                break
        state_dict = checkpoint['model_state_dict']
        
        # Create a new state_dict containing only the specified layers' weights
        filtered_state_dict = {name: weight for name, weight in state_dict.items() if any(layer_name in name for layer_name in layer_names)}

        # Load the filtered weights into the model
        model.load_state_dict(filtered_state_dict, strict=False)  # Set strict=False to ignore missing/badly shaped keys
        
        # Freeze the specified layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
        return model

def load_freezed_model(weights_filename: str=None, is_gpu: bool = True,continue_train: bool = False):
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weights_filename is not None and os.path.exists(weights_filename):
        checkpoint = torch.load(weights_filename,map_location=device)
        print('freezed model epoch is %d' % checkpoint['epoch'])
        if 'cfg' in checkpoint:
            print(checkpoint['cfg'])
            cfg = checkpoint['cfg']
            model = load_model(cfg, is_gpu=is_gpu)
            model.load_state_dict(checkpoint['model_state_dict'])

    if continue_train == False:
        for name, p in model.named_parameters():
            #if "kpts_decoder" in name: #freeze image encoder part only
            p.requires_grad = False
    return model

def load_model(cfg:CfgNode, is_gpu: bool = True, kpts_extractor_weights: str = None) -> torch.nn.Module:

    model_name = cfg.MODEL.NAME
    backbone = cfg.MODEL.BACKBONE
    num_kpts = cfg.TRAIN.NUM_KPTS
    kpts_extractor_weights = cfg.KPTS_EXTRACTOR_WEIGHTS

    print("loading network type: {}..".format(model_name))
    if model_name == 'CNNGCN_freezed':
        model = load_freezed_model(weights_filename=kpts_extractor_weights, is_gpu=is_gpu, continue_train= False)
        model.output_type = 'img2kpts'

    elif model_name == 'CNNGCN':
        model = CNN_GCN(kpt_channels=2, gcn_channels=[16, 32, 32, 48], backbone=backbone, num_kpts=num_kpts, is_gpu=is_gpu)
        model.output_type = 'img2kpts'

    elif model_name == 'CNNGCNV2':
        model = CNN_GCN(kpt_channels=2, gcn_channels=[4, 4, 8, 8, 8, 16, 16, 16], backbone=backbone, num_kpts=num_kpts, is_gpu=is_gpu)
        model.output_type = 'img2kpts'

    elif model_name == 'CNNGCNV3':
        model = CNN_GCN(kpt_channels=2, gcn_channels=[4, 8, 8, 16, 16, 32, 32, 48], backbone=backbone, num_kpts=num_kpts, is_gpu=is_gpu)
        model.output_type = 'img2kpts'

    elif model_name == "EFNet":
        model = EFNet(backbone='r3d_18', MLP_channels=[16, 32, 32, 48])
        model.output_type = 'seq2ef'

    elif model_name == "EFKptsNet":
        model = EFKptsNet(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], MLP_channels_kpts=[256, 256, 256])
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCN":
        #model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 2, 3, 3, 4, 4, 4], is_gpu=is_gpu)
        model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 3, 4], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCNTmp":
        model = EFGCNTmp(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 3, 4], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCNV2":
        #model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[2, 2, 3, 3, 4, 4, 4], is_gpu=is_gpu)
        model = EFGCN(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFGCNTmpV2":
        model = EFGCNTmp(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts'

    elif model_name == "EFKptsNetSD":
        model = EFKptsNetSDNet(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], MLP_channels_kpts=[256, 256, 256])
        model.output_type = 'seq2ef&kpts&sd'

    elif model_name == "EFGCNSD":
        model = EFGCNSD(backbone='r3d_18', MLP_channels_ef=[16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
        model.output_type = 'seq2ef&kpts&sd'

    else:
        raise NotImplementedError("Model name is not supported..")

    print('loading model {} of network type: {} and output_type {}..'.format(model_name, model.__class__.__name__, model.output_type))

    return model




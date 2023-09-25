import mindspore as ms
import os

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        if param_dict:
            param_not_load = ms.load_param_into_net(model, param_dict)
            print("param not load:", param_not_load)
    else:
        print(f"!!!Warning!!!: {ckpt} doesn't exist")

    return model


def load_full_model(config, path='./models/wukong/', verbose=False):
    model = instantiate_from_config(config.model)
    param_not_load = []
    if os.path.isdir(path):
        unet = ms.load_checkpoint(os.path.join(path, 'unet.ckpt'))
        param_not_load.extend(ms.load_param_into_net(model.model, unet))
        vae = ms.load_checkpoint(os.path.join(path, 'vae.ckpt'))
        param_not_load.extend(ms.load_param_into_net(model.first_stage_model, vae))
        text_encoder = ms.load_checkpoint(os.path.join(path, 'text_encoder.ckpt'))
        param_not_load.extend(ms.load_param_into_net(model.cond_stage_model, text_encoder))
    else:
        param_dict = ms.load_checkpoint(path)
        param_not_load.extend(ms.load_param_into_net(model, param_dict))
    print("param not load:", param_not_load)
    print("load model from", path)
    return model




if __name__ == '__main__':
    config = 'configs/v1-inference-chinese.yaml'
    config = OmegaConf.load(config)
    model1 = load_full_model(config)
    print('---------------------------------------------')
    model2 = load_full_model(config, path='./models/wukong-huahua-ms.ckpt')

    for (k1, v1), (k2, v2) in zip(model1.parameters_and_names(), model2.parameters_and_names()):
        if k1.startswith('first_stage_model.encoder.down.3.downsample'):
            continue
        if k1.startswith('first_stage_model.decoder.up.0.upsample'):
            continue
        if not (v1 == v2).all():
            print(k1, k2, v1.sum(), v2.sum())
            print('error')
            exit(0)
    print('ok')
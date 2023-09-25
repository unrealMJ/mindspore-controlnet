import mindspore as ms
import os
import numpy as np

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


def test_load_unet(config):
    model = instantiate_from_config(config.model)
    param_not_load = []
    unet = ms.load_checkpoint('/mnt/petrelfs/majie/project/minddiffusion/vision/wukong-huahua/torch2ms/ms_weight/unet.ckpt')
    param_not_load.extend(ms.load_param_into_net(model.model, unet))
    print("param not load:", param_not_load)
    print(len(param_not_load))
    print(len(unet))


def test_load_text_encoder(config):
    model = instantiate_from_config(config.model)
    param_not_load = []
    text_encoder = ms.load_checkpoint('/mnt/petrelfs/majie/project/minddiffusion/vision/wukong-huahua/torch2ms/ms_weight/text_encoder.ckpt')
    param_not_load.extend(ms.load_param_into_net(model.cond_stage_model, text_encoder))
    print("param not load:", param_not_load)
    print(len(param_not_load))
    print(len(text_encoder))


def test_load_vae(config):
    model = instantiate_from_config(config.model)
    param_not_load = []
    vae = ms.load_checkpoint('/mnt/petrelfs/majie/project/minddiffusion/vision/wukong-huahua/torch2ms/ms_weight/vae.ckpt')
    param_not_load.extend(ms.load_param_into_net(model.first_stage_model, vae))
    print("param not load:", param_not_load)
    print(len(param_not_load))
    print(len(vae))


def test_load_controlnet(config):
    model = instantiate_from_config(config.model)
    param_not_load = []
    controlnet = ms.load_checkpoint('/mnt/petrelfs/majie/project/minddiffusion/vision/wukong-huahua/torch2ms/ms_weight/controlnet.ckpt')
    param_not_load.extend(ms.load_param_into_net(model.control_model, controlnet))
    print("param not load:", param_not_load)
    print(len(param_not_load))
    print(len(controlnet))


def test_text_encoder_output(config):
    model = instantiate_from_config(config.model)
    text_encoder = ms.load_checkpoint('/mnt/petrelfs/majie/project/minddiffusion/vision/wukong-huahua/torch2ms/ms_weight/text_encoder.ckpt')
    ms.load_param_into_net(model.cond_stage_model, text_encoder)
    # input_ shape (1, 77) dtype ms.int64
    # 输入要保持一致
    input_ = ms.Tensor(np.arange(77).reshape(1, 77), dtype=ms.int64)
    output = model.cond_stage_model.transformer(input_)
    print(output.shape)
    print(output.sum(), output.min(), output.max())


def test_vae_output(config):
    model = instantiate_from_config(config.model)
    vae = ms.load_checkpoint('/mnt/petrelfs/majie/project/minddiffusion/vision/wukong-huahua/torch2ms/ms_weight/vae.ckpt')
    ms.load_param_into_net(model.first_stage_model, vae)
    # input_ shape (1, 4, 32, 32) dtype ms.float32
    # 输入要保持一致
    input_ = np.ones((1, 3, 256, 256), dtype=np.float32)
    input_ = ms.Tensor(input_, dtype=ms.float32)
    latents = model.first_stage_model.encode(input_)
    print(latents.shape)
    print(latents.sum(), latents.min(), latents.max())

    output = model.first_stage_model.decode(latents)
    print(output.shape)
    print(output.sum(), output.min(), output.max())


def test_tokenizer_output(config):
    model = instantiate_from_config(config.model)
    text = 'a photo of a girl'

    output = model.cond_stage_model.tokenize(text)
    print(output)





if __name__ == '__main__':
    config = 'configs/cldm_v15.yaml'
    config = OmegaConf.load(config)

    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        device_target="GPU",
        device_id=0,
        max_device_memory="30GB"
    )

    # model1 = load_full_model(config)
    # print('---------------------------------------------')
    # model2 = load_full_model(config, path='./models/wukong-huahua-ms.ckpt')

    # for (k1, v1), (k2, v2) in zip(model1.parameters_and_names(), model2.parameters_and_names()):
    #     if k1.startswith('first_stage_model.encoder.down.3.downsample'):
    #         continue
    #     if k1.startswith('first_stage_model.decoder.up.0.upsample'):
    #         continue
    #     if not (v1 == v2).all():
    #         print(k1, k2, v1.sum(), v2.sum())
    #         print('error')
    #         exit(0)
    # print('ok')
    # test_load_vae(config)
    # test_tokenizer_output(config=config)
    test_load_controlnet(config=config)
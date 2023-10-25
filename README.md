# Mindspore-ControlNet

## 1. Inference with pretrained ControlNet
1. download pytorch controlnet checkpoints from https://huggingface.co/lllyasviel/ControlNet/tree/main/models or https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main.

2. convert downloaded pytorch checkpoints to mindspore checkpoints, or directly download from https://huggingface.co/unrealMJ/MindSpore-ControlNet.
    ```shell
    python torch2ms/convert.py --input_path xxxx --output_path xxxx # convert full model
    python torch2ms/convert.py --input_path xxxx --output_path xxxx --only_controlnet # convert controlnet only
    ```

3. Run run_controlnet_inference.py to use controlnet.
    ```shell
    python run_controlnet_inference.py --input_path xxxx --output_path xxxx
    ```

## 2. Train ControlNet from scratch
1. Download the dataset from https://huggingface.co/datasets/fusing/fill50k


2. Run run_controlnet_train.py to train controlnet.
    ```shell
    python run_controlnet_train.py --data_path xxxx --ckpt_path xxxx
    ```

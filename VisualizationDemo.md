````markdown
# SlowFast — Clean Installation & Usage

Follow the steps for a clean installation and using the SlowFast repo.

## Installation

```bash
!pip install 'git+https://github.com/facebookresearch/fvcore'
!pip install iopath psutil tqdm timm simplejson
!pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
!git clone https://github.com/facebookresearch/detectron2.git
!python -m pip install -e detectron2
````

## Checkpoint

Download the checkpoint from:
[https://github.com/facebookresearch/SlowFast/blob/main/projects/mae/README.md](https://github.com/facebookresearch/SlowFast/blob/main/projects/mae/README.md)

---

## Method 1 — Use Original Checkpoint

We added:

```python
op.append(
    MultiScaleBlock(
        dim=headdim,
        dimout=dimout,
        input_size=featuresize,
        num_heads=dimout // 64,
        mlp_ratio=mlpratio,
        qkv_bias=qkvbias,
        separate_qkv=cfg.MVIT.SEPARATE_QKV,  # Added this boolean
        ...
    )
)
```

### Run

```bash
!python tools/visualization_reconstruction.py --video_path path_to_video \
--ckpt_path path_to_VIT_B_16x4_MAE_PT.pyth \
--cfg_file path_to_k400_VIT_B_16x4_MAE_PT.yaml \
--output_dir /content/output
```

---

## Method 2 — Fuse and Use Fused Model

**Comment out the following line: `separate_qkv= cfg.MVIT.SEPARATE_QKV,`**

File path:
[https://github.com/SharvaGogawale1/SlowFast-Representation-Learning/blob/main/slowfast/models/head_helper.py](https://github.com/SharvaGogawale1/SlowFast-Representation-Learning/blob/main/slowfast/models/head_helper.py)

Local path:
`slowfast -> models -> head_helper.py`

Then fuse the checkpoint and save the fused model and use the fused one.

### Run

```bash
!python tools/visualization_reconstruction.py --video_path /content/sample1.mp4 \
--ckpt_path /content/drive/MyDrive/Mae/data/VIT_B_16x4_MAE_PT_FUSED.pyth \
--cfg_file /content/SlowFast-Representation-Learning/configs/masked_ssl/k400_VIT_B_16x4_MAE_PT.yaml \
--output_dir /content/output
```



## Environment

conda environment reference `requirements.txt` file.

## Dataset

The data preparation follows [JODO](https://github.com/GRAPH-0/JODO?tab=readme-ov-file). 

## Pretrained MolGD

Download the pretrained model checkpoint of MolGD from [pretrained-MolGD](https://zenodo.org/records/17861895)

## Unconditional Generation

1.QM9:

```bash
#train
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_uncond.py --mode train --workdir exp_uncond/vpsde_qm9

#sample
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_uncond.py --mode eval --workdir exp_uncond/vpsde_qm9 --config.eval.ckpts your_checkpoints --config.eval.batch_size 2500 --config.sampling.steps 1000
```

- `--config.eval.ckpts` specifies the checkpoints saved during training, such as  `--config.eval.ckpts 1500000`

2.GEOM:

```bash
#train
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_geom_uncond.py --mode train --workdir exp_uncond/vpsde_geom --config.training.n_iters 2500000 --config.model.n_layers 8

#sample
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_geom_uncond.py --mode eval --workdir exp_uncond/vpsde_geom --config.eval.ckpts your_checkpoints --config.eval.batch_size 1000 --config.sampling.steps 1000 --config.model.n_layers 8 --config.seed 42
```

## Conditional Generation

1.Single-Objective Conditional Generation:

```bash
#train
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond.py --mode train --workdir exp_cond/vpsde_qm9_cond_gap --config.cond_property gap

#sample
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/vpsde_qm9_cond.py --mode eval --workdir exp_cond/vpsde_qm9_cond_gap --config.cond_property gap --config.eval.ckpts your_checkpoints
```

- Set different conditional property `--config.cond_property alpha/gap/homo/lumo/mu/Cv`
- Set different workdir `--workdir exp_cond/vpsde_qm9_cond_[alpha/gap/homo/lumo/mu/Cv]` 

2.Multi-Objective Conditional Generation:

```bash
#train
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond_multi.py --mode train --workdir exp_cond_multi/vpsde_qm9_cond_Cv_mu --config.cond_property1 Cv --config.cond_property2 mu

#sample
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/vpsde_qm9_cond_multi.py --mode eval --workdir exp_cond_multi/vpsde_qm9_cond_Cv_mu --config.cond_property1 Cv --config.cond_property2 mu --config.eval.ckpts your_checkpoints
```

- Set different conditional properties `--config.cond_property1 Cv/gap/alpha --config.cond_property2 mu`

- Set different workdir `--workdir exp_cond/vpsde_qm9_cond_[Cv_mu/gap_mu/alpha_mu]` 

## Drug-likeness optimisation generation

1.Optimise QED/SA

```bash
#Conditioned training
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond_RL.py --mode train --workdir exp_cond_RL/vpsde_qm9_cond_RL_qed --config.RL_type "qed" --config.training.n_iters 400000

#Optimised training
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond_RL.py --mode train --workdir exp_cond_RL/vpsde_qm9_cond_RL_qed --config.RL_type "qed" --config.new_train_set 10000 --config.cond_RL True --config.training.eval_batch_size 1000 --config.training.eval_samples 1000 --config.training.snapshot_freq 1000 --config.training.n_iters 480000

#sample
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond_RL.py --mode eval --workdir exp_cond_RL/vpsde_qm9_cond_RL_qed --config.eval.ckpts 480000 --config.eval.batch_size 2500 --config.sampling.steps 1000
```

- First, conduct QED conditional training to enable the model to occasionally generate molecules with specified QED values. Subsequently, perform optimisation training on the trained conditional model.
- When optimising SA, simply replace all instances of ‘qed’ in the three instructions with ‘sa’.

2.Optimise QED&SA

```bash
#Conditioned training
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond_RL_multi.py --mode train --workdir exp_cond_RL/vpsde_qm9_cond_RL_multi --config.RL_type "multi" --config.training.n_iters 400000

#Optimised training
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond_RL_multi.py --mode train --workdir exp_cond_RL/vpsde_qm9_cond_RL_multi --config.RL_type "multi" --config.new_train_set 10000 --config.cond_RL True --config.training.eval_batch_size 1000 --config.training.eval_samples 1000 --config.training.snapshot_freq 1000 --config.training.n_iters 480000

#sample
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_cond_RL_multi.py --mode eval --workdir exp_cond_RL/vpsde_qm9_cond_RL_multi --config.eval.ckpts 48000 --config.eval.batch_size 2500 --config.sampling.steps 1000
```

## Unconditional generation + Drug-likeness optimisation 

1.Optimise QED/SA

```bash
#Optimised training
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_uncond_RL.py --mode train --workdir exp_uncond_RL/vpsde_qm9_RL_qed --config.RL_type "qed" --config.new_train_set 10000 --config.training.eval_batch_size 1000 --config.training.eval_samples 1000 --config.training.snapshot_freq 1000
```

- Prior to training, we must place the model trained in the **unconditional generation** experiment into the path `exp_uncond_RL/vpsde_qm9_qed/checkpoints-meta/your_checkpoints`. Subsequently, we shall train upon this pre-trained model.
- As this experiment is solely intended to compare optimisation efficiency during training, sampling is not performed.
- When optimising SA, simply replace all instances of ‘qed’ in the three instructions with ‘sa’.

2.Optimise QED&SA

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/vpsde_qm9_uncond_RL.py --mode train --workdir exp_uncond_RL/vpsde_qm9_RL_multi --config.RL_type "multi" --config.new_train_set 10000 --config.training.eval_batch_size 1000 --config.training.eval_samples 1000 --config.training.snapshot_freq 1000
```

- Prior to training, we must place the model trained in the **unconditional generation** experiment into the path `exp_uncond_RL/vpsde_qm9_multi/checkpoints-meta/your_checkpoints`. Subsequently, we shall train upon this pre-trained model.
- As this experiment is solely intended to compare optimisation efficiency during training, sampling is not performed.

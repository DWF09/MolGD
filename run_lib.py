import os
import torch
import logging
import numpy as np
import random
import pickle
# from torch.utils import tensorboard

from datasets import get_dataset, inf_iterator, get_dataloader
from models.ema import ExponentialMovingAverage
import losses
from utils import *
from evaluation import *
import visualize
from models import *
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn, get_cond_sampling_eval_fn, get_cond_multi_sampling_eval_fn
from cond_gen import *
import matplotlib.pyplot as plt
import csv
from evaluation.rdkit_metric import get_drug_chem, mol2smiles
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.build_dataset import collate_RL
from rdkit import Chem



def set_random_seed(config):
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def vpsde_edge_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs."""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    train_loader, val_loader, test_loader = get_dataloader(train_ds, val_ds, test_ds, config)

    train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=1, device=config.device, batch_size=1000)

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step 
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
               ema.store(model.parameters())
               ema.copy_to(model.parameters())

               # Wrapper EDM sampling
               processed_mols = sampling_fn(model)

               # EDM evaluation metrics
               stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
               logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                            "complete: %.4f, unique & valid: %.4f" % (
                            step, len(sample_rdmols), stability_res['atom_stable'],
                            stability_res['mol_stable'], rdkit_res['Validity'],
                            rdkit_res['Complete'], rdkit_res['Unique']))
               mol_stability_3D = stability_res['mol_stable']
               
               # FCD metric
               fcd_res = fcd_metric(sample_rdmols)
               logging.info("3D FCD: %.4f" % (fcd_res['FCD']))
               FCD_3D = fcd_res['FCD']

               # 2D evaluations
               stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
               logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                            "complete: %.4f, unique & valid: %.4f" % (
                                step, len(sample_rdmols), stability_res['atom_stable'],
                                stability_res['mol_stable'], rdkit_res['Validity'],
                                rdkit_res['Complete'], rdkit_res['Unique']))
               fcd_res = fcd_metric(complete_rdmols)
               logging.info("2D FCD: %.4f" % (fcd_res['FCD']))
               FCD_2D = fcd_res['FCD']
               
               to_csv(config, [step, mol_stability_3D, FCD_3D, FCD_2D])
               
               ema.restore(model.parameters())

               # Visualization
               this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
               if not os.path.exists(this_sample_dir):
                   os.makedirs(this_sample_dir)

               # change `sample_rdmols` to `complete_rdmols`?
               visualize.visualize_mols(sample_rdmols, this_sample_dir, config)


def vpsde_edge_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs."""
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                      config.eval.num_samples, inverse_scaler)

    # Obtain train dataset and eval dataset
    train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=1, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols = sampling_fn(model)

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, vaild & complete: %.4f,"
                         " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                         stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                         rdkit_res['Novelty']))
            count = len(processed_mols) - rdkit_res["ChemMetrics"]["qed"].count(0)
            logging.info("3D || qed: %.4f, sa: %.4f, logp: %.4f, lipinski: %.4f" 
                         % (sum(rdkit_res["ChemMetrics"]["qed"])/count, sum(rdkit_res["ChemMetrics"]["sa"])/count,
                            sum(rdkit_res["ChemMetrics"]["logp"])/count, sum(rdkit_res["ChemMetrics"]["lipinski"])/count))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, vaild & complete: %.4f,"
                         " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                         stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                         rdkit_res['Novelty']))
            count = len(processed_mols) - rdkit_res["ChemMetrics"]["qed"].count(0)
            logging.info("2D || qed: %.4f, sa: %.4f, logp: %.4f, lipinski: %.4f" 
                         % (sum(rdkit_res["ChemMetrics"]["qed"])/count, sum(rdkit_res["ChemMetrics"]["sa"])/count,
                            sum(rdkit_res["ChemMetrics"]["logp"])/count, sum(rdkit_res["ChemMetrics"]["lipinski"])/count))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)



def vpsde_edge_cond_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']
    ## one property
    prop2idx_sub = {
        config.cond_property: prop2idx[config.cond_property]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)
    train_loader, val_loader, test_loader = get_dataloader(second_train_ds, val_ds, test_ds, config)
    train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config, prop_norms)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler, prop_dist=prop_dist)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=1, device=config.device, batch_size=1000)

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step 
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Wrapper EDM sampling
                processed_mols = sampling_fn(model)

                # EDM evaluation metrics
                stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                             step, len(sample_rdmols), stability_res['atom_stable'],
                             stability_res['mol_stable'], rdkit_res['Validity'],
                             rdkit_res['Complete'], rdkit_res['Unique']))

                # FCD metric
                fcd_res = fcd_metric(sample_rdmols)
                logging.info("3D FCD: %.4f" % (fcd_res['FCD']))

                # 2D evaluations
                stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                                 step, len(sample_rdmols), stability_res['atom_stable'],
                                 stability_res['mol_stable'], rdkit_res['Validity'],
                                 rdkit_res['Complete'], rdkit_res['Unique']))
                fcd_res = fcd_metric(complete_rdmols)
                logging.info("2D FCD: %.4f" % (fcd_res['FCD']))

                ema.restore(model.parameters())

                # Visualization
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)

                visualize.visualize_mols(sample_rdmols, this_sample_dir, config)


def vpsde_edge_cond_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    # train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']

    ## one property
    prop2idx_sub = {
        config.cond_property: prop2idx[config.cond_property]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)

    # Load property prediction model for evaluation
    property_path = os.path.join(config.data.root, 'property_classifier', f'evaluate_{config.cond_property}')
    classifier_path = os.path.join(property_path, 'best_checkpoint.npy')
    args_classifier_path = os.path.join(property_path, 'args.pickle')
    classifier = get_classifier(classifier_path, args_classifier_path).to(config.device)
    classifier = torch.nn.DataParallel(classifier)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_cond_sampling_eval_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                                config.eval.num_samples, inverse_scaler, prop_dist=prop_dist,
                                                prop_norm=prop_norms)

    # Obtain train dataset and eval dataset
    train_mols = [second_train_ds[i].rdmol for i in range(len(second_train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=1, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols, MAE_loss = sampling_fn(model, classifier)
            logging.info(f"{config.cond_property} MAE: %.4f" % MAE_loss)

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f," %
                         (stability_res['atom_stable'], stability_res['mol_stable'], rdkit_res['Validity'],
                         rdkit_res['Complete']))

            # Mose evaluation metrics
            mose_res = mose_metric(sample_rdmols)
            logging.info("Metric-3D || FCD: %.4f" % (mose_res['FCD']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                         stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                         rdkit_res['Novelty']))
            mose_res = mose_metric(complete_rdmols)
            logging.info("Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)


def vpsde_edge_cond_multi_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs with additional quantum property conditioning.
    Two conditional property"""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']
    ## two property
    prop2idx_sub = {
        config.cond_property1: prop2idx[config.cond_property1],
        config.cond_property2: prop2idx[config.cond_property2]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)
    train_loader, val_loader, test_loader = get_dataloader(second_train_ds, val_ds, test_ds, config)
    train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config, prop_norms)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler, prop_dist=prop_dist)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=1, device=config.device, batch_size=1000)

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step 
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Wrapper EDM sampling
                processed_mols = sampling_fn(model)

                # EDM evaluation metrics
                stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                             step, len(sample_rdmols), stability_res['atom_stable'],
                             stability_res['mol_stable'], rdkit_res['Validity'],
                             rdkit_res['Complete'], rdkit_res['Unique']))

                # FCD metric
                fcd_res = fcd_metric(sample_rdmols)
                logging.info("3D FCD: %.4f" % (fcd_res['FCD']))

                # 2D evaluations
                stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                             "complete: %.4f, unique & valid: %.4f" % (
                                 step, len(sample_rdmols), stability_res['atom_stable'],
                                 stability_res['mol_stable'], rdkit_res['Validity'],
                                 rdkit_res['Complete'], rdkit_res['Unique']))
                fcd_res = fcd_metric(complete_rdmols)
                logging.info("2D FCD: %.4f" % (fcd_res['FCD']))

                ema.restore(model.parameters())

                # Visualization
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)

                visualize.visualize_mols(sample_rdmols, this_sample_dir, config)


def vpsde_edge_cond_multi_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    # train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    prop2idx = dataset_info['prop2idx']

    ## two property
    prop2idx_sub = {
        config.cond_property1: prop2idx[config.cond_property1],
        config.cond_property2: prop2idx[config.cond_property2]
    }
    prop_norms = val_ds.compute_property_mean_mad(prop2idx_sub)
    prop_dist = DistributionProperty(second_train_ds, prop2idx_sub, normalizer=prop_norms)

    # Load property prediction model for evaluation
    property_path1 = os.path.join(config.data.root, 'property_classifier', f'evaluate_{config.cond_property1}')
    classifier_path1 = os.path.join(property_path1, 'best_checkpoint.npy')
    args_classifier_path1 = os.path.join(property_path1, 'args.pickle')
    classifier1 = get_classifier(classifier_path1, args_classifier_path1).to(config.device)
    classifier1 = torch.nn.DataParallel(classifier1)

    property_path2 = os.path.join(config.data.root, 'property_classifier', f'evaluate_{config.cond_property2}')
    classifier_path2 = os.path.join(property_path2, 'best_checkpoint.npy')
    args_classifier_path2 = os.path.join(property_path2, 'args.pickle')
    classifier2 = get_classifier(classifier_path2, args_classifier_path2).to(config.device)
    classifier2 = torch.nn.DataParallel(classifier2)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_cond_multi_sampling_eval_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                                      config.eval.num_samples, inverse_scaler, prop_dist=prop_dist,
                                                      prop_norm=prop_norms)

    # Obtain train dataset and eval dataset
    train_mols = [second_train_ds[i].rdmol for i in range(len(second_train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=1, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols, MAE1_loss, MAE2_loss = sampling_fn(model, classifier1, classifier2)
            logging.info(f"{config.cond_property1} MAE: %.4f" % MAE1_loss)
            logging.info(f"{config.cond_property2} MAE: %.4f" % MAE2_loss)

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f," %
                         (stability_res['atom_stable'], stability_res['mol_stable'], rdkit_res['Validity'],
                         rdkit_res['Complete']))

            # Mose evaluation metrics
            mose_res = mose_metric(sample_rdmols)
            logging.info("Metric-3D || FCD: %.4f" % (mose_res['FCD']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                         stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                         rdkit_res['Novelty']))
            mose_res = mose_metric(complete_rdmols)
            logging.info("Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)


def vpsde_edge_RL_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs."""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    train_ds, val_ds, test_ds, dataset_info = get_dataset(config)

    train_loader, val_loader, test_loader = get_dataloader(train_ds, val_ds, test_ds, config)

    # train_iter = inf_iterator(train_loader)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=1, device=config.device, batch_size=1000)
    
    T_set = []
    T_score0 = []
    T_smiles0= []
    T_property0 = []
    logging.info("compute property ...")
    for i, data in enumerate(train_ds):
        if i % 1000 == 0:
            logging.info(i)
        
        try:
            mol = data["rdmol"]
            smiles = mol2smiles(mol)
            Score = get_drug_chem(mol)
            if config.RL_type=="multi":
                score = Score["qed"] + Score["sa"]
                property = torch.Tensor([Score["qed"], Score["sa"]])
            else:
                score = Score[config.RL_type]
                property = torch.Tensor([Score[config.RL_type]])
        except:
            score = 0.
            if config.RL_type=="multi":
                property = torch.Tensor([0, 0])
            else:
                property = torch.Tensor([0])
        T_score0.append(score)
        T_smiles0.append(smiles)
        T_property0.append(property)
    logging.info("complete , mean score: %.5e"%(sum(T_score0)/len(T_score0)))   
    
    T_index = np.argsort(-np.array(T_score0))[:config.new_train_set]
    T_score = []
    for i in T_index:
        data = train_ds[i]
        T_set.append(dict(atom_one_hot = data.atom_one_hot,
                          edge_one_hot = data.edge_one_hot,
                          fc = data.fc,
                          pos = data.pos,
                          num_atom = data.num_atom,
                          smiles = T_smiles0[i],
                          score = T_score0[i],
                          property = T_property0[i]))
        T_score.append(T_score0[i])
    logging.info("new train set mean score: %.5e"%(sum(T_score)/len(T_score)))
    
    train_loader = DataLoader(T_set, batch_size=config.training.batch_size, shuffle=True,
                              collate_fn=collate_RL)
    train_iter = inf_iterator(train_loader)
    
    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step 
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
               ema.store(model.parameters())
               ema.copy_to(model.parameters())

               # Wrapper EDM sampling
               processed_mols = sampling_fn(model)

               # EDM evaluation metrics
               stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
               rdkit_res_3D = rdkit_res
               logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                            "complete: %.4f, unique & valid: %.4f" % (
                            step, len(sample_rdmols), stability_res['atom_stable'],
                            stability_res['mol_stable'], rdkit_res['Validity'],
                            rdkit_res['Complete'], rdkit_res['Unique']))
               
               mol_stability_3D = stability_res['mol_stable']
               
               # FCD metric
               fcd_res = fcd_metric(sample_rdmols)
               logging.info("3D FCD: %.4f" % (fcd_res['FCD']))
               FCD_3D = fcd_res['FCD']

               # 2D evaluations
               stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
               logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                            "complete: %.4f, unique & valid: %.4f" % (
                                step, len(sample_rdmols), stability_res['atom_stable'],
                                stability_res['mol_stable'], rdkit_res['Validity'],
                                rdkit_res['Complete'], rdkit_res['Unique']))
               
               fcd_res = fcd_metric(complete_rdmols)
               logging.info("2D FCD: %.4f" % (fcd_res['FCD']))
               
               count = len(processed_mols) - rdkit_res["ChemMetrics"]["qed"].count(0)
               logging.info("qed: %.4f, sa: %.4f, logp: %.4f, lipinski: %.4f" 
                            % (sum(rdkit_res["ChemMetrics"]["qed"])/count, sum(rdkit_res["ChemMetrics"]["sa"])/count,
                               sum(rdkit_res["ChemMetrics"]["logp"])/count, sum(rdkit_res["ChemMetrics"]["lipinski"])/count))
               
               FCD_2D = fcd_res['FCD']
               
               to_csv(config, [step, mol_stability_3D, FCD_3D, FCD_2D])
               
               ema.restore(model.parameters())

               # Visualization
               this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
               if not os.path.exists(this_sample_dir):
                   os.makedirs(this_sample_dir)

               # change `sample_rdmols` to `complete_rdmols`?
               visualize.visualize_mols(sample_rdmols, this_sample_dir, config)
               
               # update-RL
               # process generate data Remove duplicates
               G_set = []
               G_score = []
               unique = set([i["smiles"] for i in T_set])
               for i in range(len(processed_mols)):
                    if rdkit_res_3D["ChemMetrics"]["qed"][i] != 0 and rdkit_res["ChemMetrics"]["qed"][i] != 0 and rdkit_res["ChemMetrics"]["smiles"][i] not in unique:

                        if config.RL_type=="multi":
                            G_score.append(rdkit_res["ChemMetrics"]["qed"][i] + rdkit_res["ChemMetrics"]["sa"][i])
                            G_property = torch.Tensor([rdkit_res["ChemMetrics"]["qed"][i], rdkit_res["ChemMetrics"]["sa"][i]])
                        else:
                            G_score.append(rdkit_res["ChemMetrics"][config.RL_type][i])
                            G_property = torch.Tensor([rdkit_res["ChemMetrics"][config.RL_type][i]])
                            
                        unique.add(rdkit_res["ChemMetrics"]["smiles"][i])
                        
                        G_set.append(dict(mol = processed_mols[i],
                                          smiles = rdkit_res["ChemMetrics"]["smiles"][i],
                                          property = G_property))
                        
     
               T_set, T_score = update_data(T_set, T_score, G_set, G_score)
                 
               train_loader = DataLoader(T_set, batch_size=config.training.batch_size, shuffle=True,
                                           collate_fn=collate_RL)
               train_iter = inf_iterator(train_loader)
               

def vpsde_edge_RL_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs."""
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                      config.eval.num_samples, inverse_scaler)

    # Obtain train dataset and eval dataset
    train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=1, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols = sampling_fn(model)

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f," %
                         (stability_res['atom_stable'], stability_res['mol_stable'], rdkit_res['Validity'],
                          rdkit_res['Complete']))

            # Mose evaluation metrics
            mose_res = mose_metric(sample_rdmols)
            logging.info("Metric-3D || 3D FCD: %.4f" % (mose_res['FCD']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                         stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                         rdkit_res['Novelty']))
    
            mose_res = mose_metric(complete_rdmols)
            logging.info("Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))
            
            count = len(processed_mols) - rdkit_res["ChemMetrics"]["qed"].count(0)
            logging.info("2D || qed: %.4f, sa: %.4f, logp: %.4f, lipinski: %.4f" 
                         % (sum(rdkit_res["ChemMetrics"]["qed"])/count, sum(rdkit_res["ChemMetrics"]["sa"])/count,
                            sum(rdkit_res["ChemMetrics"]["logp"])/count, sum(rdkit_res["ChemMetrics"]["lipinski"])/count))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)


def vpsde_edge_cond_RL_train(config, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs."""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    train_ds, val_ds, test_ds, dataset_info = get_dataset(config)

    train_loader, val_loader, test_loader = get_dataloader(train_ds, val_ds, test_ds, config)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.training.eval_batch_size,
                                      config.training.eval_samples, inverse_scaler)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]
    fcd_metric = get_fcd_metric(test_mols, n_jobs=1, device=config.device, batch_size=1000)
    
    T_set = []
    T_score0 = []
    T_smiles0= []
    T_property0 = []
    logging.info("compute property ...")
    for i, data in enumerate(train_ds):
        if i % 1000 == 0:
            logging.info(i)
        
        try:
            mol = data["rdmol"]
            smiles = mol2smiles(mol)
            Score = get_drug_chem(mol)
            if config.RL_type=="multi":
                score = Score["qed"] + Score["sa"]
                property = torch.Tensor([Score["qed"], Score["sa"]])
            else:
                score = Score[config.RL_type]
                property = torch.Tensor([Score[config.RL_type]])
        except:
            score = 0.
            if config.RL_type=="multi":
                property = torch.Tensor([0, 0])
            else:
                property = torch.Tensor([0])
        T_score0.append(score)
        T_smiles0.append(smiles)
        T_property0.append(property)
    logging.info("complete , mean score: %.5e"%(sum(T_score0)/len(T_score0)))   
    
    T_index = np.argsort(-np.array(T_score0))[:config.new_train_set]
    T_score = []
    for i in T_index:
        data = train_ds[i]
        T_set.append(dict(atom_one_hot = data.atom_one_hot,
                          edge_one_hot = data.edge_one_hot,
                          fc = data.fc,
                          pos = data.pos,
                          num_atom = data.num_atom,
                          smiles = T_smiles0[i],
                          score = T_score0[i],
                          property = T_property0[i]))
        T_score.append(T_score0[i])
    logging.info("new train set mean score: %.5e"%(sum(T_score)/len(T_score)))
    
    train_loader = DataLoader(T_set, batch_size=config.training.batch_size, shuffle=True,
                              collate_fn=collate_RL)
    train_iter = inf_iterator(train_loader)
    
    # Training iterations
    for step in range(initial_step, num_train_steps + 1):

        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step 
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
               ema.store(model.parameters())
               ema.copy_to(model.parameters())

               # Wrapper EDM sampling
               T_mean = torch.stack([i["property"] for i in T_set], dim=0).mean(dim=0)
                                
               processed_mols = sampling_fn(model, T_mean)

               # EDM evaluation metrics
               stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
               rdkit_res_3D = rdkit_res
               logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                            "complete: %.4f, unique & valid: %.4f" % (
                            step, len(sample_rdmols), stability_res['atom_stable'],
                            stability_res['mol_stable'], rdkit_res['Validity'],
                            rdkit_res['Complete'], rdkit_res['Unique']))

               mol_stability_3D = stability_res['mol_stable']
               
               # FCD metric
               fcd_res = fcd_metric(sample_rdmols)
               logging.info("3D FCD: %.4f" % (fcd_res['FCD']))
               FCD_3D = fcd_res['FCD']

               # 2D evaluations
               stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
               logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                            "complete: %.4f, unique & valid: %.4f" % (
                                step, len(sample_rdmols), stability_res['atom_stable'],
                                stability_res['mol_stable'], rdkit_res['Validity'],
                                rdkit_res['Complete'], rdkit_res['Unique']))
               
               fcd_res = fcd_metric(complete_rdmols)
               logging.info("2D FCD: %.4f" % (fcd_res['FCD']))
               
               count = len(processed_mols) - rdkit_res["ChemMetrics"]["qed"].count(0) + 1e-8
               logging.info("qed: %.4f, sa: %.4f, logp: %.4f, lipinski: %.4f" 
                            % (sum(rdkit_res["ChemMetrics"]["qed"])/count, sum(rdkit_res["ChemMetrics"]["sa"])/count,
                               sum(rdkit_res["ChemMetrics"]["logp"])/count, sum(rdkit_res["ChemMetrics"]["lipinski"])/count))
               
               FCD_2D = fcd_res['FCD']
               
               to_csv(config, [step, mol_stability_3D, FCD_3D, FCD_2D])
               
               ema.restore(model.parameters())

               # Visualization
               this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
               if not os.path.exists(this_sample_dir):
                   os.makedirs(this_sample_dir)

               # change `sample_rdmols` to `complete_rdmols`?
               visualize.visualize_mols(sample_rdmols, this_sample_dir, config)
               
               # update-RL
               # process generate data Remove duplicates
               if config.cond_RL:
                   G_set = []
                   G_score = []
                   unique = set([i["smiles"] for i in T_set])
                   for i in range(len(processed_mols)):
                       if rdkit_res_3D["ChemMetrics"]["qed"][i] != 0 and rdkit_res["ChemMetrics"]["qed"][i] != 0 and rdkit_res["ChemMetrics"]["smiles"][i] not in unique:

                           if config.RL_type=="multi":
                               G_score.append(rdkit_res["ChemMetrics"]["qed"][i] + rdkit_res["ChemMetrics"]["sa"][i])
                               G_property = torch.Tensor([rdkit_res["ChemMetrics"]["qed"][i], rdkit_res["ChemMetrics"]["sa"][i]])
                           else:
                               G_score.append(rdkit_res["ChemMetrics"][config.RL_type][i])
                               G_property = torch.Tensor([rdkit_res["ChemMetrics"][config.RL_type][i]])
                               
                           unique.add(rdkit_res["ChemMetrics"]["smiles"][i])
                           
                           G_set.append(dict(mol = processed_mols[i],
                                             smiles = rdkit_res["ChemMetrics"]["smiles"][i],
                                             property = G_property))
                           
        
                   T_set, T_score = update_data(T_set, T_score, G_set, G_score)
                    
                   train_loader = DataLoader(T_set, batch_size=config.training.batch_size, shuffle=True,
                                              collate_fn=collate_RL)
                   train_iter = inf_iterator(train_loader)
               


def vpsde_edge_cond_RL_evaluate(config, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs."""
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                      config.eval.num_samples, inverse_scaler)

    # Obtain train dataset and eval dataset
    train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
    test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    mose_metric = get_moses_metrics(test_mols, n_jobs=1, device=config.device)
    if config.eval.sub_geometry:
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols = sampling_fn(model, torch.Tensor(config.eval_mean))

            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f," %
                         (stability_res['atom_stable'], stability_res['mol_stable'], rdkit_res['Validity'],
                          rdkit_res['Complete']))
            
            # Mose evaluation metrics
            mose_res = mose_metric(sample_rdmols)
            logging.info("Metric-3D || 3D FCD: %.4f" % (mose_res['FCD']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                         " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                         stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                         rdkit_res['Novelty']))

            mose_res = mose_metric(complete_rdmols)
            logging.info("Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                         mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))
            
            count = len(processed_mols) - rdkit_res["ChemMetrics"]["qed"].count(0)
            logging.info("qed: %.4f, sa: %.4f, logp: %.4f, lipinski: %.4f" 
                         % (sum(rdkit_res["ChemMetrics"]["qed"])/count, sum(rdkit_res["ChemMetrics"]["sa"])/count,
                            sum(rdkit_res["ChemMetrics"]["logp"])/count, sum(rdkit_res["ChemMetrics"]["lipinski"])/count))

            # save sample mols
            if config.eval.save_graph:
                sampler_name = config.sampling.method
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}_{}.pkl".format(ckpt, config.seed))
                with open(graph_file, "wb") as f:
                    pickle.dump(complete_rdmols, f)

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)

run_train_dict = {
    'vpsde_edge': vpsde_edge_train,
    'vpsde_edge_cond': vpsde_edge_cond_train,
    'vpsde_edge_cond_multi': vpsde_edge_cond_multi_train,
    'vpsde_edge_RL': vpsde_edge_RL_train,
    'vpsde_edge_cond_RL': vpsde_edge_cond_RL_train,
}


run_eval_dict = {
    'vpsde_edge': vpsde_edge_evaluate,
    'vpsde_edge_cond': vpsde_edge_cond_evaluate,
    'vpsde_edge_cond_multi': vpsde_edge_cond_multi_evaluate,
    'vpsde_edge_RL': vpsde_edge_RL_evaluate,
    'vpsde_edge_cond_RL': vpsde_edge_cond_RL_evaluate,
}


def train(config, workdir):
    run_train_dict[config.exp_type](config, workdir)


def evaluate(config, workdir, eval_folder='eval'):
    run_eval_dict[config.exp_type](config, workdir, eval_folder)


def to_csv(config, s):
    file_name = f"./{config.data.name}.csv"
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not os.path.exists(file_name) or os.stat(file_name).st_size == 0:
            writer.writerow(["step", "3D_mol_stability", "3D_FCD", "2D_FCD"])  # 写入表头
        writer.writerow(s)

def update_data(T_set, T_score, G_set, G_score):
    
    T_index = np.argsort(np.array(T_score))
    G_index = np.argsort(np.array(G_score))
    
    train_set_score = [i["score"] for i in T_set]
    logging.info("max: %.4f, min: %.4f, mean: %.4f, validity: %d, train_set_mean: %.4f" 
                  % (G_score[G_index[len(G_score)-1]], G_score[G_index[0]], np.mean(G_score), 
                    len(G_score), np.mean(train_set_score)))
    i, j = 0, 0    
    while i+j < len(G_set):
        if G_score[G_index[i+j]]>T_score[T_index[i]]:
            
            t, g = T_index[i], G_index[i+j]
            
            pos, atom_type, edge_type, fc = G_set[g]["mol"]
            
            edge_type = edge_type.unsqueeze(-1)
            edge_exist = edge_type.clone()
            edge_exist[edge_exist>0] = 1.
            
            edge = edge_type.clone()
            edge[edge==4] = 0.
            edge[edge==1] = 0.3333
            edge[edge==2] = 0.6666
            edge[edge==3] = 1.
            
            edge_one_hot = torch.cat([edge_exist, edge], dim=-1)
            
            T_set[t] = dict(atom_one_hot = F.one_hot(atom_type, num_classes=5).float(),
                              edge_one_hot = edge_one_hot,
                              fc = fc,
                              pos = pos,
                              num_atom = torch.Tensor([atom_type.shape[0]]).int(),
                              smiles = G_set[g]["smiles"],
                              property = G_set[g]["property"],
                              score = G_score[g])

            T_score[t] = G_score[g]
            
            i = i+1
        else:
            j= j+1
    
    return T_set, T_score

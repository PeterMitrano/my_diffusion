import io
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from my_image_diffusion.ddpm import Diffusion
from my_image_diffusion.models import DiffusionModel
from my_image_diffusion.utils import TrajDataset, step_epoch_generator


def train():
    np.set_printoptions(suppress=True, precision=4, linewidth=220)
    torch.set_printoptions(sci_mode=False, precision=4, linewidth=220)

    dataset_path = Path("data")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    dataset = TrajDataset(dataset_path)

    d_config = {
        "cls": DiffusionModel,
        "model_kwargs": {
            "flat_data_dim": dataset.flat_data_dim,
            "h1": 1024,
            "h2": 1024,
            "h3": 1024,
            "time_emb_dim": 128,
            "time_emb_mode": 'sin',
        }
    }
    config = {
        'lr': 5e-4,
        'batch_size': 64,
        'noise_steps': 200,
        'beta_start': 1e-7,
        'beta_end': 0.01,
        "steps": 20_000,
        "n_test_samples": 100,
        "model": d_config,
    }

    model = config['model']['cls'](**config['model']['model_kwargs'])

    opt = optim.AdamW(model.parameters(), lr=config['lr'])
    mse = nn.MSELoss()

    # use linear LR decay from the initial LR to 0
    lr_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0, total_iters=config['steps'])

    diffusion = Diffusion(shape=dataset.traj_dim, noise_steps=config['noise_steps'], beta_start=config['beta_start'],
                          beta_end=config['beta_end'])

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # setup rerun
    rr.init("traj_traj_diffusion")
    rr.connect()

    test_sample_viz_step = 0
    rr.set_time_sequence("test_sample_viz_step", test_sample_viz_step)

    config_string_io = io.StringIO()
    pprint(config, stream=config_string_io)
    rr.log("config", rr.TextDocument(config_string_io.getvalue()))

    rr.log("goal", rr.Points2D([1.75, 1.75], radii=0.03, colors=(0, 255, 0)))
    rr.log("start", rr.Points2D([0.1, 0.1], radii=0.03, colors=(255, 255, 255)))
    rr.log("obstacle", rr.Points2D([1, 1], radii=0.06, colors=(255, 0, 0)))
    rr.log("bounds", rr.Boxes2D(sizes=[2, 2], centers=[1, 1], colors=(180, 80, 180), radii=0.01))

    train_trajs = dataset.data[:config['n_test_samples']]
    plt.figure()
    for traj in train_trajs:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.01, c='k')
    rr.log("train_trajs", rr.LineStrips2D(train_trajs, colors=(0, 0, 255), radii=0.00005))

    # visualize the noise adding process
    for t in range(config['noise_steps']):
        rr.set_time_sequence("test_sample_viz_step", test_sample_viz_step)
        t_batch = (torch.ones(train_trajs.shape[0]) * t).long()
        train_trajs_noise, _ = diffusion.noise_traj(torch.tensor(train_trajs), t_batch)
        rr.log("train_trajs_noise", rr.LineStrips2D(train_trajs_noise.numpy(), colors=(128, 128, 0), radii=0.0005))
        test_sample_viz_step += 1

    for step, epoch, batch in step_epoch_generator(dataloader, config['steps']):
        rr.set_time_sequence("train_step", step)
        t = diffusion.sample_timestamps(config['batch_size'])

        x_t, noise = diffusion.noise_traj(batch, t)

        predicted_noise = model(x_t, t)
        loss = mse(predicted_noise, noise)

        # # add L1 parameter loss to encourage sparsity
        # l1_loss = torch.tensor(0.0)
        # for param in model.parameters():
        #     l1_loss += torch.norm(param, 1) * 1e-4
        # loss += l1_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()

        rr.log("epoch", rr.Scalar(epoch))
        rr.log("mse", rr.Scalar(loss.item()))
        rr.log("lr", rr.Scalar(opt.param_groups[0]['lr']))

        if step % 100 == 0:
            test_samples, all_test_samples = diffusion.sample_traj(model, n_samples=config['n_test_samples'])
            for test_samples_noise_step_t in all_test_samples:
                rr.set_time_sequence("test_sample_viz_step", test_sample_viz_step)
                rr.log("test_sampled_trajs", rr.LineStrips2D(test_samples_noise_step_t, radii=0.0005))
                test_sample_viz_step += 1

            test_sample_viz_step += 10

    # Save the model
    print("NOTE: SAVING CURRENTLY DISABLED")
    # model.config = config
    # model_ckpt_path = trial_dir / 'model.pt'
    # torch.save(model, model_ckpt_path)


if __name__ == '__main__':
    train()

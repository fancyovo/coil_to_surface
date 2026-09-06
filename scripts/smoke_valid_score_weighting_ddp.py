"""Exercise the real replay step's DDP reductions against a concatenated oracle."""
import copy
import json
import numpy as np
import torch
from torch import distributed as dist

from scripts import score_gradient_flow_rl as rl
from flow_matching.data import CoilNormalizer


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, data):
        return self.scale * data.mean(dim=(1, 2))


def main():
    rank, local_rank, device = rl.distributed_setup()
    normalizer = CoilNormalizer(mean=np.zeros(100, dtype=np.float32),
        std=np.ones(100, dtype=np.float32), current_l1_a={"8:3": 3.0}, clip=float("inf"))
    rl.permute_pair = lambda data, gradient, **kwargs: (data, gradient)
    rl.valid_flow_terms_with_transport = lambda model, data, gradient, **kwargs: (model(data), model(data) * 0)
    rl.ordinary_flow_terms = lambda model, data, **kwargs: model(data) * 0
    for scenario in ("split", "empty_rank", "all_invalid", "equal"):
        base = TinyVelocity().to(device)
        ema = copy.deepcopy(base)
        model = torch.nn.parallel.DistributedDataParallel(base, device_ids=[local_rank])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        data = (torch.arange(1, 9, device=device).float() + rank * 8)[:, None, None].expand(-1, 3, 100) / 100
        scores = torch.arange(8, device=device).float() * 3 + rank * 20 + 30
        if scenario == "equal":
            scores.fill_(60)
        valid = torch.ones(8, device=device, dtype=torch.bool)
        if scenario == "all_invalid" or (scenario == "empty_rank" and rank == 0):
            valid.fill_(False)
        pool = dict(current=data, score_gradient_flow=torch.zeros_like(data),
                    scores=scores, valid=valid, gradient_ok=valid)
        indices = torch.randint(8, (32,), device=device,
                                generator=torch.Generator(device=device).manual_seed(100 + rank))
        selected = torch.stack((data[indices].mean((1, 2)), scores[indices], valid[indices].float()))
        gathered = [torch.empty_like(selected) for _ in range(2)]
        dist.all_gather(gathered, selected)
        global_batch = torch.cat(gathered, dim=1)
        active = global_batch[2].bool()
        if active.any():
            w = torch.exp((global_batch[1, active] - global_batch[1, active].max()) / 7.5) + 0.01
            expected_gradient = (w * global_batch[0, active]).sum() / w.sum()
            expected_ess = w.sum().square() / w.square().sum()
        else:
            expected_gradient = torch.tensor(0.0, device=device)
            expected_ess = torch.tensor(0.0, device=device)
        summary = rl.train_step_from_replay(model=model, ema_model=ema, optimizer=optimizer,
            normalizer=normalizer, pool=pool, device=device, beta=0.006021959241479635,
            sample_generator=torch.Generator(device=device).manual_seed(100 + rank),
            permutation_generator=torch.Generator(device=device).manual_seed(200 + rank),
            loss_generator=torch.Generator(device=device).manual_seed(300 + rank),
            valid_weighting={"tau": 7.5, "epsilon": 0.01})
        torch.testing.assert_close(base.scale.grad, expected_gradient)
        assert abs(summary["objective"] - float(expected_gradient) * 0.25) < 1e-6
        assert abs(summary["valid_weight_ess"] - float(expected_ess)) < 1e-4
        if rank == 0:
            print(json.dumps({"event": "ddp_weighting_smoke_pass", "scenario": scenario, **summary}), flush=True)
        del model
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

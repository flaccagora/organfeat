"""Train the point descriptor (configs/train/*.yaml).

    sbatch -J tr_<name> -t 06:00:00 scripts/leonardo/gpu_run.sbatch \
        python -m ofeat.train.run --config base name=<name> loss.use_vis=false ...

Per step, for each liver of the batch (ofeat.train.data.LiverViews):
  A_vis  partial view A with its lifted DINO channel (visual-modality dropout per sample / per point)
  A_geo  the same points, geometry only (visual channel = missing vector, i.e. the CT path)
  B      an overlapping view of the same group (if any), with images
  CT     the complete CT samples (missing visual channel)
  loss = L(A_vis, CT) + L(A_geo, CT) + lambda L(A_vis, A_geo) + mu L(A_vis, B)
A geometry-only model (loss.use_vis=false) uses only A_geo, B_geo and CT.
Every eval.every steps: a proxy on val (mutual-NN inlier ratio@10 and point recall@10 on fixed pairs)
and a checkpoint. Selection uses full registration recall on val (scripts/bench/run_bench.py).
-> outputs/train/<name>/{config.yaml, log.jsonl, step_<k>.pt}
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ofeat.data import liver_meshes
from ofeat.data.points import load_config as points_config
from ofeat.data.points import noisy_depth
from ofeat.eval.metrics import correct
from ofeat.eval.pairs import pairs
from ofeat.features.geom import geom_features
from ofeat.match import match
from ofeat.mesh import REPO, data_root
from ofeat.models.descriptor import build
from ofeat.models.infer import VisualChannel
from ofeat.models.losses import infonce
from ofeat.train.data import AugConfig, LiverViews, collate


def load_cfg(name: str, overrides: list[str]):
    cfg = OmegaConf.load(REPO / "configs" / "train" / f"{name}.yaml")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def photometric(img: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """img (3, H, W) in [0, 1]: brightness, contrast, saturation, hue, gamma, blur, noise."""
    import torchvision.transforms.v2.functional as TF

    def u(a, b):
        return float(a + (b - a) * torch.rand(1, generator=g))

    img = TF.adjust_brightness(img, u(0.7, 1.3))
    img = TF.adjust_contrast(img, u(0.7, 1.3))
    img = TF.adjust_saturation(img, u(0.7, 1.3))
    img = TF.adjust_hue(img, u(-0.03, 0.03))
    img = img.clamp(0, 1) ** u(0.8, 1.25)
    if u(0, 1) < 0.3:
        img = TF.gaussian_blur(img, [7, 7], [u(0.3, 1.5)])
    return (img + 0.01 * torch.randn(img.shape, generator=g).to(img.device)).clamp(0, 1)


class Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dev = "cuda"
        self.out = Path("outputs/train") / cfg.name
        self.out.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.out / "config.yaml")
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.model = build(OmegaConf.to_container(cfg.model), pretrained=cfg.model.init == "sonata",
                           weights_dir=str(data_root() / "weights" / "sonata")).to(self.dev)
        self.amp = bool(cfg.optim.amp)
        self.use_vis = bool(cfg.loss.use_vis)
        self.visual = VisualChannel(device=self.dev) if self.use_vis else None
        if hasattr(self.model, "enc"):  # Sonata: pretrained encoder at encoder_lr
            enc = list(self.model.enc.net.parameters())
            enc_ids = {id(p) for p in enc}
            rest = [p for p in self.model.parameters() if id(p) not in enc_ids]
            # the visual/geometric adapter lives inside the wrapped embedding: give it the head lr
            adapter = list(self.model.enc.net.embedding.adapter.parameters())
            ad_ids = {id(p) for p in adapter}
            enc = [p for p in enc if id(p) not in ad_ids]
        else:  # local PPF network: trained from scratch, one lr
            enc, rest, adapter = [], list(self.model.parameters()), []
        groups = [{"params": enc, "lr": cfg.optim.encoder_lr}, {"params": rest + adapter, "lr": cfg.optim.lr}]
        self.opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=cfg.optim.weight_decay)
        warm = max(1, int(cfg.optim.warmup_frac * cfg.optim.steps))

        def lr_lambda(s):
            if s < warm:
                return (s + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, cfg.optim.steps - warm)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.pcfg = points_config()
        val = liver_meshes.subjects("val")[: cfg.eval.livers]
        sig, holes = cfg.eval.get("sigma100", 0.0), cfg.eval.get("holes", 0.0)
        noise = (lambda d, r: noisy_depth(d, r, sig, holes)) if sig or holes else None
        self.val_pairs = list(pairs("val", val, self.pcfg, views=slice(0, cfg.eval.views), depth_noise=noise))

    # -- inputs --------------------------------------------------------------------------------------
    def _geom(self, c: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = c["xyz"].to(self.dev, non_blocking=True)
        n = c["normals"].to(self.dev, non_blocking=True)
        return x, n, geom_features(x, n, tuple(self.cfg.model.geom_radii))

    def _vis(self, clouds: list[dict], train: bool) -> list[torch.Tensor]:
        imgs = [c["rgb"].to(self.dev).permute(2, 0, 1).float() / 255 for c in clouds]
        if train and self.cfg.loss.photometric:
            imgs = [photometric(im, self.gen) for im in imgs]
        tok = self.visual.tokens(torch.stack(imgs))
        H, W = imgs[0].shape[1:]
        return [self.visual.project(t, c["pix"].to(self.dev), (W, H)) for t, c in zip(tok, clouds)]

    def _drop(self, n: int) -> torch.Tensor:
        L = self.cfg.loss
        if float(torch.rand(1, generator=self.gen)) < L.vis_drop_sample:
            return torch.zeros(n, dtype=torch.bool, device=self.dev)
        return torch.rand(n, generator=self.gen).to(self.dev) >= L.vis_drop_point

    # -- step ----------------------------------------------------------------------------------------
    def step(self, batch: list[dict]) -> dict:
        L = self.cfg.loss
        xs, ns, gs, vs, oks, roles = [], [], [], [], [], []
        with_b = [s for s in batch if "B" in s]
        vis_clouds = [s["A"] for s in batch] + [s["B"] for s in with_b]
        vis_feats = self._vis(vis_clouds, train=True) if self.use_vis else [None] * len(vis_clouds)
        vis_a = dict(zip(range(len(batch)), vis_feats[: len(batch)]))
        vis_b = dict(zip([id(s) for s in with_b], vis_feats[len(batch):]))
        for i, s in enumerate(batch):
            xa, na, ga = self._geom(s["A"])
            if self.use_vis:
                for key, v, ok in (("A_vis", vis_a[i], self._drop(len(xa))), ("A_geo", None, None)):
                    xs.append(xa), ns.append(na), gs.append(ga), vs.append(v), oks.append(ok)
                    roles.append((i, key))
            else:
                xs.append(xa), ns.append(na), gs.append(ga), vs.append(None), oks.append(None)
                roles.append((i, "A_geo"))
            if "B" in s:
                xb, nb, gb = self._geom(s["B"])
                vb = vis_b[id(s)] if self.use_vis else None
                xs.append(xb), ns.append(nb), gs.append(gb), vs.append(vb)
                oks.append(self._drop(len(xb)) if self.use_vis else None)
                roles.append((i, "B"))
            xc, nc, gc = self._geom(s["CT"])
            xs.append(xc), ns.append(nc), gs.append(gc), vs.append(None), oks.append(None)
            roles.append((i, "CT"))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            zs = self.model(xs, ns, gs, vs, oks)
        z = {r: zz for r, zz in zip(roles, zs)}
        terms, stats = {}, {}

        def add(key, za, zb, pa, pb, w):
            if w <= 0:
                return
            loss, st = infonce(za, zb, pa, pb, L.tau, L.pos_radius, L.safe_radius, L.n_anchor)
            terms.setdefault(key, []).append(w * loss)
            stats.setdefault(key + "_acc10", []).append(st["acc10"])

        for i, s in enumerate(batch):
            pa = s["A"]["world"].to(self.dev)
            pc = s["CT"]["world"].to(self.dev)
            a_main = z[(i, "A_vis")] if self.use_vis else z[(i, "A_geo")]
            add("ct_geo", z[(i, "A_geo")], z[(i, "CT")], pa, pc, 1.0)
            if self.use_vis:
                add("ct_vis", z[(i, "A_vis")], z[(i, "CT")], pa, pc, 1.0)
                add("cross", z[(i, "A_vis")], z[(i, "A_geo")], pa, pa, L.lambda_cross)
            if (i, "B") in z:
                add("pairs", a_main, z[(i, "B")], pa, s["B"]["world"].to(self.dev), L.mu_pairs)
        loss = sum(torch.stack(v).mean() for v in terms.values())
        self.opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):  # spconv runs fp16 kernels under autocast (defreg results 014)
            self.sched.step()
            return {"skipped": 1.0}
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        self.opt.step()
        self.sched.step()
        out = {"loss": float(loss.detach()), "grad_norm": float(gn)}
        out.update({k: float(torch.stack(v).mean()) for k, v in terms.items()})
        out.update({k: float(np.mean(v)) for k, v in stats.items()})
        return out

    # -- proxy eval ----------------------------------------------------------------------------------
    @torch.no_grad()
    def proxy(self) -> dict:
        rng_state = torch.get_rng_state(), torch.cuda.get_rng_state()
        self.model.eval()
        res = {"geo": [], "vis": []}
        for p in self.val_pairs:
            cs = {"xyz": torch.from_numpy(p.P), "normals": torch.from_numpy(p.nP), "rgb": torch.from_numpy(p.rgb),
                  "pix": torch.from_numpy(p.pix).long()}
            ct = {"xyz": torch.from_numpy(p.Q), "normals": torch.from_numpy(p.nQ)}
            xa, na, ga = self._geom(cs)
            xc, nc, gc = self._geom(ct)
            variants = [("geo", None)]
            if self.use_vis:
                variants.append(("vis", self._vis([cs], train=False)[0]))
            for key, v in variants:
                torch.manual_seed(0)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    za, zc = self.model([xa, xc], [na, nc], [ga, gc], [v, None])
                m = match(za.float(), zc.float())
                ok = correct(p.G, p.Q, m.src, m.tgt, 10.0)
                res[key].append((float(ok.mean()) if len(ok) else 0.0,
                                 float(correct(p.G, p.Q, np.arange(len(p.P)), m.nn, 10.0).mean())))
        self.model.train()
        torch.set_rng_state(rng_state[0])
        torch.cuda.set_rng_state(rng_state[1])
        out = {}
        for key, r in res.items():
            if r:
                out[f"val_{key}_inlier10"] = float(np.mean([a for a, _ in r]))
                out[f"val_{key}_recall10"] = float(np.mean([b for _, b in r]))
        return out

    def save(self, step: int) -> None:
        torch.save({"model": self.model.state_dict(), "config": OmegaConf.to_container(self.cfg),
                    "step": step}, self.out / f"step_{step:06d}.pt")

    def fit(self) -> None:
        cfg = self.cfg
        subjects = liver_meshes.subjects("train")
        if cfg.data.frac_livers < 1:
            rng = np.random.default_rng(cfg.seed)
            subjects = sorted(rng.choice(subjects, int(round(cfg.data.frac_livers * len(subjects))),
                                         replace=False).tolist())
        aug = AugConfig(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in OmegaConf.to_container(cfg.data.aug).items()})
        ds = LiverViews(subjects, self.pcfg, aug, seed=cfg.seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=True,
                                         num_workers=cfg.data.workers, collate_fn=collate, drop_last=True,
                                         persistent_workers=True, prefetch_factor=4,
                                         worker_init_fn=lambda _: torch.set_num_threads(1))
        log = open(self.out / "log.jsonl", "a")
        rec = {"step": 0, **self.proxy()}
        log.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        step, t0, acc = 0, time.time(), []
        self.model.train()
        while step < cfg.optim.steps:
            for batch in dl:
                acc.append(self.step(batch))
                step += 1
                if step % 50 == 0:
                    rec = {"step": step, "lr": self.sched.get_last_lr()[0], "sec_per_step": (time.time() - t0) / 50,
                           **{k: float(np.mean([a[k] for a in acc if k in a])) for k in set().union(*acc)}}
                    t0, acc = time.time(), []
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(rec, flush=True)
                if step % cfg.eval.every == 0 or step == cfg.optim.steps:
                    rec = {"step": step, **self.proxy()}
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(rec, flush=True)
                    self.save(step)
                if step >= cfg.optim.steps:
                    break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    Trainer(load_cfg(a.config, a.overrides)).fit()


if __name__ == "__main__":
    main()

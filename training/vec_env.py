"""
training/vec_env.py  (v2 – distributed rollout)
────────────────────────────────────────────────────────────────────────────
Each worker owns one OverrideEnv AND collects a full ROLLOUT_STEPS-step
rollout locally using a CPU copy of the current policy.

IPC per update cycle
  OLD (step-by-step): 512 steps × 32 workers × ~1.7 ms recv latency = 27 s
  NEW (distributed):  1 send (weights) + 1 recv (buffer) per worker   ≈ 1.8 s

The main process serialises policy weights → sends once → workers compute
512 steps in parallel → main receives completed numpy buffers → GAE + PPO.
"""

import os
import warnings
import multiprocessing as mp
import numpy as np
from typing import Dict, List

_CMD_COLLECT = 0
_CMD_CLOSE   = 1


# ─────────────────────────────────────────────────────────────────────────────
def _load_cpu_net(cls, sd_np: dict):
    """Reconstruct a network on CPU from a {name: numpy_array} state-dict."""
    import torch
    net = cls()
    net.load_state_dict({k: torch.from_numpy(v) for k, v in sd_np.items()})
    net.eval()
    return net


def _worker_fn(conn, seed: int) -> None:
    """
    Worker process entry point.

    Waits for _CMD_COLLECT, loads CPU policies, collects n_steps of experience
    with local inference, then sends back packed numpy buffer data.
    """
    # ── GPU isolation: workers are CPU-only. ─────────────────────────────────
    # Without this, `import torch` in every worker enumerates CUDA devices,
    # sending dozens of simultaneous requests to the GPU driver — the root
    # cause of the driver_power_state_failure BSOD on Windows.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    warnings.filterwarnings("ignore", category=UserWarning)

    import torch
    # Each worker gets exactly 1 torch thread.
    # Without this, all 16 workers compete for all 24 CPU threads simultaneously,
    # causing torch matrix-multiply contention that inflates per-step time ~5×.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from training.env_wrapper import OverrideEnv
    from training.network    import Policy, CentralizedCritic

    env   = OverrideEnv(headless=True, seed=seed)
    obs   = env.reset()
    masks = env.get_action_masks()

    conn.send("ready")   # signal: fully initialised

    try:
        while True:
            cmd, data = conn.recv()

            if cmd == _CMD_CLOSE:
                break

            if cmd == _CMD_COLLECT:
                n_steps, pd = data   # pd = serialised policy dicts

                # ── Load CPU policies (fresh weights each update cycle) ────────
                red_policy  = _load_cpu_net(Policy,           pd["red_policy"])
                blue_policy = _load_cpu_net(Policy,           pd["blue_policy"])
                red_critic  = _load_cpu_net(CentralizedCritic, pd["red_critic"])
                blue_critic = _load_cpu_net(CentralizedCritic, pd["blue_critic"])

                # ── Rollout accumulators ──────────────────────────────────────
                _FIELDS = ["obs0","obs1","cont0","cont1","disc0","disc1",
                           "lp0","lp1","mask0","mask1","reward","value","done"]
                rd = {f: [] for f in _FIELDS}
                bd = {f: [] for f in _FIELDS}
                episodes = []

                for _ in range(n_steps):
                    with torch.no_grad():
                        # Batch both alliance robots in one forward pass each
                        # (2 policy calls per step instead of 4)
                        ro = torch.FloatTensor(
                            np.stack([obs["red1"],  obs["red2"]]))   # (2, OBS)
                        rm = torch.BoolTensor(
                            np.stack([masks["red1"], masks["red2"]]))# (2, DISC)
                        bo = torch.FloatTensor(
                            np.stack([obs["blue1"], obs["blue2"]]))
                        bm = torch.BoolTensor(
                            np.stack([masks["blue1"],masks["blue2"]]))

                        rc, r_disc_t, rlp, _ = red_policy.get_action(ro, rm, False)
                        bc, bd_,     blp, _ = blue_policy.get_action(bo, bm, False)

                        r1o_v = ro[0]; r2o_v = ro[1]
                        b1o_v = bo[0]; b2o_v = bo[1]
                        rv = red_critic( r1o_v.unsqueeze(0), r2o_v.unsqueeze(0)).item()
                        bv = blue_critic(b1o_v.unsqueeze(0), b2o_v.unsqueeze(0)).item()

                    r1c, r2c = rc[0].numpy(),       rc[1].numpy()
                    r1d, r2d = r_disc_t[0].numpy(), r_disc_t[1].numpy()
                    b1c, b2c = bc[0].numpy(),  bc[1].numpy()
                    b1d, b2d = bd_[0].numpy(), bd_[1].numpy()
                    r1lp, r2lp = float(rlp[0]), float(rlp[1])
                    b1lp, b2lp = float(blp[0]), float(blp[1])

                    step_actions = {
                        "red1":  (r1c, r1d),
                        "red2":  (r2c, r2d),
                        "blue1": (b1c, b1d),
                        "blue2": (b2c, b2d),
                    }
                    next_obs, rewards, done, info = env.step(step_actions)
                    next_masks = env.get_action_masks()

                    red_r  = (rewards.get("red1",  0.0) + rewards.get("red2",  0.0)) / 2.0
                    blue_r = (rewards.get("blue1", 0.0) + rewards.get("blue2", 0.0)) / 2.0

                    # Store pre-step obs (current obs at inference time)
                    rd["obs0"].append(obs["red1"]);   rd["obs1"].append(obs["red2"])
                    rd["cont0"].append(r1c);  rd["cont1"].append(r2c)
                    rd["disc0"].append(r1d);  rd["disc1"].append(r2d)
                    rd["lp0"].append(float(r1lp));    rd["lp1"].append(float(r2lp))
                    rd["mask0"].append(masks["red1"]); rd["mask1"].append(masks["red2"])
                    rd["reward"].append(red_r);       rd["value"].append(rv)
                    rd["done"].append(done)

                    bd["obs0"].append(obs["blue1"]);  bd["obs1"].append(obs["blue2"])
                    bd["cont0"].append(b1c);  bd["cont1"].append(b2c)
                    bd["disc0"].append(b1d);  bd["disc1"].append(b2d)
                    bd["lp0"].append(float(b1lp));    bd["lp1"].append(float(b2lp))
                    bd["mask0"].append(masks["blue1"]); bd["mask1"].append(masks["blue2"])
                    bd["reward"].append(blue_r);      bd["value"].append(bv)
                    bd["done"].append(done)

                    if done:
                        if "red_score" in info:
                            episodes.append((info["red_score"], info["blue_score"]))
                        obs   = env.reset()
                        masks = env.get_action_masks()
                    else:
                        obs   = next_obs
                        masks = next_masks

                # ── Bootstrap last values ─────────────────────────────────────
                with torch.no_grad():
                    r1o = torch.FloatTensor(obs["red1"])
                    r2o = torch.FloatTensor(obs["red2"])
                    b1o = torch.FloatTensor(obs["blue1"])
                    b2o = torch.FloatTensor(obs["blue2"])
                    last_rv = red_critic( r1o.unsqueeze(0), r2o.unsqueeze(0)).item()
                    last_bv = blue_critic(b1o.unsqueeze(0), b2o.unsqueeze(0)).item()

                # ── Pack into numpy arrays for efficient IPC ──────────────────
                def arr(lst, dtype=np.float32):
                    return np.array(lst, dtype=dtype)
                def stk(lst):
                    return np.stack(lst).astype(np.float32)

                conn.send({
                    "red": {
                        "obs0":  stk(rd["obs0"]),  "obs1":  stk(rd["obs1"]),
                        "cont0": stk(rd["cont0"]), "cont1": stk(rd["cont1"]),
                        "disc0": stk(rd["disc0"]), "disc1": stk(rd["disc1"]),
                        "lp0":   arr(rd["lp0"]),   "lp1":   arr(rd["lp1"]),
                        "mask0": stk(rd["mask0"]), "mask1": stk(rd["mask1"]),
                        "reward":arr(rd["reward"]), "value": arr(rd["value"]),
                        "done":  arr(rd["done"], np.bool_),
                    },
                    "blue": {
                        "obs0":  stk(bd["obs0"]),  "obs1":  stk(bd["obs1"]),
                        "cont0": stk(bd["cont0"]), "cont1": stk(bd["cont1"]),
                        "disc0": stk(bd["disc0"]), "disc1": stk(bd["disc1"]),
                        "lp0":   arr(bd["lp0"]),   "lp1":   arr(bd["lp1"]),
                        "mask0": stk(bd["mask0"]), "mask1": stk(bd["mask1"]),
                        "reward":arr(bd["reward"]), "value": arr(bd["value"]),
                        "done":  arr(bd["done"], np.bool_),
                    },
                    "last_rv": last_rv,
                    "last_bv": last_bv,
                    "episodes": episodes,
                })

    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try: env.close()
        except Exception: pass
        try: conn.close()
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
class SubprocVecEnv:
    """
    N OverrideEnv instances running in N separate processes.

    Each call to collect_rollouts() sends current policy weights to every
    worker, waits for all workers to return completed rollout buffers, and
    returns the list of result dicts.  No per-step IPC.
    """

    def __init__(self, num_envs: int, base_seed: int = 42) -> None:
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")

        self._conns: List[mp.connection.Connection] = []
        self._procs: List[mp.Process] = []

        print(f"[Train] Spawning {num_envs} worker processes (staggered) …",
              flush=True)
        import time as _time
        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_worker_fn,
                args=(child_conn, base_seed + i * 1337),
                daemon=True,
                name=f"EnvWorker-{i}",
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)
            # Stagger by 0.4 s per worker: prevents 16 Python interpreters
            # from all importing torch/pygame simultaneously, which would
            # spike CPU to 100 % and stress the GPU driver.
            _time.sleep(0.4)

        ready = sum(1 for conn in self._conns if conn.recv() == "ready")
        print(f"[Train] {ready}/{num_envs} workers ready.", flush=True)

    def collect_rollouts(self, policy_data: dict, n_steps: int) -> list:
        """
        Fan policy weights out to all workers; collect completed rollout buffers.

        policy_data = {
            'red_policy':  {name: np_array},
            'blue_policy': {name: np_array},
            'red_critic':  {name: np_array},
            'blue_critic': {name: np_array},
        }
        Returns list[dict] — one result per worker.
        """
        for conn in self._conns:
            conn.send((_CMD_COLLECT, (n_steps, policy_data)))
        return [conn.recv() for conn in self._conns]

    def close(self) -> None:
        for conn in self._conns:
            try: conn.send((_CMD_CLOSE, None))
            except Exception: pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            try: conn.close()
            except Exception: pass

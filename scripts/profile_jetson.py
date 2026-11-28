import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from model      import AgenticViT
from controller import AgenticController, ControllerConfig, ControllerState


ACTIONS = ["Infer", "Enhance", "FeatureSend", "Adapt", "Defer"]


def read_ina3221_mW(channel: int = 0) -> float:
    import glob
    path    = f"/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/curr{channel + 1}_input"
    matches = glob.glob(path)
    if not matches:
        return -1.0
    try:
        with open(matches[0]) as f:
            curr_mA = float(f.read())
        volt_path = matches[0].replace(f"curr{channel + 1}", f"in{channel + 1}")
        with open(volt_path) as f:
            volt_mV = float(f.read())
        return curr_mA * volt_mV / 1e6
    except Exception:
        return -1.0


def benchmark_action(model, action: str, device,
                     n_runs: int = 1000, n_warmup: int = 20) -> dict:
    model.eval()
    dummy_frames = torch.randn(1, 16, 3, 224, 224, device=device)
    latencies    = []

    for i in range(n_warmup + n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if action in ("Infer", "Defer"):
            with torch.no_grad():
                model(dummy_frames)

        elif action == "Enhance":
            enhanced = dummy_frames + 0.01 * torch.randn_like(dummy_frames)
            with torch.no_grad():
                model(enhanced)

        elif action == "FeatureSend":
            with torch.no_grad():
                _, probs, h = model(dummy_frames)
            import hashlib
            _ = hashlib.sha256(h.squeeze(0).cpu().numpy().tobytes()).digest()

        elif action == "Adapt":
            with torch.no_grad():
                model(dummy_frames)
            adapter_params = [p for n, p in model.named_parameters() if "adapter" in n]
            if adapter_params:
                opt = torch.optim.Adam(adapter_params, lr=1e-4)
                for _ in range(10):
                    opt.zero_grad()
                    _, probs, _ = model(dummy_frames)
                    loss = -probs.log().mean()
                    loss.backward()
                    opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= n_warmup:
            latencies.append((t1 - t0) * 1000)

    lat = np.array(latencies)
    return {
        "action":  action,
        "mean_ms": round(float(lat.mean()), 2),
        "p95_ms":  round(float(np.percentile(lat, 95)), 2),
        "p99_ms":  round(float(np.percentile(lat, 99)), 2),
        "max_ms":  round(float(lat.max()), 2),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AgenticViT(num_classes=2, L=16).to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)

    results = []
    for action in ACTIONS:
        print(f"  Profiling {action}...", end=" ", flush=True)
        res = benchmark_action(model, action, device,
                               n_runs=args.n_runs, n_warmup=args.n_warmup)
        results.append(res)
        print(f"mean={res['mean_ms']:.1f}ms  P95={res['p95_ms']:.1f}ms")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "profiling.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--n_runs",     type=int, default=1000)
    p.add_argument("--n_warmup",   type=int, default=20)
    main(p.parse_args())

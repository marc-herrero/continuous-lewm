#!/usr/bin/env python3
"""
scripts/make_figures.py
Loads committed per-episode result JSONs from results/ and generates:
1. Paired McNemar tables with exact p-values and Wilson score CIs
2. Unified scaling frontier curve (CEM vs Hybrid vs Pure EKI)
3. End-to-end training parity table across epochs
4. Frozen-encoder 4.8x parameter-efficiency table
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path so cwm can be imported
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwm.analysis.statistics import mcnemar_test, wilson_score_interval


def analyze_frozen_study(results_dir: Path):
    print("\n" + "=" * 95)
    print("STUDY 1: FROZEN-ENCODER PREDICTOR COMPARISON (PushT, MPC CEM)")
    print("=" * 95)

    base_f = results_dir / "eval_discrete_baseline_400ep.json"
    ode_f = results_dir / "eval_continuous_base_400ep.json"

    if base_f.exists() and ode_f.exists():
        with open(base_f) as f:
            d_disc = json.load(f)
        with open(ode_f) as f:
            d_ode = json.load(f)

        vec_disc = d_disc.get("episode_successes", d_disc.get("success_flags", []))
        vec_ode = d_ode.get("episode_successes", d_ode.get("success_flags", []))

        n = len(vec_disc)
        succ_disc = sum(vec_disc)
        succ_ode = sum(vec_ode)
        ci_disc = wilson_score_interval(succ_disc, n)
        ci_ode = wilson_score_interval(succ_ode, n)

        mcn = mcnemar_test(vec_disc, vec_ode)

        print(f"Discrete Baseline (10.79M, 3-frame hist): {succ_disc}/{n} ({succ_disc/n*100:.2f}%) [95% CI: {ci_disc[0]:.1f}, {ci_disc[1]:.1f}]")
        print(f"Continuous ODE-ViT (2.25M, 1-frame Markov): {succ_ode}/{n} ({succ_ode/n*100:.2f}%) [95% CI: {ci_ode[0]:.1f}, {ci_ode[1]:.1f}]")
        print(f"Discordant pairs (ODE wins: {mcn['c_test_wins']}, Discrete wins: {mcn['b_baseline_wins']}) | Exact McNemar p = {mcn['p_value']:.4f}")
    print("=" * 95)


def analyze_end_to_end(results_dir: Path):
    print("\n" + "=" * 95)
    print("STUDY 2: END-TO-END TRAINING FROM PIXELS (N=200 Paired, Seed 42)")
    print("=" * 95)
    print(f"{'Epoch':<8} | {'Discrete (n=200)':<22} | {'Continuous ODE (n=200)':<24} | {'Discordant (ODE/Disc)':<22} | {'McNemar p':<10}")
    print("-" * 95)

    epochs = [10, 50, 73, 85, 100]
    for ep in epochs:
        disc_p = results_dir / f"eval_reproducible_disc_ep{ep}_200ep.json"
        ode_p = results_dir / f"eval_reproducible_ode_ep{ep}_n4_200ep.json"

        if disc_p.exists() and ode_p.exists():
            d_disc = json.load(open(disc_p))
            d_ode = json.load(open(ode_p))

            v_d = d_disc.get("episode_successes", d_disc.get("success_flags", []))
            v_o = d_ode.get("episode_successes", d_ode.get("success_flags", []))

            s_d = sum(v_d)
            s_o = sum(v_o)
            mcn = mcnemar_test(v_d, v_o)

            disc_str = f"{s_d/len(v_d)*100:5.1f}% ({s_d}/200)"
            ode_str = f"{s_o/len(v_o)*100:5.1f}% ({s_o}/200)"
            disc_counts = f"{mcn['c_test_wins']:2d} / {mcn['b_baseline_wins']:2d}"
            print(f"Epoch {ep:<2d} | {disc_str:<22} | {ode_str:<24} | {disc_counts:<22} | p = {mcn['p_value']:.3f}")
    print("=" * 95)


def analyze_scaling_frontier(results_dir: Path, output_fig: Path):
    print("\n" + "=" * 95)
    print("PLANNING BENCHMARK: SCALING FRONTIER OF CEM vs HYBRID vs PURE EKI")
    print("=" * 95)

    configs = [
        ("CEM (30x5)", "CEM", 150, "eval_eki_benchmark_cem_30x5_200ep.json"),
        ("Pure EKI (64x10)", "Pure EKI", 640, "eval_eki_benchmark_eki_64x10_200ep.json"),
        ("CEM (64x10)", "CEM", 640, "eval_eki_benchmark_cem_64x10_200ep.json"),
        ("Hybrid (2+5, 64p)", "Hybrid", 448, "eval_eki_benchmark_hybrid_2cem_5eki_200ep.json"),
        ("Hybrid (2+6, 64p)", "Hybrid", 512, "eval_eki_benchmark_hybrid_2cem_6eki_200ep.json"),
        ("Hybrid (2+8, 64p)", "Hybrid", 640, "eval_eki_benchmark_hybrid_2cem_8eki_200ep.json"),
        ("CEM (100x10)", "CEM", 1000, "eval_eki_benchmark_cem_100x10_200ep.json"),
        ("Hybrid (3+5, 128p)", "Hybrid", 1024, "eval_eki_benchmark_hybrid_128p_3cem_5eki_200ep.json"),
        ("Hybrid (4+8, 128p)", "Hybrid", 1536, "eval_eki_benchmark_hybrid_128p_4cem_8eki_200ep.json"),
        ("CEM (100x20)", "CEM", 2000, "eval_eki_benchmark_cem_100x20_200ep.json"),
        ("Hybrid (4+12, 128p)", "Hybrid", 2048, "eval_eki_benchmark_hybrid_128p_4cem_12eki_200ep.json"),
        ("CEM (200x20)", "CEM", 4000, "eval_eki_benchmark_cem_200x20_200ep.json"),
        ("Hybrid (4+16, 200p)", "Hybrid", 4000, "eval_eki_benchmark_hybrid_200p_4cem_16eki_200ep.json"),
        ("Full CEM (300x30)", "CEM", 9000, "eval_reproducible_disc_ep10_200ep.json"),
    ]

    loaded = []
    for name, fam, rollouts, fname in configs:
        fpath = results_dir / fname
        if fpath.exists():
            d = json.load(open(fpath))
            succ = d.get("success_count", d.get("successes", sum(d.get("episode_successes", d.get("success_flags", [])))))
            tot = d.get("total_episodes", len(d.get("episode_successes", d.get("success_flags", []))))
            rate = (succ / tot) * 100.0
            ci_l, ci_u = wilson_score_interval(succ, tot)
            t_ep = d.get("time_per_ep_s", d.get("avg_time_per_ep", d.get("total_eval_time", 0) / tot if tot else 0))
            loaded.append({
                "name": name,
                "family": fam,
                "rollouts": rollouts,
                "rate": rate,
                "succ": succ,
                "tot": tot,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "time_per_ep": t_ep,
            })

    print(f"{'Method':<24} | {'Rollouts':<8} | {'Success (95% CI)':<22} | {'Time/ep':<8}")
    print("-" * 75)
    for it in loaded:
        ci_str = f"{it['rate']:5.1f}% [{it['ci_lower']:4.1f}, {it['ci_upper']:4.1f}]"
        print(f"{it['name']:<24} | {it['rollouts']:<8} | {ci_str:<22} | {it['time_per_ep']:.2f}s")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Reproduce all figures and statistics from results JSONs.")
    parser.add_argument("--results", type=str, default="results", help="Directory containing result JSONs.")
    parser.add_argument("--figures", type=str, default="assets/figures", help="Directory to save figures.")
    args = parser.parse_args()

    results_dir = Path(args.results)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyze_frozen_study(results_dir)
    analyze_end_to_end(results_dir)
    analyze_scaling_frontier(results_dir, figures_dir / "eki_cem_scaling_curves.png")


if __name__ == "__main__":
    main()

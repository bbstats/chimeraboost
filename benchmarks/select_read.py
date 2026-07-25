"""SELECT Phase 0 reader: frontier position of the pinned fast arms.

Evaluates the bars pre-registered in benchmarks/SELECT_PLAN.md against a
single results JSON, so every quantity comes from one field.

  Bar A  cost      candidate slowdown <= --cost-bar (default 2.5x)
  Bar B  frontier  candidate win rate strictly above the chord from the
                   reference-fast model (LightGBM) to the reference-strong
                   model (ChimeraBoost), evaluated at the candidate's own
                   slowdown. An interior point is not a frontier point.

Bar C (candidate genuinely beats LightGBM on the primary sign test) is not
computed here -- compare_runs.py already does paired sign tests and now
carries the near-solved guard, so run it with the same JSON twice:

  python benchmarks/compare_runs.py RUN.json RUN.json LightGBM CAND \
      --model LightGBM --model-new CAND

Usage:  python benchmarks/select_read.py [RUN.json] [--cost-bar 2.5]
"""
import argparse

import make_pareto
import summarize


def chord(wr_fast, s_fast, wr_strong, s_strong, s):
    """Win rate the LightGBM->ChimeraBoost interpolation reaches at slowdown s."""
    if s_strong == s_fast:
        return None
    t = (s - s_fast) / (s_strong - s_fast)
    return wr_fast + (wr_strong - wr_fast) * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default=None)
    ap.add_argument("--cost-bar", type=float, default=2.5)
    ap.add_argument("--fast-ref", default="LightGBM")
    ap.add_argument("--strong-ref", default="ChimeraBoost")
    ap.add_argument("--candidates", nargs="+",
                    default=["ChimeraBoostOne", "ChimeraBoostOneLin",
                             "ChimeraBoostSel25"])
    args = ap.parse_args()

    path = args.json_path or summarize.latest_json()
    data = summarize.load(path)
    scored, meta, primary = make_pareto.score_models(data)
    front = make_pareto.pareto_frontier(scored)

    print(f"run: {path}")
    print(f"datasets scored head-to-head: {meta.get('n_h2h')}  "
          f"exact ties: {meta.get('n_ties')}")
    print()
    print(f"{'model':<22}{'win rate':>10}{'95% CI':>16}{'slowdown':>11}"
          f"{'frontier':>10}")
    for m, s in sorted(scored.items(), key=lambda kv: -(kv[1]["winrate"] or 0)):
        wr, lo, hi = s["winrate"], s["wr_lo"], s["wr_hi"]
        if wr is None or s["slowdown"] is None:
            continue
        print(f"{m:<22}{wr*100:>9.1f}%"
              f"{'[%.1f, %.1f]' % (lo*100, hi*100):>16}"
              f"{s['slowdown']:>10.2f}x"
              f"{'yes' if m in front else '-':>10}")
    print()

    ref_f, ref_s = scored.get(args.fast_ref), scored.get(args.strong_ref)
    if not ref_f or not ref_s or ref_f["winrate"] is None:
        print("missing reference arms; cannot evaluate the chord.")
        return
    wr_f, s_f = ref_f["winrate"], ref_f["slowdown"]
    wr_s, s_s = ref_s["winrate"], ref_s["slowdown"]
    print(f"chord anchors: {args.fast_ref} {wr_f*100:.1f}% @ {s_f:.2f}x  ->  "
          f"{args.strong_ref} {wr_s*100:.1f}% @ {s_s:.2f}x")
    print()

    for cand in args.candidates:
        s = scored.get(cand)
        if not s or s["winrate"] is None or s["slowdown"] is None:
            print(f"{cand}: absent from this run")
            continue
        wr, sl = s["winrate"], s["slowdown"]
        need = chord(wr_f, s_f, wr_s, s_s, sl)
        bar_a = sl <= args.cost_bar
        bar_b = need is not None and wr > need
        print(f"{cand}")
        print(f"   win rate {wr*100:.1f}%  slowdown {sl:.2f}x")
        print(f"   Bar A (cost <= {args.cost_bar}x):        "
              f"{'PASS' if bar_a else 'FAIL'}")
        print(f"   Bar B (chord needs > {need*100:.1f}%):   "
              f"{'PASS' if bar_b else 'FAIL'}"
              f"   margin {(wr - need)*100:+.1f} pts")
        print(f"   on Pareto frontier: {'yes' if cand in front else 'no'}")
        print()


if __name__ == "__main__":
    main()

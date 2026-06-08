"""Consolidated, phone-friendly reports for one idea's cascade run.

Pure formatting -- takes the verdict dicts the tier runners produce and renders
them as compact text (curves + deltas + verdict), in the spirit of the existing
/bench summary.
"""


def preregister(name, spec):
    lines = [
        "=" * 64,
        f"IDEA: {name}   [{spec['category']}]",
        f"PARAMS: {spec['params']}",
        "PRE-REGISTERED HYPOTHESIS:",
    ]
    # wrap the hypothesis to ~72 cols
    words, line = spec["hypothesis"].split(), "  "
    for w in words:
        if len(line) + len(w) + 1 > 72:
            lines.append(line)
            line = "  " + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        lines.append(line)
    lines.append("=" * 64)
    return "\n".join(lines)


def fast_tier(v, label=None):
    lines = []
    if label:
        lines.append(f"-- {label} --")
    lines.append(f"{'dataset':<26}{'bestD%':>9}{'domin':>8}{'early':>10}")
    lines.append("-" * 53)
    for r in sorted(v["rows"], key=lambda r: r["best_val_delta_pct"]):
        lines.append(f"{r['dataset']:<26}"
                     f"{100*r['best_val_delta_pct']:>+8.2f}%"
                     f"{r['dominance']:>8.2f}"
                     f"{r['early_signal']:>+10.4f}")
    lines.append("-" * 53)
    lines.append(
        f"favorable {v['favorable']}/{v['n']} (need {v['need']}) | "
        f"mean bestD {100*v['mean_best_delta_pct']:+.2f}% | "
        f"mean dominance {v['mean_dominance']:.2f} | {v['seconds']:.1f}s")
    lines.append(f"GATE T0->T1: {'PASS' if v['passed'] else 'KILL'}")
    return "\n".join(lines)


def promo_tier(name, v):
    st = v["sign_test"]
    lines = [f"{'dataset':<26}{'task':>11}{'rel%':>9}{'base':>9}{'var':>9}"]
    lines.append("-" * 64)
    for r in sorted(v["rows"], key=lambda r: r["rel"]):
        lines.append(f"{r['dataset']:<26}{r['task']:>11}"
                     f"{100*r['rel']:>+8.2f}%"
                     f"{r['base']['primary']:>9.4f}"
                     f"{r['variant']['primary']:>9.4f}")
    lines.append("-" * 64)
    lines.append(
        f"[{name}] wins {st['wins']} / losses {st['losses']} / ties {st['ties']}"
        f"  (n={st['n']}, p={st['p']:.3f}) | mean rel {100*v['mean_rel']:+.2f}% "
        f"| tree ratio {v['mean_tree_ratio']:.2f}x | {v['seconds']:.1f}s")
    lines.append(f"GATE {name}: {'PASS' if v['passed'] else 'STOP'}")
    return "\n".join(lines)


def final(out):
    lines = ["", "#" * 64, f"# VERDICT [{out['idea']}]: {out['verdict']}"]
    for tier, v in out["tiers"].items():
        if "sign_test" in v:
            st = v["sign_test"]
            lines.append(f"#   {tier}: wins {st['wins']}/{st['losses']} "
                         f"p={st['p']:.3f} mean rel {100*v['mean_rel']:+.2f}%")
        else:
            lines.append(f"#   {tier}: favorable {v['favorable']}/{v['n']} "
                         f"mean bestD {100*v['mean_best_delta_pct']:+.2f}%")
    lines.append("#" * 64)
    return "\n".join(lines)

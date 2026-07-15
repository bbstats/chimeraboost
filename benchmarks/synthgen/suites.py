"""Frozen suite id lists + verified canary ids. smoke is a subset of screen is
a subset of full (asserted in tests) so per-dataset pairing stays valid across
tiers.

CANARIES = saturated ids whose ceiling the default baseline PROVABLY reaches
(freeze-time fit check, filters.at_ceiling). Dataset meta must stay a pure
function of (VERSION, id), so this freeze-time knowledge lives here, not in
meta; backtest.py / synth_report.py read it from here.

Regenerate with freeze.py after ANY change to the generator (and bump
recipe.VERSION -- the version lives inside every dataset key).
"""
from .recipe import VERSION  # noqa: F401  (re-exported)

# Frozen 2026-07-14 (v2) by freeze.py --count 1000 (row budgets 400K/1.6M;
# scan: 789/1000 accepted, canaries 58/100 saturated verified at ceiling; screen
# 136 sets / 401K rows, n<2000 share 0.35, 6 cat-bearing canaries; full 211
# sets / 1.61M rows).
SUITES = {
    "smoke": [340, 382, 618, 775, 792, 948],
    "screen": [5, 8, 14, 15, 17, 19, 25, 48, 55, 61, 66, 69, 74, 78, 94, 110, 112, 116, 117, 125, 128, 130, 132, 134, 147, 167, 173, 182, 187, 189, 190, 191, 204, 211, 214, 215, 225, 236, 249, 251, 252, 255, 262, 271, 273, 295, 297, 300, 303, 313, 315, 317, 323, 326, 329, 340, 342, 346, 355, 357, 368, 372, 382, 390, 401, 414, 440, 441, 447, 453, 459, 464, 465, 472, 478, 479, 484, 500, 505, 531, 534, 537, 548, 553, 555, 557, 561, 562, 569, 589, 590, 595, 614, 618, 621, 639, 648, 663, 667, 686, 697, 699, 700, 729, 730, 731, 733, 737, 739, 747, 749, 764, 765, 768, 775, 784, 792, 793, 801, 806, 813, 842, 845, 857, 871, 878, 879, 885, 921, 935, 945, 947, 948, 953, 976, 993],
    "full": [1, 5, 8, 11, 14, 15, 17, 19, 20, 25, 35, 42, 48, 55, 59, 61, 62, 66, 69, 74, 78, 94, 95, 102, 107, 110, 111, 112, 116, 117, 125, 128, 130, 132, 134, 137, 139, 147, 156, 167, 173, 182, 187, 189, 190, 191, 204, 205, 209, 211, 214, 215, 225, 228, 231, 236, 249, 251, 252, 255, 262, 271, 273, 276, 284, 295, 296, 297, 300, 302, 303, 313, 315, 317, 323, 324, 326, 327, 329, 332, 337, 340, 342, 345, 346, 355, 357, 368, 372, 377, 382, 388, 390, 401, 412, 414, 426, 440, 441, 446, 447, 448, 453, 459, 464, 465, 467, 472, 478, 479, 484, 496, 497, 500, 502, 505, 518, 527, 531, 533, 534, 537, 548, 550, 553, 555, 557, 561, 562, 569, 570, 572, 575, 589, 590, 595, 610, 611, 614, 618, 621, 622, 639, 640, 648, 652, 663, 667, 672, 678, 685, 686, 697, 699, 700, 701, 709, 719, 726, 729, 730, 731, 733, 737, 739, 744, 745, 746, 747, 749, 764, 765, 768, 775, 784, 787, 788, 792, 793, 794, 801, 802, 806, 813, 819, 823, 833, 842, 845, 851, 857, 871, 878, 879, 880, 885, 893, 908, 921, 935, 945, 947, 948, 949, 953, 960, 970, 972, 976, 993, 996],
}

# Re-verified 2026-07-15 with the tightened 3-seed check (freeze.py
# --canaries-only): 18 -> 8; the drops had real multi-seed headroom.
CANARIES = {17, 107, 117, 317, 327, 527, 747, 787}


def frozen_keys(suite):
    from .api import key_for
    ids = SUITES[suite]
    return [key_for(i) for i in ids]


def all_frozen_keys():
    from .api import key_for
    seen, out = set(), []
    for name in ("smoke", "screen", "full"):
        for i in SUITES[name]:
            if i not in seen:
                seen.add(i)
                out.append(key_for(i))
    return out

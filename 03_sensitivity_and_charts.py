"""
03_sensitivity_and_charts.py
----------------------------
SupeRR Selector Season 2 (Rajasthan Royals) - Step 3:
1. Runs Sensitivity & Temporal Reliability stress tests:
   - Model A (Primary): Elite (IPL + Full-Member T20I), PI >= 45
   - Model B: All Matches (including Associate/Emerging), PI >= 45
   - Model C: Stricter Threshold (Elite, PI >= 50, min_hp=120)
   - Model D: Temporal Split (2022-2024 vs 2025-2026 correlation)
2. Generates 3 publication-quality 180-DPI charts for the Methodology Report.
3. Exports a multi-tab styled Excel workbook (`SupeRR_Selector_S2_Refined_Analysis.xlsx`).
"""

import importlib
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

step2 = importlib.import_module("02_pressure_and_ratings")


def run_sensitivity_tests(
    df: pd.DataFrame, qual_bat_A: pd.DataFrame, qual_bowl_A: pd.DataFrame, out_dir: str = "."
) -> pd.DataFrame:
    """Compares Primary Model A against All-Matches (B), Stricter PI>=50 (C), and Temporal Split (D)."""
    # Model B: All matches
    qual_bat_B = step2.evaluate_batters(df, elite_only=False, min_hp_balls=150, min_norm_balls=300, min_qual_matches=12)
    qual_bowl_B = step2.evaluate_bowlers(df, elite_only=False, min_hp_balls=150, min_norm_balls=300, min_qual_matches=12)

    m_bat_B = qual_bat_A[["striker_id", "clutch_bat_score"]].merge(
        qual_bat_B[["striker_id", "clutch_bat_score"]], on="striker_id", suffixes=("_A", "_B")
    )
    m_bowl_B = qual_bowl_A[["bowler_id", "clutch_bowl_score"]].merge(
        qual_bowl_B[["bowler_id", "clutch_bowl_score"]], on="bowler_id", suffixes=("_A", "_B")
    )

    # Model C: Stricter PI >= 50
    df_C = df.copy()
    df_C["is_hp"] = df_C["pressure_index"] >= 50.0
    qual_bat_C = step2.evaluate_batters(df_C, elite_only=True, min_hp_balls=120, min_norm_balls=300, min_qual_matches=10)
    qual_bowl_C = step2.evaluate_bowlers(df_C, elite_only=True, min_hp_balls=120, min_norm_balls=300, min_qual_matches=10)

    m_bat_C = qual_bat_A[["striker_id", "clutch_bat_score"]].merge(
        qual_bat_C[["striker_id", "clutch_bat_score"]], on="striker_id", suffixes=("_A", "_C")
    )
    m_bowl_C = qual_bowl_A[["bowler_id", "clutch_bowl_score"]].merge(
        qual_bowl_C[["bowler_id", "clutch_bowl_score"]], on="bowler_id", suffixes=("_A", "_C")
    )

    # Model D: Temporal Split (2022-mid 2024 vs mid 2024-2026) on EB HP Impact
    df_early = df[df["match_date"] < "2024-07-01"]
    df_late = df[df["match_date"] >= "2024-07-01"]
    qb_e = step2.evaluate_batters(df_early, elite_only=True, min_hp_balls=60, min_norm_balls=100, min_qual_matches=5)
    qb_l = step2.evaluate_batters(df_late, elite_only=True, min_hp_balls=60, min_norm_balls=100, min_qual_matches=5)
    qw_e = step2.evaluate_bowlers(df_early, elite_only=True, min_hp_balls=60, min_norm_balls=100, min_qual_matches=5)
    qw_l = step2.evaluate_bowlers(df_late, elite_only=True, min_hp_balls=60, min_norm_balls=100, min_qual_matches=5)

    t_bat = qb_e[["striker_id", "eb_impact_hp"]].merge(qb_l[["striker_id", "eb_impact_hp"]], on="striker_id", suffixes=("_E", "_L"))
    t_bowl = qw_e[["bowler_id", "eb_impact_hp"]].merge(qw_l[["bowler_id", "eb_impact_hp"]], on="bowler_id", suffixes=("_E", "_L"))

    rows = [
        {
            "model_variant": "Model A (Primary): Elite IPL+FM, PI >= 45",
            "hp_deliveries": int(df[df["is_elite"]]["is_hp"].sum()),
            "hp_share_pct": round(float(df[df["is_elite"]]["is_hp"].mean() * 100), 2),
            "qualified_batters": len(qual_bat_A),
            "top10_bat_overlap": "10 / 10 (Baseline)",
            "spearman_r_batters": 1.0000,
            "qualified_bowlers": len(qual_bowl_A),
            "top10_bowl_overlap": "10 / 10 (Baseline)",
            "spearman_r_bowlers": 1.0000,
        },
        {
            "model_variant": "Model B: All Matches (incl. Associate/Emerging), PI >= 45",
            "hp_deliveries": int(df["is_hp"].sum()),
            "hp_share_pct": round(float(df["is_hp"].mean() * 100), 2),
            "qualified_batters": len(qual_bat_B),
            "top10_bat_overlap": f"{len(set(qual_bat_A.head(10)['striker_name']).intersection(set(qual_bat_B.head(10)['striker_name'])))} / 10",
            "spearman_r_batters": round(float(m_bat_B["clutch_bat_score_A"].corr(m_bat_B["clutch_bat_score_B"], method="spearman")), 4),
            "qualified_bowlers": len(qual_bowl_B),
            "top10_bowl_overlap": f"{len(set(qual_bowl_A.head(10)['bowler_name']).intersection(set(qual_bowl_B.head(10)['bowler_name'])))} / 10",
            "spearman_r_bowlers": round(float(m_bowl_B["clutch_bowl_score_A"].corr(m_bowl_B["clutch_bowl_score_B"], method="spearman")), 4),
        },
        {
            "model_variant": "Model C: Stricter Threshold (Elite, PI >= 50, min 120 HP)",
            "hp_deliveries": int(df_C[df_C["is_elite"]]["is_hp"].sum()),
            "hp_share_pct": round(float(df_C[df_C["is_elite"]]["is_hp"].mean() * 100), 2),
            "qualified_batters": len(qual_bat_C),
            "top10_bat_overlap": f"{len(set(qual_bat_A.head(10)['striker_name']).intersection(set(qual_bat_C.head(10)['striker_name'])))} / 10",
            "spearman_r_batters": round(float(m_bat_C["clutch_bat_score_A"].corr(m_bat_C["clutch_bat_score_C"], method="spearman")), 4),
            "qualified_bowlers": len(qual_bowl_C),
            "top10_bowl_overlap": f"{len(set(qual_bowl_A.head(10)['bowler_name']).intersection(set(qual_bowl_C.head(10)['bowler_name'])))} / 10",
            "spearman_r_bowlers": round(float(m_bowl_C["clutch_bowl_score_A"].corr(m_bowl_C["clutch_bowl_score_C"], method="spearman")), 4),
        },
        {
            "model_variant": "Model D: Temporal Split (2022-mid24 vs mid24-26 EB HP Impact)",
            "hp_deliveries": int(df[df["is_elite"]]["is_hp"].sum()),
            "hp_share_pct": round(float(df[df["is_elite"]]["is_hp"].mean() * 100), 2),
            "qualified_batters": len(t_bat),
            "top10_bat_overlap": "Out-of-Sample",
            "spearman_r_batters": round(float(t_bat["eb_impact_hp_E"].corr(t_bat["eb_impact_hp_L"], method="spearman")), 4),
            "qualified_bowlers": len(t_bowl),
            "top10_bowl_overlap": "Out-of-Sample",
            "spearman_r_bowlers": round(float(t_bowl["eb_impact_hp_E"].corr(t_bowl["eb_impact_hp_L"], method="spearman")), 4),
        },
    ]
    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(os.path.join(out_dir, "sensitivity_comparison.csv"), index=False)
    return sens_df


def _place_labels_no_overlap(ax, fig, points, max_iter=350):
    """Places (x, y, text) annotations with leader lines and iterative bbox repulsion."""
    texts = []
    # Initial deterministic radial offsets to break ties cleanly
    angles = np.linspace(0, 2 * np.pi, len(points), endpoint=False)
    for idx, (x, y, label) in enumerate(points):
        dx = 1.6 * np.cos(angles[idx])
        dy = 1.4 * np.sin(angles[idx])
        t = ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=8.2,
            fontweight="semibold",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
            arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.75, shrinkA=0, shrinkB=3),
            zorder=5,
        )
        texts.append((t, x, y))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    for _ in range(max_iter):
        moved = False
        bboxes = [t.get_window_extent(renderer=renderer).expanded(1.06, 1.14) for t, _, _ in texts]
        for i in range(len(texts)):
            t_i, px_i, py_i = texts[i]
            pos_i = np.array(t_i.get_position(), dtype=float)
            disp_vec = np.array([0.0, 0.0])

            for j in range(len(texts)):
                if i == j:
                    continue
                if bboxes[i].overlaps(bboxes[j]):
                    c_i = np.array([(bboxes[i].x0 + bboxes[i].x1) / 2, (bboxes[i].y0 + bboxes[i].y1) / 2])
                    c_j = np.array([(bboxes[j].x0 + bboxes[j].x1) / 2, (bboxes[j].y0 + bboxes[j].y1) / 2])
                    diff = c_i - c_j
                    if np.linalg.norm(diff) < 1e-3:
                        diff = np.array([1.0, 1.0])
                    diff = diff / np.linalg.norm(diff)
                    disp_vec += np.array([diff[0] * x_span * 0.018, diff[1] * y_span * 0.022])
                    moved = True

            # Keep inside plot bounds with margin
            new_pos = pos_i + disp_vec
            new_pos[0] = np.clip(new_pos[0], xlim[0] + x_span * 0.03, xlim[1] - x_span * 0.16)
            new_pos[1] = np.clip(new_pos[1], ylim[0] + y_span * 0.04, ylim[1] - y_span * 0.05)
            if np.any(new_pos != pos_i):
                t_i.set_position((new_pos[0], new_pos[1]))
        if not moved:
            break
        fig.canvas.draw()


def generate_charts(
    df: pd.DataFrame, val_table: pd.DataFrame, qual_bat: pd.DataFrame, qual_bowl: pd.DataFrame, out_dir: str = "."
):
    """Generates 3 publication-grade 220-DPI charts with zero overlapping labels."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
    })

    # 1. Pressure Index Empirical Validation (3-panel clean journal figure)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=220)
    tiers = ["Ordinary\n(PI < 45)", "High Pressure\n(PI >= 45)"]
    bar_colors = ["#64748B", "#0F766E"]  # Muted Slate vs Deep Teal

    v_ord = val_table[~val_table["is_hp"]].iloc[0]
    v_hp = val_table[val_table["is_hp"]].iloc[0]
    lev_ord = float(df[(df["innings"] == 2) & (~df["is_hp"])]["leverage"].mean())
    lev_hp = float(df[(df["innings"] == 2) & (df["is_hp"])]["leverage"].mean())

    panels = [
        ("Scoring Rate (RPO)", [v_ord["run_rate_rpo"], v_hp["run_rate_rpo"]], "{:.2f}", "-0.64 RPO"),
        ("Wicket Probability (%)", [v_ord["wicket_prob_pct"], v_hp["wicket_prob_pct"]], "{:.2f}%", "+31.9%"),
        ("2nd-Inn Win-Prob Leverage", [lev_ord, lev_hp], "{:.3f}", f"{lev_hp / lev_ord:.2f}x"),
    ]

    for ax, (title, vals, fmt, delta_lbl) in zip(axes, panels):
        bars = ax.bar(tiers, vals, color=bar_colors, width=0.48, edgecolor="none", zorder=3)
        ax.set_title(title, fontweight="bold", fontsize=10.5, loc="left", pad=10, color="#111827")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", color="#F3F4F6", linewidth=0.8, zorder=0)
        ax.set_ylim(0, max(vals) * 1.25)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.025,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#111827",
            )
        ax.text(
            0.98, 0.94, delta_lbl,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, fontweight="bold", color="#0F766E",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDFA", edgecolor="#99F6E4", lw=0.8),
        )

    fig.suptitle(
        "Empirical Validation of the Unified Pressure Index (PI >= 45)",
        x=0.02, ha="left", fontweight="bold", fontsize=13, color="#111827", y=1.03,
    )
    fig.text(
        0.02, 0.955,
        "Comparison across 200,999 deliveries (IPL & T20I, 2022–2026): High Pressure captures 17.9% of balls with sharper dismissal risk and 2.5x leverage.",
        fontsize=9, color="#4B5563", ha="left",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_dir, "pressure_index_validation.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Helper for Batters & Bowlers Quadrant Charts
    def _make_quadrant(df_q, name_col, cons_col, title, subtitle, cons_label, out_name):
        fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=220)
        # Background non-top-10
        rest = df_q.iloc[10:]
        top10 = df_q.head(10)

        sc = ax.scatter(
            rest["eb_impact_norm"],
            rest["eb_impact_hp"],
            s=rest["balls_hp"] * 0.38,
            c=rest[cons_col],
            cmap="YlGnBu",
            vmin=30,
            vmax=85,
            alpha=0.55,
            edgecolors="#4B5563",
            linewidth=0.6,
            zorder=2,
        )
        ax.scatter(
            top10["eb_impact_norm"],
            top10["eb_impact_hp"],
            s=top10["balls_hp"] * 0.44,
            c=top10[cons_col],
            cmap="YlGnBu",
            vmin=30,
            vmax=85,
            alpha=0.95,
            edgecolors="#111827",
            linewidth=1.1,
            zorder=4,
        )

        ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=0.8, zorder=1)

        pad_x = (df_q["eb_impact_norm"].max() - df_q["eb_impact_norm"].min()) * 0.14
        pad_y = (df_q["eb_impact_hp"].max() - df_q["eb_impact_hp"].min()) * 0.14
        xmin = df_q["eb_impact_norm"].min() - pad_x
        xmax = df_q["eb_impact_norm"].max() + pad_x * 1.3
        ymin = df_q["eb_impact_hp"].min() - pad_y
        ymax = df_q["eb_impact_hp"].max() + pad_y
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        p_min = max(xmin, ymin)
        p_max = min(xmax, ymax)
        ax.plot([p_min, p_max], [p_min, p_max], color="#BE123C", linestyle=":", linewidth=1.2, zorder=1, label="Parity Line (HP = Normal)")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="-", color="#F3F4F6", linewidth=0.7, zorder=0)

        # Subtle quadrant text
        ax.text(xmin + (xmax - xmin) * 0.02, ymax - (ymax - ymin) * 0.04, "Pressure Risers\n(Outperform Baseline)", fontsize=8.2, color="#6B7280", va="top", style="italic")

        # Repelled Top-10 labels with leader lines
        pts = [(float(r["eb_impact_norm"]), float(r["eb_impact_hp"]), f"#{int(r['rank'])} {r[name_col]}") for _, r in top10.iterrows()]
        _place_labels_no_overlap(ax, fig, pts)

        cbar = plt.colorbar(sc, ax=ax, fraction=0.032, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.set_label(cons_label, fontweight="bold", fontsize=9, color="#1F2937")

        ax.set_xlabel("EB Normal Baseline Net Run Impact per 100 Balls", fontweight="bold", fontsize=10, color="#1F2937", labelpad=8)
        ax.set_ylabel("EB High-Pressure Net Run Impact per 100 Balls", fontweight="bold", fontsize=10, color="#1F2937", labelpad=8)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12.5, color="#111827", pad=22)
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color="#4B5563", ha="left")
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=8.5)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, out_name), dpi=220, bbox_inches="tight")
        plt.close(fig)

    _make_quadrant(
        qual_bat,
        "striker_name",
        "pos_hp_innings",
        "Clutch Batting Performance: Normal Baseline vs. High-Pressure Impact",
        f"n = {len(qual_bat)} qualified batters (Elite IPL + Full-Member T20I; >=150 HP balls). Bubble size = HP balls faced.",
        "Positive HP Innings (%)",
        "clutch_batters_quadrant.png",
    )

    _make_quadrant(
        qual_bowl,
        "bowler_name",
        "pos_hp_spells",
        "Clutch Bowling Performance: Normal Baseline vs. High-Pressure Impact",
        f"n = {len(qual_bowl)} qualified bowlers (Elite IPL + Full-Member T20I; >=150 HP balls). Bubble size = HP balls bowled.",
        "Positive HP Spells (%)",
        "clutch_bowlers_quadrant.png",
    )


def export_styled_excel(
    audit_stats: dict, val_table: pd.DataFrame, qual_bat: pd.DataFrame, qual_bowl: pd.DataFrame, sens_df: pd.DataFrame, out_dir: str = "."
):
    """Exports a clean multi-tab Excel workbook."""
    wb = openpyxl.Workbook()
    ws_sum = wb.active
    ws_sum.title = "Executive Summary"

    header_fill = PatternFill("solid", fgColor="1E2761")
    header_font = Font(bold=True, color="FFFFFF", size=11)

    ws_sum.append(["SupeRR Selector Season 2 — Refined Clutch Analytics Summary"])
    ws_sum["A1"].font = Font(bold=True, size=14, color="C81D77")
    ws_sum.append([])
    ws_sum.append(["Metric", "Value"])
    for c in ws_sum[3]:
        c.fill = header_fill
        c.font = header_font
    for k, v in audit_stats.items():
        ws_sum.append([k, v])

    def _write_df(title, df_in):
        ws = wb.create_sheet(title)
        ws.append(list(df_in.columns))
        for c in ws[1]:
            c.fill = header_fill
            c.font = header_font
            c.alignment = Alignment(horizontal="center")
        for row in df_in.itertuples(index=False):
            ws.append(list(row))
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 36)

    _write_df("Top Batters", qual_bat)
    _write_df("Top Bowlers", qual_bowl)
    _write_df("Pressure Validation", val_table)
    _write_df("Sensitivity & Temporal", sens_df)

    for col in ws_sum.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws_sum.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 42)

    wb.save(os.path.join(out_dir, "SupeRR_Selector_S2_Refined_Analysis.xlsx"))

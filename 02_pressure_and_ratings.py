"""
02_pressure_and_ratings.py
--------------------------
SupeRR Selector Season 2 (Rajasthan Royals) - Step 2:
1. Fits an Empirical 2nd-Innings Win Probability Model & computes Ball Leverage.
2. Computes the Unified Pressure Index (0 to 100; High Pressure = PI >= 45).
3. Derives Empirical Wicket Run Cost (WRC) from the 20-over Run Expectancy Matrix.
4. Computes Venue- & Context-Adjusted Net Run Impact per 100 balls for Batters & Bowlers.
5. Applies Empirical Bayes Shrinkage (k = 120), Floored Uplift vs Normal Baseline,
   and Match-by-Match Consistency, then verifies names against Unstop mapping.
"""

import importlib
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

step1 = importlib.import_module("01_data_audit_and_state")


def _make_wp_features(
    sub: pd.DataFrame, runs_delta: float = 0.0, wkt_delta: int = 0, ball_delta: int = 0
) -> np.ndarray:
    b_rem = np.maximum(1.0, sub["balls_remaining_before"].values - ball_delta)
    r_need = np.maximum(0.0, sub["runs_needed_before"].values - runs_delta)
    w_hand = np.maximum(0.0, sub["wickets_in_hand_before"].values - wkt_delta)
    rrr = (r_need / b_rem) * 6.0
    par_rpo = np.where(sub["is_ipl"].values, 8.95, 8.20) * sub["venue_factor"].values
    return np.column_stack(
        [
            rrr - par_rpo,
            r_need / np.sqrt(b_rem),
            w_hand,
            np.log1p(b_rem),
            (rrr - par_rpo) * (120.0 - np.minimum(120.0, b_rem)) / 120.0,
            w_hand * np.log1p(b_rem),
        ]
    )


def compute_pressure_and_expectancy(
    df: pd.DataFrame, pi_threshold: float = 45.0
) -> tuple[pd.DataFrame, dict]:
    """Computes Win Prob Leverage, Pressure Index (0-100), and Empirical WRC."""
    df = df.copy()

    par_table = (
        df[df["innings"].isin([1, 2])]
        .groupby(["is_ipl", "innings", "overid"])
        .agg(
            par_rpo=("total_ball_runs", lambda x: x.mean() * 6.0),
            par_wkts=("cum_wkts_before", "mean"),
        )
        .reset_index()
    )
    df = df.merge(par_table, on=["is_ipl", "innings", "overid"], how="left")

    # 1. Empirical 2nd-Innings Win Probability Model
    c_mask = (df["innings"] == 2) & (df["win_team_id"].notna())
    c_df = df[c_mask].copy()
    c_df["chase_won"] = (c_df["batting_team_id"] == c_df["win_team_id"]).astype(int)

    lr = LogisticRegression(max_iter=1000)
    X_train = _make_wp_features(c_df)
    y_train = c_df["chase_won"].values
    lr.fit(X_train, y_train)
    train_probs = lr.predict_proba(X_train)[:, 1]
    wp_metrics = {
        "wp_auc": round(float(roc_auc_score(y_train, train_probs)), 4),
        "wp_brier": round(float(brier_score_loss(y_train, train_probs)), 4),
    }

    inn2_mask = df["innings"] == 2
    inn2 = df[inn2_mask]
    wp_curr = lr.predict_proba(_make_wp_features(inn2))[:, 1]
    wp_four = lr.predict_proba(
        _make_wp_features(inn2, runs_delta=4.0, wkt_delta=0, ball_delta=1)
    )[:, 1]
    wp_wkt = np.where(
        inn2["wickets_in_hand_before"].values <= 1,
        0.0,
        lr.predict_proba(
            _make_wp_features(inn2, runs_delta=0.0, wkt_delta=1, ball_delta=1)
        )[:, 1],
    )
    leverage_inn2 = np.maximum(0.0, wp_four - wp_wkt)

    df["win_prob"] = np.nan
    df.loc[inn2_mask, "win_prob"] = wp_curr
    df["leverage"] = 0.0
    df.loc[inn2_mask, "leverage"] = leverage_inn2

    # 2. Unified Pressure Index (0-100)
    par_rpo_v = np.where(df["is_ipl"], 8.95, 8.20) * df["venue_factor"]
    live_chase = df["win_prob"].between(0.04, 0.96)
    lev_pts_2 = np.where(live_chase, np.clip(df["leverage"] / 0.25 * 48.0, 0, 50), 2.0)
    rrr_pts_2 = np.where(
        live_chase, np.clip((df["rrr_before"] - (par_rpo_v - 0.5)) * 3.5, 0, 25), 2.0
    )

    excess_wkts = np.maximum(0, df["cum_wkts_before"] - df["par_wkts"])
    wkt_pts_2 = np.clip(excess_wkts * 4.0 + df["wkt_last_12b"] * 4.5, 0, 15)

    wkt_pts_1 = np.clip(
        excess_wkts * 8.0
        + df["wkt_last_12b"] * 7.5
        + np.where(df["cum_wkts_before"] >= 5, (df["cum_wkts_before"] - 4) * 3.0, 0),
        0,
        38,
    )
    deficit_pts_1 = np.where(
        df["legal_balls_before"] >= 12,
        np.clip((par_rpo_v + 0.5 - df["crr_before"]) * 5.5, 0, 25),
        6.0,
    ) + np.where(
        (df["overid"] >= 16)
        & (df["cum_wkts_before"].between(2, 7))
        & (df["crr_before"].between(6.5, 11.0)),
        14.0,
        0.0,
    )
    stage_pts_1 = np.where(
        df["overid"] >= 17,
        15.0,
        np.where(df["overid"] >= 13, 8.0, np.where(df["overid"] <= 6, 5.0, 3.0)),
    )

    stakes_pts = np.where(
        df["knock_out"] == 1,
        10.0,
        np.where(~df["stage"].isin(["League", "Group"]), 6.0, 0.0),
    )

    pi_inn2 = lev_pts_2 + rrr_pts_2 + wkt_pts_2 + stakes_pts
    pi_inn1 = wkt_pts_1 + deficit_pts_1 + stage_pts_1 + stakes_pts

    df["pressure_index"] = np.where(
        df["innings"] >= 3,
        100.0,
        np.clip(np.where(df["innings"] == 2, pi_inn2, pi_inn1), 0, 100),
    )
    df["is_hp"] = df["pressure_index"] >= pi_threshold

    # 3. Empirical Wicket Run Cost from 20-over Run Expectancy Matrix
    inn_tot = df.groupby(["match_id", "innings"])["total_ball_runs"].transform("sum")
    df["runs_remaining"] = inn_tot - df["cum_runs_before"]
    full_inn = df[(df["innings"].isin([1, 2])) & (df["max_balls"] == 120)]
    re_w = (
        full_inn.groupby(["overid", "wkt_tier"], observed=True)["runs_remaining"]
        .mean()
        .unstack()
    )
    tiers = ["0-1", "2-3", "4-5", "6+"]
    wrc_map = {}
    for ov in range(1, 21):
        for i, t in enumerate(tiers):
            if i < len(tiers) - 1:
                diff = (re_w.loc[ov, tiers[i]] - re_w.loc[ov, tiers[i + 1]]) / 2.0
            else:
                diff = re_w.loc[ov, tiers[i]] * 0.18
            wrc_map[(ov, t)] = float(
                np.clip(diff if pd.notna(diff) else 10.0, 3.5, 22.0)
            )
    df["wkt_run_cost"] = [
        wrc_map.get((int(ov), str(wt)), 10.0)
        for ov, wt in zip(df["overid"], df["wkt_tier"])
    ]
    return df, wp_metrics


def evaluate_batters(
    df: pd.DataFrame,
    elite_only: bool = True,
    min_hp_balls: int = 150,
    min_norm_balls: int = 300,
    min_qual_matches: int = 12,
    k_shrink: float = 120.0,
) -> pd.DataFrame:
    """Ranks qualified batters on EB HP Impact, Baseline Uplift, and Match Consistency."""
    sub = df[df["is_elite"]].copy() if elite_only else df.copy()
    bdf = sub[sub["is_batter_ball"]].copy()

    b_ctx = (
        bdf.groupby(["is_ipl", "innings_grp", "overid", "wkt_tier"], observed=True)
        .agg(
            exp_bat_runs=("runs", "mean"),
            exp_bat_out=("is_striker_out", "mean"),
        )
        .reset_index()
    )
    bdf = bdf.merge(b_ctx, on=["is_ipl", "innings_grp", "overid", "wkt_tier"], how="left")
    bdf["exp_bat_runs_adj"] = bdf["exp_bat_runs"] * bdf["venue_factor"]
    bdf["net_run_impact"] = (bdf["runs"] - bdf["exp_bat_runs_adj"]) - bdf[
        "wkt_run_cost"
    ] * (bdf["is_striker_out"].astype(float) - bdf["exp_bat_out"])
    bdf["is_bnd"] = bdf["is_four"] | bdf["is_six"]
    bdf["is_dot"] = bdf["runs"] == 0

    bsum = (
        bdf.groupby(["striker_id", "striker_name", "is_hp"])
        .agg(
            balls=("runs", "count"),
            innings_cnt=("match_id", "nunique"),
            runs=("runs", "sum"),
            exp_runs=("exp_bat_runs_adj", "sum"),
            outs=("is_striker_out", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            bnds=("is_bnd", "sum"),
            dots=("is_dot", "sum"),
            net_impact_sum=("net_run_impact", "sum"),
        )
        .unstack(fill_value=0)
    )
    bsum.columns = [f"{col}_{'hp' if hp else 'norm'}" for col, hp in bsum.columns]
    bsum = bsum.reset_index()

    qual = bsum[
        (bsum["balls_hp"] >= min_hp_balls) & (bsum["balls_norm"] >= min_norm_balls)
    ].copy()

    qual["sr_norm"] = (qual["runs_norm"] / qual["balls_norm"]) * 100.0
    qual["sr_hp"] = (qual["runs_hp"] / qual["balls_hp"]) * 100.0
    qual["sr_delta"] = qual["sr_hp"] - qual["sr_norm"]

    qual["avg_norm"] = qual["runs_norm"] / np.maximum(1, qual["outs_norm"])
    qual["avg_hp"] = qual["runs_hp"] / np.maximum(1, qual["outs_hp"])
    qual["avg_delta"] = qual["avg_hp"] - qual["avg_norm"]

    qual["bnd_pct_norm"] = (qual["bnds_norm"] / qual["balls_norm"]) * 100.0
    qual["bnd_pct_hp"] = (qual["bnds_hp"] / qual["balls_hp"]) * 100.0
    qual["dot_pct_norm"] = (qual["dots_norm"] / qual["balls_norm"]) * 100.0
    qual["dot_pct_hp"] = (qual["dots_hp"] / qual["balls_hp"]) * 100.0

    qual["impact_100_norm"] = (qual["net_impact_sum_norm"] / qual["balls_norm"]) * 100.0
    qual["impact_100_hp"] = (qual["net_impact_sum_hp"] / qual["balls_hp"]) * 100.0

    pop_hp = (qual["net_impact_sum_hp"].sum() / qual["balls_hp"].sum()) * 100.0
    pop_norm = (qual["net_impact_sum_norm"].sum() / qual["balls_norm"].sum()) * 100.0
    qual["eb_impact_hp"] = (qual["balls_hp"] * qual["impact_100_hp"] + k_shrink * pop_hp) / (
        qual["balls_hp"] + k_shrink
    )
    qual["eb_impact_norm"] = (
        qual["balls_norm"] * qual["impact_100_norm"] + k_shrink * pop_norm
    ) / (qual["balls_norm"] + k_shrink)
    qual["eb_delta"] = qual["eb_impact_hp"] - qual["eb_impact_norm"]
    qual["uplift_vs_par"] = qual["eb_impact_hp"] - np.maximum(0.0, qual["eb_impact_norm"])

    hp_inn = (
        bdf[bdf["is_hp"]]
        .groupby(["striker_id", "match_id"])
        .agg(hp_balls=("runs", "count"), hp_impact=("net_run_impact", "sum"))
        .reset_index()
    )
    cons = (
        hp_inn[hp_inn["hp_balls"] >= 5]
        .groupby("striker_id")
        .agg(
            qual_hp_innings=("match_id", "count"),
            pos_hp_innings=("hp_impact", lambda x: (x > 0).mean() * 100.0),
        )
        .reset_index()
    )
    qual = qual.merge(cons, on="striker_id", how="left")
    qual = qual[qual["qual_hp_innings"] >= min_qual_matches].copy()

    for c in ["eb_impact_hp", "uplift_vs_par", "pos_hp_innings"]:
        qual[f"z_{c}"] = (qual[c] - qual[c].mean()) / qual[c].std()

    qual["clutch_bat_score"] = 50.0 + 15.0 * (
        0.60 * qual["z_eb_impact_hp"]
        + 0.15 * qual["z_uplift_vs_par"]
        + 0.25 * qual["z_pos_hp_innings"]
    )
    qual = qual.sort_values("clutch_bat_score", ascending=False).reset_index(drop=True)
    qual.insert(0, "rank", np.arange(1, len(qual) + 1))
    return qual


def evaluate_bowlers(
    df: pd.DataFrame,
    elite_only: bool = True,
    min_hp_balls: int = 150,
    min_norm_balls: int = 300,
    min_qual_matches: int = 12,
    k_shrink: float = 120.0,
) -> pd.DataFrame:
    """Ranks qualified bowlers on EB HP Impact, Baseline Uplift, and Spell Consistency."""
    sub = df[df["is_elite"]].copy() if elite_only else df.copy()
    wdf = sub[sub["extras_type"] != "Penalty"].copy()

    w_ctx = (
        wdf.groupby(["is_ipl", "innings_grp", "overid", "wkt_tier"], observed=True)
        .agg(
            tot_bowler_runs=("bowler_runs", "sum"),
            tot_legal_balls=("is_bowler_ball", "sum"),
            tot_bowler_wkts=("is_bowler_wicket", "sum"),
        )
        .reset_index()
    )
    w_ctx["exp_bowl_rpb"] = w_ctx["tot_bowler_runs"] / np.maximum(
        1, w_ctx["tot_legal_balls"]
    )
    w_ctx["exp_bowl_wpb"] = w_ctx["tot_bowler_wkts"] / np.maximum(
        1, w_ctx["tot_legal_balls"]
    )

    wdf = wdf.merge(
        w_ctx[
            ["is_ipl", "innings_grp", "overid", "wkt_tier", "exp_bowl_rpb", "exp_bowl_wpb"]
        ],
        on=["is_ipl", "innings_grp", "overid", "wkt_tier"],
        how="left",
    )
    wdf["exp_runs_ball"] = (
        wdf["is_bowler_ball"].astype(float) * wdf["exp_bowl_rpb"] * wdf["venue_factor"]
    )
    wdf["exp_wkts_ball"] = wdf["is_bowler_ball"].astype(float) * wdf["exp_bowl_wpb"]
    wdf["net_bowl_impact"] = (wdf["exp_runs_ball"] - wdf["bowler_runs"]) + wdf[
        "wkt_run_cost"
    ] * (wdf["is_bowler_wicket"].astype(float) - wdf["exp_wkts_ball"])
    wdf["is_dot_bowl"] = wdf["is_bowler_ball"] & (wdf["bowler_runs"] == 0)
    wdf["is_bnd_bowl"] = wdf["is_four"] | wdf["is_six"]

    wsum = (
        wdf.groupby(["bowler_id", "bowler_name", "is_hp"])
        .agg(
            balls=("is_bowler_ball", "sum"),
            matches_cnt=("match_id", "nunique"),
            runs=("bowler_runs", "sum"),
            exp_runs=("exp_runs_ball", "sum"),
            wkts=("is_bowler_wicket", "sum"),
            exp_wkts=("exp_wkts_ball", "sum"),
            dots=("is_dot_bowl", "sum"),
            bnds=("is_bnd_bowl", "sum"),
            net_impact_sum=("net_bowl_impact", "sum"),
        )
        .unstack(fill_value=0)
    )
    wsum.columns = [f"{col}_{'hp' if hp else 'norm'}" for col, hp in wsum.columns]
    wsum = wsum.reset_index()

    qual = wsum[
        (wsum["balls_hp"] >= min_hp_balls) & (wsum["balls_norm"] >= min_norm_balls)
    ].copy()

    qual["econ_norm"] = (qual["runs_norm"] / qual["balls_norm"]) * 6.0
    qual["econ_hp"] = (qual["runs_hp"] / qual["balls_hp"]) * 6.0
    qual["econ_delta"] = qual["econ_hp"] - qual["econ_norm"]

    qual["exp_econ_hp"] = (qual["exp_runs_hp"] / qual["balls_hp"]) * 6.0
    qual["econ_saved_hp"] = qual["exp_econ_hp"] - qual["econ_hp"]

    qual["sr_norm"] = np.where(
        qual["wkts_norm"] > 0, qual["balls_norm"] / qual["wkts_norm"], np.nan
    )
    qual["sr_hp"] = np.where(
        qual["wkts_hp"] > 0, qual["balls_hp"] / qual["wkts_hp"], np.nan
    )

    qual["dot_pct_norm"] = (qual["dots_norm"] / qual["balls_norm"]) * 100.0
    qual["dot_pct_hp"] = (qual["dots_hp"] / qual["balls_hp"]) * 100.0

    qual["impact_100_norm"] = (qual["net_impact_sum_norm"] / qual["balls_norm"]) * 100.0
    qual["impact_100_hp"] = (qual["net_impact_sum_hp"] / qual["balls_hp"]) * 100.0

    pop_hp = (qual["net_impact_sum_hp"].sum() / qual["balls_hp"].sum()) * 100.0
    pop_norm = (qual["net_impact_sum_norm"].sum() / qual["balls_norm"].sum()) * 100.0
    qual["eb_impact_hp"] = (qual["balls_hp"] * qual["impact_100_hp"] + k_shrink * pop_hp) / (
        qual["balls_hp"] + k_shrink
    )
    qual["eb_impact_norm"] = (
        qual["balls_norm"] * qual["impact_100_norm"] + k_shrink * pop_norm
    ) / (qual["balls_norm"] + k_shrink)
    qual["eb_delta"] = qual["eb_impact_hp"] - qual["eb_impact_norm"]
    qual["uplift_vs_par"] = qual["eb_impact_hp"] - np.maximum(0.0, qual["eb_impact_norm"])

    hp_spell = (
        wdf[wdf["is_hp"]]
        .groupby(["bowler_id", "match_id"])
        .agg(hp_balls=("is_bowler_ball", "sum"), hp_impact=("net_bowl_impact", "sum"))
        .reset_index()
    )
    cons = (
        hp_spell[hp_spell["hp_balls"] >= 5]
        .groupby("bowler_id")
        .agg(
            qual_hp_spells=("match_id", "count"),
            pos_hp_spells=("hp_impact", lambda x: (x > 0).mean() * 100.0),
        )
        .reset_index()
    )
    qual = qual.merge(cons, on="bowler_id", how="left")
    qual = qual[qual["qual_hp_spells"] >= min_qual_matches].copy()

    for c in ["eb_impact_hp", "uplift_vs_par", "pos_hp_spells"]:
        qual[f"z_{c}"] = (qual[c] - qual[c].mean()) / qual[c].std()

    qual["clutch_bowl_score"] = 50.0 + 15.0 * (
        0.60 * qual["z_eb_impact_hp"]
        + 0.15 * qual["z_uplift_vs_par"]
        + 0.25 * qual["z_pos_hp_spells"]
    )
    qual = qual.sort_values("clutch_bowl_score", ascending=False).reset_index(drop=True)
    qual.insert(0, "rank", np.arange(1, len(qual) + 1))
    return qual


def run_ratings_pipeline(
    data_dir: str = r"C:\Users\Asus\Downloads\6af458274c1e8_data_pack\Data Pack",
    out_dir: str = ".",
):
    df_raw, audit_stats = step1.load_and_reconstruct_state(data_dir)
    df, wp_metrics = compute_pressure_and_expectancy(df_raw, pi_threshold=45.0)
    audit_stats.update(wp_metrics)

    val_table = (
        df.groupby("is_hp")
        .agg(
            total_balls=("match_id", "count"),
            run_rate_rpo=("total_ball_runs", lambda x: x.mean() * 6.0),
            wicket_prob_pct=("is_wicket", lambda x: x.mean() * 100.0),
            dot_ball_pct=("total_ball_runs", lambda x: (x == 0).mean() * 100.0),
            boundary_pct=(
                "is_four",
                lambda x: (df.loc[x.index, "is_four"] | df.loc[x.index, "is_six"]).mean()
                * 100.0,
            ),
            mean_wp_leverage=("leverage", "mean"),
        )
        .reset_index()
    )
    val_table["context_tier"] = np.where(
        val_table["is_hp"], "High Pressure (PI >= 45)", "Ordinary (PI < 45)"
    )
    val_table["share_of_dataset_pct"] = (
        val_table["total_balls"] / val_table["total_balls"].sum()
    ) * 100.0

    qual_bat = evaluate_batters(df)
    qual_bowl = evaluate_bowlers(df)

    mapping = pd.read_csv(os.path.join(data_dir, "competition_team_player_mapping.csv"))
    id_to_map_name = (
        mapping.groupby("player_id")["player_name"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    qual_bat["unstop_dropdown_name"] = qual_bat["striker_id"].map(id_to_map_name)
    qual_bowl["unstop_dropdown_name"] = qual_bowl["bowler_id"].map(id_to_map_name)

    os.makedirs(out_dir, exist_ok=True)
    qual_bat.to_csv(os.path.join(out_dir, "top_batters_under_pressure.csv"), index=False)
    qual_bowl.to_csv(os.path.join(out_dir, "top_bowlers_under_pressure.csv"), index=False)
    val_table.to_csv(os.path.join(out_dir, "pressure_validation_summary.csv"), index=False)

    return df, audit_stats, val_table, qual_bat, qual_bowl

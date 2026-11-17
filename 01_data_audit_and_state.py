"""
01_data_audit_and_state.py
--------------------------
SupeRR Selector Season 2 (Rajasthan Royals) - Step 1:
Audits the raw CSV files, fixes the `extras1_runs` and `revised_overs` (balls)
quirks, computes shrunk venue scoring factors, flags Elite (IPL + Full-Member T20I)
matches, and reconstructs exact pre-ball scoreboard states for every delivery.
"""

import os
import numpy as np
import pandas as pd

FULL_MEMBERS = {
    "India", "Australia", "England", "South Africa", "New Zealand", "Pakistan",
    "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan", "Zimbabwe", "Ireland",
}

IPL_TEAMS = {
    "Rajasthan Royals", "Mumbai Indians", "Chennai Super Kings",
    "Royal Challengers Bengaluru", "Kolkata Knight Riders", "Sunrisers Hyderabad",
    "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants",
}

ELITE_TEAMS = FULL_MEMBERS | IPL_TEAMS


def load_and_reconstruct_state(data_dir: str = ".") -> tuple[pd.DataFrame, dict]:
    """Loads raw CSVs, fixes extras/DLS quirks, and builds pre-ball match states."""
    bbb = pd.read_csv(os.path.join(data_dir, "ball_by_ball.csv"))
    matches = pd.read_csv(os.path.join(data_dir, "matches.csv"))
    comps = pd.read_csv(os.path.join(data_dir, "competitions.csv"))

    df = bbb.merge(
        matches[
            [
                "match_id", "comp_id", "match_date", "stage", "venue_id",
                "knock_out", "home_team", "away_team",
                "first_inn_revised_overs", "second_inn_revised_overs",
                "revised_target", "win_team_id", "result",
            ]
        ],
        on="match_id",
        how="left",
    )
    df = df.merge(
        comps[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left",
    )

    # Sort strictly chronologically within each match and innings
    df = df.sort_values(
        ["match_id", "innings", "innings_ball_num", "overid", "over_ball_num"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Data Audit Fix 1: `extras1_runs` records actual primary extra runs (e.g.,
    # 1-5 for wides/byes/legbyes, 1 for Overstep, 5 for Penalty). `extras_2_runs`
    # records secondary extra runs (e.g., byes/legbyes off a no-ball).
    # -------------------------------------------------------------------------
    df["total_ball_runs"] = (
        df["runs"] + df["extras1_runs"].fillna(0) + df["extras_2_runs"].fillna(0)
    )

    # Legal deliveries for over progression & bowler balls
    df["is_legal_ball"] = ~df["extras_type"].isin(
        ["wide", "Overstep", "other", "Penalty"]
    )
    # Batter balls faced (No-balls, byes, legbyes count; wides & Penalty do not)
    df["is_batter_ball"] = ~df["extras_type"].isin(["wide", "Penalty"])

    # Bowler runs conceded (wides, Overstep, other charged; byes/legbyes/Penalty excluded)
    df["bowler_runs"] = df["runs"] + np.where(
        df["extras_type"].isin(["wide", "Overstep", "other"]),
        df["extras1_runs"].fillna(0),
        0,
    )
    df["is_bowler_ball"] = df["is_legal_ball"]

    # Bowler-credited dismissals (exclude RunOut, Retire Out, Obstructing the field)
    bowler_dismissals = ["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"]
    df["is_bowler_wicket"] = df["is_wicket"] & df["dismissal_type"].isin(
        bowler_dismissals
    )
    df["is_striker_out"] = df["is_wicket"] & (
        df["dismissed_player_id"] == df["striker_id"]
    )

    # Rolling cumulative state BEFORE each delivery is bowled
    df["cum_runs_after"] = df.groupby(["match_id", "innings"])["total_ball_runs"].cumsum()
    df["cum_runs_before"] = df["cum_runs_after"] - df["total_ball_runs"]

    df["cum_wkts_after"] = df.groupby(["match_id", "innings"])["is_wicket"].cumsum()
    df["cum_wkts_before"] = df["cum_wkts_after"] - df["is_wicket"].astype(int)

    df["legal_balls_after"] = df.groupby(["match_id", "innings"])["is_legal_ball"].cumsum()
    df["legal_balls_before"] = df["legal_balls_after"] - df["is_legal_ball"].astype(int)

    # -------------------------------------------------------------------------
    # Data Audit Fix 2: `first_inn_revised_overs` and `second_inn_revised_overs`
    # in matches.csv are stored in BALLS (e.g., 90, 48, 114), NOT overs.
    # -------------------------------------------------------------------------
    df["max_balls"] = np.where(
        df["innings"] == 1,
        df["first_inn_revised_overs"].fillna(120),
        np.where(
            df["innings"] == 2,
            df["second_inn_revised_overs"].fillna(120),
            6,
        ),
    )
    df["balls_remaining_before"] = np.maximum(
        1, df["max_balls"] - df["legal_balls_before"]
    )
    df["wickets_in_hand_before"] = np.maximum(1, 10 - df["cum_wkts_before"])

    # 1st Innings target & 2nd Innings Required Run Rate (RRR)
    inn1_totals = (
        df[df["innings"] == 1]
        .groupby("match_id")["total_ball_runs"]
        .sum()
        .rename("inn1_total")
    )
    df = df.merge(inn1_totals, on="match_id", how="left")

    df["target_runs"] = np.where(
        df["innings"] == 2, df["revised_target"].fillna(df["inn1_total"] + 1), np.nan
    )
    df["runs_needed_before"] = np.where(
        df["innings"] == 2,
        np.maximum(0, df["target_runs"] - df["cum_runs_before"]),
        np.nan,
    )
    df["rrr_before"] = np.where(
        df["innings"] == 2,
        (df["runs_needed_before"] / df["balls_remaining_before"]) * 6.0,
        np.nan,
    )
    df["crr_before"] = np.where(
        df["legal_balls_before"] > 0,
        (df["cum_runs_before"] / df["legal_balls_before"]) * 6.0,
        8.0,
    )

    df["is_ipl"] = df["comp_name"].str.contains("IPL", na=False)
    df["is_elite"] = df["home_team"].isin(ELITE_TEAMS) & df["away_team"].isin(ELITE_TEAMS)
    df["innings_grp"] = np.where(df["innings"] == 1, 1, 2)
    df["phase"] = pd.cut(
        df["overid"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"],
    )

    # Shrunk Venue Scoring Factor (normalizes flat vs sluggish pitches)
    inn_rpb = (
        df[df["innings"].isin([1, 2])]
        .groupby(["match_id", "innings", "venue_id", "is_ipl"])
        .agg(r=("total_ball_runs", "sum"), b=("is_legal_ball", "sum"))
        .reset_index()
    )
    inn_rpb["rpb"] = inn_rpb["r"] / np.maximum(1, inn_rpb["b"])
    inn_rpb["fmt_rpb"] = inn_rpb.groupby("is_ipl")["rpb"].transform("mean")
    inn_rpb["rel_rpb"] = inn_rpb["rpb"] / inn_rpb["fmt_rpb"]
    v_stats = (
        inn_rpb.groupby("venue_id")
        .agg(n=("rel_rpb", "count"), mean_rel=("rel_rpb", "mean"))
        .reset_index()
    )
    v_stats["venue_factor"] = (v_stats["n"] * v_stats["mean_rel"] + 10.0 * 1.0) / (
        v_stats["n"] + 10.0
    )
    df = df.merge(v_stats[["venue_id", "venue_factor"]], on="venue_id", how="left")
    df["venue_factor"] = df["venue_factor"].fillna(1.0)

    # Recent wicket shock (wickets fallen in the preceding 12 deliveries)
    df["wkt_last_12b"] = df.groupby(["match_id", "innings"])["is_wicket"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).sum().fillna(0)
    )
    df["wkt_tier"] = pd.cut(
        df["cum_wkts_before"], bins=[-1, 1, 3, 5, 10], labels=["0-1", "2-3", "4-5", "6+"]
    )

    audit_stats = {
        "total_matches_metadata": int(len(matches)),
        "total_matches_bbb": int(df["match_id"].nunique()),
        "total_deliveries": int(len(df)),
        "elite_matches": int(df[df["is_elite"]]["match_id"].nunique()),
        "elite_balls": int(df["is_elite"].sum()),
        "ipl_matches": int(df[df["is_ipl"]]["match_id"].nunique()),
        "ipl_balls": int(df["is_ipl"].sum()),
        "t20i_matches": int(df[df["match_format"] == "T20I"]["match_id"].nunique()),
        "t20i_balls": int((df["match_format"] == "T20I").sum()),
    }
    return df, audit_stats

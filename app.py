import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
import streamlit as st

st.set_page_config(page_title="ORE-CAST v1 | Autonomous EPL Predictor", layout="wide", page_icon="⚽")

# Canonical club name mapping for cross-API consistency
CLUB_NAME_MAP = {
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Brighton": "Brighton & Hove Albion",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Bournemouth": "AFC Bournemouth",
    "Leeds": "Leeds United",
    "Hull": "Hull City",
    "Coventry": "Coventry City",
    "Sunderland": "Sunderland AFC",
    "Ipswich": "Ipswich Town",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Leicester": "Leicester City"
}

def clean_team_name(name: str) -> str:
    cleaned = name.replace(" FC", "").replace(" AFC", "").strip()
    return CLUB_NAME_MAP.get(cleaned, cleaned)

# ==============================================================================
# 1. LIVE PREMIER LEAGUE FIXTURES & RESULTS (ZERO LOCAL TXT FILES)
# ==============================================================================
@st.cache_data(ttl=600)  # Re-queries every 10 minutes for live scores
def fetch_live_epl_fixtures():
    """
    Pulls the full 380-match schedule and live match results directly
    from the official open Premier League API endpoints.
    """
    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"

    try:
        b_res = requests.get(bootstrap_url, timeout=7).json()
        f_res = requests.get(fixtures_url, timeout=7).json()

        team_dict = {t["id"]: clean_team_name(t["name"]) for t in b_res["teams"]}

        played, upcoming = [], []
        for f in f_res:
            event = f.get("event")
            if not event:
                continue

            h_team = team_dict.get(f["team_h"], f"Team {f['team_h']}")
            a_team = team_dict.get(f["team_a"], f"Team {f['team_a']}")
            kickoff = f.get("kickoff_time", "TBD")
            date_str = kickoff[:10] if kickoff != "TBD" else "TBD"
            time_str = kickoff[11:16] if kickoff != "TBD" else "TBD"

            if f.get("finished", False):
                played.append({
                    "matchday": int(event),
                    "date": date_str,
                    "time": time_str,
                    "home_team": h_team,
                    "away_team": a_team,
                    "home_goals": int(f.get("team_h_score", 0)),
                    "away_goals": int(f.get("team_a_score", 0))
                })
            else:
                upcoming.append({
                    "matchday": int(event),
                    "date": date_str,
                    "time": time_str,
                    "home_team": h_team,
                    "away_team": a_team
                })

        df_played = pd.DataFrame(played)
        df_upcoming = pd.DataFrame(upcoming)

        # Autonomous gameweek detection: lowest matchday containing unfinished fixtures
        if not df_upcoming.empty:
            active_md = int(df_upcoming["matchday"].min())
        else:
            active_md = int(df_played["matchday"].max()) if not df_played.empty else 1

        return df_played, df_upcoming, active_md

    except Exception:
        return pd.DataFrame(), pd.DataFrame(), 1

# ==============================================================================
# 2. HISTORICAL H2H DATA LOADER (PRIOR 3 SEASONS)
# ==============================================================================
@st.cache_data(ttl=86400)
def fetch_historical_fixtures():
    """
    Retrieves previous complete Premier League season results to populate
    the 10-match H2H matrix for teams that have met fewer than 10 times this season.
    """
    season_codes = ["2324", "2425", "2526"]
    frames = []

    for s in season_codes:
        url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
        try:
            raw = pd.read_csv(url, usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            raw["HomeTeam"] = raw["HomeTeam"].apply(clean_team_name)
            raw["AwayTeam"] = raw["AwayTeam"].apply(clean_team_name)
            raw.rename(columns={
                "HomeTeam": "home_team", "AwayTeam": "away_team",
                "FTHG": "home_goals", "FTAG": "away_goals", "Date": "date"
            }, inplace=True)
            frames.append(raw.dropna())
        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals"])

# ==============================================================================
# 3. LIVE AUTOMATED INJURY & RATING FEED
# ==============================================================================
@st.cache_data(ttl=1800)
def fetch_live_injuries():
    """
    Pulls active injuries, suspensions, and player form ratings from the Premier League API.
    Zero human input required.
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        res = requests.get(url, timeout=7).json()
        teams = {t["id"]: clean_team_name(t["name"]) for t in res["teams"]}
        players = res["elements"]

        injury_records = []
        for p in players:
            if p["status"] in ["i", "d", "s"]:
                team_name = teams.get(p["team"], "Unknown")
                rating = float(p.get("form", 5.0))
                mins = p.get("minutes", 450)
                injury_records.append({
                    "team": team_name,
                    "player": f"{p['first_name']} {p['second_name']}",
                    "status": "Injured" if p["status"] == "i" else "Doubtful",
                    "news": p.get("news", "Sidelined"),
                    "rating": max(6.0, rating),
                    "minutes": mins
                })
        return pd.DataFrame(injury_records)
    except Exception:
        return pd.DataFrame(columns=["team", "player", "status", "news", "rating", "minutes"])

def compute_team_deficit(team_name, injuries_df):
    if injuries_df.empty:
        return 0.0, []
    team_absences = injuries_df[injuries_df["team"] == team_name]
    if team_absences.empty:
        return 0.0, []

    total_deficit = 0.0
    absences = []
    for _, r in team_absences.iterrows():
        min_weight = min(1.0, r["minutes"] / 450.0)
        quality = (r["rating"] / 10.0) * 0.12
        impact = min_weight * quality
        total_deficit += impact
        absences.append(f"{r['player']} ({r['news']}) — Impact: -{impact * 100:.1f}%")

    return min(0.35, total_deficit), absences

# ==============================================================================
# 4. STRICT 10-MATCH H2H & 5-MATCH FORM ENGINES
# ==============================================================================
def get_h2h_data(h_team, a_team, combined_played_df):
    matches = combined_played_df[
        ((combined_played_df["home_team"] == h_team) & (combined_played_df["away_team"] == a_team)) |
        ((combined_played_df["home_team"] == a_team) & (combined_played_df["away_team"] == h_team))
    ].tail(10)  # Strict limit: previous 10 meetings only

    n = len(matches)
    if n == 0:
        return {"h_wins": 0, "draws": 0, "a_wins": 0, "ppg": 1.35, "encounters": 0}

    h_wins, draws, a_wins = 0, 0, 0
    points = []
    weights = np.linspace(0.65, 1.0, n)

    for _, r in matches.iterrows():
        if r["home_goals"] == r["away_goals"]:
            draws += 1
            pts = 1
        elif r["home_team"] == h_team:
            if r["home_goals"] > r["away_goals"]:
                h_wins += 1
                pts = 3
            else:
                a_wins += 1
                pts = 0
        else:
            if r["away_goals"] > r["home_goals"]:
                h_wins += 1
                pts = 3
            else:
                a_wins += 1
                pts = 0
        points.append(pts)

    return {
        "h_wins": h_wins, "draws": draws, "a_wins": a_wins,
        "ppg": float(np.average(points, weights=weights)), "encounters": n
    }

def get_form_data(team, played_df):
    matches = played_df[(played_df["home_team"] == team) | (played_df["away_team"] == team)].tail(5)
    n = len(matches)
    if n == 0:
        return {"ppg": 1.35, "gf": 1.3, "ga": 1.3, "gd": 0.0, "pld": 0}

    pts, gf, ga = 0, 0, 0
    for _, r in matches.iterrows():
        f = r["home_goals"] if r["home_team"] == team else r["away_goals"]
        a = r["away_goals"] if r["home_team"] == team else r["home_goals"]
        gf += f
        ga += a
        if f > a: pts += 3
        elif f == a: pts += 1

    return {"ppg": pts / n, "gf": gf / n, "ga": ga / n, "gd": (gf - ga) / n, "pld": n}

# ==============================================================================
# 5. POISSON PREDICTION MODEL
# ==============================================================================
def project_match(h_team, a_team, played_df, combined_history_df, injuries_df):
    h_form = get_form_data(h_team, played_df)
    a_form = get_form_data(a_team, played_df)
    h2h = get_h2h_data(h_team, a_team, combined_history_df)

    h_def, h_absences = compute_team_deficit(h_team, injuries_df)
    a_def, a_absences = compute_team_deficit(a_team, injuries_df)

    base_h, base_a = 1.45, 1.15

    h_att = max(0.5, h_form["gf"] / base_h)
    a_vuln = max(0.5, a_form["ga"] / base_h)
    a_att = max(0.5, a_form["gf"] / base_a)
    h_vuln = max(0.5, h_form["ga"] / base_a)

    h2h_skew = (h2h["ppg"] - 1.35) * 0.12

    lambda_h = max(0.3, (base_h * h_att * a_vuln + h2h_skew) * (1.0 - h_def))
    lambda_a = max(0.3, (base_a * a_att * h_vuln - h2h_skew) * (1.0 - a_def))

    max_g = 8
    h_pmf = [poisson.pmf(i, lambda_h) for i in range(max_g)]
    a_pmf = [poisson.pmf(j, lambda_a) for j in range(max_g)]
    mat = np.outer(h_pmf, a_pmf)

    p_h = float(np.sum(np.tril(mat, -1)))
    p_d = float(np.sum(np.diag(mat)))
    p_a = float(np.sum(np.triu(mat, 1)))

    odds_h = round(1.0 / p_h, 2) if p_h > 0 else 99.0
    odds_d = round(1.0 / p_d, 2) if p_d > 0 else 99.0
    odds_a = round(1.0 / p_a, 2) if p_a > 0 else 99.0

    score = np.unravel_index(np.argmax(mat), mat.shape)

    return {
        "p_h": p_h, "p_d": p_d, "p_a": p_a,
        "odds_h": odds_h, "odds_d": odds_d, "odds_a": odds_a,
        "score": f"{score[0]} - {score[1]}",
        "lambda_h": lambda_h, "lambda_a": lambda_a,
        "h_def": h_def, "a_def": a_def,
        "h_abs": h_absences, "a_abs": a_absences,
        "h2h": h2h, "h_form": h_form, "a_form": a_form
    }

# ==============================================================================
# 6. STREAMLIT APPLICATION DASHBOARD
# ==============================================================================
df_played_current, df_upcoming_current, active_gameweek = fetch_live_epl_fixtures()
df_past_history = fetch_historical_fixtures()
df_injuries = fetch_live_injuries()

# Build unified matches pool for H2H calculations
combined_history = pd.concat([
    df_past_history[["home_team", "away_team", "home_goals", "away_goals"]],
    df_played_current[["home_team", "away_team", "home_goals", "away_goals"]]
], ignore_index=True)

st.title("⚽ ORE-CAST v1 | Premier League Match Engine")
st.caption(f"Fully Autonomous Cloud Ingestion | Live on **Matchday {active_gameweek}**")

tab_preds, tab_inj, tab_h2h = st.tabs([
    "🔮 Active Matchday Predictions",
    "🏥 Live Injury & Absence Deficits",
    "📜 H2H (Last 10) & Form (Last 5) Inspector"
])

# TAB 1: ACTIVE MATCHDAY PREDICTIONS
with tab_preds:
    st.subheader(f"Gameweek {active_gameweek} Probabilities & Fair Market Odds")
    st.info(f"🔒 **Round Locked:** Matches for Matchday {active_gameweek + 1} remain locked until all Matchday {active_gameweek} fixtures officially finish.")

    active_fixtures = df_upcoming_current[df_upcoming_current["matchday"] == active_gameweek]

    if active_fixtures.empty:
        st.success(f"All Matchday {active_gameweek} fixtures have concluded. Model is preparing next round.")
    else:
        rows = []
        for _, fix in active_fixtures.iterrows():
            res = project_match(fix["home_team"], fix["away_team"], df_played_current, combined_history, df_injuries)
            rows.append({
                "Date": fix["date"],
                "Kickoff": fix["time"],
                "Home Team": fix["home_team"],
                "Away Team": fix["away_team"],
                "Home Win %": f"{res['p_h'] * 100:.1f}%",
                "Draw %": f"{res['p_d'] * 100:.1f}%",
                "Away Win %": f"{res['p_a'] * 100:.1f}%",
                "Fair Decimal Odds (H/D/A)": f"{res['odds_h']:.2f} | {res['odds_d']:.2f} | {res['odds_a']:.2f}",
                "Projected Score": res["score"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# TAB 2: LIVE SQUAD ABSENCES
with tab_inj:
    st.subheader("Live Premier League Injury Tracker & Squad Penalties")
    all_teams = sorted(list(set(df_played_current["home_team"]).union(set(df_upcoming_current["home_team"]))))

    if all_teams:
        inspected = st.selectbox("Inspect Team Squad Penalty", all_teams, index=0)
        def_val, abs_list = compute_team_deficit(inspected, df_injuries)
        c1, c2 = st.columns(2)
        c1.metric("Automated Attack/Defense Deficit", f"-{def_val * 100:.1f}%")
        with c2:
            if abs_list:
                st.markdown("**Sidelined Players Detected by API:**")
                for item in abs_list:
                    st.write(f"* {item}")
            else:
                st.success("No critical starter injuries detected for this squad.")

# TAB 3: H2H & FORM INSPECTOR
with tab_h2h:
    st.subheader("Strict 10-Game H2H & 5-Game Form Audit")
    if len(all_teams) >= 2:
        col_x, col_y = st.columns(2)
        with col_x:
            t1 = st.selectbox("Club A", all_teams, index=0)
        with col_y:
            t2 = st.selectbox("Club B", all_teams, index=1 if len(all_teams) > 1 else 0)

        if t1 == t2:
            st.warning("Please choose two distinct clubs.")
        else:
            audit = project_match(t1, t2, df_played_current, combined_history, df_injuries)
            cx, cy = st.columns(2)
            with cx:
                st.markdown(f"#### Last {audit['h2h']['encounters']} H2H Meetings")
                st.write(f"* **{t1} Wins:** {audit['h2h']['h_wins']}")
                st.write(f"* **Draws:** {audit['h2h']['draws']}")
                st.write(f"* **{t2} Wins:** {audit['h2h']['a_wins']}")
                st.write(f"* **Decayed Weighted PPG for {t1}:** {audit['h2h']['ppg']:.2f}")

            with cy:
                st.markdown("#### Rolling 5-Match Form")
                st.write(f"* **{t1} Form PPG:** {audit['h_form']['ppg']:.2f} (GD: {audit['h_form']['gd']:+.2f}/game)")
                st.write(f"* **{t2} Form PPG:** {audit['a_form']['ppg']:.2f} (GD: {audit['a_form']['gd']:+.2f}/game)")

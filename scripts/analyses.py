"""Analyses module for the sexual-assault datasets.

This file contains a working change-point analysis implementation (analysis 1)
and stubs for the other analyses. Run the analyses from the command line or
via the Flask UI which calls run_all().
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt

sns.set(style="whitegrid")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def find_csv_by_keyword(keywords):
    """Return first csv Path whose name contains any keyword (case-insensitive)."""
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix.lower() == ".csv":
            name = p.name.lower()
            for kw in keywords:
                if kw.lower() in name:
                    return p
    return None


def load_state_wise_1999_2013():
    """Load the state-wise file (1999-2013) and return a cleaned DataFrame.

    Strategy: find a CSV with 'state' and '1999' in name; load it, normalize
    column names, ensure 'year' and 'state' columns exist, and create a
    numeric 'cases' column by summing numeric columns per row when explicit
    total isn't present.
    """
    p = find_csv_by_keyword(["state wise", "state wise sexual", "1999 - 2013", "1999-2013"])
    if p is None:
        # fallback: try the explicit filename that exists in the repo
        candidate = ROOT / "State wise Sexual Assault (Detailed) 1999 - 2013.csv"
        if candidate.exists():
            p = candidate
    if p is None:
        raise FileNotFoundError("Could not find the state-wise 1999-2013 CSV in repo root")

    df = pd.read_csv(p, low_memory=False)
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    cols_low = {c.lower(): c for c in df.columns}
    # find year and state columns
    year_col = cols_low.get("year") or cols_low.get("year ") or None
    state_col = None
    for cand in ["state/ut", "state/ut ", "state/ut ", "state", "state/ut/city", "states/ ut"]:
        if cand in cols_low:
            state_col = cols_low[cand]
            break
    if state_col is None:
        # fallback: first column that contains 'state'
        for c in df.columns:
            if "state" in c.lower():
                state_col = c
                break
    if year_col is None:
        for c in df.columns:
            if "year" in c.lower():
                year_col = c
                break

    if year_col is None or state_col is None:
        raise ValueError("Could not detect year or state columns in state-wise file")

    # Coerce year
    df["year"] = pd.to_numeric(df[year_col].astype(str).str.extract(r"(\d{4})")[0], errors="coerce").astype('Int64')
    df["state"] = df[state_col].astype(str).str.strip()

    # create 'cases' by summing numeric columns (excluding year)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in [year_col]]
    if numeric_cols:
        df["cases"] = df[numeric_cols].sum(axis=1)
    else:
        # if there are numeric-looking strings, coerce them
        possible = []
        for c in df.columns:
            if c.lower() not in [year_col.lower(), state_col.lower()]:
                coerced = pd.to_numeric(df[c].astype(str).str.replace(r"[.,]", "", regex=True), errors="coerce")
                if coerced.notna().sum() > 0:
                    possible.append(coerced)
        if possible:
            df["cases"] = pd.concat(possible, axis=1).sum(axis=1)
        else:
            df["cases"] = 0

    # drop rows without year
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df[["state", "year", "cases"]]


def change_point_analysis(output_dir: Path = None, n_bkps=1):
    """Detect a single change point per state in the 1999-2013 dataset.

    Saves:
      - outputs/change_points.csv
      - outputs/change_point_<state>.png for top states by effect size
    """
    if output_dir is None:
        output_dir = OUT
    output_dir.mkdir(exist_ok=True)

    df = load_state_wise_1999_2013()
    # aggregate to year-state total (in case dataset has multiple rows per state-year)
    agg = df.groupby(["state", "year"], dropna=True)["cases"].sum().reset_index()

    states = agg["state"].unique()
    results = []
    plots = []

    for st in states:
        s = agg[agg["state"] == st].sort_values("year")
        years = s["year"].values
        vals = s["cases"].fillna(0).astype(float).values
        if len(vals) < 6:
            continue
        # use ruptures Binseg to detect n_bkps breakpoints
        try:
            algo = rpt.Binseg(model="l2").fit(vals)
            bkps = algo.predict(n_bkps=n_bkps)
            # bkps is list of break indexes (ending indices). Convert first break to year
            bp_idx = bkps[0] if bkps else None
            if bp_idx is None or bp_idx >= len(years):
                continue
            # breakpoint year is the year at that index (we draw vertical line after the point)
            change_year = int(years[bp_idx-1]) if bp_idx-1 < len(years) else int(years[-1])
            before_mean = vals[:bp_idx].mean() if bp_idx > 0 else np.nan
            after_mean = vals[bp_idx:].mean() if bp_idx < len(vals) else np.nan
            effect = after_mean - before_mean
            results.append({"state": st, "change_year": change_year, "before_mean": float(before_mean), "after_mean": float(after_mean), "effect": float(effect)})

            # plot
            plt.figure(figsize=(7,3.5))
            plt.plot(years, vals, marker="o")
            plt.axvline(change_year, color="red", linestyle="--", label=f"change ~ {change_year}")
            plt.title(f"{st} — change {change_year} (effect {effect:.1f})")
            plt.xlabel("Year")
            plt.ylabel("Inferred cases")
            plt.tight_layout()
            outp = output_dir / f"change_point_{re.sub(r'[^a-zA-Z0-9-_]', '_', st)}.png"
            plt.savefig(outp, dpi=150)
            plt.close()
            plots.append(str(outp.name))
        except Exception:
            continue

    df_res = pd.DataFrame(results).sort_values("effect", key=lambda s: s.abs(), ascending=False)
    df_res.to_csv(output_dir / "change_points.csv", index=False)
    print(f"Wrote {output_dir / 'change_points.csv'} and {len(plots)} plots")

    # also create a small summary plot for top 6 by absolute effect
    top6 = df_res.head(6)
    if not top6.empty:
        plt.figure(figsize=(10, 6))
        for i, row in top6.iterrows():
            st = row["state"]
            s = agg[agg["state"] == st].sort_values("year")
            plt.plot(s["year"], s["cases"].fillna(0), marker="o", label=f"{st} ({int(row['change_year'])})")
        plt.legend(fontsize="small")
        plt.title("Top 6 states by change-point effect (1999-2013)")
        plt.xlabel("Year")
        plt.ylabel("Inferred cases")
        plt.tight_layout()
        plt.savefig(output_dir / "change_points_top6.png", dpi=150)
        plt.close()

    return df_res


def under_reporting_heuristic():
    """Estimate possible under-reporting per state using age-share and known-offender share.

    Strategy (heuristic):
    - Use the detailed 2001-2008 file to compute each state's share of victims in child age-bands (<=17).
    - Use the state-wise 1999-2013 file to find a column mentioning "known" offenders and compute known-offender share.
    - Use the summary 2015-2020 file to get a recent-year total (2020 if available, else latest column).
    - Flag states where both child_share and known_offender_share are substantially below national medians.
    - Estimate a lower-bound of "hidden cases" by scaling recent_total so the child_share would match the national median
      (simple proportional adjustment). This is a heuristic and must be interpreted cautiously.

    Outputs written to outputs/under_reporting_estimates.csv and outputs/under_reporting_heatmap.png
    """
    # helpers
    def find_csv(keywords):
        p = find_csv_by_keyword(keywords)
        if p:
            return p
        # try literal known filenames
        candidates = [
            ROOT / "Detailed Cases (Registered) sexual Assault 2001-2008.csv",
            ROOT / "State wise Sexual Assault (Detailed) 1999 - 2013.csv",
            ROOT / "Summary of cases (rape) 2015-2020.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    # 1) load detailed 2001-2008 for age bands
    p_det = find_csv(['2001-2008', 'detailed cases'])
    age_shares = {}
    if p_det is not None:
        df_det = pd.read_csv(p_det, low_memory=False)
        df_det.columns = [c.strip() for c in df_det.columns]
        # find state column
        state_col = None
        for c in df_det.columns:
            if 'state' in c.lower() or 'states' in c.lower() or 'ut' in c.lower():
                state_col = c
                break
        # identify age-related columns
        age_cols = []
        for c in df_det.columns:
            low = c.lower()
            # common patterns: '0-6', '7-12', '13-18', 'below 18', 'under 18', 'age'
            if re.search(r'\b(under|below)\b.*18', low) or re.search(r'\d{1,2}\s*[-to]\s*\d{1,2}', low) or 'age' in low:
                # skip columns which are just text like 'age group' header
                age_cols.append(c)
        # find a total cases column if present
        total_col = None
        for c in df_det.columns:
            if 'rape' in c.lower() and 'total' in c.lower():
                total_col = c
                break
        # fallback: pick numeric columns whose header contains 'cases'
        if total_col is None:
            for c in df_det.columns:
                if 'case' in c.lower() and pd.api.types.is_numeric_dtype(df_det[c]):
                    total_col = c
                    break

        if state_col is not None and total_col is not None and age_cols:
            # compute per-row child count by selecting age columns that appear to be <=17
            def age_col_max(cname):
                nums = re.findall(r"(\d{1,2})", cname)
                if not nums:
                    return None
                try:
                    return max(int(n) for n in nums)
                except Exception:
                    return None

            child_cols = []
            for c in age_cols:
                m = age_col_max(c)
                if m is None:
                    # include if column explicitly says 'under 18' or 'below 18'
                    if re.search(r'\b(under|below)\b.*18', c.lower()):
                        child_cols.append(c)
                else:
                    if m <= 17:
                        child_cols.append(c)

            if not child_cols:
                # as fallback, include any age cols that mention 'child' or 'minor'
                for c in age_cols:
                    if 'child' in c.lower() or 'minor' in c.lower():
                        child_cols.append(c)

            if child_cols:
                # aggregate per state
                df_det[total_col] = pd.to_numeric(df_det[total_col].astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce').fillna(0)
                for c in child_cols:
                    df_det[c] = pd.to_numeric(df_det[c].astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce').fillna(0)
                grouped = df_det.groupby(state_col).agg({total_col: 'sum', **{c: 'sum' for c in child_cols}})
                grouped['child_sum'] = grouped[child_cols].sum(axis=1)
                grouped['child_share'] = grouped['child_sum'] / grouped[total_col].replace({0: pd.NA})
                for st, row in grouped[['child_share', total_col]].iterrows():
                    age_shares[st.strip()] = {'child_share': float(row['child_share']) if pd.notna(row['child_share']) else pd.NA, 'det_total': float(row[total_col])}
    else:
        print('Detailed 2001-2008 CSV not found; skipping age-share calculation')

    # 2) load offender-known shares from state-wise 1999-2013 file
    p_state = find_csv(['state wise', '1999'])
    known_shares = {}
    if p_state is not None:
        raw = pd.read_csv(p_state, low_memory=False)
        raw.columns = [c.strip() for c in raw.columns]
        # try to find state and total and 'known to the victims' column
        state_col = None
        for c in raw.columns:
            if 'state' in c.lower() or 'state/ut' in c.lower():
                state_col = c
                break
        known_col = None
        total_col = None
        for c in raw.columns:
            low = c.lower()
            if 'known to the victims' in low or ('known' in low and 'victim' in low):
                known_col = c
            if 'total' in low and ('case' in low or 'cases' in low):
                total_col = c
        # fallback: if known_col not found, try columns that mention 'known' or 'parents'/'relatives' and sum them
        if state_col is not None:
            if known_col is None:
                candidates = [c for c in raw.columns if 'known' in c.lower() or 'relative' in c.lower() or 'parent' in c.lower() or 'neighbour' in c.lower()]
                if candidates:
                    raw[candidates] = raw[candidates].apply(lambda s: pd.to_numeric(s.astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce').fillna(0))
                    raw['known_sum_tmp'] = raw[candidates].sum(axis=1)
                    known_col = 'known_sum_tmp'
            if total_col is None:
                # fallback: any numeric column likely total
                for c in raw.columns:
                    if pd.api.types.is_numeric_dtype(raw[c]) and c != known_col:
                        total_col = c
                        break

            if known_col is not None and total_col is not None:
                raw[known_col] = pd.to_numeric(raw[known_col].astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce').fillna(0)
                raw[total_col] = pd.to_numeric(raw[total_col].astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce').fillna(0)
                grp = raw.groupby(state_col).agg({known_col: 'sum', total_col: 'sum'})
                grp['known_share'] = grp[known_col] / grp[total_col].replace({0: pd.NA})
                for st, row in grp[['known_share']].iterrows():
                    known_shares[st.strip()] = float(row['known_share']) if pd.notna(row['known_share']) else pd.NA
    else:
        print('State-wise CSV not found; skipping known-offender share calculation')

    # 3) load recent summary to get recent totals (prefer 2020)
    p_sum = find_csv(['2015-2020', 'summary of cases'])
    recent_totals = {}
    if p_sum is not None:
        s = pd.read_csv(p_sum, low_memory=False)
        s.columns = [c.strip() for c in s.columns]
        # state column
        state_col = None
        for c in s.columns:
            if 'state' in c.lower():
                state_col = c
                break
        # find 2020 column
        year_col = None
        for c in s.columns:
            if '2020' in c:
                year_col = c
                break
        if year_col is None:
            # choose the last numeric-year-like column
            yr_cols = [c for c in s.columns if re.search(r'\b20\d{2}\b', c)]
            if yr_cols:
                year_col = yr_cols[-1]

        if state_col is not None and year_col is not None:
            s[year_col] = pd.to_numeric(s[year_col].astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce').fillna(0)
            for idx, row in s.iterrows():
                recent_totals[str(row[state_col]).strip()] = float(row[year_col])
    else:
        print('Summary 2015-2020 CSV not found; recent totals will be empty')

    # merge states observed across the above sources
    states = set(list(age_shares.keys()) + list(known_shares.keys()) + list(recent_totals.keys()))
    rows = []
    for st in sorted(states):
        cs = age_shares.get(st, {})
        child_share = cs.get('child_share', pd.NA)
        known_share = known_shares.get(st, pd.NA)
        recent = recent_totals.get(st, pd.NA)
        rows.append({'state': st, 'child_share': child_share, 'known_share': known_share, 'recent_total': recent})

    df_out = pd.DataFrame(rows)

    # compute medians (skip NaN)
    median_child = float(df_out['child_share'].dropna().median()) if df_out['child_share'].dropna().size>0 else pd.NA
    median_known = float(df_out['known_share'].dropna().median()) if df_out['known_share'].dropna().size>0 else pd.NA

    ests = []
    for _, r in df_out.iterrows():
        st = r['state']
        cs = r['child_share']
        ks = r['known_share']
        recent = r['recent_total']
        flag = False
        est_hidden = 0.0
        reason = ''
        if pd.notna(cs) and pd.notna(ks) and pd.notna(median_child) and pd.notna(median_known):
            if cs < median_child and ks < median_known:
                flag = True
                reason = 'low_child_and_known_share'
                # estimate hidden: scale recent upwards so child_share matches median_child
                if pd.notna(recent) and pd.notna(cs) and cs>0:
                    est_hidden = max(0.0, recent * (median_child / cs - 1.0))
                else:
                    est_hidden = pd.NA
        ests.append({'state': st, 'child_share': cs, 'known_share': ks, 'recent_total': recent, 'flag_underreport': flag, 'est_hidden_cases': est_hidden, 'reason': reason})

    df_est = pd.DataFrame(ests)
    df_est.to_csv(OUT / 'under_reporting_estimates.csv', index=False)
    print(f'Wrote {OUT / "under_reporting_estimates.csv"}')

    # heatmap: show normalized est_hidden (log) or flag
    try:
        import matplotlib.colors as mcolors
        vis = df_est.copy()
        vis['score'] = vis['est_hidden_cases'].replace({pd.NA:0}).fillna(0).astype(float)
        vis = vis.sort_values('score', ascending=False).head(40)
        plt.figure(figsize=(6, max(4, 0.18*len(vis))))
        sns.barplot(x='score', y='state', data=vis, palette='magma')
        plt.xlabel('Estimated hidden cases (heuristic)')
        plt.title('Under-reporting heuristic — top states')
        plt.tight_layout()
        plt.savefig(OUT / 'under_reporting_heatmap.png', dpi=150)
        plt.close()
        print(f'Wrote {OUT / "under_reporting_heatmap.png"}')
    except Exception as e:
        print('Failed to create heatmap:', e)

    return df_est


def composite_vulnerability_index():
    """Compute a Composite Vulnerability Index (CVI) per state.

    Components (heuristic):
      - recent_total (reported cases, recent year from summary file) [higher -> more vulnerable]
      - child_share (share of victims <=17) [higher -> more vulnerable]
      - known_offender_share inverse (1 - known_share) [higher -> more vulnerable]
      - recent trend slope (positive increases) [higher -> more vulnerable]

    Outputs:
      - outputs/cvi_rankings.csv
      - outputs/cvi_dashboard.png (top-10 bar chart)

    Notes:
      - This is an index for prioritization, not a causal measure. Per-capita normalization is optional
        (not applied unless a population file is provided). If you want per-capita, provide a CSV with
        state and population columns and I'll re-run the CVI.
    """
    # reuse under-reporting to obtain child_share and known_share and recent_total where possible
    try:
        df_est = under_reporting_heuristic()
    except Exception:
        # fallback: try reading existing output
        est_path = OUT / 'under_reporting_estimates.csv'
        if est_path.exists():
            df_est = pd.read_csv(est_path)
        else:
            raise

    # load state-year data to compute trend slopes
    try:
        df_state = load_state_wise_1999_2013()
    except Exception:
        df_state = None

    # compute trend slope per state (linear fit on recent years available)
    slopes = {}
    if df_state is not None:
        for st in df_state['state'].unique():
            s = df_state[df_state['state'] == st].sort_values('year')
            # take last up to 6 years
            s_recent = s.tail(6)
            if len(s_recent) >= 3:
                # fit linear slope
                yrs = s_recent['year'].values.astype(float)
                vals = s_recent['cases'].fillna(0).values.astype(float)
                # if all zeros, slope 0
                if np.all(vals == 0):
                    slope = 0.0
                else:
                    try:
                        coeffs = np.polyfit(yrs, vals, 1)
                        slope = float(coeffs[0])
                    except Exception:
                        slope = 0.0
                slopes[st] = slope

    # merge metrics
    df = df_est.copy()
    df = df.rename(columns={'state': 'state'})
    df['trend_slope'] = df['state'].map(slopes).fillna(0.0)

    # Transform/normalize components
    # recent_total: use log1p to reduce skew
    df['recent_total_log'] = df['recent_total'].fillna(0).apply(lambda x: np.log1p(x))
    # child_share and known_share: already between 0-1 but may have NaN
    df['child_share_f'] = df['child_share'].fillna(0.0)
    df['known_share_f'] = df['known_share'].fillna(0.0)
    df['known_inv'] = 1.0 - df['known_share_f']
    df['trend_pos'] = df['trend_slope'].apply(lambda x: x if x>0 else 0.0)

    def minmax(series):
        s = series.dropna().astype(float)
        if s.empty:
            return series.apply(lambda _: 0.0)
        mn = s.min(); mx = s.max()
        if mx == mn:
            return series.apply(lambda _: 0.0)
        return (series - mn) / (mx - mn)

    df['c_recent'] = minmax(df['recent_total_log'])
    df['c_child'] = minmax(df['child_share_f'])
    df['c_known_inv'] = minmax(df['known_inv'])
    df['c_trend'] = minmax(df['trend_pos'])

    # weights
    w_recent = 0.4
    w_child = 0.25
    w_known_inv = 0.2
    w_trend = 0.15

    df['cvi_score'] = (w_recent*df['c_recent'] + w_child*df['c_child'] + w_known_inv*df['c_known_inv'] + w_trend*df['c_trend'])
    df = df.sort_values('cvi_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    out_cols = ['state', 'recent_total', 'child_share', 'known_share', 'trend_slope', 'c_recent', 'c_child', 'c_known_inv', 'c_trend', 'cvi_score', 'rank']
    df[out_cols].to_csv(OUT / 'cvi_rankings.csv', index=False)
    print(f'Wrote {OUT / "cvi_rankings.csv"}')

    # plot top 10
    top10 = df.head(10)
    plt.figure(figsize=(8,4.5))
    sns.barplot(x='cvi_score', y='state', data=top10, palette='rocket')
    plt.title('Composite Vulnerability Index — top 10 states')
    plt.xlabel('CVI score (0-1)')
    plt.tight_layout()
    plt.savefig(OUT / 'cvi_dashboard.png', dpi=150)
    plt.close()
    print(f'Wrote {OUT / "cvi_dashboard.png"}')

    return df


def network_analysis():
    """Build a state similarity network from offender-category shares and detect communities.

    Steps:
      - Load the state-wise 1999-2013 CSV and identify offender-category columns (parents, relatives, neighbours, other known, unknown)
      - Aggregate counts by state across years and compute category proportions per state
      - Compute pairwise cosine similarity between states on the offender-share vectors
      - Create a graph connecting states with similarity above a threshold or top-k neighbors
      - Run community detection (greedy modularity) and compute centrality measures
      - Save `outputs/network_clusters.csv` (state, cluster, centralities) and `outputs/network_clusters.png` (graph)

    Returns the cluster DataFrame.
    """
    try:
        p_state = find_csv_by_keyword(['state wise', '1999', 'State wise Sexual Assault'])
        if p_state is None:
            p_state = ROOT / 'State wise Sexual Assault (Detailed) 1999 - 2013.csv'
        if not p_state or not p_state.exists():
            raise FileNotFoundError('State-wise CSV not found for network analysis')

        raw = pd.read_csv(p_state, low_memory=False)
        raw.columns = [c.strip() for c in raw.columns]

        # find state column
        state_col = None
        for c in raw.columns:
            if 'state' in c.lower():
                state_col = c
                break
        if state_col is None:
            raise ValueError('State column not found in state-wise CSV')

        # candidate offender-related columns
        offender_cands = []
        for c in raw.columns:
            low = c.lower()
            if any(term in low for term in ['parent', 'family', 'relative', 'relatives', 'neighbour', 'neighbourhood', 'neighbourhood', 'neighbour', 'other known', 'other_known', 'known to the victims', 'known']):
                offender_cands.append(c)

        # Also include columns that explicitly mention 'offenders were' patterns
        for c in raw.columns:
            if 'offender' in c.lower() and ('were' in c.lower() or 'known' in c.lower()):
                if c not in offender_cands:
                    offender_cands.append(c)

        # If not found, fall back to columns that mention 'parents /' or explicit known categories
        if not offender_cands:
            offender_cands = [c for c in raw.columns if any(k in c.lower() for k in ['parents', 'relative', 'neighbour', 'known'])]

        if not offender_cands:
            raise ValueError('No offender-category columns detected in state-wise CSV')

        # Coerce numeric for these columns
        for c in offender_cands:
            raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(r"[.,]", "", regex=True), errors='coerce').fillna(0)

        # Aggregate across years (sum per state)
        agg = raw.groupby(state_col)[offender_cands].sum()
        agg.index = agg.index.map(lambda s: str(s).strip())

        # Compute proportions per state
        agg['total_offender_counts'] = agg.sum(axis=1)
        # avoid divide by zero
        for c in offender_cands:
            agg[c + '_prop'] = agg[c] / agg['total_offender_counts'].replace({0: pd.NA})

        prop_cols = [c + '_prop' for c in offender_cands]
        prop_df = agg[prop_cols].fillna(0).astype(float)

        # similarity matrix (cosine)
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(prop_df.values)
        states = list(prop_df.index)

        import networkx as nx
        G = nx.Graph()
        # add nodes with total counts
        totals = agg['total_offender_counts'].to_dict()
        for st in states:
            G.add_node(st, total=int(totals.get(st, 0)))

        # add edges: connect if similarity > threshold OR top-3 neighbors per state
        thresh = 0.5
        for i, s1 in enumerate(states):
            # top-k neighbors (excluding self)
            sims = list(enumerate(sim[i]))
            sims = [(j, val) for j, val in sims if j != i]
            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
            # add top 3
            for j, val in sims_sorted[:3]:
                if val > 0.05:  # small floor
                    G.add_edge(states[i], states[j], weight=float(val))
            # add any above threshold
            for j, val in sims:
                if j > i and val >= thresh:
                    G.add_edge(states[i], states[j], weight=float(val))

        # community detection
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G, weight='weight'))
        except Exception:
            # fallback to label propagation
            from networkx.algorithms.community import asyn_lpa_communities
            communities = list(asyn_lpa_communities(G))

        # map state -> cluster id
        cluster_map = {}
        for cid, comm in enumerate(communities, start=1):
            for st in comm:
                cluster_map[st] = cid

        # compute centralities
        degree = nx.degree_centrality(G)
        betw = nx.betweenness_centrality(G, weight='weight')

        rows = []
        for st in states:
            rows.append({'state': st, 'cluster': cluster_map.get(st, 0), 'degree_centrality': degree.get(st, 0.0), 'betweenness': betw.get(st, 0.0), 'total_offender_counts': int(totals.get(st, 0))})

        df_clusters = pd.DataFrame(rows).sort_values(['cluster', 'degree_centrality'], ascending=[True, False])
        df_clusters.to_csv(OUT / 'network_clusters.csv', index=False)
        print(f'Wrote {OUT / "network_clusters.csv"}')

        # plot network
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        # color nodes by cluster
        cmap = plt.get_cmap('tab20')
        node_colors = [cmap((cluster_map.get(n, 0) % 20) / 20) for n in G.nodes()]
        node_sizes = [max(50, int(200 * (G.nodes[n]['total'] / (max(1, max(totals.values())))))) for n in G.nodes()]
        # draw edges with alpha by weight
        edges = G.edges(data=True)
        weights = [d['weight'] for (_, _, d) in edges]
        nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray', width=[max(0.5, w*3) for w in weights])
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title('State similarity network (offender-category shares) — colored by community')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(OUT / 'network_clusters.png', dpi=200)
        plt.close()
        print(f'Wrote {OUT / "network_clusters.png"}')

        return df_clusters
    except Exception as e:
        print('Network analysis failed:', e)
        raise


def age_cohort_analysis():
    """Analyze changes in age-band shares across time windows and produce plots.

    Outputs:
      - outputs/age_cohort_changes.csv  (state, age_band, share_early, share_late, pct_point_change)
      - outputs/age_cohort_top_changes.png  (bar chart of largest absolute changes)
      - outputs/age_cohort_stack_national.png (national stacked area by year for age bands)
    """
    # find detailed file
    p_det = find_csv_by_keyword(['2001-2008', 'detailed cases', 'Detailed Cases'])
    if p_det is None:
        p_det = ROOT / 'Detailed Cases (Registered) sexual Assault 2001-2008.csv'
    if not p_det.exists():
        raise FileNotFoundError('Detailed 2001-2008 CSV not found for age-cohort analysis')

    df = pd.read_csv(p_det, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # detect state and year
    state_col = None
    year_col = None
    for c in df.columns:
        low = c.lower()
        if state_col is None and ('state' in low or 'states' in low or 'ut' in low):
            state_col = c
        if year_col is None and 'year' in low:
            year_col = c
    if state_col is None:
        # try common alternative
        for c in df.columns:
            if 'states/ ut' in c.lower() or 'states/ ut/cities' in c.lower():
                state_col = c
                break
    if year_col is None:
        # attempt to find a column with a 4-digit sample
        for c in df.columns:
            sample = df[c].astype(str).head(50).str.extract(r'(\d{4})')[0]
            if sample.notna().sum() > 5:
                year_col = c
                break

    if state_col is None or year_col is None:
        raise ValueError('Could not detect state/year columns in detailed 2001-2008 file')

    # detect age-band columns: look for patterns like '0-6', '7-12', '13-18', 'below 18', 'under 18', 'age'
    age_cols = []
    for c in df.columns:
        low = c.lower()
        if re.search(r'\d{1,2}\s*[-to]\s*\d{1,2}', low) or re.search(r'under|below|age', low):
            # exclude columns that are not numeric by sampling
            sample = df[c].astype(str).str.replace(r"[,.]", "", regex=True)
            coerced = pd.to_numeric(sample, errors='coerce')
            if coerced.notna().sum() > 10:
                age_cols.append(c)

    if not age_cols:
        raise ValueError('No age-band numeric columns detected in detailed file')

    # coerce numeric on selected columns
    for c in age_cols + [year_col]:
        if c != year_col:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce').fillna(0)
        else:
            df[year_col] = pd.to_numeric(df[year_col].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').astype('Int64')

    df = df.dropna(subset=[year_col]).copy()
    df[year_col] = df[year_col].astype(int)

    # define early / late windows (split years into two halves)
    years = sorted(df[year_col].unique())
    if len(years) < 4:
        raise ValueError('Not enough years for cohort comparison')
    mid = len(years) // 2
    early_years = years[:mid]
    late_years = years[mid:]

    # aggregate sums per state per window
    agg_early = df[df[year_col].isin(early_years)].groupby(state_col)[age_cols].sum()
    agg_late = df[df[year_col].isin(late_years)].groupby(state_col)[age_cols].sum()

    states = sorted(set(agg_early.index.tolist()) | set(agg_late.index.tolist()))
    rows = []
    for st in states:
        e = agg_early.loc[st] if st in agg_early.index else pd.Series({c:0 for c in age_cols})
        l = agg_late.loc[st] if st in agg_late.index else pd.Series({c:0 for c in age_cols})
        e_total = e.sum()
        l_total = l.sum()
        for c in age_cols:
            share_e = float(e[c] / e_total) if e_total>0 else pd.NA
            share_l = float(l[c] / l_total) if l_total>0 else pd.NA
            pct_point = (share_l - share_e) if (pd.notna(share_l) and pd.notna(share_e)) else pd.NA
            pct_rel = (pct_point / share_e) if (pd.notna(pct_point) and share_e and share_e>0) else pd.NA
            rows.append({'state': str(st).strip(), 'age_band': c, 'share_early': share_e, 'share_late': share_l, 'pct_point_change': pct_point, 'pct_relative_change': pct_rel})

    df_changes = pd.DataFrame(rows)
    df_changes.to_csv(OUT / 'age_cohort_changes.csv', index=False)
    print(f'Wrote {OUT / "age_cohort_changes.csv"}')

    # compute top absolute changes (by absolute point change) for plotting
    df_abs = df_changes.dropna(subset=['pct_point_change']).copy()
    df_abs['abs_pp'] = df_abs['pct_point_change'].abs()
    top = df_abs.sort_values('abs_pp', ascending=False).head(20)

    # bar chart of top changes
    plt.figure(figsize=(8, max(4, 0.35*len(top))))
    sns.barplot(x='pct_point_change', y='state', hue='age_band', data=top, dodge=False)
    plt.axvline(0, color='k', linewidth=0.6)
    plt.xlabel('Percentage-point change (late - early)')
    plt.title('Top state × age-band changes between periods')
    plt.legend(title='age band', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUT / 'age_cohort_top_changes.png', dpi=150)
    plt.close()
    print(f'Wrote {OUT / "age_cohort_top_changes.png"}')

    # national stacked area by year for age bands
    yearly = df.groupby(year_col)[age_cols].sum()
    yearly_norm = yearly.div(yearly.sum(axis=1), axis=0).fillna(0)
    plt.figure(figsize=(10,5))
    yearly_norm.plot(kind='area', stacked=True, colormap='tab20', alpha=0.85)
    plt.ylabel('Share')
    plt.xlabel('Year')
    plt.title('National distribution of age-bands over time')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(OUT / 'age_cohort_stack_national.png', dpi=150)
    plt.close()
    print(f'Wrote {OUT / "age_cohort_stack_national.png"}')

    return df_changes


def international_context():
    """Create a quick international context comparison using `World Wide Cases detail.csv`.

    Strategy:
      - Load the world file and look for indicators containing 'rape', 'sexual' or 'violent' (in that order).
      - Pick the most recent year available for India for that indicator, and compare India's value to regional medians and country peers.
      - Produce CSV (`outputs/international_context.csv`) with per-region medians and India value, and a PNG (`outputs/international_context.png`) showing India vs region medians.
      - Save a short cautionary note (`outputs/international_context_note.txt`).

    Returns the DataFrame used for plotting.
    """
    p = find_csv_by_keyword(['world wide', 'world wide cases', 'World Wide Cases detail'])
    if p is None:
        p = ROOT / 'World Wide Cases detail.csv'
    if not p.exists():
        raise FileNotFoundError('World Wide Cases detail CSV not found')

    try:
        w = pd.read_csv(p, low_memory=False)
    except UnicodeDecodeError:
        w = pd.read_csv(p, low_memory=False, encoding='latin1')
    # normalize columns
    w.columns = [c.strip() for c in w.columns]
    cols_low = {c.lower(): c for c in w.columns}

    # find key columns used in the summary (from repo context: Country, Region, Indicator, Year, VALUE)
    country_col = cols_low.get('country') or cols_low.get('country name') or cols_low.get('country_name') or cols_low.get('country/territory')
    region_col = cols_low.get('region') or cols_low.get('region name')
    indicator_col = cols_low.get('indicator')
    year_col = cols_low.get('year')
    value_col = cols_low.get('value') or cols_low.get('VALUE')

    for c in [country_col, region_col, indicator_col, year_col, value_col]:
        if c is None:
            raise ValueError('Could not find expected columns in world CSV; available: ' + ','.join(w.columns.tolist()))

    # normalize text
    w[country_col] = w[country_col].astype(str).str.strip()
    w[region_col] = w[region_col].astype(str).str.strip()
    w[indicator_col] = w[indicator_col].astype(str).str.strip()

    # try to find an indicator: prefer 'rape' or 'sexual', else 'violent'
    cand = None
    for kw in ['rape', 'sexual']:
        mask = w[indicator_col].str.lower().str.contains(kw, na=False)
        if mask.any():
            cand = w[mask].copy()
            chosen_kw = kw
            break
    if cand is None:
        mask = w[indicator_col].str.lower().str.contains('violent', na=False)
        if mask.any():
            cand = w[mask].copy()
            chosen_kw = 'violent'

    if cand is None or cand.empty:
        # fallback: pick the largest-group indicator by count
        top_ind = w[indicator_col].value_counts().idxmax()
        cand = w[w[indicator_col] == top_ind].copy()
        chosen_kw = top_ind

    # coerce year and value
    cand[year_col] = pd.to_numeric(cand[year_col].astype(str).str.extract(r"(\d{4})")[0], errors='coerce').astype('Int64')
    cand[value_col] = pd.to_numeric(cand[value_col].astype(str).str.replace(r"[,.]", "", regex=True), errors='coerce')

    # find most recent year where India has a value
    india_rows = cand[cand[country_col].str.lower() == 'india']
    india_recent_year = None
    if not india_rows.empty:
        india_recent_year = int(india_rows[year_col].dropna().max())
    else:
        # fallback: use global most recent year
        india_recent_year = int(cand[year_col].dropna().max())

    # build comparison: for india_recent_year, get country values and region medians
    df_year = cand[cand[year_col] == india_recent_year].copy()
    df_year = df_year.dropna(subset=[value_col])
    if df_year.empty:
        raise ValueError(f'No values for chosen indicator in year {india_recent_year}')

    # compute region medians and counts
    region_stats = df_year.groupby(region_col)[value_col].agg(['median', 'count']).reset_index().rename(columns={'median': 'region_median', 'count': 'n_countries'})

    # extract India value
    india_val = None
    india_entry = df_year[df_year[country_col].str.lower() == 'india']
    if not india_entry.empty:
        india_val = float(india_entry.iloc[0][value_col])

    # prepare CSV
    out = region_stats.copy()
    out['indicator'] = chosen_kw
    out['year'] = india_recent_year
    out['india_value'] = india_val
    out = out[[region_col, 'n_countries', 'region_median', 'india_value', 'indicator', 'year']]
    out.to_csv(OUT / 'international_context.csv', index=False)
    print(f'Wrote {OUT / "international_context.csv"} (indicator ~ {chosen_kw}, year {india_recent_year})')

    # plot: India vs region median
    try:
        plt.figure(figsize=(8,5))
        # order regions by median
        out_sorted = out.sort_values('region_median', ascending=False)
        sns.barplot(x='region_median', y=region_col, data=out_sorted, color='lightgray')
        if pd.notna(india_val):
            # find India's region (if present)
            india_region = None
            if not india_entry.empty:
                india_region = india_entry.iloc[0][region_col]
            # annotate India value on plot as a red dot at the appropriate region row
            # if India's region present, highlight its bar
            for i, r in out_sorted.iterrows():
                if india_region and r[region_col] == india_region:
                    plt.barh(r[region_col], r['region_median'], color='salmon')
        plt.axvline(india_val if india_val is not None else 0, color='red', linestyle='--', label=f'India ({india_recent_year})')
        plt.xlabel('Region median (indicator units)')
        plt.ylabel('Region')
        plt.title(f'International context — indicator ~ "{chosen_kw}" — year {india_recent_year}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / 'international_context.png', dpi=150)
        plt.close()
        print(f'Wrote {OUT / "international_context.png"}')
    except Exception as e:
        print('Failed to create international plot:', e)

    # save note
    note = (
        f"Indicator selection: '{chosen_kw}'.\n"
        f"Year used: {india_recent_year}.\n"
        "Caution: the world dataset contains many different indicators and units; this plot is a simple, exploratory comparison. "
        "Reported counts are not directly comparable across countries without harmonized definitions and population normalization.\n"
        "Interpret with caution; this chart is for contextual framing only."
    )
    with open(OUT / 'international_context_note.txt', 'w') as fh:
        fh.write(note)

    return out


def run_all():
    """Run all available analyses (for now, change-point) and return a dict of results."""
    out = {}
    try:
        cp = change_point_analysis()
        out['change_points'] = cp
    except Exception as e:
        out['change_points_error'] = str(e)
    return out


if __name__ == "__main__":
    print("Running change-point analysis...")
    res = change_point_analysis()
    print(res.head())

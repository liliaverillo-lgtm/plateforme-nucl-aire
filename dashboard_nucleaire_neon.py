#!/usr/bin/env python3
"""
Dashboard — Modulation nucléaire par réacteur (France)
Normalisation par la puissance nominale IAEA PRIS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MISE EN PLACE NEON (une seule fois, 2 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1) Créer un compte gratuit sur https://neon.tech
 2) "Create project" → nommer "nucleaire-cache"
 3) Dashboard → "Connection Details" → copier la chaîne
    qui commence par : postgresql://...
 4) Dans Streamlit Cloud → Settings → Secrets, coller :
      DATABASE_URL = "postgresql://user:pass@xxx.neon.tech/neondb?sslmode=require"

 En local : créer .streamlit/secrets.toml avec la même clé.

 REQUIREMENTS
━━━━━━━━━━━━━
 entsoe-py
 pandas
 plotly
 streamlit
 psycopg2-binary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
warnings.filterwarnings("ignore")

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import psycopg2
import psycopg2.extras
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from entsoe import EntsoePandasClient


# ══════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════

API_KEY               = "c5cb3857-bc40-4f4c-a4db-088946785b4a"
COUNTRY               = "FR"
TZ                    = "Europe/Paris"
SEUIL_ON_PCT          = 5
N_COLS_SPARKLINES     = 4
MAX_WORKERS_API       = 4
REFRESH_TODAY_MINUTES = 30

PUISSANCE_NOMINALE_MW = {
    "BUGEY 2": 910,  "BUGEY 3": 910,  "BUGEY 4": 880,  "BUGEY 5": 880,
    "BLAYAIS 1": 910,  "BLAYAIS 2": 910,  "BLAYAIS 3": 910,  "BLAYAIS 4": 910,
    "CHINON 1": 905,   "CHINON 2": 905,   "CHINON 3": 905,   "CHINON 4": 905,
    "CRUAS 1": 915,    "CRUAS 2": 915,    "CRUAS 3": 915,    "CRUAS 4": 915,
    "DAMPIERRE 1": 890,"DAMPIERRE 2": 890,"DAMPIERRE 3": 890,"DAMPIERRE 4": 890,
    "GRAVELINES 1": 910,"GRAVELINES 2": 910,"GRAVELINES 3": 910,
    "GRAVELINES 4": 910,"GRAVELINES 5": 910,"GRAVELINES 6": 910,
    "ST LAURENT 1": 915,"ST LAURENT 2": 915,
    "TRICASTIN 1": 915,"TRICASTIN 2": 915,"TRICASTIN 3": 915,"TRICASTIN 4": 915,
    "FLAMANVILLE 1": 1310,"FLAMANVILLE 2": 1310,
    "PALUEL 1": 1330,  "PALUEL 2": 1330,  "PALUEL 3": 1330,  "PALUEL 4": 1330,
    "ST ALBAN 1": 1335,"ST ALBAN 2": 1335,
    "BELLEVILLE 1": 1310,"BELLEVILLE 2": 1310,
    "CATTENOM 1": 1300,"CATTENOM 2": 1300,"CATTENOM 3": 1300,"CATTENOM 4": 1300,
    "GOLFECH 1": 1310, "GOLFECH 2": 1310,
    "NOGENT 1": 1310,  "NOGENT 2": 1310,
    "PENLY 1": 1320,   "PENLY 2": 1320,
    "CHOOZ 1": 1500,   "CHOOZ 2": 1500,
    "CIVAUX 1": 1495,  "CIVAUX 2": 1495,
    "FLAMANVILLE 3": 1630,
}

AUJOURDHUI = datetime.now().date()
HIER       = AUJOURDHUI - timedelta(days=1)
_db_lock   = threading.Lock()


# ══════════════════════════════════════════════════════════════════
# 1. EXTRACTION ENTSO-E
# ══════════════════════════════════════════════════════════════════

def extraire_actual_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex ENTSO-E → DataFrame wide (réacteur → MW)."""
    if isinstance(df.columns, pd.MultiIndex):
        niv0 = df.columns.get_level_values(0).astype(str)
        niv1 = df.columns.get_level_values(1).astype(str)
        m1   = niv1.str.contains("Aggregated", case=False, na=False)
        m0   = niv0.str.contains("Aggregated", case=False, na=False)
        if m1.any():
            out = df.loc[:, m1].copy(); out.columns = out.columns.droplevel(1)
        elif m0.any():
            out = df.loc[:, m0].copy(); out.columns = out.columns.droplevel(0)
        else:
            out = df.copy(); out.columns = niv0
    else:
        out = df.copy()
        out.columns = [str(c) for c in out.columns]
    return out


# ══════════════════════════════════════════════════════════════════
# 2. BASE DE DONNÉES NEON (PostgreSQL)
# ══════════════════════════════════════════════════════════════════

def _conn() -> psycopg2.extensions.connection:
    """Nouvelle connexion PostgreSQL (légère, fermée après usage)."""
    try:
        return psycopg2.connect(st.secrets["DATABASE_URL"], connect_timeout=10)
    except KeyError:
        st.error(
            "🔑 **Secret manquant.** "
            "Ajoutez `DATABASE_URL` dans les secrets Streamlit.\n\n"
            "Voir les instructions en haut du fichier source."
        )
        st.stop()


def init_db() -> None:
    """Crée les tables si elles n'existent pas (idempotent)."""
    conn = _conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS production (
                        ts       TEXT NOT NULL,
                        reacteur TEXT NOT NULL,
                        mw       REAL,
                        PRIMARY KEY (ts, reacteur)
                    )
                """)
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prod_ts ON production(ts)"
                )
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS jours_charges (
                        jour        TEXT    PRIMARY KEY,
                        charge_ts   TEXT    NOT NULL,
                        est_complet INTEGER NOT NULL DEFAULT 1
                    )
                """)
    finally:
        conn.close()


def jour_est_cache(jour: date) -> bool:
    """True si le jour est en DB et frais."""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT charge_ts, est_complet FROM jours_charges WHERE jour = %s",
                (str(jour),)
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return False
    charge_ts_str, est_complet = row
    if not est_complet:
        age_min = (datetime.now() - datetime.fromisoformat(charge_ts_str)).total_seconds() / 60
        return age_min < REFRESH_TODAY_MINUTES
    return True


def sauvegarder_en_db(jour: date, df_wide: pd.DataFrame) -> None:
    """Persiste df_wide dans Neon (PostgreSQL)."""
    if df_wide.empty:
        return

    idx = df_wide.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ts_utc = idx.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S")

    df_tmp = df_wide.copy()
    df_tmp.index      = ts_utc
    df_tmp.index.name = "ts"
    df_long = (
        df_tmp.reset_index()
              .melt(id_vars="ts", var_name="reacteur", value_name="mw")
              .dropna(subset=["mw"])
    )
    rows = [tuple(r) for r in df_long[["ts", "reacteur", "mw"]].values.tolist()]
    if not rows:
        return

    est_complet = 1 if jour < datetime.now().date() else 0

    with _db_lock:
        conn = _conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    # Upsert PostgreSQL (équivalent de INSERT OR REPLACE)
                    psycopg2.extras.execute_values(
                        cur,
                        """INSERT INTO production (ts, reacteur, mw) VALUES %s
                           ON CONFLICT (ts, reacteur) DO UPDATE SET mw = EXCLUDED.mw""",
                        rows
                    )
                    cur.execute(
                        """INSERT INTO jours_charges (jour, charge_ts, est_complet)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (jour) DO UPDATE
                           SET charge_ts = EXCLUDED.charge_ts,
                               est_complet = EXCLUDED.est_complet""",
                        (str(jour), datetime.now().isoformat(), est_complet)
                    )
        finally:
            conn.close()


def charger_depuis_db(start: date, end: date) -> pd.DataFrame:
    """Charge [start, end] depuis Neon."""
    start_sql = (datetime.combine(start, datetime.min.time()) - timedelta(hours=3)) \
                .strftime("%Y-%m-%dT%H:%M:%S")
    end_sql   = (datetime.combine(end, datetime.min.time()) + timedelta(hours=27)) \
                .strftime("%Y-%m-%dT%H:%M:%S")

    conn = _conn()
    try:
        df_long = pd.read_sql_query(
            "SELECT ts, reacteur, mw FROM production WHERE ts >= %s AND ts <= %s ORDER BY ts",
            conn, params=(start_sql, end_sql)
        )
    finally:
        conn.close()

    if df_long.empty:
        return pd.DataFrame()

    df_long["ts"] = pd.to_datetime(df_long["ts"], utc=True).dt.tz_convert(TZ)
    borne_start   = pd.Timestamp(str(start),             tz=TZ)
    borne_end     = pd.Timestamp(str(end) + " 23:59:59", tz=TZ)
    df_long = df_long[(df_long["ts"] >= borne_start) & (df_long["ts"] <= borne_end)]

    if df_long.empty:
        return pd.DataFrame()

    df_wide = df_long.pivot_table(index="ts", columns="reacteur", values="mw", aggfunc="mean")
    df_wide.columns.name = None
    df_wide.index.name   = None
    return df_wide


@st.cache_data(ttl=30, show_spinner=False)
def stats_cache() -> dict:
    """Statistiques rapides sur le cache (sidebar)."""
    try:
        conn = _conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT jour FROM jours_charges ORDER BY jour")
                jours = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()
        return {"n": len(jours), "min": jours[0] if jours else None, "max": jours[-1] if jours else None}
    except Exception:
        return {"n": 0, "min": None, "max": None}


def purger_periode_db(start: date, end: date) -> int:
    """Supprime une période du cache Neon."""
    jours     = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    start_sql = (datetime.combine(start, datetime.min.time()) - timedelta(hours=3)) \
                .strftime("%Y-%m-%dT%H:%M:%S")
    end_sql   = (datetime.combine(end, datetime.min.time()) + timedelta(hours=27)) \
                .strftime("%Y-%m-%dT%H:%M:%S")
    with _db_lock:
        conn = _conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM production WHERE ts >= %s AND ts <= %s",
                        (start_sql, end_sql)
                    )
                    cur.executemany(
                        "DELETE FROM jours_charges WHERE jour = %s",
                        [(str(j),) for j in jours]
                    )
        finally:
            conn.close()
    stats_cache.clear()
    return len(jours)


# ══════════════════════════════════════════════════════════════════
# 3. API ENTSO-E
# ══════════════════════════════════════════════════════════════════

def api_telecharger_jour(jour: date) -> bool:
    """Télécharge un jour depuis ENTSO-E et le sauvegarde dans Neon."""
    client   = EntsoePandasClient(api_key=API_KEY)
    start_ts = pd.Timestamp(str(jour) + " 00:00", tz=TZ)
    end_ts   = pd.Timestamp(str(jour) + " 23:59", tz=TZ)
    df_raw   = client.query_generation_per_plant(
        country_code=COUNTRY, start=start_ts, end=end_ts, psr_type="B14"
    )
    if df_raw is None or df_raw.empty:
        return False
    sauvegarder_en_db(jour, extraire_actual_aggregated(df_raw))
    return True


# ══════════════════════════════════════════════════════════════════
# 4. INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════

init_db()

st.set_page_config(page_title="☢️ Modulation nucléaire France", layout="wide", page_icon="☢️")
st.title("☢️ Modulation nucléaire par réacteur — France")
st.caption("Production normalisée par la puissance nominale (IAEA PRIS) · Cache : Neon · Données : ENTSO-E")

with st.sidebar:
    st.header("📅 Période")
    start_date = st.date_input("Début", value=HIER - timedelta(days=6), max_value=AUJOURDHUI)
    end_date   = st.date_input("Fin",   value=HIER,                      max_value=AUJOURDHUI)
    nb_jours   = (end_date - start_date).days + 1
    st.info(f"📆 {nb_jours} jour(s)")
    if nb_jours > 31:
        st.warning("⚠️ Au-delà de 31 jours, le premier chargement peut être long.")
    lancer = st.button("🔄 Rafraîchir", type="primary", use_container_width=True)
    with st.expander("🗑️ Gestion du cache"):
        st.caption("Force un re-téléchargement de la période sélectionnée.")
        if st.button("Purger la période", use_container_width=True):
            n = purger_periode_db(start_date, end_date)
            st.toast(f"🗑️ {n} jour(s) supprimés", icon="✅")
    st.markdown("---")
    info = stats_cache()
    if info["n"] == 0:
        st.caption("📂 Base Neon vide — premier lancement.")
    else:
        st.caption(f"☁️ Neon : **{info['n']} jours** en cache\n\nDu {info['min']} au {info['max']}")
    st.markdown("**Source Pnom** : IAEA PRIS · [pris.iaea.org](https://pris.iaea.org/pris/CountryStatistics/CountryDetails.aspx?current=FR)")

if "premier_chargement" not in st.session_state:
    st.session_state.premier_chargement = True
    lancer = True

if not lancer:
    st.stop()

if start_date > end_date:
    st.error("La date de début doit être antérieure à la date de fin.")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# 5. CHARGEMENT
# ══════════════════════════════════════════════════════════════════

def charger_periode(start: date, end: date) -> tuple[pd.DataFrame, int, int]:
    tous_les_jours  = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    jours_a_fetcher = [j for j in tous_les_jours if not jour_est_cache(j)]
    nb_cache        = len(tous_les_jours) - len(jours_a_fetcher)
    echecs: list[tuple[str, str]] = []
    lock    = threading.Lock()
    counter = {"n": 0}

    if jours_a_fetcher:
        barre = st.progress(0.0, text="⚡ Téléchargement des jours manquants…")

        def _fetch(j: date) -> tuple[date, bool]:
            ok = api_telecharger_jour(j)
            with lock:
                counter["n"] += 1
                barre.progress(
                    counter["n"] / len(jours_a_fetcher),
                    text=f"⚡ Téléchargé {counter['n']}/{len(jours_a_fetcher)} jour(s)…",
                )
            return j, ok

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as ex:
            futures = {ex.submit(_fetch, j): j for j in jours_a_fetcher}
            try:
                for fut in as_completed(futures, timeout=120):
                    try:
                        j, ok = fut.result(timeout=60)
                        if not ok:
                            echecs.append((str(j), "Aucune donnée retournée par l'API"))
                    except Exception as exc:
                        echecs.append((str(futures[fut]), str(exc)))
            except Exception:
                for fut in futures:
                    fut.cancel()
                echecs += [
                    (str(futures[fut]), "Timeout — ENTSO-E n'a pas répondu")
                    for fut in futures if not fut.done()
                ]

        barre.empty()
        if echecs:
            with st.expander(f"⚠️ {len(echecs)} jour(s) en erreur"):
                for j, err in echecs:
                    st.write(f"**{j}** : {err}")

    df     = charger_depuis_db(start, end)
    nb_api = len(jours_a_fetcher) - len(echecs)
    return df, nb_cache, nb_api


with st.spinner("⏳ Chargement des données…"):
    try:
        df_brut, nb_cache, nb_api = charger_periode(start_date, end_date)
    except Exception as exc:
        st.error(f"Erreur : {exc}")
        st.stop()

if df_brut is None or df_brut.empty:
    st.error("Aucune donnée disponible pour cette période.")
    st.stop()

stats_cache.clear()
st.success(
    f"✅ Données chargées — {start_date} → {end_date}  ·  "
    f"☁️ {nb_cache} jour(s) depuis Neon  ·  "
    f"🌐 {nb_api} jour(s) téléchargé(s) depuis l'API"
)


# ══════════════════════════════════════════════════════════════════
# 6. TRAITEMENT
# ══════════════════════════════════════════════════════════════════

df_nuc = extraire_actual_aggregated(df_brut)
df_nuc = df_nuc.dropna(axis=1, how="all")
if df_nuc.columns.duplicated().any():
    df_nuc = df_nuc.T.groupby(level=0).max().T
df_nuc = df_nuc.resample("1h").mean().ffill().fillna(0)
df_nuc = df_nuc[sorted(df_nuc.columns)]

if df_nuc.empty or df_nuc.shape[1] == 0:
    st.error("Aucune donnée disponible après traitement.")
    with st.expander("Debug"):
        st.write(list(df_brut.columns)[:20])
    st.stop()

reacteurs     = df_nuc.columns.tolist()
serie_pnom    = pd.Series({r: PUISSANCE_NOMINALE_MW.get(r, max(df_nuc[r].max(), 900.0)) for r in reacteurs})
df_taux       = (df_nuc.div(serie_pnom) * 100).clip(upper=105)
taux_derniere = df_taux.iloc[-1]
prod_derniere = df_nuc.iloc[-1]
reacteurs_on  = int((taux_derniere >= SEUIL_ON_PCT).sum())
reacteurs_off = int((taux_derniere <  SEUIL_ON_PCT).sum())
taux_moyen    = taux_derniere[taux_derniere >= SEUIL_ON_PCT].mean()


# ══════════════════════════════════════════════════════════════════
# 7. MÉTRIQUES
# ══════════════════════════════════════════════════════════════════

st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("☢️ Production totale",       f"{prod_derniere.sum():,.0f} MW")
c2.metric("✅ En marche",               f"{reacteurs_on} réacteurs")
c3.metric("🔴 Arrêtés / < 5 %",        f"{reacteurs_off} réacteurs")
c4.metric("📊 Taux de charge moyen",    f"{taux_moyen:.1f} %")
c5.metric("⚡ Puissance nominale parc", f"{serie_pnom.sum() / 1e3:.1f} GW")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# 8. HEATMAP
# ══════════════════════════════════════════════════════════════════

st.subheader("🔲 Heatmap — Taux de charge par réacteur (% Pnom)")
st.caption("🟢 Vert = puissance nominale · ⚫ Noir = arrêt · 🟡 intermédiaire = modulation")

COLORSCALE = [
    [0.00, "rgb(5,5,5)"],      [0.04, "rgb(40,5,5)"],
    [0.15, "rgb(120,20,0)"],   [0.30, "rgb(180,60,0)"],
    [0.45, "rgb(200,120,0)"],  [0.60, "rgb(210,190,0)"],
    [0.75, "rgb(170,210,30)"], [0.88, "rgb(80,200,40)"],
    [0.95, "rgb(30,220,60)"],  [0.99, "rgb(10,230,70)"],
    [1.00, "rgb(0,255,80)"],
]

fig_heatmap = go.Figure(go.Heatmap(
    z=df_taux[reacteurs].T.values, x=df_taux.index, y=reacteurs,
    colorscale=COLORSCALE, zmin=0, zmax=100, hoverongaps=False,
    hovertemplate="%{y}<br>%{x}<br><b>%{z:.1f} % Pnom</b><extra></extra>",
    colorbar=dict(title="% Pnom", ticksuffix=" %", tickvals=[0,25,50,75,100], tickfont=dict(size=10)),
))
fig_heatmap.update_layout(
    xaxis_title="",
    yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    template="plotly_dark",
    height=max(420, len(reacteurs) * 14),
    margin=dict(l=140, r=90, t=20, b=40),
)
st.plotly_chart(fig_heatmap, use_container_width=True, theme=None)


# ══════════════════════════════════════════════════════════════════
# 9. SPARKLINES
# ══════════════════════════════════════════════════════════════════

st.subheader("📈 Courbes individuelles — Taux de charge par réacteur")
st.caption("🟢 Vert = en marche · 🔴 Rouge = arrêté · Axe Y = % Pnom (IAEA PRIS)")

n_rows_spark = max(1, math.ceil(len(reacteurs) / N_COLS_SPARKLINES))
titres = [f"{r}<br>{serie_pnom[r]:.0f} MW" for r in reacteurs]

fig_spark = make_subplots(
    rows=n_rows_spark, cols=N_COLS_SPARKLINES,
    subplot_titles=titres, shared_xaxes=True,
    vertical_spacing=0.03, horizontal_spacing=0.06,
)

for idx, reacteur in enumerate(reacteurs):
    row = idx // N_COLS_SPARKLINES + 1
    col = idx %  N_COLS_SPARKLINES + 1
    serie_pct = df_taux[reacteur]
    en_marche = serie_pct.iloc[-1] >= SEUIL_ON_PCT
    couleur   = "#00C853" if en_marche else "#E53935"
    fill_col  = "rgba(0,200,83,0.15)" if en_marche else "rgba(229,57,53,0.15)"
    fig_spark.add_trace(go.Scatter(
        x=serie_pct.index, y=serie_pct.values,
        mode="lines", line=dict(color=couleur, width=1.2),
        fill="tozeroy", fillcolor=fill_col,
        name=reacteur, showlegend=False,
        customdata=df_nuc[reacteur].values,
        hovertemplate=(
            f"<b>{reacteur}</b> (Pnom {serie_pnom[reacteur]:.0f} MW)<br>"
            "%{x}<br><b>%{customdata:.0f} MW</b> · %{y:.1f} % Pnom<extra></extra>"
        ),
    ), row=row, col=col)

shapes_hline = []
for idx in range(len(reacteurs)):
    ax_idx = idx + 1
    shapes_hline.append(dict(
        type="line", x0=0, x1=1, y0=100, y1=100,
        xref="x domain" if ax_idx == 1 else f"x{ax_idx} domain",
        yref="y"        if ax_idx == 1 else f"y{ax_idx}",
        line=dict(dash="dot", color="rgba(255,255,255,0.2)", width=0.8),
    ))

fig_spark.update_layout(
    template="plotly_dark", height=max(800, n_rows_spark * 200),
    hovermode="closest", margin=dict(l=30, r=20, t=60, b=20), shapes=shapes_hline,
)
fig_spark.update_annotations(font_size=9)
fig_spark.update_xaxes(showticklabels=False, showspikes=False, showgrid=False)
fig_spark.update_yaxes(
    showticklabels=True, ticksuffix="%", nticks=3,
    tickfont=dict(size=9, color="#CCCCCC"),
    gridcolor="rgba(180,180,180,0.3)", gridwidth=0.5,
    showgrid=True, zeroline=False, rangemode="tozero", showspikes=False,
)
st.plotly_chart(fig_spark, use_container_width=True, theme=None)


# ══════════════════════════════════════════════════════════════════
# 10. TABLEAU & TÉLÉCHARGEMENT
# ══════════════════════════════════════════════════════════════════

with st.expander("📋 Tableau — taux de charge par réacteur (dernière valeur)"):
    df_table = pd.DataFrame({
        "Pnom (MWe)"        : serie_pnom,
        "Production (MW)"   : prod_derniere.round(0),
        "Taux de charge (%)": taux_derniere.round(1),
        "État": taux_derniere.apply(lambda x: "✅ En marche" if x >= SEUIL_ON_PCT else "🔴 Arrêté"),
    }).sort_values("Taux de charge (%)", ascending=False)
    st.dataframe(df_table, use_container_width=True)

with st.expander("📋 Télécharger les données (taux de charge %)"):
    csv = df_taux.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ CSV — taux de charge horaire par réacteur", csv,
        file_name=f"modulation_nucleaire_FR_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

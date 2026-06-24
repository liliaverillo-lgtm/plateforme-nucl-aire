#!/usr/bin/env python3
"""
Dashboard — Modulation nucléaire par réacteur (France)
Normalisation par la puissance nominale IAEA PRIS

Logique de données :
  1. Lecture depuis nucleaire_FR_historique.parquet (même dossier que le script)
  2. Jours manquants dans le parquet → téléchargés depuis ENTSO-E API
  3. Nouveaux jours ajoutés au parquet pour les prochaines fois
  4. Traitement et affichage

Usage :
  pip install entsoe-py pandas pyarrow plotly streamlit
  streamlit run dashboard_electricite_france_optimise.py
"""

import warnings
warnings.filterwarnings("ignore")

import math
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from entsoe import EntsoePandasClient

# ═══════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
API_KEY           = "c5cb3857-bc40-4f4c-a4db-088946785b4a"
COUNTRY           = "FR"
TZ                = "Europe/Paris"
SEUIL_ON_PCT      = 5
N_COLS_SPARKLINES = 4
MAX_WORKERS_API   = 4

# Résolution du chemin vers le parquet — essaie plusieurs emplacements
# pour fonctionner aussi bien en local que sur Streamlit Cloud
def _trouver_parquet() -> Path:
    candidats = [
        Path(__file__).parent / "nucleaire_FR_historique.parquet",  # relatif au script
        Path.cwd() / "nucleaire_FR_historique.parquet",              # répertoire de travail
    ]
    for p in candidats:
        if p.exists():
            return p
    return candidats[0]  # chemin par défaut si aucun trouvé

FICHIER_PARQUET = _trouver_parquet()

PUISSANCE_NOMINALE_MW = {
    "BUGEY 2": 910,      "BUGEY 3": 910,      "BUGEY 4": 880,      "BUGEY 5": 880,
    "BLAYAIS 1": 910,    "BLAYAIS 2": 910,    "BLAYAIS 3": 910,    "BLAYAIS 4": 910,
    "CHINON 1": 905,     "CHINON 2": 905,     "CHINON 3": 905,     "CHINON 4": 905,
    "CRUAS 1": 915,      "CRUAS 2": 915,      "CRUAS 3": 915,      "CRUAS 4": 915,
    "DAMPIERRE 1": 890,  "DAMPIERRE 2": 890,  "DAMPIERRE 3": 890,  "DAMPIERRE 4": 890,
    "GRAVELINES 1": 910, "GRAVELINES 2": 910, "GRAVELINES 3": 910,
    "GRAVELINES 4": 910, "GRAVELINES 5": 910, "GRAVELINES 6": 910,
    "ST LAURENT 1": 915, "ST LAURENT 2": 915,
    "TRICASTIN 1": 915,  "TRICASTIN 2": 915,  "TRICASTIN 3": 915,  "TRICASTIN 4": 915,
    "FLAMANVILLE 1": 1310, "FLAMANVILLE 2": 1310, "FLAMANVILLE 3": 1630,
    "PALUEL 1": 1330,    "PALUEL 2": 1330,    "PALUEL 3": 1330,    "PALUEL 4": 1330,
    "ST ALBAN 1": 1335,  "ST ALBAN 2": 1335,
    "BELLEVILLE 1": 1310, "BELLEVILLE 2": 1310,
    "CATTENOM 1": 1300,  "CATTENOM 2": 1300,  "CATTENOM 3": 1300,  "CATTENOM 4": 1300,
    "GOLFECH 1": 1310,   "GOLFECH 2": 1310,
    "NOGENT 1": 1310,    "NOGENT 2": 1310,
    "PENLY 1": 1320,     "PENLY 2": 1320,
    "CHOOZ 1": 1500,     "CHOOZ 2": 1500,
    "CIVAUX 1": 1495,    "CIVAUX 2": 1495,
}

AUJOURDHUI = datetime.now().date()
HIER       = AUJOURDHUI - timedelta(days=1)


# ═══════════════════════════════════════════════════════════════════
# 1. PARQUET — lecture / écriture
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def lire_parquet() -> pd.DataFrame:
    """Lit le parquet local. Résultat mis en cache — invalidé par lire_parquet.clear()."""
    if not FICHIER_PARQUET.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(FICHIER_PARQUET)
    except Exception:
        return pd.DataFrame()


def jours_dans_parquet(df: pd.DataFrame) -> set:
    """Retourne l'ensemble des dates présentes dans le DataFrame."""
    if df.empty:
        return set()
    return set(pd.to_datetime(df.index).normalize().date)


def ajouter_au_parquet(df_nouveau: pd.DataFrame) -> None:
    """
    Fusionne df_nouveau dans le parquet existant et sauvegarde.
    Invalide le cache de lire_parquet() pour que la prochaine lecture
    reflète les nouvelles données.
    """
    if df_nouveau.empty:
        return
    df_existant = lire_parquet()
    df_final    = df_nouveau if df_existant.empty else pd.concat([df_existant, df_nouveau])
    df_final    = df_final[~df_final.index.duplicated(keep="last")].sort_index()
    df_final.to_parquet(FICHIER_PARQUET)
    lire_parquet.clear()  # invalide le cache


# ═══════════════════════════════════════════════════════════════════
# 2. API ENTSO-E
# ═══════════════════════════════════════════════════════════════════

def api_telecharger_jour(jour_str: str) -> pd.DataFrame:
    """Télécharge un jour depuis ENTSO-E (psr_type B14 = Nuclear)."""
    client   = EntsoePandasClient(api_key=API_KEY)
    start_ts = pd.Timestamp(jour_str + " 00:00", tz=TZ)
    end_ts   = pd.Timestamp(jour_str + " 23:59", tz=TZ)
    return client.query_generation_per_plant(
        country_code=COUNTRY, start=start_ts, end=end_ts, psr_type="B14"
    )


# ═══════════════════════════════════════════════════════════════════
# 3. CHARGEMENT — parquet en priorité, API pour les jours manquants
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def charger_periode(start_str: str, end_str: str) -> pd.DataFrame:
    """
    Pour chaque jour de [start, end] :
      - Présent dans le parquet → lecture locale (instantané)
      - Absent → téléchargement API (en parallèle) + ajout au parquet
    Retourne le DataFrame brut (MultiIndex colonnes) de toute la période.
    """
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end   = datetime.strptime(end_str,   "%Y-%m-%d").date()
    nb    = (end - start).days + 1

    df_parquet   = lire_parquet()
    jours_caches = jours_dans_parquet(df_parquet)

    # Séparer : jours déjà en cache / jours à télécharger
    jours_a_fetcher = []
    jour = start
    while jour <= end:
        if jour not in jours_caches:
            jours_a_fetcher.append(jour)
        jour += timedelta(days=1)

    # ── Extraire les jours présents depuis le parquet ──────────────
    morceaux = []
    if not df_parquet.empty and len(jours_a_fetcher) < nb:
        start_ts = pd.Timestamp(str(start), tz=TZ)
        end_ts   = pd.Timestamp(str(end) + " 23:59", tz=TZ)
        masque   = (df_parquet.index >= start_ts) & (df_parquet.index <= end_ts)
        morceaux.append(df_parquet.loc[masque])

    # ── Télécharger les jours manquants en parallèle ───────────────
    if jours_a_fetcher:
        barre    = st.progress(0.0, text="⏳ Téléchargement des jours manquants…")
        compteur = {"n": 0}
        echecs   = []

        def fetch_et_sauvegarder(j: date):
            df_j = api_telecharger_jour(str(j))
            if df_j is not None and not df_j.empty:
                ajouter_au_parquet(df_j)   # écrit dans le parquet immédiatement
            return df_j

        with ThreadPoolExecutor(max_workers=MAX_WORKERS_API) as pool:
            futures = {pool.submit(fetch_et_sauvegarder, j): j for j in jours_a_fetcher}
            for future in as_completed(futures):
                j = futures[future]
                try:
                    df_j = future.result()
                    if df_j is not None and not df_j.empty:
                        morceaux.append(df_j)
                except Exception as e:
                    echecs.append((str(j), str(e)))
                compteur["n"] += 1
                barre.progress(
                    compteur["n"] / len(jours_a_fetcher),
                    text=f"✅ {compteur['n']}/{len(jours_a_fetcher)} jours téléchargés"
                )

        barre.empty()
        if echecs:
            with st.expander(f"⚠️ {len(echecs)} jour(s) en erreur"):
                for j, err in echecs:
                    st.write(f"**{j}** : {err}")

    if not morceaux:
        return pd.DataFrame()

    df_total = pd.concat(morceaux)
    df_total = df_total[~df_total.index.duplicated(keep="last")].sort_index()
    return df_total


# ═══════════════════════════════════════════════════════════════════
# 4. INTERFACE
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="☢️ Modulation nucléaire France",
                   layout="wide", page_icon="☢️")
st.title("☢️ Modulation nucléaire par réacteur — France")
st.caption("Production normalisée par la puissance nominale (IAEA PRIS) · "
           "Source : parquet local + ENTSO-E API pour les jours manquants")

with st.sidebar:
    st.header("📅 Période")
    start_date = st.date_input("Début", value=HIER - timedelta(days=6), max_value=AUJOURDHUI)
    end_date   = st.date_input("Fin",   value=HIER,                     max_value=AUJOURDHUI)
    nb_jours   = (end_date - start_date).days + 1
    st.info(f"📆 {nb_jours} jour(s)")
    if nb_jours > 31:
        st.warning("⚠️ Au-delà de 31 jours, le premier chargement peut être long.")
    lancer = st.button("🔄 Rafraîchir", type="primary", use_container_width=True)

    st.markdown("---")
    df_info = lire_parquet()
    if df_info.empty:
        st.error("❌ Parquet non trouvé")
        st.caption(f"**Script :** `{Path(__file__).parent}`")
        st.caption(f"**CWD :** `{Path.cwd()}`")
        st.caption(f"**Chemin testé :** `{FICHIER_PARQUET}`")
        st.caption(f"**Fichier existe :** `{FICHIER_PARQUET.exists()}`")
    else:
        jours_dispo = jours_dans_parquet(df_info)
        st.caption(f"💾 **{len(jours_dispo)} jours** en cache local\n\n"
                   f"Du {min(jours_dispo)} au {max(jours_dispo)}")
    st.markdown("**Pnom** : IAEA PRIS · "
                "[pris.iaea.org](https://pris.iaea.org/pris/CountryStatistics/"
                "CountryDetails.aspx?current=FR)")

# Chargement automatique à la première visite
if "premier_chargement" not in st.session_state:
    st.session_state.premier_chargement = True
    lancer        = True
    premiere_visite = True
else:
    premiere_visite = False

if not lancer:
    st.stop()

if start_date > end_date:
    st.error("La date de début doit être antérieure à la date de fin.")
    st.stop()

# Sur clic explicite : vider le cache pour forcer un rechargement frais
if lancer and not premiere_visite:
    st.cache_data.clear()

# ─── Chargement ──────────────────────────────────────────────────
with st.spinner("⏳ Chargement…"):
    df_brut = charger_periode(str(start_date), str(end_date))

if df_brut is None or df_brut.empty:
    st.error("Aucune donnée disponible pour cette période.")
    st.stop()

# Résumé parquet vs API
df_apres      = lire_parquet()
jours_apres   = jours_dans_parquet(df_apres)
n_parquet     = sum(1 for j in pd.date_range(start_date, end_date).date if j in jours_apres)
n_api         = nb_jours - n_parquet
st.success(f"✅ {start_date} → {end_date} · "
           f"💾 {n_parquet} jour(s) depuis le parquet · "
           f"🌐 {n_api} jour(s) téléchargé(s) depuis l'API")


# ═══════════════════════════════════════════════════════════════════
# 5. TRAITEMENT
# ═══════════════════════════════════════════════════════════════════

def extraire_actual_aggregated(df: pd.DataFrame) -> pd.DataFrame:
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


df_nuc = extraire_actual_aggregated(df_brut)
df_nuc = df_nuc.dropna(axis=1, how="all")
if df_nuc.columns.duplicated().any():
    df_nuc = df_nuc.T.groupby(level=0).max().T
df_nuc = df_nuc.resample("1h").mean().ffill().fillna(0)
df_nuc = df_nuc[sorted(df_nuc.columns)]

if df_nuc.empty or df_nuc.shape[1] == 0:
    st.error("Aucune donnée après traitement.")
    st.stop()

reacteurs     = df_nuc.columns.tolist()
serie_pnom    = pd.Series(
    {r: PUISSANCE_NOMINALE_MW.get(r, max(df_nuc[r].max(), 900.0)) for r in reacteurs},
    name="Pnom (MWe)"
)
df_taux       = (df_nuc.div(serie_pnom) * 100).clip(upper=105)
taux_derniere = df_taux.iloc[-1]
prod_derniere = df_nuc.iloc[-1]
reacteurs_on  = int((taux_derniere >= SEUIL_ON_PCT).sum())
reacteurs_off = int((taux_derniere <  SEUIL_ON_PCT).sum())
taux_moyen    = taux_derniere[taux_derniere >= SEUIL_ON_PCT].mean()


# ═══════════════════════════════════════════════════════════════════
# 6. MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("☢️ Production totale",       f"{prod_derniere.sum():,.0f} MW")
c2.metric("✅ En marche",               f"{reacteurs_on} réacteurs")
c3.metric("🔴 Arrêtés / < 5 %",        f"{reacteurs_off} réacteurs")
c4.metric("📊 Taux de charge moyen",   f"{taux_moyen:.1f} %")
c5.metric("⚡ Puissance nominale parc", f"{serie_pnom.sum() / 1e3:.1f} GW")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════
# 7. HEATMAP
# ═══════════════════════════════════════════════════════════════════
st.subheader("🔲 Heatmap — Taux de charge par réacteur (% Pnom)")
st.caption("🟢 Vert = puissance nominale · ⚫ Noir = arrêt · 🟡 intermédiaire = modulation")

COLORSCALE = [
    [0.00, "rgb(5,5,5)"],      [0.04, "rgb(40,5,5)"],
    [0.15, "rgb(120,20,0)"],   [0.30, "rgb(180,60,0)"],
    [0.45, "rgb(200,120,0)"],  [0.60, "rgb(210,190,0)"],
    [0.75, "rgb(170,210,30)"], [0.88, "rgb(80,200,40)"],
    [0.95, "rgb(30,220,60)"],  [1.00, "rgb(0,255,80)"],
]
fig_heatmap = go.Figure(go.Heatmap(
    z=df_taux[reacteurs].T.values, x=df_taux.index, y=reacteurs,
    colorscale=COLORSCALE, zmin=0, zmax=100, hoverongaps=False,
    hovertemplate="%{y}<br>%{x}<br><b>%{z:.1f} % Pnom</b><extra></extra>",
    colorbar=dict(title="% Pnom", ticksuffix=" %",
                  tickvals=[0, 25, 50, 75, 100], tickfont=dict(size=10)),
))
fig_heatmap.update_layout(
    yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    template="plotly_dark",
    height=max(420, len(reacteurs) * 14),
    margin=dict(l=140, r=90, t=20, b=40),
)
st.plotly_chart(fig_heatmap, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════
# 8. SPARKLINES
# ═══════════════════════════════════════════════════════════════════
st.subheader("📈 Courbes individuelles — Taux de charge par réacteur")
st.caption("🟢 Vert = en marche · 🔴 Rouge = arrêté · Axe Y = % Pnom (IAEA PRIS)")

n_rows_spark = max(1, math.ceil(len(reacteurs) / N_COLS_SPARKLINES))
fig_spark    = make_subplots(
    rows=n_rows_spark, cols=N_COLS_SPARKLINES,
    subplot_titles=[f"{r}<br>{serie_pnom[r]:.0f} MW" for r in reacteurs],
    shared_xaxes=True, vertical_spacing=0.03, horizontal_spacing=0.06,
)

shapes_100pct = []
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
        hovertemplate=(f"<b>{reacteur}</b> (Pnom {serie_pnom[reacteur]:.0f} MW)<br>"
                       "%{x}<br><b>%{customdata:.0f} MW</b> · %{y:.1f} % Pnom<extra></extra>"),
    ), row=row, col=col)

    # Ligne de référence à 100 % (construites en une passe, plus rapide)
    n = idx + 1
    shapes_100pct.append(dict(
        type="line", x0=0, x1=1, y0=100, y1=100,
        xref=("x domain" if n == 1 else f"x{n} domain"),
        yref=("y"        if n == 1 else f"y{n}"),
        line=dict(dash="dot", color="rgba(255,255,255,0.2)", width=0.8),
    ))

fig_spark.update_layout(
    template="plotly_dark",
    height=max(800, n_rows_spark * 200),
    hovermode="closest",
    margin=dict(l=30, r=20, t=60, b=20),
    shapes=shapes_100pct,
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


# ═══════════════════════════════════════════════════════════════════
# 9. TABLEAU & TÉLÉCHARGEMENT
# ═══════════════════════════════════════════════════════════════════
with st.expander("📋 Tableau — taux de charge par réacteur (dernière valeur)"):
    df_table = pd.DataFrame({
        "Pnom (MWe)"         : serie_pnom,
        "Production (MW)"    : prod_derniere.round(0),
        "Taux de charge (%)" : taux_derniere.round(1),
        "État"               : taux_derniere.apply(
            lambda x: "✅ En marche" if x >= SEUIL_ON_PCT else "🔴 Arrêté"
        ),
    }).sort_values("Taux de charge (%)", ascending=False)
    st.dataframe(df_table, use_container_width=True)

with st.expander("📋 Télécharger les données (taux de charge %)"):
    st.download_button(
        "⬇️ CSV — taux de charge horaire par réacteur",
        df_taux.to_csv().encode("utf-8"),
        file_name=f"modulation_nucleaire_FR_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

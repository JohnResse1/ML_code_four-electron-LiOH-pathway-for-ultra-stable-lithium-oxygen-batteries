#!/usr/bin/env python3
"""Export exact CSV source data for panels a-f of the 3x2 overview figure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproducibility import DESCRIPTOR_COLUMNS, add_record_ids  # noqa: E402

OUT = ROOT / "results" / "checklist_reproduction" / "comprehensive_3x2_source_data"
METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
    "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu",
}


def elements(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [token for token in str(value).replace(" ", "").split("-") if token and token != "**"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = add_record_ids(pd.read_csv(ROOT / "dataset.csv"))
    frame["Elements"] = frame["Compounds(M1-M2-N)"].map(elements)
    y1 = pd.to_numeric(frame["Overpotential(V)"], errors="coerce")
    y2 = pd.to_numeric(frame["Cycling_performance"], errors="coerce")
    x = frame[DESCRIPTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    pd.DataFrame({"record_id": frame.record_id, "Overpotential(V)": y1}).dropna().to_csv(
        OUT / "panel_a_overpotential_distribution.csv", index=False
    )

    counts = pd.Series([item for row in frame.Elements for item in row]).value_counts()
    plotted_elements = [item for item in counts.index if item in METALS]
    if len(plotted_elements) < 2:
        plotted_elements = counts.index.tolist()
    index = {element: i for i, element in enumerate(plotted_elements)}
    cooccurrence = np.zeros((len(index), len(index)), dtype=int)
    for row in frame.Elements:
        positions = [index[item] for item in row if item in index]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                cooccurrence[positions[i], positions[j]] += 1
                cooccurrence[positions[j], positions[i]] += 1
    co_long = (
        pd.DataFrame(cooccurrence, index=plotted_elements, columns=plotted_elements)
        .rename_axis("element_1").reset_index()
        .melt(id_vars="element_1", var_name="element_2", value_name="cooccurrence_count")
    )
    co_long.to_csv(OUT / "panel_b_element_cooccurrence.csv", index=False)

    groups = frame.Elements.map(lambda row: "-".join(sorted(set(row))) if row else "Unknown")
    panel_c = groups.value_counts().drop(labels="Unknown", errors="ignore").head(20).rename("frequency").reset_index()
    panel_c.columns = ["element_group", "frequency"]
    panel_c["cumulative_count"] = panel_c.frequency.cumsum()
    panel_c["cumulative_percent"] = 100 * panel_c.cumulative_count / panel_c.frequency.sum()
    panel_c.to_csv(OUT / "panel_c_element_group_pareto.csv", index=False)

    pd.DataFrame({"record_id": frame.record_id, "Overpotential(V)": y1, "Cycling_performance": y2}).dropna().to_csv(
        OUT / "panel_d_overpotential_vs_cycling.csv", index=False
    )

    ranking = []
    for column in DESCRIPTOR_COLUMNS:
        rho, p_value = spearmanr(x[column], y1.fillna(y1.median()))
        ranking.append({"feature": column, "spearman_rho_for_selection": rho, "p_value": p_value, "abs_rho": abs(rho)})
    ranking = pd.DataFrame(ranking).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT / "panel_e_feature_selection_ranking.csv", index=False)
    top = ranking.head(15).feature.tolist()
    correlation_input = x.copy()
    correlation_input["Overpotential(V)"] = y1.fillna(y1.median()).values
    correlation_input[top + ["Overpotential(V)"]].corr(method="pearson").to_csv(
        OUT / "panel_e_pearson_correlation_matrix.csv", index_label="variable"
    )

    normalized = StandardScaler().fit_transform(x)
    perplexity = max(5, min(30, (len(frame) - 1) // 3))
    embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=42).fit_transform(normalized)
    pd.DataFrame({
        "record_id": frame.record_id,
        "tSNE_1": embedded[:, 0],
        "tSNE_2": embedded[:, 1],
        "Overpotential(V)": y1,
    }).to_csv(OUT / "panel_f_tsne_embedding.csv", index=False)

    print(f"Exported 7 CSV files to {OUT.relative_to(ROOT)}")
    for path in sorted(OUT.glob("*.csv")):
        print(f"{path.name}: {len(pd.read_csv(path))} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Streamlit app for lichen richness/density:
- Choose Region (USA, CA, LA County, Pasadena)
- Time filter (year range)
- Taxon filters (phylum, class, genus, species)
- Grid size and metric (records vs species richness)
- Map contrast (none / percentile / log)
- Species list for current view, with optional subset display

Run:
  streamlit run streamlit_app.py -- --data-dir /path/to/data_dir
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import Polygon

# ---------- helpers (same contrast function as before) ----------

def apply_contrast(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        return np.log10(values.astype(float) + 1.0)
    if mode == "pct":
        vmin, vmax = np.percentile(values, 2), np.percentile(values, 98)
        return np.clip(values, vmin, vmax)
    return values

def build_equal_angle_grid(bbox, cell):
    import math
    import geopandas as gpd
    from shapely.geometry import Polygon

    xmin, ymin, xmax, ymax = bbox
    lon0 = (np.floor(xmin / cell) * cell)
    lat0 = (np.floor(ymin / cell) * cell)
    lon_edges = np.arange(lon0, xmax + cell, cell)
    lat_edges = np.arange(lat0, ymax + cell, cell)

    recs = []
    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            poly = Polygon([
                (lon_edges[j],     lat_edges[i]),
                (lon_edges[j + 1], lat_edges[i]),
                (lon_edges[j + 1], lat_edges[i + 1]),
                (lon_edges[j],     lat_edges[i + 1]),
            ])
            recs.append({"row": i, "col": j, "geometry": poly})
    gdf = gpd.GeoDataFrame(recs, crs=4326)
    gdf.attrs.update(lon0=lon0, lat0=lat0, cell=cell,
                     lon_edges=lon_edges, lat_edges=lat_edges)
    return gdf

@st.cache_data(show_spinner=False)
def load_regions(data_dir: Path) -> Dict[str, gpd.GeoDataFrame]:
    regions = {}
    for name in ("USA", "California", "Los_Angeles_County", "Pasadena"):
        gpkg = data_dir / "regions" / f"{name}.gpkg"
        if gpkg.exists():
            regions[name] = gpd.read_file(gpkg)
    return regions

@st.cache_data(show_spinner=False)
def list_years(data_dir: Path, region: str) -> list[int]:
    root = data_dir / "occurrences.parquet" / f"region={region}"
    years = []
    if root.exists():
        for ydir in sorted(root.glob("year=*")):
            try:
                years.append(int(ydir.name.split("=")[1]))
            except Exception:
                pass
    return years

@st.cache_data(show_spinner=False)
def load_species_catalog(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "species_catalog.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["speciesKey","species","genus","family","order","class","phylum"])

@st.cache_data(show_spinner=True)
def load_points(data_dir: Path, region: str, years: list[int]) -> pd.DataFrame:
    """Load points for selected region & years, concatenated."""
    root = data_dir / "occurrences.parquet" / f"region={region}"
    dfs = []
    for y in years:
        part = root / f"year={y}" / "part.parquet"
        if part.exists():
            dfs.append(pd.read_parquet(part))
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        return df
    return pd.DataFrame(
        columns=["gbifID","speciesKey","species","genus","family","order","class","phylum",
                 "decimalLatitude","decimalLongitude","eventDate","year","basisOfRecord"]
    )

def make_grid_agg(df: pd.DataFrame, poly: gpd.GeoDataFrame, cell: float,
                  tax_filters: dict | None) -> gpd.GeoDataFrame:
    import geopandas as gpd
    # Taxon filters
    if tax_filters:
        for k, vals in tax_filters.items():
            if vals:
                df = df[df[k].isin(vals)]
    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry","n_records","n_species"], geometry="geometry", crs=4326)

    # bbox and grid
    bbox = tuple(poly.total_bounds)
    grid = build_equal_angle_grid(bbox, cell)
    # fast prefilter by bbox
    lat = df["decimalLatitude"].to_numpy()
    lon = df["decimalLongitude"].to_numpy()
    mask = (lat >= bbox[1]) & (lat <= bbox[3]) & (lon >= bbox[0]) & (lon <= bbox[2])
    df = df.loc[mask, ["speciesKey","decimalLatitude","decimalLongitude"]]
    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry","n_records","n_species"], geometry="geometry", crs=4326)

    # bin to indices
    rows = np.floor((df["decimalLatitude"].to_numpy() - grid.attrs["lat0"]) / cell).astype(int)
    cols = np.floor((df["decimalLongitude"].to_numpy() - grid.attrs["lon0"]) / cell).astype(int)
    nrows = int(round((grid.attrs["lat_edges"][-1] - grid.attrs["lat_edges"][0]) / cell))
    ncols = int(round((grid.attrs["lon_edges"][-1] - grid.attrs["lon_edges"][0]) / cell))
    valid = (rows>=0) & (rows<nrows) & (cols>=0) & (cols<ncols)

    if not valid.any():
        return gpd.GeoDataFrame(columns=["geometry","n_records","n_species"], geometry="geometry", crs=4326)

    idx = rows[valid] * ncols + cols[valid]
#    spk = df["speciesKey"].to_numpy()[valid].astype("int64")
    # Drop NaNs and safely cast to int64
    spk_raw = df["speciesKey"].to_numpy()[valid]
    spk = spk_raw[~pd.isna(spk_raw)].astype("int64", copy=False)

    # aggregate
    from collections import defaultdict
    counts = defaultdict(int)
    spp = defaultdict(set)
    for i, k in zip(idx, spk):
        counts[i] += 1
        spp[i].add(int(k))

    # build geodf
    grid = gpd.overlay(grid, poly, how="intersection", keep_geom_type=False)
    flat_index = (grid["row"].to_numpy() * ncols) + grid["col"].to_numpy()
    n_records = np.array([counts.get(i, 0) for i in flat_index], dtype=int)
    n_species = np.array([len(spp.get(i, set())) for i in flat_index], dtype=int)
    grid["n_records"] = n_records
    grid["n_species"] = n_species
    return grid


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="Lichen Explorer", layout="wide")
    st.title("🧫 Lichen Explorer — USA / CA / LA / Pasadena")
    st.markdown(
        "<div style='text-align:center; font-size:14px; color:gray;'>"
        "Created by <b>Ashish Mahabal</b> with GBIF data using ChatGPT 5"
        "</div>",
        unsafe_allow_html=True,
    )


#    # Parse CLI flag passed after '--'
#    parser = argparse.ArgumentParser(add_help=False)
#    parser.add_argument("--data-dir", required=True)
#    args, _ = parser.parse_known_args()
#    data_dir = Path(args.data_dir)

    import os

    # Detect data directory automatically
    default_data = Path(__file__).parent / "data_dir"
    data_dir_env = os.getenv("DATA_DIR", str(default_data))
    
    data_dir = Path(data_dir_env)
    if not data_dir.exists():
        st.error(f"Data directory not found: {data_dir}")
        st.stop()


    regions = load_regions(data_dir)
    if not regions:
        st.error("No regions found. Did you run the preprocessor and point --data-dir correctly?")
        st.stop()

    # Sidebar — region and years
    region = st.sidebar.selectbox("Region", list(regions.keys()), index=list(regions.keys()).index("California") if "California" in regions else 0)
    years_available = list_years(data_dir, region)
    if not years_available:
        st.warning(f"No yearly partitions found for region={region}.")
        st.stop()

    year_min, year_max = min(years_available), max(years_available)
    year_range = st.sidebar.slider("Year range", min_value=int(year_min), max_value=int(year_max),
                                   value=(int(max(year_min, year_max-10)), int(year_max)), step=1)
    years_selected = [y for y in years_available if year_range[0] <= y <= year_range[1]]

    # ---- Robust taxonomy filters ----
    cat = load_species_catalog(data_dir)
    
    def _subset(df: pd.DataFrame, col: str, selected: list[str] | list[int] | set) -> pd.DataFrame:
        """Return df filtered by df[col] ∈ selected if selected is non-empty; else df unchanged."""
        if selected:
            return df[df[col].isin(selected)]
        return df
    
    # Start from full catalog; progressively narrow it based on prior selections
    phyla_all   = sorted(cat["phylum"].dropna().unique().tolist())
    phyla_sel   = st.sidebar.multiselect("Phylum", phyla_all, default=[])
    
    cat_p       = _subset(cat, "phylum", phyla_sel)
    
    classes_all = sorted(cat_p["class"].dropna().unique().tolist())
    classes_sel = st.sidebar.multiselect("Class", classes_all, default=[])
    
    cat_pc      = _subset(cat_p, "class", classes_sel)
    
    genera_all  = sorted(cat_pc["genus"].dropna().unique().tolist())
    genera_sel  = st.sidebar.multiselect("Genus", genera_all, default=[])
    
    cat_pcg     = _subset(cat_pc, "genus", genera_sel)
    
    species_choices = (
        cat_pcg[["speciesKey", "species"]]
        .dropna(subset=["speciesKey", "species"])
        .drop_duplicates()
        .sort_values("species")
    )
    species_map = dict(zip(species_choices["species"], species_choices["speciesKey"]))
    species_names_sel = st.sidebar.multiselect("Species (optional)", list(species_map.keys()), default=[])
    
    tax_filters: dict = {}
    if phyla_sel:
        tax_filters["phylum"] = phyla_sel
    if classes_sel:
        tax_filters["class"] = classes_sel
    if genera_sel:
        tax_filters["genus"] = genera_sel
    if species_names_sel:
        sel_keys = {species_map[n] for n in species_names_sel}
        tax_filters["speciesKey"] = sel_keys  # applied to points before grid-agg


    cell = st.sidebar.select_slider("Grid size (deg)", options=[0.5, 0.25, 0.1, 0.05, 0.02, 0.01], value=0.05)
    metric = st.sidebar.radio("Metric", ["Species richness (unique species)", "Occurrence density (records)"])
    contrast = st.sidebar.radio("Contrast", ["pct", "log", "none"], index=0)

    # Load points
    df = load_points(data_dir, region, years_selected)

    # Apply explicit species selection first (if any)
    if "speciesKey" in tax_filters:
        df = df[df["speciesKey"].isin(tax_filters["speciesKey"])]
        tax_filters.pop("speciesKey", None)  # remaining filters handled inside make_grid_agg

    if df.empty:
        st.warning("No points for selected filters.")
        st.stop()

    # Apply speciesKey filter if provided
    if "speciesKey" in tax_filters:
        df = df[df["speciesKey"].isin(tax_filters["speciesKey"])]
        tax_filters.pop("speciesKey", None)

    # Region polygon and grid-agg
    poly = regions[region]
    grid = make_grid_agg(df, poly, cell=float(cell), tax_filters=tax_filters)

    # Plot
    if grid.empty:
        st.warning("No cells after filters.")
        st.stop()

    col = "n_species" if metric.startswith("Species") else "n_records"
    grid = grid.copy()
    grid["_plot"] = apply_contrast(grid[col].to_numpy(), contrast)

    # Static map (geopandas -> matplotlib) inside Streamlit
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection=proj)
    xmin, ymin, xmax, ymax = poly.total_bounds
    pad_x, pad_y = 0.05 * (xmax - xmin), 0.05 * (ymax - ymin)
    ax.set_extent([xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y], crs=proj)
    grid.plot(ax=ax, transform=proj, column="_plot", legend=True, edgecolor="none", cmap="viridis", alpha=0.92)
    poly.boundary.plot(ax=ax, transform=proj, linewidth=0.8, color="black")
#    ax.set_title(f"{metric} — {region} — {year_range[0]}–{year_range[1]} (grid={cell}°)")
#    st.pyplot(fig, clear_figure=True)
    # Main title
    ax.set_title(
        f"{metric} — {region} — {year_range[0]}–{year_range[1]} (grid={cell}°)",
        fontsize=13,
        weight="bold"
    )

    # Credit line
    plt.text(
        0.5, -0.12,
        "Created by Ashish Mahabal with GBIF data using ChatGPT 5",
        fontsize=9,
        ha="center",
        transform=ax.transAxes,
        color="gray",
        alpha=0.8
    )

    st.pyplot(fig, clear_figure=True)


    # Species list in current selection (fast)
    # We recompute species list using the same filters & bounding box of the region
    with st.expander("Show species currently in view"):
        spp = df
        if tax_filters:
            for k, vals in tax_filters.items():
                if vals:
                    spp = spp[spp[k].isin(vals)]
        spp = spp[["speciesKey","species","genus","family","order","class","phylum"]].dropna(subset=["speciesKey"]).drop_duplicates().sort_values("species")
        st.write(f"{len(spp)} species in current view.")
        st.dataframe(spp, use_container_width=True)

    # Download buttons
    import io

    # --- Gridded GeoParquet ---
    buf_parquet = io.BytesIO()
    grid.to_parquet(buf_parquet, index=False)
    st.download_button(
        "Download gridded GeoParquet",
        data=buf_parquet.getvalue(),
        file_name=f"lichen_grid_{region}_{year_range[0]}_{year_range[1]}.parquet",
        mime="application/octet-stream",
    )

    # --- Species CSV ---
    buf_csv = io.StringIO()
    spp.to_csv(buf_csv, index=False)
    st.download_button(
        "Download species list (CSV)",
        data=buf_csv.getvalue(),
        file_name=f"species_{region}_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()


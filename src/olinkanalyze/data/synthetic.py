"""Synthetic NPX datasets used as ``npx_data1`` and ``npx_data2``."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from olinkanalyze._constants import BRIDGE_SAMPLE_IDS
from olinkanalyze._utils import attach_check_log

_DATA_DIR = Path(__file__).resolve().parent

_ASSAYS = (
    ("OID20001", "P05231", "IL6", "Olink Cardiometabolic"),
    ("OID20002", "P01375", "TNF", "Olink Cardiometabolic"),
    ("OID20003", "P10145", "CXCL8", "Olink Cardiometabolic"),
    ("OID20004", "P13500", "CCL2", "Olink Cardiometabolic"),
    ("OID20005", "Q14116", "IL18", "Olink Cardiometabolic"),
    ("OID20006", "P01133", "EGF", "Olink Cardiometabolic"),
    ("OID20007", "P15692", "VEGFA", "Olink Cardiometabolic"),
    ("OID20008", "P01137", "TGFB1", "Olink Cardiometabolic"),
    ("OID20009", "P01584", "IL1B", "Olink Cardiometabolic"),
    ("OID20010", "P05112", "IL4", "Olink Cardiometabolic"),
    ("OID20011", "P05231", "IL6", "Olink Inflammation"),
    ("OID20012", "P01375", "TNF", "Olink Inflammation"),
    ("OID20013", "P10145", "CXCL8", "Olink Inflammation"),
    ("OID20014", "P13500", "CCL2", "Olink Inflammation"),
    ("OID20015", "Q14116", "IL18", "Olink Inflammation"),
    ("OID20016", "P01133", "EGF", "Olink Inflammation"),
    ("OID20017", "P15692", "VEGFA", "Olink Inflammation"),
    ("OID20018", "P01137", "TGFB1", "Olink Inflammation"),
    ("OID20019", "P01584", "IL1B", "Olink Inflammation"),
    ("OID20020", "P05112", "IL4", "Olink Inflammation"),
)

# Assays with a planted Treatment effect (Treated higher NPX).
_EFFECT_ASSAYS = {"OID20001", "OID20002", "OID20003", "OID20011", "OID20012"}
_PROJECT_SHIFT = {"data1": 0.0, "data2": 0.8}


def _load_meta(project: str) -> pd.DataFrame:
    filename = "npx_data1_meta.csv" if project == "data1" else "npx_data2_meta.csv"
    path = _DATA_DIR / filename
    if path.exists():
        meta = pd.read_csv(path, sep=";")
        meta.columns = [c.strip().strip('"') for c in meta.columns]
        for column in meta.columns:
            if meta[column].dtype == object:
                meta[column] = meta[column].astype(str).str.strip().str.strip('"')
                meta.loc[meta[column].isin({"", "nan", "None"}), column] = pd.NA
        return meta
    return _fallback_meta(project)


def _fallback_meta(project: str) -> pd.DataFrame:
    subjects = [f"ID{i}" for i in range(1, 21)]
    times = ["Baseline", "Week.6", "Week.12"]
    rows = []
    idx = 1
    for subject in subjects:
        treatment = "Treated" if int(subject[2:]) > 10 else "Untreated"
        for time in times:
            rows.append(
                {
                    "SampleID": f"A{idx}",
                    "Subject": subject,
                    "Treatment": treatment,
                    "Site": "Site_A",
                    "Time": time,
                    "Project": "20200001" if project == "data1" else "20200002",
                }
            )
            idx += 1
    for control in ("CONTROL_SAMPLE_AS 1", "CONTROL_SAMPLE_AS 2"):
        rows.append(
            {
                "SampleID": control,
                "Subject": pd.NA,
                "Treatment": pd.NA,
                "Site": pd.NA,
                "Time": pd.NA,
                "Project": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def make_npx_data(project: str = "data1", seed: int = 1) -> pd.DataFrame:
    """Build a compact long-format NPX table with metadata columns."""
    rng = np.random.default_rng(seed)
    meta = _load_meta(project)
    if "SampleID" not in meta.columns:
        raise ValueError("Metadata file is missing SampleID.")

    # Keep the well-known bridge samples in both projects when present.
    if project == "data2":
        extra = pd.DataFrame(
            {
                "SampleID": [sid for sid in BRIDGE_SAMPLE_IDS if sid not in set(meta["SampleID"])],
                "Subject": pd.NA,
                "Treatment": "Untreated",
                "Site": "Site_B",
                "Time": "Baseline",
                "Project": "20200002",
            }
        )
        if not extra.empty:
            meta = pd.concat([meta, extra], ignore_index=True)

    rows: list[dict] = []
    shift = _PROJECT_SHIFT[project]
    for sample in meta.itertuples(index=False):
        sample_id = str(sample.SampleID)
        is_control = "CONTROL" in sample_id.upper()
        treatment = getattr(sample, "Treatment", pd.NA)
        for index, (oid, uniprot, assay, panel) in enumerate(_ASSAYS, start=1):
            npx = float(rng.normal(0.0, 0.8)) + shift
            if not is_control and str(treatment) == "Treated" and oid in _EFFECT_ASSAYS:
                npx += 1.6
            qc = "PASS"
            if sample_id.endswith("1") and index == 1:
                qc = "FAIL"
            rows.append(
                {
                    "SampleID": sample_id,
                    "Index": index,
                    "OlinkID": oid,
                    "UniProt": uniprot,
                    "Assay": assay,
                    "MissingFreq": 0.0,
                    "Panel": panel,
                    "Panel_Lot_Nr": "B04401",
                    "PlateID": "P1",
                    "QC_Warning": qc,
                    "LOD": -3.0,
                    "NPX": npx,
                    "Normalization": "Intensity",
                    "Assay_Warning": "PASS",
                    "AssayType": "assay",
                    "SampleType": "CONTROL" if is_control else "SAMPLE",
                    "ExploreVersion": "1.7.0",
                    "Subject": getattr(sample, "Subject", pd.NA),
                    "Treatment": treatment,
                    "Site": getattr(sample, "Site", pd.NA),
                    "Time": getattr(sample, "Time", pd.NA),
                    "Project": getattr(sample, "Project", pd.NA),
                }
            )

    # Duplicate control samples to match the bundled example datasets.
    frame = pd.DataFrame(rows)
    controls = frame.loc[frame["SampleID"].str.contains("CONTROL", case=False, na=False)]
    if not controls.empty:
        frame = pd.concat([frame, controls], ignore_index=True)

    from olinkanalyze.qc.check_npx import check_npx
    from olinkanalyze.qc.column_names import categorize_npx_columns

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = check_npx(frame)
    frame = categorize_npx_columns(frame, col_names=log["col_names"])
    return attach_check_log(frame, log, copy=False)

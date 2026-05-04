"""
data_preprocessing.py — Schema-agnostic data cleaning & feature extraction.
Auto-detects student name, department, attendance, marks, and subject columns.
NOW SUPPORTS long-format data (one row per subject) → auto-pivots to wide format.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from utils import (
    COLUMN_ALIASES,
    detect_column,
    detect_subject_columns,
    normalise_col,
)

logger = logging.getLogger("university_agent.preprocessing")


# ── Public API ────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> dict:
    """
    Clean df and extract a metadata dictionary used by all other modules.
    If data is in long format (one row per subject), auto-pivots to wide format.
    """
    df = df.copy()
    df = _coerce_numeric(df)

    meta: dict = {}

    # ── Detect if data is long format ─────────────────────────────────────────
    long_info = _detect_long_format(df)
    meta["is_long_format"]    = long_info["is_long"]
    meta["subject_col_long"]  = long_info.get("subject_col")
    meta["marks_col_long"]    = long_info.get("marks_col")
    meta["df_long"]           = df.copy()

    if long_info["is_long"]:
        logger.info("Long format detected — pivoting to wide format...")
        df = _pivot_long_to_wide(df, long_info)
        logger.info("Pivoted shape: %s", df.shape)

    df = _fill_missing(df)

    # ── Detect semantic columns ───────────────────────────────────────────────
    meta["name_col"]   = detect_column(df, "student_name")
    meta["dept_col"]   = detect_column(df, "department")
    meta["attend_col"] = detect_column(df, "attendance")
    meta["roll_col"]   = detect_column(df, "roll_no")
    meta["year_col"]   = detect_column(df, "year")

    meta["subject_cols"]  = detect_subject_columns(df)
    meta["numeric_cols"]  = list(df.select_dtypes(include=[np.number]).columns)
    meta["df"]            = df

    # Count unique students
    id_col = meta.get("roll_col") or meta.get("name_col")
    meta["n_students"] = int(df[id_col].nunique()) if id_col else len(df)

    # Department stats
    if meta["dept_col"]:
        depts = df[meta["dept_col"]].dropna().unique().tolist()
        meta["departments"]   = [str(d) for d in depts]
        meta["n_departments"] = len(depts)
    else:
        meta["departments"]   = []
        meta["n_departments"] = 0

    meta["has_attendance"] = meta["attend_col"] is not None
    meta["has_subjects"]   = len(meta["subject_cols"]) > 0

    _add_derived_columns(df, meta)

    logger.info(
        "Preprocessing done — students=%d  depts=%d  subjects=%d  long_format=%s",
        meta["n_students"], meta["n_departments"],
        len(meta["subject_cols"]), meta["is_long_format"],
    )
    return meta


# ── Long format detection ─────────────────────────────────────────────────────
def _detect_long_format(df: pd.DataFrame) -> dict:
    subject_col = None
    marks_col   = None

    subject_hints = ["subject", "course", "module", "paper", "subject_name"]
    marks_hints   = ["marks", "score", "mark", "obtained"]

    for col in df.columns:
        norm = normalise_col(col)
        if any(h == norm or norm.startswith(h) for h in subject_hints):
            subject_col = col
            break

    for col in df.columns:
        norm = normalise_col(col)
        if any(h == norm or norm.startswith(h) for h in marks_hints):
            if pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.7:
                marks_col = col
                break

    if subject_col is None or marks_col is None:
        return {"is_long": False}

    # Find student ID column
    id_col = None
    id_hints = ["student_id", "roll_no", "rollno", "roll", "id", "sid", "reg"]
    for hint in id_hints:
        for col in df.columns:
            if hint in normalise_col(col):
                id_col = col
                break
        if id_col:
            break

    # Fallback: use name column
    if id_col is None:
        for col in df.columns:
            if "name" in normalise_col(col):
                id_col = col
                break

    if id_col:
        n_rows   = len(df)
        n_unique = df[id_col].nunique()
        if n_rows / max(n_unique, 1) > 2:
            return {
                "is_long": True,
                "subject_col": subject_col,
                "marks_col":   marks_col,
                "id_col":      id_col,
            }

    return {"is_long": False}


def _pivot_long_to_wide(df: pd.DataFrame, long_info: dict) -> pd.DataFrame:
    subject_col = long_info["subject_col"]
    marks_col   = long_info["marks_col"]
    id_col      = long_info["id_col"]

    # Columns to carry forward (everything except subject name & marks value)
    carry_cols = [c for c in df.columns if c not in [subject_col, marks_col]]

    try:
        pivoted = df.pivot_table(
            index=id_col,
            columns=subject_col,
            values=marks_col,
            aggfunc="mean",
        ).reset_index()
        pivoted.columns.name = None

        # Merge back meta columns
        info_cols = [c for c in carry_cols if c != id_col]
        if info_cols:
            info_df = df.groupby(id_col)[info_cols].first().reset_index()
            pivoted = pivoted.merge(info_df, on=id_col, how="left")

    except Exception as e:
        logger.error("Pivot failed: %s", e)
        return df

    logger.info("Pivot complete: %d students x %d columns", len(pivoted), len(pivoted.columns))
    return pivoted


# ── Internal helpers ──────────────────────────────────────────────────────────
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.7:
            df[col] = converted
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    return df


def _add_derived_columns(df: pd.DataFrame, meta: dict) -> None:
    scols = meta["subject_cols"]
    if not scols:
        return

    if "Total" not in df.columns:
        df["Total"] = df[scols].sum(axis=1)
        meta["numeric_cols"].append("Total")

    if "Average" not in df.columns:
        df["Average"] = df[scols].mean(axis=1).round(2)
        meta["numeric_cols"].append("Average")

    if "Grade" not in df.columns:
        df["Grade"] = df["Average"].apply(_grade)

    meta["df"] = df


def _grade(avg: float) -> str:
    if avg >= 90:  return "A+"
    if avg >= 80:  return "A"
    if avg >= 70:  return "B"
    if avg >= 60:  return "C"
    if avg >= 50:  return "D"
    return "F"


# ── Convenience re-export ─────────────────────────────────────────────────────
def get_student_row(meta: dict, name: str) -> pd.DataFrame:
    df, name_col = meta["df"], meta["name_col"]
    if name_col is None:
        return pd.DataFrame()
    mask = df[name_col].astype(str).str.lower().str.contains(
        re.escape(name.lower()), na=False
    )
    return df[mask].reset_index(drop=True)

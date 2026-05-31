"""
Excel workbook builder (pandas + xlsxwriter).

Given an ordered list of "sections" (each a named DataFrame, optionally with a
chart spec), this writes a polished workbook with:
  - a Legend / data-dictionary sheet first,
  - a hover comment on every column header,
  - frozen header row, auto-filter, sensible column widths and number formats,
  - optional embedded charts.

It never touches a database. It only formats data already pulled by queries.py.
"""

import pandas as pd

from column_definitions import define


def section(name, fn, *args, charts=None):
    """
    Run one query function and package it as a section for write_workbook().
    If the query fails, the section is marked with an error string so the
    workbook shows a clear note on that sheet instead of crashing.
    """
    try:
        df = fn(*args)
        return {"name": name, "df": df, "charts": charts or []}
    except Exception as e:  # noqa: BLE001 - we want to capture anything
        return {"name": name, "df": None, "error": f"{type(e).__name__}: {e}"}


# Columns whose values should render as percentages / minutes.
_PERCENT_HINT = ("%",)
_MINUTE_HINT = ("Minutes", "Median")


def _is_percent_col(name):
    return any(h in name for h in _PERCENT_HINT)


def _is_minute_col(name):
    return any(name.endswith(h) or h in name for h in _MINUTE_HINT)


def _build_legend_df(sections):
    """Build the data-dictionary rows from every column actually written."""
    seen = set()
    rows = []
    for sec in sections:
        df = sec["df"]
        if df is None:
            continue
        for col in df.columns:
            key = (sec["name"], col)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "Sheet": sec["name"],
                "Column": col,
                "Meaning": define(col) or "(no description)",
            })
    return pd.DataFrame(rows, columns=["Sheet", "Column", "Meaning"])


def write_workbook(path, sections, title):
    """
    path     : output .xlsx path
    sections : list of dicts: {
                   "name": sheet name,
                   "df": DataFrame (or None if the query failed),
                   "error": optional error string,
                   "charts": optional list of chart specs (see _add_chart),
               }
    title    : workbook title shown on the Legend sheet
    """
    writer = pd.ExcelWriter(path, engine="xlsxwriter")
    book = writer.book

    # --- reusable cell formats ---
    fmt_header = book.add_format({
        "bold": True, "bg_color": "#1F4E78", "font_color": "white",
        "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True,
    })
    fmt_title = book.add_format({"bold": True, "font_size": 14})
    fmt_sub = book.add_format({"italic": True, "font_color": "#666666"})
    fmt_pct = book.add_format({"num_format": "0.0%"})
    fmt_min = book.add_format({"num_format": "0.00"})
    fmt_int = book.add_format({"num_format": "#,##0"})
    fmt_wrap = book.add_format({"text_wrap": True, "valign": "top"})

    # ---------- 1) Legend sheet first ----------
    legend_df = _build_legend_df(sections)
    legend_df.to_excel(writer, sheet_name="Legend", startrow=2, index=False)
    ws = writer.sheets["Legend"]
    ws.write(0, 0, title, fmt_title)
    ws.write(1, 0, "Column dictionary - hover any column header in the data "
                   "sheets to see the same description.", fmt_sub)
    for c, col in enumerate(legend_df.columns):
        ws.write(2, c, col, fmt_header)
    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 26)
    ws.set_column(2, 2, 80, fmt_wrap)
    ws.freeze_panes(3, 0)

    # ---------- 2) data sheets ----------
    for sec in sections:
        name = sec["name"][:31]  # Excel sheet-name limit
        df = sec.get("df")
        err = sec.get("error")

        if df is None or err:
            # The query failed - write a clear note instead of crashing.
            ph = book.add_worksheet(name)
            ph.set_column(0, 0, 100, fmt_wrap)
            ph.write(0, 0, "This section could not be generated.", fmt_title)
            ph.write(2, 0, "Reason: " + (err or "no data returned"))
            ph.write(4, 0, "The rest of the report is unaffected. This usually "
                           "means a column or table name differs on this server "
                           "- tell the developer the reason text above.")
            continue

        df.to_excel(writer, sheet_name=name, startrow=0, index=False)
        ws = writer.sheets[name]

        # header styling + hover comments
        for c, col in enumerate(df.columns):
            ws.write(0, c, col, fmt_header)
            meaning = define(col)
            if meaning:
                ws.write_comment(0, c, meaning, {"x_scale": 2.2, "y_scale": 1.4})

        # column widths + number formats
        for c, col in enumerate(df.columns):
            width = max(len(str(col)) + 2, 12)
            try:
                longest = df[col].astype(str).map(len).max()
                width = max(width, min(int(longest) + 2, 45))
            except Exception:
                pass
            if _is_percent_col(col):
                ws.set_column(c, c, width, fmt_pct)
            elif _is_minute_col(col):
                ws.set_column(c, c, width, fmt_min)
            else:
                ws.set_column(c, c, width)

        if len(df) > 0:
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        ws.freeze_panes(1, 0)

        for chart in sec.get("charts", []) or []:
            try:
                _add_chart(book, ws, name, df, chart)
            except Exception:
                pass  # a chart never breaks the report

    writer.close()


def _add_chart(book, worksheet, sheet_name, df, spec):
    """
    spec = {
        "type": "column"|"bar"|"line",
        "title": str,
        "categories": column name for x-axis labels,
        "values": column name for the series,
        "max_rows": optional cap on rows charted,
        "anchor": optional cell like "H2",
    }
    """
    cat = spec["categories"]
    val = spec["values"]
    if cat not in df.columns or val not in df.columns:
        return

    n = len(df)
    if "max_rows" in spec:
        n = min(n, spec["max_rows"])
    if n == 0:
        return

    cat_idx = list(df.columns).index(cat)
    val_idx = list(df.columns).index(val)

    chart = book.add_chart({"type": spec.get("type", "column")})
    chart.add_series({
        "name": val,
        "categories": [sheet_name, 1, cat_idx, n, cat_idx],
        "values": [sheet_name, 1, val_idx, n, val_idx],
        "data_labels": {"value": True},
    })
    chart.set_title({"name": spec.get("title", val)})
    chart.set_legend({"none": True})
    chart.set_size({"width": 640, "height": 360})
    worksheet.insert_chart(spec.get("anchor", "J2"), chart)

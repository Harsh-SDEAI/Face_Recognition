"""
CLIENT season report -> one Excel workbook.

Leads with value delivered: games & photos processed, matches/suggestions,
player coverage, and turnaround time. Reliability is framed positively
(success rate). Internal noise (errors, retries, data-quality) is NOT here -
that lives in internal_report.py.

Run:
    python client_report.py [output_folder]

Strictly read-only. No database is ever written to.
"""

import os
import sys
from datetime import datetime

import db
import queries
from excel_writer import write_workbook, section


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"CDP_FaceMatching_Client_Report_{datetime.now():%Y%m%d}.xlsx"
    )

    dpp = db.connect_dpp()
    cdp2000 = db.connect_cdp2000()
    try:
        names = queries.get_tournament_names(cdp2000)

        sections = [
            section("Overview", queries.overview, dpp),
            section(
                "By Tournament", queries.by_tournament, dpp, names,
                charts=[{
                    "type": "column",
                    "title": "Games processed per tournament",
                    "categories": "TournamentName",
                    "values": "Games",
                    "anchor": "M2",
                }],
            ),
            section("By Team", queries.by_team, dpp, cdp2000, names),
            section("By Game", queries.by_game, dpp, names),
        ]

        write_workbook(out_path, sections,
                       "CDP AI Face Matching - Season Report (Client)")
        print(f"Client report written: {out_path}")
    finally:
        try:
            dpp.close()
        except Exception:
            pass
        try:
            cdp2000.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

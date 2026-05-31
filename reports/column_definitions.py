"""
Single source of truth for what every report column means.

This dictionary feeds BOTH:
  - the hover comment on each column header in the Excel sheets, and
  - the "Legend" sheet (the printable data dictionary).

Keep wording plain enough for a client; the internal-only columns are at the
bottom and may be more technical.
"""

COLUMN_DEFS = {
    # ---- shared identity columns ----
    "TournamentID": "Internal numeric ID of the tournament.",
    "TournamentName": "Name of the tournament (looked up from tournament records).",
    "GameNumber": "Unique number identifying a single game.",
    "TeamKey": "Internal unique ID of a team.",
    "TeamNumber": "The team's display number.",
    "TeamName": "Name of the team.",
    "Status": "Final processing status of the game (completed / error / etc.).",

    # ---- volume / value metrics (client-facing) ----
    "Games": "Number of games processed.",
    "Completed": "Games that finished processing successfully.",
    "Errored": "Games that ended in an error after exhausting retries.",
    "Photos": "Total game photos processed.",
    "PhotosWithMatch": "Game photos in which at least one player was matched.",
    "MatchRate%": "Share of game photos that produced at least one player match "
                  "(umpire-only photos excluded).",
    "Faces": "Total number of faces detected across all game photos.",
    "Suggestions": "Total player suggestions (matches) the system generated.",
    "AISuggestions": "Suggestions generated automatically by the AI.",
    "PlayersMatched": "Number of distinct players found in at least one photo.",
    "RosterSize": "Number of players on the team's roster.",
    "Coverage%": "Share of the team's roster players who were found in at least "
                 "one game photo.",
    "GroupMatches": "Player matches that came from group photos (more than one "
                    "face in the photo).",
    "GroupPhotos": "Photos containing more than one detected face.",
    "UmpirePhotos": "Photos flagged as umpire photos (no roster players expected).",
    "HomeTeamPhotos": "Photos flagged as belonging to the home team.",
    "VisitorTeamPhotos": "Photos flagged as belonging to the visiting team.",

    # ---- timing (client-friendly) ----
    "AvgMinutes": "Average time taken to process one game, in minutes.",
    "MinMinutes": "Fastest game processing time, in minutes.",
    "MaxMinutes": "Slowest game processing time, in minutes.",
    "ProcessMinutes": "Time taken to process this specific game, in minutes.",

    # ---- overview sheet ----
    "Metric": "The measured quantity.",
    "Value": "The value of the measured quantity.",

    # ---- internal / technical columns ----
    "ManualResults": "Suggestions that were added or corrected manually by a "
                     "human, rather than by the AI. A high share can indicate "
                     "the AI missed matches.",
    "ManualRate%": "Share of suggestions that were manual rather than automatic.",
    "RetryCount": "How many times the game had to be retried before finishing.",
    "Median": "Middle value: half of games were faster than this.",
    "p90Minutes": "90% of games finished faster than this (minutes).",
    "p95Minutes": "95% of games finished faster than this (minutes).",
    "p99Minutes": "99% of games finished faster than this (minutes).",
    "CompletedGames": "Number of games that completed successfully.",
    "Day": "Calendar day on which games were processed.",
    "GamesProcessed": "Number of games processed on that day.",
    "MissingImages": "Photos whose image file path was missing / unreadable.",
    "PlayersNoUsableStudioFace": "Players whose studio photo produced no usable "
                                 "face (so they could not be matched).",
    "ZeroFacePhotos": "Game photos in which no face was detected at all.",
    "FailedPermanently": "Games that ended in error after using all their retries.",
    "PhotosNoMatch": "Game photos that produced no player match.",
}


def define(column_name):
    """Return the definition for a column, or empty string if undefined."""
    return COLUMN_DEFS.get(column_name, "")

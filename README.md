# DIII XC Selection Tools

Interactive tools for projecting the NCAA Division III cross country championship field, built from TFRRS results.

- **Selection Machine** (`selection-machine/`): set each region's finishing order and see which teams fill the 10 automatic and 22 at-large bids, pick by pick, plus the 70 individual qualifiers. Two methods: common opponents in the field (default) or a pairwise model that also uses team strength.
- **Kolas Tracker** (`kolas-tracker/`): a points-based selection system (wins over teams in the field, then head-to-head, common opponents, regional place), shown after each weekend of the 2026 season, with simulated regionals you can roll or reorder.
- **Report** (`reports/`): proposal for incentives to increase interregional racing.

Both pages are self-contained HTML with the data inlined; open them directly or serve the repo with GitHub Pages.

## How selection is modeled

Follows the NCAA DIII cross country selection criteria (2023-24 manual): regional champions get automatic bids, then at-large teams are taken from the "board" (the highest remaining team from each region), compared on head-to-head and common opponents. Only results from September 22 on count, and a result counts for a team only if it ran at least 5 of its projected regional top 7.

Backtests on 2022-2025 at-large selections (held out): the common-opponents model matches the committee on 90% of picks, the pairwise model with team strength on 91.5%, and the Kolas points system on 91%.

## Repo layout

| Path | What it is |
|---|---|
| `index.html` | Landing page |
| `selection-machine/index.html` | Selection Machine |
| `kolas-tracker/index.html` | Kolas Tracker |
| `data/` | The JSON inlined in each page (2026 season, through Nov 1) |
| `scripts/` | Python analysis and backtest scripts |
| `reports/` | PDF write-ups |

## Scripts

The Python scripts expect a working directory with the full TFRRS results table (`all/results.csv`, about 200 MB, not included) and are run as `python3 scripts/<name>.py <workdir>`. `team_adjust.py` holds the shared runner-rating code most scripts import. They are included for transparency; they are not a packaged pipeline.

## Notes

Projections are unofficial and not affiliated with the NCAA or TFRRS.

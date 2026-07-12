# O*NET 30.3 Database

**Source:** U.S. Department of Labor, Employment and Training Administration (USDOL/ETA)
**License:** CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
**Attribution required:** "This page includes information from the O*NET 30.3
Database by the U.S. Department of Labor, Employment and Training Administration
(USDOL/ETA). Used under the CC BY 4.0 license. O*NET® is a trademark of USDOL/ETA."

## Download

Full text-file bundle (all 45 files, tab-delimited):
https://www.onetcenter.org/dl_files/database/db_30_3_text.zip

Or individual files (Excel format), the two we need first:
- Occupation Data: https://www.onetcenter.org/dl_files/database/db_30_3_excel/Occupation%20Data.xlsx
- Essential Skills: https://www.onetcenter.org/dl_files/database/db_30_3_excel/Essential%20Skills.xlsx

Useful later:
- Job Zones (education/experience proxy): https://www.onetcenter.org/dl_files/database/db_30_3_excel/Job%20Zones.xlsx
- Task Statements: https://www.onetcenter.org/dl_files/database/db_30_3_excel/Task%20Statements.xlsx

Data dictionary: https://www.onetcenter.org/dictionary/30.3/excel/

## Placement

Drop downloaded files here as:
```
data/raw/onet/occupation_data.xlsx
data/raw/onet/essential_skills.xlsx
data/raw/onet/job_zones.xlsx        # optional, later
data/raw/onet/task_statements.xlsx  # optional, later
```

These are gitignored — not committed to the repo (data is public/re-downloadable,
no need to bloat repo size or duplicate a licensed dataset).

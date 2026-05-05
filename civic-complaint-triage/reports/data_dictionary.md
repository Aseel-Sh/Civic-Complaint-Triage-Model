# Data Dictionary

Beginner-friendly reference for the Civic Complaint Triage Model. Columns are grouped by how they are used in the pipeline.

## Source columns used from the Philadelphia L&I complaint data
These fields come from the raw complaint dataset and are carried forward (sometimes cleaned or normalized).

| Column name | Meaning | Source or generated | Used in model | Notes |
| --- | --- | --- | --- | --- |
| complaint_type | Human-readable complaint category (e.g., VACANT LOTS (CLIP)) | Source (normalized to uppercase) | Yes | Derived from the most readable complaint label column in the raw data. |
| complaint_source | Intake source or channel (e.g., 311) | Source (normalized to uppercase) | Yes | Pulled from the raw source/system-of-record field. |
| zip_code | 5-digit ZIP code | Source (cleaned) | Yes | Extracted from raw ZIP or postal fields. |
| council_district | City council district | Source | Yes | Included if present in raw data; used as a geography signal. |
| lat | Latitude | Source | Yes | From raw latitude or geometry-derived field. |
| lng | Longitude | Source | Yes | From raw longitude or geometry-derived field. |
| opened_date | Complaint opened/created date | Source (parsed) | No | Parsed from the raw opened/created date field for time-based features. |
| closed_date | Complaint closed/resolved date | Source (parsed) | No | Parsed from the raw closed/resolved date field for target creation. |

## Generated target columns
Targets are computed after cleaning and are used only for training/evaluation.

| Column name | Meaning | Source or generated | Used in model | Notes |
| --- | --- | --- | --- | --- |
| days_to_resolution | Days between open and close | Generated | No | Used to create target labels. |
| delayed | 1 if days_to_resolution exceeds the median, else 0 | Generated | No | Sensitivity target. |
| delayed_30 | 1 if days_to_resolution > 30 days, else 0 | Generated | No | Primary target used for model training. |
| delayed_top25 | 1 if days_to_resolution is in the top 25% | Generated | No | Sensitivity target for comparison. |

## Engineered features
These features are derived from source data for modeling.

| Column name | Meaning | Source or generated | Used in model | Notes |
| --- | --- | --- | --- | --- |
| submitted_month | Month of complaint submission | Generated | Yes | From opened_date. |
| submitted_dayofweek | Day of week (0=Mon) | Generated | Yes | From opened_date. |
| is_weekend | 1 if Saturday/Sunday, else 0 | Generated | Yes | From submitted_dayofweek. |
| complaint_type_total_count | Total complaints for this type | Generated | Yes | Count of complaint_type in the dataset. |
| zip_total_complaints | Total complaints in the ZIP | Generated | Yes | Count of zip_code in the dataset. |
| zip_type_complaint_count | Complaints for this type within ZIP | Generated | Yes | Count by zip_code and complaint_type. |

## Safe scoring features
Only these inputs are accepted for scoring, and only these are used for the model.

| Column name | Meaning | Source or generated | Used in model | Notes |
| --- | --- | --- | --- | --- |
| complaint_type | Complaint category | Source | Yes | Normalized to uppercase. |
| complaint_source | Intake source | Source | Yes | Normalized to uppercase. |
| zip_code | ZIP code | Source | Yes | Cleaned to 5 digits when possible. |
| council_district | Council district | Source | Yes | Allowed input if present. |
| lat | Latitude | Source | Yes | Geographic signal. |
| lng | Longitude | Source | Yes | Geographic signal. |
| submitted_month | Month of submission | Generated | Yes | Derived from opened_date. |
| submitted_dayofweek | Day of week | Generated | Yes | Derived from opened_date. |
| is_weekend | Weekend flag | Generated | Yes | Derived from opened_date. |
| complaint_type_total_count | Type volume | Generated | Yes | Filled using historical aggregates with fallback. |
| zip_total_complaints | ZIP volume | Generated | Yes | Filled using historical aggregates with fallback. |
| zip_type_complaint_count | Type volume in ZIP | Generated | Yes | Filled using historical aggregates with fallback. |

## Columns excluded from modeling (IDs, raw dates, addresses, geometry, leakage)
These columns are either identifiers, raw timestamps, addresses, geometry fields, or post-outcome fields that could leak the target. They are excluded from modeling and scoring.

| Column name | Meaning | Source or generated | Used in model | Notes |
| --- | --- | --- | --- | --- |
| the_geom | Geometry field | Source | No | Raw spatial geometry. |
| the_geom_webmercator | Geometry field (web mercator) | Source | No | Raw spatial geometry. |
| objectid | Internal row ID | Source | No | Identifier only. |
| addressobjectid | Address ID | Source | No | Identifier only. |
| address | Street address | Source | No | Potentially sensitive and not scoring-safe. |
| unit_type | Unit type | Source | No | Address detail. |
| unit_num | Unit number | Source | No | Address detail. |
| zip | Raw ZIP field | Source | No | Replaced by cleaned zip_code. |
| censustract | Census tract | Source | No | Geographic ID not used for scoring. |
| parcel_id_num | Parcel ID | Source | No | Identifier only. |
| opa_account_num | OPA account ID | Source | No | Identifier only. |
| opa_owner | OPA owner name | Source | No | Sensitive/identifying field. |
| complaintnumber | Complaint number | Source | No | Identifier only. |
| casenumber | Case number | Source | No | Identifier only. |
| complaintcode | Complaint code | Source | No | Coded ID, not a readable feature. |
| complaintcodename | Complaint code name | Source | No | Superseded by complaint_type normalization. |
| unitresponsible | Responsible unit | Source | No | Post-outcome or operational detail. |
| ticket_num_311 | 311 ticket number | Source | No | Identifier only. |
| complaintdate | Raw complaint date | Source | No | Raw timestamp replaced by opened_date. |
| initialinvestigation_date | Investigation date | Source | No | Post-outcome leakage risk. |
| systemofrecord | System of record | Source | No | Raw field replaced by complaint_source normalization. |
| geocode_x | Geocode X | Source | No | Raw coordinate field. |
| geocode_y | Geocode Y | Source | No | Raw coordinate field. |
| posse_jobid | POSSE job ID | Source | No | Identifier only. |

Notes:
- The exact raw column names can vary slightly in the source export; the pipeline normalizes them to the cleaned columns above.
- Only the safe scoring features are accepted as input when using the scoring script.

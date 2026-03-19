# Container Scan (CS) — Implementation Plan (Cyber Alignment + New Grouping)

**Status:** Ready to execute in phases — **pending Prashant’s sign-off** on case grouping (see §2).  
**Last updated:** March 2026  
**Stakeholders:** UPM (implementation), Cyber (Zach — decisions captured), Cyber leadership (Prashant — product alignment).

---

## 1. Executive summary

This plan implements the **Cyber-aligned container scan pipeline**:

| Area | Direction |
|------|-----------|
| **Source query** | `cvulnerabilities_cluster_container` + **`current_open_vulns`**-style joins (container, `cluster_container`, cluster, `cvulnerabilities`) + **`vuln_soft`** pattern via `csoftware_container` / `csoftware` for Proof (paths + package details). |
| **Filters** | C/H/M, `cvuln.resolveable = true`, `container.resolvable = true`, temporal: `container.updated_at` and `cluster_container.updated_at` = export day. |
| **Paths** | **Optional** for ticketing — ticket all in-scope findings; show paths when present (Zach). |
| **Container identity** | Use **`container.container_id`** from vmtautomation **as-is** (Zach). |
| **Owner** | **`risk_owner`** is primary for assignment and “unknown owner” rules (Zach). Exclude rows where **`risk_owner`** is NULL/empty after the agreed COALESCE rule (see §3.3). |
| **Case grouping** | **Uniqueness:** `severity` + `cluster_id` + `container_id` (Zach: `container_id` is unique enough vs old `image_repository`). **`risk_owner`** on the case for routing; confirm with Prashant whether **`risk_owner`** is also part of **`case_key`** or only stored on the case row (§2). |
| **Unknown owners** | No cases/tickets; out of scope for weekly leadership stakeholder reports (agreed with Zach). |

**VM pipeline (traditional host vulns):** This work touches **only the CS path** (`cs_*` tables, CS DAGs, `cs_*` Python modules). **`vm_latest_vfe` / `vm_case_master` / VM ingestion jobs are out of scope** unless we discover a shared utility that must stay backward-compatible — see §10.

---

## 2. Product gate — Prashant sign-off

Before merging schema or grouping logic:

- [ ] **Prashant** confirms case grouping for CS: **`severity` + `cluster_id` + `container_id`** (and how **`risk_owner`** participates — key field vs display-only).
- [ ] Document the decision in this file (checkbox → **Done** with date).

**If Prashant requests changes** (e.g. keep `cluster_name` on the case for readability while key uses IDs), capture here and adjust Phase 3–4 accordingly.

---

## 3. Decisions log (Cyber — Zach)

| # | Topic | Decision |
|---|--------|----------|
| 1 | Paths | Optional — ticket all in scope; show paths when available. |
| 2 | Grouping uniqueness | `cluster_id` + `container_id` (+ `severity` for case buckets) sufficient vs old repo-based grouping. |
| 3 | Owner field | **`risk_owner`** (not `system_owner` as primary for ticketing / unknowns). |
| 4 | Container id | **`container.container_id`** as stored in DB — do not rebuild registry/repo/name/digest in SQL unless needed for display-only. |
| 5 | Unknown owners | Exclude — no tickets; not in weekly leadership reports. |

**Open implementation detail:** Whether **`risk_owner`** in `cluster_container` should be combined with **`container.risk_owner`** via `COALESCE(cc.risk_owner, c.risk_owner)` — match PowerBI / Silpa’s query patterns; default to **COALESCE** for richness, then **filter NULL/empty**.

---

## 4. Architecture overview (target state)

```text
vmtautomation (read-only)
  container_export  → checkpoint / export_day
  container, cluster_container, cluster
  cvulnerabilities_cluster_container  ← fact
  cvulnerabilities
  csoftware_container, csoftware, cvulnerabilities_csoftware
        ↓
  CS load (streaming) → cs_latest_vfe  (+ cluster_id, + software JSON / Proof fields)
        ↓
  CS ingestion (validate, Graph email enrich on owner fields, group by new case_key)
        ↓
  cs_case_master + cs_vulnerability_record
        ↓
  cs_message_queue → ADO processor (payload includes software / paths per record)
```

---

## 5. Database & schema changes (Themis)

### 5.1 `cs_latest_vfe`

**Add (minimum):**

- `cluster_id` (VARCHAR) — from `cluster` / join path.
- Software / Proof columns (names TBD): e.g. `affected_software` (JSONB or TEXT), `fixed_by_versions` (TEXT), optional `proof` TEXT if we mirror PowerBI string.
- Keep **`system_owner`**, **`risk_owner`**, **`cluster_name`**, image fields for display, email enrichment, and reporting even if not in **case_key**.

**Indexes:** Replace or add indexes that match new query filters and dedup: e.g. `(cluster_id, container_id, cvulnerabilities_id)`, `(risk_owner)`, `(day)`.

### 5.2 `cs_case_master` (breaking change)

**Today:** `cluster_name`, `system_owner`, `severity`, `image_repository`, `case_key`, …

**Target (conceptual):**

- Store **`severity`**, **`cluster_id`**, **`container_id`** (and optionally **`cluster_name`** for human-readable ADO/title — not part of uniqueness if Zach’s model is ID-based).
- Store **`risk_owner`** (NOT NULL for any row we insert — we pre-filter unknowns).
- **`case_key`:** deterministic string from the agreed tuple, e.g. `severity|cluster_id|container_id` or include `risk_owner` if Prashant requires it.

**Migration strategy (DV1 / local):**

1. **Truncate** CS processing tables in dependency order (see `cs_cvc_switch_implementation_plan.md` §8 / §11): `cs_vulnerability_record`, `cs_message_queue`, `cs_case_master`, then `cs_latest_vfe`, `cs_bad_message`.
2. **DDL:** `ALTER TABLE cs_case_master` — add new columns, backfill not applicable if truncate-first.
3. **Unique index:** replace `idx_cs_cm_grouping` on old four columns with new partial unique index on **open** cases matching new grouping columns.

**cgac-api / Django:** If `CSCaseMaster` is mirrored outside this repo, coordinate ORM + migrations there in the same release window.

### 5.3 `cs_vulnerability_record`

- **`record` JSONB:** Add software fields (`affected_software`, `fixed_by_versions`, paths per package as designed).
- Unique constraint `(cs_case_master_id, cvulnerabilities_id, container_id)` — still valid if **`container_id`** is globally unique; confirm same CVE cannot appear twice in one case for same container (CVC grain should prevent).

### 5.4 Field length / charter limits

- **`container_id`** can be long (registry/repo/name/digest) — ensure VARCHAR length in DDL (e.g. 1024+ if not already TEXT).
- **`cluster_id`** — same review.
- **`case_key`** — bounded string; if concatenation grows, cap component lengths in `generate_case_key` or use hash (last resort).

---

## 6. Query work (`cs_load_findings.py`)

### 6.1 Replace fact table

- From `cvulnerabilities_container` → **`cvulnerabilities_cluster_container`**.
- Join order aligned with **`current_open_vulns`**: `latest_date` / `:export_day` → `container` → `cluster_container` → `cluster` → CVC → `cvulnerabilities` → LEFT JOIN **`vuln_soft`**.

### 6.2 `vuln_soft` CTE

- Aggregate per `(cvulnerabilities_id, container_id)` from `csoftware_container` → `cvulnerabilities_csoftware` → `csoftware`.
- Include **paths** in output when present (optional ticketing — still join LEFT so findings without paths load).
- Unnest **`fixed_by_versions`** safely for empty JSONB (see existing plan in `cs_cvc_switch_implementation_plan.md`).

### 6.3 Filters

- `vc.severity` → `cvc.severity` IN C/H/M.
- `cvuln.resolveable = true`, `container.resolvable = true`.
- `container.updated_at = :export_day`, `cluster_container.updated_at = :export_day`.

### 6.4 Exclude unknown `risk_owner`

- **WHERE** clause (or post-filter in Python for clarity): exclude rows where **`COALESCE(cc.risk_owner, c.risk_owner)`** is NULL or blank (exact predicate to match validation).

### 6.5 Checkpoint

- Consider aligning `container_export` selection with Cyber: `processed AND has_data AND date IS NOT NULL` vs `MAX(day)` — follow-up if counts drift.

---

## 7. Ingestion & validation (`cs_ingestion_job.py`, `ingestion.py`)

### 7.1 `VULN_COLUMNS`

- Add **`cluster_id`**, software columns.
- Keep **`risk_owner`**, **`system_owner`** for email enrichment and display.

### 7.2 `REQUIRED_FIELDS`

- Remove dependency on **`cluster_name`**, **`system_owner`**, **`image_repository`** as *required for processing* if new rule is **`risk_owner`** + IDs.
- **New minimum:** e.g. `cvulnerabilities_id`, `container_id`, `severity`, `cluster_id`, `risk_owner` (after COALESCE).

### 7.3 `CS_DEDUP_ORDER` / `CS_BATCH_QUERY` / `DISTINCT ON`

- Replace grouping columns with **`cluster_id`, `risk_owner`, `severity`, `container_id`** (or whatever final key Prashant confirms).
- Include **`cluster_id`** and **`cvulnerabilities_id`** in dedup for CVC grain.

### 7.4 Email enrichment (`CS_NAME_FIELDS`)

- Add **`risk_owner`** (and keep **`system_owner`** if still needed for comms). **Primary assignee for Graph lookup** should match **ticket owner** (likely **`risk_owner`** per Zach).

### 7.5 Blacklist / “Unknown Owner”

- Align **`CS_NAMES_BLACKLIST`** with exclusion rules — rows with NULL **`risk_owner`** should not reach grouping; blacklist remains for bad placeholder names if still needed.

---

## 8. Grouping & case key (`common.py`, `GroupedCase`, `updates.py`)

- **`generate_case_key(...)`** — new signature and ordering (normalize: lower/strip like today).
- **`GroupedCase`** — hold **`cluster_id`**, **`container_id`**, **`risk_owner`**, **`severity`**; retain **`cluster_name`** for display if loaded.
- **`CsCaseMasterUpdate`** / upsert SQL — column list and conflict target updated.

---

## 9. ADO pipeline (`ado_messages.py`, `cs_ado_cases.py`, templates)

### 9.1 Payload (`_build_cs_ado_payload`)

- Case-level fields: expose **`cluster_id`**, **`container_id`**, **`risk_owner`** (and **`cluster_name`** for humans).
- Per-container / per-record: add **`affected_software`**, **`fixed_by_versions`**, path snippets — **cap list size and string length** per field to avoid ADO / gateway limits (see megacase work — batch size, truncate long strings).

### 9.2 Templates

- Update **`cs_case_ado_*.html`** (or equivalent) to show software + paths when present.

### 9.3 Limits & megacases

- **Many CVEs per case** — same as today: batch ADO messages, commit per batch, cap JSON size per work item field if needed.
- **Char limits:** Align with DB VARCHAR limits and ADO field limits; centralize caps (e.g. namespace truncation pattern already used).

---

## 10. VM pipeline boundary (no accidental breakage)

| Item | Action |
|------|--------|
| **`vm_airflow/jobs/load_findings.py` / `ingestion_job.py`** | **Do not change** for CS work. |
| **`vm_case_master` / `vm_vulnerability_record`** | **Out of scope.** |
| **Shared code** | **`utils/db.py`**, **`ConfigManager`**, **`DatabaseManager`** — only change if CS needs new config keys; avoid breaking VM DAG imports. |
| **Tests** | Run **VM unit tests** if shared modules change; keep CS tests isolated under `tests/.../cs_*`. |

If **cgac-api** shares models with VM + CS, split releases or feature-flag only CS endpoints.

---

## 11. Phased execution (iterations)

Execute in order; each phase should be shippable or behind a feature flag where noted.

| Phase | Scope | Deliverables |
|-------|--------|--------------|
| **0** | **Gate + env** | Prashant confirmation §2; DV1 truncate plan; optional feature flag `CS_USE_CVC_QUERY` for safe rollback. |
| **1** | **Load query** | New `CS_FINDINGS_QUERY` + `CS_VFE_COLUMNS`; `vuln_soft`; filters + **`risk_owner`** exclusion; streaming load tested on sample day. |
| **2** | **Schema** | `cs_latest_vfe` + `cs_case_master` DDL (init scripts + migration notes); indexes. |
| **3** | **Ingestion** | `VULN_COLUMNS`, dedup, `REQUIRED_FIELDS`, validation, **`generate_case_key`**, `GroupedCase`, upserts. |
| **4** | **Email** | Graph enrichment for **`risk_owner`** (and system_owner if retained). |
| **5** | **ADO** | Payload + templates + caps + processor compatibility. |
| **6** | **Tests & docs** | Unit/integration tests; update `cs_cvc_switch_implementation_plan.md` / alignment docs; runbook for truncate order. |
| **7** | **Hardening** | Performance (query duration, indexes on vmtautomation side if Cyber approves read-only hints); monitor load time. |

**Rollback:** Revert to previous `CS_FINDINGS_QUERY` + old grouping; truncate CS tables; re-run (documented in existing rollback section).

---

## 12. Edge cases checklist

- [ ] **Empty `paths` in `csoftware_container`** — Proof still has package name/version; paths optional.
- [ ] **NULL `fixed_by_versions`** — handle JSONB null/empty arrays in `vuln_soft`.
- [ ] **Same `container_id` on two clusters** — CVC row is per `(cluster_id, container_id, vuln_id)`; case key includes **`cluster_id`**.
- [ ] **`risk_owner` populated but `system_owner` null** — Zach said **`risk_owner`** primary; ticket still assignable.
- [ ] **Both owners null** — excluded at load or validation.
- [ ] **Very long `container_id` / `cluster_id`** — DDL + ADO truncation.
- [ ] **Megacase** — many vulns per case; keep batching and payload caps.
- [ ] **Export day mismatch** — checkpoint vs Cyber `processed/has_data` — document drift if observed.

---

## 13. References (existing docs in repo)

- `docs/cs_cvc_switch_implementation_plan.md` — CVC + vuln_soft + temporal filters (merge technical detail into implementation).
- `docs/cs_silpa_powerbi_query_dissection.md` — query semantics.
- `docs/cs_zach_meeting_summary_and_next_steps.md` — Zach discussion (superseded by Slack decisions; keep for history).
- `docs/cs_cyber_meeting_questions.md` — Cyber Q&A context.

---

## 14. Open questions (for you or Cyber)

1. **Prashant:** Final shape of **`case_key`** — include **`risk_owner`** or only **`severity|cluster_id|container_id`**?
2. **Cyber:** Optional alignment of **`container_export`** checkpoint query with PowerBI.
3. **cgac-api:** Any **Django admin / API** that assumes old **`cs_case_master`** columns — coordinate migration.

---

**End of plan.** Update §2 and §14 as decisions land; use Phase table as the execution checklist.

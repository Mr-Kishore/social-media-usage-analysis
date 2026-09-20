# Social Media Usage Analysis System

**Undergraduate Mini Project**  
*Department of Computer Technology*  
*Dr. N.G.P. Arts and Science College (Autonomous), Coimbatore*  
*(Affiliated with Bharathiar University)*

---

## 1. Project Overview

The **Social Media Usage Analysis System** is an end-to-end Python analytics pipeline designed to investigate the behavioral trade-offs between daily social media screen time, academic study hours, sleep duration, and standardized academic performance among higher education students.

### Key Capabilities:
1. **Dual Data Sourcing:**
   * **Synthetic Parametric Generation:** Produces a deterministic, statistically parameterized student cohort ($N = 1,000$, random seed `42`) with realistic Gaussian disturbance terms ($\epsilon_{\text{sleep}}$, $\epsilon_{\text{study}}$, $\epsilon_{\text{perf}}$) for privacy-preserving research.
   * **Empirical Survey Ingestion:** Ingests, validates, and normalizes real student survey records ($N = 5,000$) from `dataset/Student Social Media And Mental Health Impact.csv` into a standardized canonical format.
2. **Automated Exploratory Data Analysis (EDA):** Computes descriptive statistics (mean, std, quartiles), categorical frequency counts, and multivariate Pearson correlation matrices ($4 \times 4$).
3. **Publication-Ready Visualizations:** Automatically exports 6 high-resolution (300 DPI) analytical figures directly into `charts/` adhering to academic viva and thesis reporting standards.
4. **High Performance:** Complete end-to-end execution completes in **under 3.5 seconds** (passing the $< 5.0\text{s}$ KPI threshold).

---

## 2. System Architecture & Directory Layout

```
mini-project/
├── charts/                                            # Automatically exported 300 DPI figures
│   ├── chart1_platform_usage.png                     # Descending platform frequency countplot
│   ├── chart2_daily_usage_hist.png                   # Daily usage histogram with KDE curve
│   ├── chart3_age_group_usage.png                    # Mean usage across age cohorts (<18, 18-21, 22+)
│   ├── chart4_study_vs_usage.png                     # Study hours vs. daily usage regression
│   ├── chart5_sleep_vs_usage.png                     # Sleep duration vs. daily usage regression
│   └── chart6_usage_vs_performance.png               # Academic performance vs. daily usage regression
├── data/                                             # Canonical output directory
│   └── social_media_usage_dataset.csv                # Standardized 7-attribute analytical dataset
├── dataset/                                          # Raw empirical data repository
│   └── Student Social Media And Mental Health Impact.csv # Empirical survey records (N = 5,000)
├── src/                                              # Modular source code
│   ├── __init__.py                                   # Package initialization
│   ├── generate_data.py                              # Dual generator (Synthetic & Real Ingestion)
│   └── analyze.py                                    # Automated EDA & visualization engine
├── PRD.md                                            # Product Requirements Document
├── SRS.md                                            # Software Requirements Specification
├── README.md                                         # Project documentation & user guide
├── requirements.txt                                  # Dependency requirements
└── run.py                                            # Unified CLI orchestrator entrypoint
```

---

## 3. Data Dictionary & Canonical Schema

Both synthetic and empirical data streams are harmonized into a standardized **7-attribute schema**:

| Attribute Name | Data Type | Constraint Range | Description |
| :--- | :--- | :--- | :--- |
| `Age` | Integer | $17 \le x \le 25$ | Student age in completed years |
| `Gender` | Categorical | `Male`, `Female`, `Other` | Self-reported gender identity |
| `Platform` | Categorical | Instagram, YouTube, etc. | Primary preferred social media platform |
| `Daily_Usage_Hours` | Float (2 dec) | $0.5 \le x \le 9.8$ | Average daily screen time spent on social media |
| `Study_Hours` | Float (2 dec) | $0.5 \le x \le 8.5$ | Daily academic study duration |
| `Sleep_Hours` | Float (2 dec) | $3.5 \le x \le 9.5$ | Average nocturnal sleep duration |
| `Academic_Performance`| Float (1 dec) | $35.0 \le x \le 98.5$ | Standardized exam performance index percentage |

### Empirical Dataset Mapping:
When ingesting `dataset/Student Social Media And Mental Health Impact.csv`:
* `Age` $\rightarrow$ `Age` (clipped to $[17, 25]$)
* `Gender` $\rightarrow$ `Gender` (`Male`, `Female`, `Other`)
* `Most_Used_Platform` $\rightarrow$ `Platform`
* `Avg_Daily_Usage_Hours` $\rightarrow$ `Daily_Usage_Hours`
* `Study_Hours` $\rightarrow$ `Study_Hours`
* `Sleep_Hours_Per_Night` $\rightarrow$ `Sleep_Hours`
* `Mental_Health_Score` $\rightarrow$ `Academic_Performance` (rescaled via $\text{Score} \times 10$, mapped to $36.0\text{--}94.0\%$)

---

## 4. Installation & Setup

### Prerequisites
* **Python Runtime:** Python 3.10, 3.11, 3.12, 3.13, or 3.14
* **Operating System:** Windows 10/11, macOS, or Linux

### Install Dependencies
Clone or navigate to the project directory, then install the required libraries:
```bash
pip install -r requirements.txt
```

---

## 5. Usage Guide

### Unified Execution (`run.py`)
### Interactive Analytical Dashboard (Streamlit UI)
To launch the full interactive web application without default branding or loading emojis:

```bash
streamlit run app.py
```
* **Real-time Dual Mode Switching:** Seamlessly toggle between Synthetic Cohort ($N=1,000$, seed=42) and Empirical Survey ($N=5,000$).
* **Dynamic Interactive Charts:** Hover inspection, zoom, and direct value tooltips for all figures.
* **Interactive Live Filtering:** Filter by platform, gender, and age cohort in real time.
* **'What-If' Academic Simulator:** Adjust daily screen time, study hours, and sleep duration to project academic performance with live gauge indicators.

### Command-Line Execution (`run.py`)
You can also run the pipeline headlessly via the terminal:

```bash
# Mode 1: Run with Synthetic Data (Default, N=1,000, seed=42)
python run.py --data synthetic

# Mode 2: Run with Empirical Survey Ingestion (N=5,000)
python run.py --data real

# Default Execution (runs synthetic mode with complete banner)
python run.py
```

### Optional Execution Flags:
```text
--data {synthetic,real}   Select data source (default: synthetic)
--input PATH              Path to input raw CSV (default: dataset/Student Social Media And Mental Health Impact.csv)
--output-data PATH        Destination path for canonical dataset (default: data/social_media_usage_dataset.csv)
--charts-dir DIR          Destination directory for 300 DPI figures (default: charts/)
--n INT                   Sample size for synthetic data (default: 1000)
--seed INT                PRNG seed for synthetic reproducibility (default: 42)
```

### Running Individual Modules Directly
You can also run modules independently:

```bash
# Generate synthetic data only
python src/generate_data.py --mode synthetic --n 1000 --seed 42

# Ingest real survey dataset only
python src/generate_data.py --mode real

# Run EDA and chart generation on existing canonical CSV
python src/analyze.py --data data/social_media_usage_dataset.csv --charts charts
```

### Automated Testing Suite
Run the 15-point unit and integration test suite via `pytest`:

```bash
pytest tests/ -v
```

### Continuous Integration (CI/CD Pipeline)
An enterprise GitHub Actions pipeline is configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) to automatically validate every push and pull request to `main`:
* **Matrix Testing:** Executes across **Python 3.10, 3.11, and 3.12** on `ubuntu-latest`.
* **Static Code Analysis:** Python compilation and syntax verification.
* **Unit & Integration Tests:** Executes the full `pytest` suite testing data generation, boundary constraints, statistical validations, and EDA engines.
* **Pipeline Verification:** Executes both `--data synthetic` and `--data real` end-to-end runs.
* **Artifact Validation:** Verifies all 6 required 300 DPI analytical charts are generated and uploads them as build artifacts.

---

## 6. Generated Visual Artifacts

All figures are automatically saved to `charts/` in high-resolution **300 DPI**:

1. **`chart1_platform_usage.png`**: Descending frequency countplot displaying platform preference distribution with direct value annotations.
2. **`chart2_daily_usage_hist.png`**: Distribution of daily social media usage across 16 discrete bins with an overlaid Kernel Density Estimation (KDE) curve and marked mean/median lines.
3. **`chart3_age_group_usage.png`**: Mean daily usage hours partitioned into demographic cohorts (`<18`, `18–21`, `22+`) with sample counts.
4. **`chart4_study_vs_usage.png`**: Linear regression scatter plot illustrating academic study duration versus daily social media screen time.
5. **`chart5_sleep_vs_usage.png`**: Linear regression scatter plot illustrating sleep duration versus daily social media screen time.
6. **`chart6_usage_vs_performance.png`**: Linear regression scatter plot evaluating academic exam performance against daily social media screen time.

---

## 7. Statistical Findings & Acceptance Criteria

| Statistical Validation Metric | Synthetic Model ($N=1,000$) | Empirical Dataset ($N=5,000$) | Target Acceptance Threshold | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Usage vs. Sleep ($r$)** | **$-0.7198$** | **$-0.8084$** | $r \le -0.40$ | **VERIFIED** |
| **Usage vs. Study ($r$)** | **$-0.7441$** | **$-0.8786$** | Inverse ($r < 0$) | **VERIFIED** |
| **Usage vs. Performance ($r$)** | **$-0.8665$** | **$-0.8165$** | Inverse ($r < 0$) | **VERIFIED** |
| **Execution Latency** | **$2.58\text{ s}$** | **$3.25\text{ s}$** | $< 5.0\text{ s}$ | **VERIFIED** |
| **Dataset Completeness** | 1,000 rows (0 nulls) | 5,000 rows (0 nulls) | 100% Non-null | **VERIFIED** |

---

## 8. Academic Declaration

* **Student:** Gopi
* **Degree:** Bachelor of Science in Computer Technology
* **Institution:** Dr. N.G.P. Arts and Science College (Autonomous), Coimbatore
* **Affiliation:** Bharathiar University

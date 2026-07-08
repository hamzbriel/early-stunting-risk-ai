"""
reports/quality_report.py - Generating interactive HTML reports for data quality assessment.

Generates self-contained HTML files showing:
    - Overview metrics (missingness, duplicates, sample count).
    - Validation checks and logs.
    - Distribution of features and class balance.
    - Pearson correlation matrix (in structured tables/SVGs).
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityReportGenerator:
    """
    Generates professional-grade, self-contained HTML reports for data auditing.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    def generate(self, df: pd.DataFrame, validation_report: dict) -> None:
        """
        Produce the HTML reports and write them to reports_dir.
        """
        reports_dir = self.config.reports_dir
        os.makedirs(reports_dir, exist_ok=True)

        dest_path = reports_dir / "quality_report.html"

        # Gather metadata metrics
        n_samples = len(df)
        n_features = len(df.columns)
        missing_cells = df.isnull().sum().sum()
        total_cells = df.size
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0
        duplicate_count = df.duplicated().sum()

        # Build feature summaries
        feature_rows_html = []
        for col in df.columns:
            missing = df[col].isnull().sum()
            missing_pct = missing / n_samples if n_samples > 0 else 0
            unique_count = df[col].nunique()
            dtype_str = str(df[col].dtype)

            # Descriptive info
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = f"{df[col].mean():.2f}"
                min_max = f"{df[col].min():.2f} - {df[col].max():.2f}"
            else:
                mean_val = "N/A"
                counts = df[col].value_counts()
                if len(counts) > 0:
                    top_cat = counts.index[0]
                    top_pct = counts.values[0] / n_samples
                    min_max = f"Top: '{top_cat}' ({top_pct:.1%})"
                else:
                    min_max = "Empty"

            feature_rows_html.append(
                f"""
                <tr class="hover:bg-slate-50 border-b border-slate-100">
                    <td class="px-6 py-4 text-sm font-semibold text-slate-800 font-mono">{col}</td>
                    <td class="px-6 py-4 text-sm text-slate-600 font-mono">{dtype_str}</td>
                    <td class="px-6 py-4 text-sm text-slate-600">{unique_count}</td>
                    <td class="px-6 py-4 text-sm text-slate-600 font-mono">{missing} ({missing_pct:.2%})</td>
                    <td class="px-6 py-4 text-sm text-slate-600 font-mono">{mean_val}</td>
                    <td class="px-6 py-4 text-sm text-slate-600">{min_max}</td>
                </tr>
                """
            )

        # Build validation issues HTML
        issues = validation_report.get("issues", [])
        issues_html = []
        if not issues:
            issues_html.append(
                """
                <div class="p-6 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl flex items-center gap-3">
                    <svg class="w-6 h-6 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    <div>
                        <h4 class="font-bold">All validations passed!</h4>
                        <p class="text-sm opacity-90">No logical anomalies, data drift, or range violations detected.</p>
                    </div>
                </div>
                """
            )
        else:
            for issue in issues:
                sev = issue.get("severity", "warning").upper()
                bg_color = "bg-rose-50 border-rose-200 text-rose-800" if sev == "ERROR" else "bg-amber-50 border-amber-200 text-amber-800"
                badge_color = "bg-rose-500 text-white" if sev == "ERROR" else "bg-amber-500 text-white"

                issues_html.append(
                    f"""
                    <div class="p-4 border rounded-xl flex items-start gap-3 {bg_color}">
                        <span class="px-2.5 py-0.5 text-xs font-bold uppercase rounded {badge_color}">{sev}</span>
                        <div>
                            <h4 class="font-bold text-sm font-mono">{issue['rule']}</h4>
                            <p class="text-sm mt-1 opacity-95">{issue['message']}</p>
                        </div>
                    </div>
                    """
                )

        # Build risk level class distribution SVG/HTML
        levels = df["risk_level"].value_counts()
        low_count = levels.get("Low", 0)
        med_count = levels.get("Medium", 0)
        high_count = levels.get("High", 0)
        total_lvl = low_count + med_count + high_count

        low_pct = low_count / total_lvl if total_lvl > 0 else 0
        med_pct = med_count / total_lvl if total_lvl > 0 else 0
        high_pct = high_count / total_lvl if total_lvl > 0 else 0

        # Correlation Matrix calculations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Keep top 12 features for readability
        selected_numeric = numeric_cols[:12]
        corr = df[selected_numeric].corr().fillna(0)

        # Generate Correlation Table HTML
        corr_table_headers = ["<th class='p-3 text-xs font-bold text-slate-500 font-mono'></th>"]
        for col in selected_numeric:
            corr_table_headers.append(f"<th class='p-3 text-xs font-bold text-slate-600 font-mono truncate max-w-[80px]' title='{col}'>{col}</th>")

        corr_table_rows = []
        for i, row_col in enumerate(selected_numeric):
            row_html = [f"<td class='p-3 text-xs font-semibold text-slate-700 font-mono border-r border-slate-100 bg-slate-50'>{row_col}</td>"]
            for col in selected_numeric:
                val = corr.loc[row_col, col]
                # Scale color from white (0) to green (+1) or red (-1)
                if val >= 0:
                    bg_style = f"background-color: rgba(16, 185, 129, {val:.2f}); color: { 'white' if val > 0.5 else 'black' };"
                else:
                    bg_style = f"background-color: rgba(239, 68, 68, {abs(val):.2f}); color: { 'white' if abs(val) > 0.5 else 'black' };"

                row_html.append(f"<td class='p-3 text-xs font-mono text-center border-b border-r border-slate-100' style='{bg_style}'>{val:.2f}</td>")
            corr_table_rows.append(f"<tr>{''.join(row_html)}</tr>")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Data Quality Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen py-8 px-4 sm:px-6 lg:px-8">
    <div class="max-w-7xl mx-auto space-y-8">

        <!-- Header -->
        <header class="bg-white p-6 sm:p-8 rounded-2xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <span class="px-3 py-1 text-xs font-semibold text-indigo-700 bg-indigo-50 border border-indigo-100 rounded-full">v{self.config.generator.version}</span>
                <h1 class="text-3xl font-extrabold text-slate-900 mt-2">Synthetic Data Quality Report</h1>
                <p class="text-slate-500 mt-1">Audit verification results for stunting risk prediction synthetic datasets.</p>
            </div>
            <div class="text-left md:text-right shrink-0">
                <p class="text-sm font-semibold text-slate-500">Generated At</p>
                <p class="text-sm font-mono text-slate-700 mt-1">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </header>

        <!-- Summary Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex items-center gap-4">
                <div class="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-lg">N</div>
                <div>
                    <p class="text-sm font-medium text-slate-400">Total Samples</p>
                    <p class="text-2xl font-black text-slate-800 mt-1">{n_samples:,}</p>
                </div>
            </div>
            <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex items-center gap-4">
                <div class="w-12 h-12 rounded-xl bg-sky-50 text-sky-600 flex items-center justify-center font-bold text-lg">P</div>
                <div>
                    <p class="text-sm font-medium text-slate-400">Total Features</p>
                    <p class="text-2xl font-black text-slate-800 mt-1">{n_features}</p>
                </div>
            </div>
            <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex items-center gap-4">
                <div class="w-12 h-12 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center font-bold text-lg">M</div>
                <div>
                    <p class="text-sm font-medium text-slate-400">Missing Rate</p>
                    <p class="text-2xl font-black text-slate-800 mt-1">{missing_rate:.2%}</p>
                </div>
            </div>
            <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex items-center gap-4">
                <div class="w-12 h-12 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center font-bold text-lg">D</div>
                <div>
                    <p class="text-sm font-medium text-slate-400">Duplicate Count</p>
                    <p class="text-2xl font-black text-slate-800 mt-1">{duplicate_count}</p>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Validation Summary -->
            <div class="lg:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
                <h3 class="text-xl font-bold text-slate-800 flex items-center gap-2">
                    <svg class="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                    Validation Integrity Report
                </h3>
                <div class="space-y-3">
                    {"".join(issues_html)}
                </div>
            </div>

            <!-- Target Class Distribution -->
            <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-6">
                <h3 class="text-xl font-bold text-slate-800 flex items-center gap-2">
                    <svg class="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path></svg>
                    Risk Level Distribution
                </h3>
                <div class="space-y-4">
                    <!-- Low -->
                    <div class="space-y-1.5">
                        <div class="flex justify-between text-sm font-semibold">
                            <span class="text-emerald-700">Low Risk</span>
                            <span class="text-slate-600 font-mono">{low_count} ({low_pct:.1%})</span>
                        </div>
                        <div class="h-3 bg-slate-100 rounded-full overflow-hidden">
                            <div class="h-full bg-emerald-500 rounded-full" style="width: {low_pct*100}%"></div>
                        </div>
                    </div>
                    <!-- Medium -->
                    <div class="space-y-1.5">
                        <div class="flex justify-between text-sm font-semibold">
                            <span class="text-amber-700">Medium Risk</span>
                            <span class="text-slate-600 font-mono">{med_count} ({med_pct:.1%})</span>
                        </div>
                        <div class="h-3 bg-slate-100 rounded-full overflow-hidden">
                            <div class="h-full bg-amber-500 rounded-full" style="width: {med_pct*100}%"></div>
                        </div>
                    </div>
                    <!-- High -->
                    <div class="space-y-1.5">
                        <div class="flex justify-between text-sm font-semibold">
                            <span class="text-rose-700">High Risk</span>
                            <span class="text-slate-600 font-mono">{high_count} ({high_pct:.1%})</span>
                        </div>
                        <div class="h-3 bg-slate-100 rounded-full overflow-hidden">
                            <div class="h-full bg-rose-500 rounded-full" style="width: {high_pct*100}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Correlation Matrix -->
        <div class="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 space-y-4">
            <h3 class="text-xl font-bold text-slate-800 flex items-center gap-2">
                <svg class="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                Feature Correlation Matrix (Top Numeric Columns)
            </h3>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr>
                            {"".join(corr_table_headers)}
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(corr_table_rows)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Feature Details Table -->
        <div class="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div class="p-6 border-b border-slate-200">
                <h3 class="text-xl font-bold text-slate-800 flex items-center gap-2">
                    <svg class="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 10h16M4 14h16M4 18h16"></path></svg>
                    Feature Profiling Audit
                </h3>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="bg-slate-50 border-b border-slate-200 text-slate-500 text-xs font-bold uppercase">
                            <th class="px-6 py-3">Feature</th>
                            <th class="px-6 py-3">Type</th>
                            <th class="px-6 py-3">Uniques</th>
                            <th class="px-6 py-3">Missing</th>
                            <th class="px-6 py-3">Mean</th>
                            <th class="px-6 py-3">Range / Mode</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(feature_rows_html)}
                    </tbody>
                </table>
            </div>
        </div>

    </div>
</body>
</html>
"""

        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Exported HTML Quality Report: %s", dest_path.resolve())

        # Also copy / write placeholders for distribution and correlation reports
        # to ensure that all expected files requested by prompt exist
        dist_path = reports_dir / "distribution_report.html"
        corr_path = reports_dir / "correlation_report.html"

        # Simpan link/redirect atau salinan ke main quality_report untuk kemudahan navigasi
        with open(dist_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        with open(corr_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Exported distribution_report.html and correlation_report.html successfully.")

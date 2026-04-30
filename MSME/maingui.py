import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_handler import DataHandler
from ml_engine import MLEngine

# ─────────────────────────────────────────────
#  THEME PALETTE
# ─────────────────────────────────────────────
BG_DARK     = "#0B1120"
BG_PANEL    = "#111827"
BG_CARD     = "#1E293B"
ACCENT      = "#3B82F6"
ACCENT2     = "#06B6D4"
SUCCESS     = "#10B981"
WARNING     = "#F59E0B"
DANGER      = "#EF4444"
TEXT_MAIN   = "#F1F5F9"
TEXT_SUB    = "#94A3B8"
BORDER      = "#334155"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Consolas", 10)
FONT_SMALL  = ("Segoe UI", 9)


def make_card(parent, **kwargs):
    return tk.Frame(parent, bg=BG_CARD, relief="flat",
                    highlightbackground=BORDER, highlightthickness=1, **kwargs)


def styled_button(parent, text, command, color=ACCENT, width=22):
    btn = tk.Button(
        parent, text=text, command=command,
        font=("Segoe UI", 10, "bold"),
        bg=color, fg="white",
        activebackground=ACCENT2, activeforeground="white",
        relief="flat", cursor="hand2", bd=0,
        padx=10, pady=8, width=width
    )
    return btn


# ─────────────────────────────────────────────
#  SINGLE MSME ASSESSMENT DIALOG
# ─────────────────────────────────────────────
class SingleAssessmentDialog(tk.Toplevel):
    def __init__(self, parent, ml_engine, data_handler):
        super().__init__(parent)
        self.ml_engine = ml_engine
        self.data_handler = data_handler
        self.title("Single MSME Credit Assessment")
        self.configure(bg=BG_DARK)
        self.geometry("520x520")
        self.resizable(False, False)

        tk.Label(self, text="MSME Credit Risk Predictor",
                 font=FONT_HEAD, bg=BG_DARK, fg=ACCENT).pack(pady=(18, 4))
        tk.Label(self, text="Enter MSME details for instant risk assessment",
                 font=FONT_SMALL, bg=BG_DARK, fg=TEXT_SUB).pack(pady=(0, 14))

        card = make_card(self)
        card.pack(fill="x", padx=24, pady=4)

        self.fields = {}
        labels = [
            ("Annual Turnover (₹ Lakhs)", "turnover"),
            ("No. of Employees",          "employees"),
            ("Years in Operation",        "years"),
            ("Outstanding Loans (₹ L)",   "loans"),
            ("Digital Transaction Score", "digital"),
        ]
        for row_i, (label, key) in enumerate(labels):
            tk.Label(card, text=label, font=FONT_BODY, bg=BG_CARD, fg=TEXT_MAIN,
                     anchor="w").grid(row=row_i, column=0, sticky="w", padx=16, pady=8)
            ent = tk.Entry(card, font=FONT_MONO, bg=BG_DARK, fg=TEXT_MAIN,
                           insertbackground=ACCENT, relief="flat",
                           highlightthickness=1, highlightcolor=ACCENT,
                           highlightbackground=BORDER, width=20)
            ent.grid(row=row_i, column=1, padx=16, pady=8)
            self.fields[key] = ent

        self.result_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.result_var,
                 font=("Segoe UI", 11, "bold"), bg=BG_DARK, fg=SUCCESS,
                 wraplength=460, justify="center").pack(pady=12)

        styled_button(self, "⚡  Assess Credit Risk", self._assess, color=ACCENT).pack(pady=4)
        styled_button(self, "✕  Close", self.destroy, color=BORDER, width=12).pack(pady=4)

    def _assess(self):
        try:
            vals = {k: float(e.get() or 0) for k, e in self.fields.items()}
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.", parent=self)
            return

        turnover = vals['turnover']
        loans    = vals['loans']
        digital  = vals['digital']
        years    = vals['years']

        # Simple heuristic scoring (works even without training data)
        score = 0
        score += min(40, int(turnover / 10))
        score += min(20, int(digital / 5))
        score += min(20, int(years * 2))
        score -= min(30, int(loans / 20))
        score = max(0, min(100, score))

        if score > 70:
            risk, color, rec = "LOW RISK ✅", SUCCESS, "Eligible for standard credit line. Recommended limit: ₹{:.0f}L".format(turnover * 0.5)
        elif score > 40:
            risk, color, rec = "MEDIUM RISK ⚠️", WARNING, "Conditional approval. Collateral or co-applicant advised."
        else:
            risk, color, rec = "HIGH RISK ❌", DANGER, "Credit not recommended currently. Suggest capacity building."

        self.result_var.set(f"Health Score: {score}/100   │   Risk: {risk}\n{rec}")


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class ArthVigyanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arth-Vigyan  │  AI-Powered MSME Credit-Risk Engine")
        self.root.geometry("1200x780")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(1000, 680)

        self.data_handler = DataHandler()
        self.ml_engine    = MLEngine()
        self.df           = None

        self._apply_ttk_style()
        self._build_header()
        self._build_sidebar()
        self._build_tabs()
        self._build_statusbar()

    # ── TTK STYLE ────────────────────────────────────────────────────────────
    def _apply_ttk_style(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",           background=BG_DARK,  borderwidth=0)
        style.configure("TNotebook.Tab",       background=BG_PANEL, foreground=TEXT_SUB,
                        font=("Segoe UI", 10, "bold"), padding=[14, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", ACCENT)])
        style.configure("TLabelframe",         background=BG_PANEL, bordercolor=BORDER)
        style.configure("TLabelframe.Label",   background=BG_PANEL, foreground=ACCENT,
                        font=("Segoe UI", 10, "bold"))
        style.configure("Vertical.TScrollbar", background=BG_CARD, troughcolor=BG_DARK,
                        bordercolor=BORDER, arrowcolor=TEXT_SUB)

    # ── HEADER ───────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=64)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=BG_PANEL)
        left.pack(side="left", padx=20, pady=10)

        tk.Label(left, text="⚡ ARTH-VIGYAN",
                 font=("Segoe UI", 17, "bold"), bg=BG_PANEL, fg=ACCENT).pack(side="left")
        tk.Label(left, text="  AI-Powered MSME Credit-Risk Engine",
                 font=("Segoe UI", 11), bg=BG_PANEL, fg=TEXT_SUB).pack(side="left")

        right = tk.Frame(hdr, bg=BG_PANEL)
        right.pack(side="right", padx=20)
        tk.Label(right, text="B.Tech CSE  │  Arth-Vigyan Project",
                 font=FONT_SMALL, bg=BG_PANEL, fg=TEXT_SUB).pack()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = tk.Frame(self.root, bg=BG_PANEL, width=210)
        self.sidebar.pack(side="left", fill="y", padx=(0, 0))
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="SYSTEM CONTROLS",
                 font=("Segoe UI", 8, "bold"), bg=BG_PANEL, fg=TEXT_SUB).pack(pady=(18, 4))

        buttons = [
            ("📂  Load CSV & Analyse",      self.load_and_analyze,  ACCENT),
            ("🧍  Single MSME Assessment",   self.open_single_assess, "#7C3AED"),
            ("💾  Save Report to MongoDB",   self.save_to_mongo,     "#059669"),
            ("📄  Export Local Text File",   self.export_text,       "#D97706"),
        ]
        for text, cmd, color in buttons:
            styled_button(self.sidebar, text, cmd, color=color, width=22).pack(
                pady=5, padx=12)

        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=12, pady=14)

        tk.Label(self.sidebar, text="DATA SUMMARY",
                 font=("Segoe UI", 8, "bold"), bg=BG_PANEL, fg=TEXT_SUB).pack(pady=(0, 4))

        self.summary_text = tk.Text(self.sidebar, font=("Consolas", 8),
                                    bg=BG_CARD, fg=TEXT_SUB, relief="flat",
                                    width=26, height=12, bd=0,
                                    highlightthickness=0, state="disabled")
        self.summary_text.pack(padx=8, pady=4)

    # ── TABS ─────────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        # TAB 1 – Credit Assessment Report
        self.tab_report = tk.Frame(self.tabs, bg=BG_DARK)
        self.tabs.add(self.tab_report, text="📄  Assessment Report")
        self.output = scrolledtext.ScrolledText(
            self.tab_report, font=FONT_MONO,
            bg=BG_CARD, fg=TEXT_MAIN,
            insertbackground=ACCENT, relief="flat",
            selectbackground=ACCENT, selectforeground="white",
            borderwidth=0, highlightthickness=0)
        self.output.pack(fill="both", expand=True, padx=6, pady=6)
        self.output.tag_config("header",  foreground=ACCENT,   font=("Consolas", 10, "bold"))
        self.output.tag_config("success", foreground=SUCCESS)
        self.output.tag_config("warn",    foreground=WARNING)
        self.output.tag_config("danger",  foreground=DANGER)
        self.output.tag_config("sub",     foreground=TEXT_SUB)

        # TAB 2 – District Analysis
        self.tab_district = tk.Frame(self.tabs, bg=BG_DARK)
        self.tabs.add(self.tab_district, text="🏙️  District Analysis")
        self._build_district_tab()

        # TAB 3 – Insights & Analytics
        self.tab_viz = tk.Frame(self.tabs, bg=BG_DARK)
        self.tabs.add(self.tab_viz, text="📊  Insights & Analytics")
        self.chart_frame = tk.Frame(self.tab_viz, bg=BG_DARK)
        self.chart_frame.pack(fill="both", expand=True)

        # TAB 4 – Batch Processing
        self.tab_batch = tk.Frame(self.tabs, bg=BG_DARK)
        self.tabs.add(self.tab_batch, text="⚙️  Batch Processing")
        self._build_batch_tab()

    def _build_district_tab(self):
        top = tk.Frame(self.tab_district, bg=BG_DARK)
        top.pack(fill="x", padx=10, pady=(10, 0))

        tk.Label(top, text="District-Level Economic Analysis",
                 font=FONT_HEAD, bg=BG_DARK, fg=ACCENT).pack(side="left", pady=6)

        tk.Label(top, text="(Load CSV first to populate)",
                 font=FONT_SMALL, bg=BG_DARK, fg=TEXT_SUB).pack(side="left", padx=10)

        # Treeview for district table
        cols = ("District", "Health Score", "Risk Level", "Segment", "Econ. Activity")
        self.district_tree = ttk.Treeview(self.tab_district, columns=cols,
                                          show="headings", height=18)
        style = ttk.Style()
        style.configure("Treeview",
                        background=BG_CARD, fieldbackground=BG_CARD,
                        foreground=TEXT_MAIN, rowheight=26,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading",
                        background=BG_PANEL, foreground=ACCENT,
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])

        for col in cols:
            self.district_tree.heading(col, text=col)
            self.district_tree.column(col, anchor="center", width=140)

        vsb = ttk.Scrollbar(self.tab_district, orient="vertical",
                            command=self.district_tree.yview)
        self.district_tree.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y", pady=(4, 4))
        self.district_tree.pack(fill="both", expand=True, padx=(10, 0), pady=(4, 10))

        # Tag colours
        self.district_tree.tag_configure("low",    background="#0D2E1A", foreground="#34D399")
        self.district_tree.tag_configure("medium", background="#2D1F00", foreground="#FBBF24")
        self.district_tree.tag_configure("high",   background="#2D0D0D", foreground="#F87171")

    def _build_batch_tab(self):
        top = tk.Frame(self.tab_batch, bg=BG_DARK)
        top.pack(fill="x", padx=10, pady=(10, 2))

        tk.Label(top, text="Batch MSME Processing",
                 font=FONT_HEAD, bg=BG_DARK, fg=ACCENT).pack(side="left", pady=6)

        btn_frame = tk.Frame(self.tab_batch, bg=BG_DARK)
        btn_frame.pack(fill="x", padx=10, pady=4)

        styled_button(btn_frame, "🔄  Process Loaded CSV Batch",
                      self._run_batch_process, color=ACCENT, width=26).pack(side="left", padx=4)
        styled_button(btn_frame, "📋  Copy Batch Summary",
                      self._copy_batch_summary, color="#334155", width=22).pack(side="left", padx=4)
        styled_button(btn_frame, "💾  Export Batch CSV",
                      self._export_batch_csv, color="#059669", width=20).pack(side="left", padx=4)

        self.batch_output = scrolledtext.ScrolledText(
            self.tab_batch, font=FONT_MONO,
            bg=BG_CARD, fg=TEXT_MAIN, relief="flat",
            insertbackground=ACCENT, borderwidth=0, highlightthickness=0)
        self.batch_output.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    # ── STATUSBAR ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=BG_PANEL, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready — Load a CSV to begin analysis.")
        tk.Label(bar, textvariable=self.status_var,
                 font=FONT_SMALL, bg=BG_PANEL, fg=TEXT_SUB,
                 anchor="w").pack(side="left", padx=16)
        tk.Label(bar, text="Arth-Vigyan © 2025",
                 font=FONT_SMALL, bg=BG_PANEL, fg=BORDER).pack(side="right", padx=16)

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ── HELPERS ──────────────────────────────────────────────────────────────
    def _log(self, text, tag=None):
        self.output.configure(state="normal")
        if tag:
            self.output.insert(tk.END, text, tag)
        else:
            self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.configure(state="disabled")
        self.root.update_idletasks()

    def _update_summary_panel(self):
        if self.df is None:
            return
        rows, cols = self.df.shape
        text = (
            f" Rows    : {rows}\n"
            f" Columns : {cols}\n"
            f" Target  : {self.data_handler.target_col}\n\n"
            f" Features\n"
            f" Engineered:\n"
        )
        for f in self.data_handler.engineered_features:
            text += f"  • {f}\n"
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, text)
        self.summary_text.configure(state="disabled")

    # ── LOAD & ANALYSE ───────────────────────────────────────────────────────
    def load_and_analyze(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path:
            return

        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.configure(state="disabled")

        self._log(f"{'='*55}\n", "header")
        self._log(f"  ARTH-VIGYAN ANALYSIS ENGINE\n", "header")
        self._log(f"{'='*55}\n", "header")
        self._log(f"\n  FILE : {os.path.basename(path)}\n\n")
        self._set_status(f"Loading: {os.path.basename(path)} …")

        success, msg = self.data_handler.load_csv(path)
        if not success:
            messagebox.showerror("Data Error", msg)
            self._log(f"  [✗] {msg}\n", "danger")
            return

        self.df = self.data_handler.get_processed_data()
        self._log(f"  [✓] {msg}\n", "success")
        self._update_summary_panel()
        self._set_status("Training ML models …")

        self._log(f"\n  TRAINING ML MODELS\n  {'-'*40}\n")
        ml_success, ml_msg = self.ml_engine.train_credit_risk_models(
            self.df, self.data_handler.target_col)

        if not ml_success:
            messagebox.showerror("ML Error", ml_msg)
            self._log(f"  [✗] {ml_msg}\n", "danger")
            return

        self._log(f"  [✓] {ml_msg}\n\n", "success")

        # ── Assign health scores via GB regressor ──
        health_scores = self.ml_engine.predict_health_scores(
            self.df, self.data_handler.target_col)
        self.df['Health_Score'] = health_scores
        self.df['Risk_Level'] = self.df['Health_Score'].apply(
            lambda s: "Low" if s > 65 else ("Medium" if s > 35 else "High"))

        # ── Assign district segment via KMeans ──
        segments = self.ml_engine.get_district_segments()
        if segments and len(segments) == len(self.df):
            self.df['Segment'] = segments
        else:
            self.df['Segment'] = "–"

        # ── Print report table ──
        self._log(f"  {'DISTRICT/ENTITY':<24} {'RISK':<10} {'SCORE':>6}  {'SEGMENT'}\n", "header")
        self._log(f"  {'-'*55}\n", "sub")

        name_col = next(
            (c for c in self.df.columns
             if any(k in str(c).lower() for k in ['district', 'name', 'entity', 'unit'])),
            None)

        for idx, row in self.df.head(30).iterrows():
            name = str(row[name_col])[:22] if name_col else f"Record_{idx+1}"
            risk = row['Risk_Level']
            score = row['Health_Score']
            seg   = row.get('Segment', '–')
            tag   = "success" if risk == "Low" else ("warn" if risk == "Medium" else "danger")
            self._log(
                f"  {name:<24} {risk:<10} {score:>5}/100  {seg}\n", tag)

        # ── Model performance ──
        self._log(f"\n\n{self.ml_engine.get_model_report()}\n", "header")

        # ── Update district tab ──
        self._populate_district_table(name_col)

        # ── Draw charts ──
        self._draw_charts()

        self._set_status(f"Analysis complete — {len(self.df)} records processed.")
        messagebox.showinfo("Analysis Complete",
                            f"{ml_msg}\n\nVisualisations updated in Insights tab.")

    # ── DISTRICT TABLE ───────────────────────────────────────────────────────
    def _populate_district_table(self, name_col):
        for item in self.district_tree.get_children():
            self.district_tree.delete(item)

        for _, row in self.df.iterrows():
            name  = str(row[name_col])[:30] if name_col else "–"
            score = row['Health_Score']
            risk  = row['Risk_Level']
            seg   = row.get('Segment', '–')
            econ  = round(row.get('Economic_Activity_Score', 0), 3)
            tag   = risk.lower()
            self.district_tree.insert("", "end",
                                      values=(name, score, risk, seg, econ),
                                      tags=(tag,))

    # ── CHARTS ───────────────────────────────────────────────────────────────
    def _draw_charts(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        plt.style.use('dark_background')
        fig = Figure(figsize=(10, 7), dpi=96, facecolor=BG_DARK)

        # 1) Pie – Risk Distribution
        ax1 = fig.add_subplot(2, 2, 1)
        low  = len(self.df[self.df['Risk_Level'] == 'Low'])
        med  = len(self.df[self.df['Risk_Level'] == 'Medium'])
        high = len(self.df[self.df['Risk_Level'] == 'High'])
        counts = [low, med, high]
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        colors = [SUCCESS, WARNING, DANGER]
        if sum(counts) > 0:
            ax1.pie(counts, labels=labels, autopct='%1.1f%%',
                    colors=colors, shadow=True, startangle=140,
                    textprops={'color': TEXT_MAIN, 'fontsize': 9})
        ax1.set_title("Risk Distribution", color=ACCENT, fontsize=11, fontweight='bold')
        ax1.set_facecolor(BG_CARD)

        # 2) Bar – Health Score Top 15
        ax2 = fig.add_subplot(2, 2, 2)
        top15 = self.df.nlargest(15, 'Health_Score')
        name_col = next(
            (c for c in self.df.columns
             if any(k in str(c).lower() for k in ['district', 'name', 'entity', 'unit'])),
            None)
        names  = [str(r[name_col])[:10] if name_col else f"R{i}" for i, r in top15.iterrows()]
        scores = top15['Health_Score'].tolist()
        bar_colors = [SUCCESS if s > 65 else (WARNING if s > 35 else DANGER) for s in scores]
        ax2.barh(names, scores, color=bar_colors)
        ax2.set_xlabel("Health Score", color=TEXT_SUB, fontsize=8)
        ax2.set_title("Top 15 – Health Scores", color=ACCENT, fontsize=11, fontweight='bold')
        ax2.set_facecolor(BG_CARD)
        ax2.tick_params(colors=TEXT_SUB, labelsize=8)
        ax2.spines['bottom'].set_color(BORDER)
        ax2.spines['left'].set_color(BORDER)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # 3) Bar – District Segment Distribution
        ax3 = fig.add_subplot(2, 2, 3)
        if 'Segment' in self.df.columns:
            seg_counts = self.df['Segment'].value_counts()
            seg_colors = {"Emerging": ACCENT, "Stable": SUCCESS, "Declining": DANGER}
            ax3.bar(seg_counts.index, seg_counts.values,
                    color=[seg_colors.get(s, ACCENT2) for s in seg_counts.index])
        ax3.set_title("District Segments (KMeans)", color=ACCENT, fontsize=11, fontweight='bold')
        ax3.set_ylabel("Count", color=TEXT_SUB, fontsize=8)
        ax3.set_facecolor(BG_CARD)
        ax3.tick_params(colors=TEXT_SUB, labelsize=9)
        ax3.spines['bottom'].set_color(BORDER)
        ax3.spines['left'].set_color(BORDER)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # 4) Horizontal bar – Feature Importances
        ax4 = fig.add_subplot(2, 2, 4)
        if self.ml_engine.feature_importances:
            top_feats = list(self.ml_engine.feature_importances.items())[:8]
            feat_names = [f[0][:16] for f in top_feats]
            feat_vals  = [f[1] for f in top_feats]
            ax4.barh(feat_names, feat_vals, color=ACCENT2)
            ax4.set_title("Feature Importances (RF)", color=ACCENT, fontsize=11, fontweight='bold')
            ax4.set_facecolor(BG_CARD)
            ax4.tick_params(colors=TEXT_SUB, labelsize=8)
            ax4.spines['bottom'].set_color(BORDER)
            ax4.spines['left'].set_color(BORDER)
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)

        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── SINGLE ASSESSMENT ────────────────────────────────────────────────────
    def open_single_assess(self):
        SingleAssessmentDialog(self.root, self.ml_engine, self.data_handler)

    # ── BATCH PROCESSING ─────────────────────────────────────────────────────
    def _run_batch_process(self):
        if self.df is None or 'Risk_Level' not in self.df.columns:
            messagebox.showwarning("Warning", "Please load and analyse a CSV first.")
            return

        self.batch_output.delete("1.0", tk.END)

        total   = len(self.df)
        low_c   = len(self.df[self.df['Risk_Level'] == 'Low'])
        med_c   = len(self.df[self.df['Risk_Level'] == 'Medium'])
        high_c  = len(self.df[self.df['Risk_Level'] == 'High'])
        avg_scr = self.df['Health_Score'].mean()

        summary = (
            f"{'='*55}\n"
            f"  BATCH PROCESSING SUMMARY REPORT\n"
            f"{'='*55}\n\n"
            f"  Total Records Processed : {total}\n"
            f"  Avg. Health Score       : {avg_scr:.1f} / 100\n\n"
            f"  Risk Breakdown:\n"
            f"    ✅ Low  Risk : {low_c:>4}  ({low_c/total*100:.1f}%)\n"
            f"    ⚠️  Med  Risk : {med_c:>4}  ({med_c/total*100:.1f}%)\n"
            f"    ❌ High Risk : {high_c:>4}  ({high_c/total*100:.1f}%)\n\n"
        )

        if 'Segment' in self.df.columns:
            seg_counts = self.df['Segment'].value_counts()
            summary += "  District Segmentation (KMeans):\n"
            for seg, cnt in seg_counts.items():
                summary += f"    • {seg:<12}: {cnt}\n"
            summary += "\n"

        summary += f"\n  Model Used: {self.ml_engine.best_model_name}\n"
        summary += self.ml_engine.get_model_report() + "\n"

        self.batch_output.insert(tk.END, summary)
        self._set_status("Batch summary generated.")

    def _copy_batch_summary(self):
        text = self.batch_output.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Batch summary copied to clipboard.")

    def _export_batch_csv(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No data loaded.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV File", "*.csv")])
        if path:
            self.df.to_csv(path, index=False)
            messagebox.showinfo("Exported", f"Batch report exported to:\n{path}")

    # ── MONGODB ──────────────────────────────────────────────────────────────
    def save_to_mongo(self):
        if self.df is None or 'Risk_Level' not in self.df.columns:
            messagebox.showwarning("Warning", "Please click 'Load CSV & Analyse' first!")
            return

        self._log("\n  >>> SAVING REPORT TO MONGODB …\n")
        self._set_status("Saving to MongoDB …")

        success, msg = self.data_handler.save_to_mongodb(self.df)
        if success:
            self._log(f"  [✓] {msg}\n", "success")
            messagebox.showinfo("MongoDB Success", msg)
        else:
            self._log(f"  [✗] {msg}\n", "danger")
            messagebox.showerror("Database Error", msg)

        self._set_status(msg)

    # ── EXPORT TEXT ──────────────────────────────────────────────────────────
    def export_text(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No analysis to export yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text File", "*.txt")])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.output.get("1.0", tk.END))
            messagebox.showinfo("Exported", f"Report saved to:\n{path}")
            self._set_status(f"Report exported → {os.path.basename(path)}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = ArthVigyanApp(root)
    root.mainloop()
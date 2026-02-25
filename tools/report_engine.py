"""
tools/report_engine.py — Excel + professional PDF report generator.

Reads from SharedState (populated by all other agents) and produces:
  • Multi-sheet Excel workbook  (.xlsx)
  • Formatted investment PDF report (.pdf)

════════════════════════════════════════════════════════════════════
在此处添加新的 API Key 和工具函数（例如上传至云端 / 发送邮件）:

  import os
  SMTP_HOST     = os.getenv("SMTP_HOST", "")
  SMTP_USER     = os.getenv("SMTP_USER", "")
  SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

  def email_report(self, recipient: str) -> str:
      # 将生成的 PDF 报告通过 SMTP 发送给指定邮箱
      ...
════════════════════════════════════════════════════════════════════
"""

import os
from datetime import datetime

import pandas as pd
from agno.utils.log import logger

from tools.base_tool import BaseTool
from state import get_state


class ReportEngine(BaseTool):
    tool_name = "report_engine"
    tool_description = (
        "Generates Excel and PDF investment reports from SharedState data. "
        "Call save_full_report() to produce both files at once."
    )

    def __init__(self, output_dir: str = "reports"):
        super().__init__(name=self.tool_name)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.register(self.save_full_report)
        self.register(self.save_to_excel)
        self.register(self.save_to_pdf)

        # ════════════════════════════════════════════════════════════════
        # 在此处添加新的 API Key（报告分发渠道）
        # self.smtp_host     = os.getenv("SMTP_HOST", "")
        # self.smtp_user     = os.getenv("SMTP_USER", "")
        # self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        # self.s3_bucket     = os.getenv("AWS_S3_BUCKET", "")
        # ════════════════════════════════════════════════════════════════

    # ── Main entry point ────────────────────────────────────────────────

    def save_full_report(self, ticker: str = "") -> str:
        """
        Generates both Excel and PDF reports from the current SharedState.
        This is the primary method the ReportManager agent should call.

        Args:
            ticker: Override ticker if needed (default: uses SharedState.ticker).
        """
        state = get_state()
        if ticker:
            state.ticker = ticker.upper()

        results = []
        results.append(self.save_to_excel())
        results.append(self.save_to_pdf())
        return "\n".join(results)

    # ── Excel ────────────────────────────────────────────────────────────

    def save_to_excel(self, filename: str = "") -> str:
        """
        Writes all SharedState tabular data to a multi-sheet Excel workbook.

        Args:
            filename: Custom filename (auto-generated from ticker if omitted).
        """
        try:
            state = get_state()
            if not filename:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{state.ticker or 'report'}_{ts}.xlsx"

            filepath = os.path.join(self.output_dir, filename)
            sheets = state.to_excel_sheets()

            if not sheets:
                return "No data in SharedState to export."

            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                for sheet_name, content in sheets.items():
                    name = str(sheet_name)[:31]
                    if isinstance(content, pd.DataFrame):
                        df = content
                    elif isinstance(content, list) and content and isinstance(content[0], dict):
                        df = pd.DataFrame(content)
                    elif isinstance(content, dict):
                        df = pd.DataFrame([content])
                    else:
                        df = pd.DataFrame({"Value": [str(content)]})
                    df.to_excel(writer, sheet_name=name, index=True)

            state.excel_path = filepath
            logger.info(f"[report_engine] Excel saved: {filepath}")
            return f"✅ Excel report saved: {filepath}"
        except Exception as e:
            logger.error(f"[report_engine] save_to_excel error: {e}")
            return f"Error saving Excel: {e}"

    # ── PDF ──────────────────────────────────────────────────────────────

    def save_to_pdf(self, filename: str = "") -> str:
        """
        Generates a professional PDF investment report from SharedState.
        Uses reportlab for layout; falls back to .txt if reportlab unavailable.

        Args:
            filename: Custom filename (auto-generated from ticker if omitted).
        """
        state = get_state()
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{state.ticker or 'report'}_{ts}.pdf"

        try:
            return self._build_pdf(state, filename)
        except ImportError:
            return self._build_txt_fallback(state, filename)
        except Exception as e:
            logger.error(f"[report_engine] save_to_pdf error: {e}")
            return f"Error saving PDF: {e}"

    # ── PDF builder (reportlab) ──────────────────────────────────────────

    def _build_pdf(self, state, filename: str) -> str:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
            Table, TableStyle, KeepTogether,
        )

        filepath = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        # ── Colour palette ──────────────────────────────────────────────
        NAVY   = colors.HexColor("#0d1b2a")
        BLUE   = colors.HexColor("#1e3a5f")
        ACCENT = colors.HexColor("#2196f3")
        GREEN  = colors.HexColor("#1b5e20")
        GREEN_BG = colors.HexColor("#e8f5e9")
        RED    = colors.HexColor("#b71c1c")
        RED_BG = colors.HexColor("#ffebee")
        AMBER  = colors.HexColor("#e65100")
        AMBER_BG = colors.HexColor("#fff3e0")
        GREY   = colors.HexColor("#546e7a")
        LIGHT  = colors.HexColor("#eceff1")

        styles = getSampleStyleSheet()
        def _s(name, parent="Normal", **kw):
            return ParagraphStyle(name, parent=styles[parent], **kw)

        S = {
            "title":    _s("T", "Title",   fontSize=22, textColor=NAVY, spaceAfter=4),
            "subtitle": _s("S", "Normal",  fontSize=10, textColor=GREY, spaceAfter=2),
            "h2":       _s("H2","Heading2", fontSize=13, textColor=BLUE, spaceBefore=14, spaceAfter=4),
            "h3":       _s("H3","Heading3", fontSize=10, textColor=BLUE, spaceBefore=8, spaceAfter=2),
            "body":     _s("B", "Normal",  fontSize=9,  leading=14, spaceAfter=3),
            "mono":     _s("M", "Normal",  fontSize=8,  fontName="Courier", leading=11),
            "disc":     _s("D", "Normal",  fontSize=7,  textColor=GREY, spaceAfter=0),
        }

        def rec_style(rec: str):
            r = rec.upper()
            if "BUY" in r:   return _s("REC","Normal", fontSize=14, fontName="Helvetica-Bold",
                                        textColor=GREEN, backColor=GREEN_BG, borderPad=8)
            if "SELL" in r:  return _s("REC","Normal", fontSize=14, fontName="Helvetica-Bold",
                                        textColor=RED, backColor=RED_BG, borderPad=8)
            return _s("REC","Normal", fontSize=14, fontName="Helvetica-Bold",
                       textColor=AMBER, backColor=AMBER_BG, borderPad=8)

        def _p(text, style="body"):
            safe = (str(text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
            return Paragraph(safe, S[style])

        def _hr(color=ACCENT, thickness=1):
            return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=4)

        story = []

        # ── Header ──────────────────────────────────────────────────────
        rec = state.recommendation or "—"
        story += [
            _p(f"Investment Analysis Report", "title"),
            _p(f"{state.company_name or state.ticker}  ({state.ticker})  ·  {state.timestamp}", "subtitle"),
            _hr(ACCENT, 1.5),
            Spacer(1, 0.3*cm),
        ]

        # Recommendation banner
        if state.recommendation:
            banner_text = (f"{'✅' if 'BUY' in rec.upper() else '🔴' if 'SELL' in rec.upper() else '🟡'}  "
                           f"RECOMMENDATION: {rec.upper()}"
                           f"{'  |  Target: ' + state.target_price if state.target_price else ''}"
                           f"{'  |  Conviction: ' + state.conviction if state.conviction else ''}")
            story.append(Paragraph(banner_text, rec_style(rec)))
            story.append(Spacer(1, 0.4*cm))

        # ── Key Metrics table ────────────────────────────────────────────
        if state.raw_metrics:
            story.append(_p("Key Metrics", "h2"))
            metrics = state.raw_metrics
            items = [(k, str(v)) for k, v in metrics.items() if k not in ("Ticker","Company")]
            half = len(items) // 2 + len(items) % 2
            col1, col2 = items[:half], items[half:]
            tdata = [["Metric", "Value", "Metric", "Value"]]
            for i in range(half):
                left_k, left_v = col1[i]
                right_k, right_v = col2[i] if i < len(col2) else ("", "")
                tdata.append([left_k, left_v, right_k, right_v])
            tbl = Table(tdata, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 3.5*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), BLUE),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",   (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
                ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cfd8dc")),
                ("PADDING",    (0,0), (-1,-1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.3*cm))

        # ── Financial summary ────────────────────────────────────────────
        if state.company_summary:
            story.append(_hr(GREY, 0.5))
            story.append(_p("Fundamental Analysis", "h2"))
            for line in state.company_summary.split("\n"):
                if line.strip():
                    story.append(_p(line))

        # ── Macro & Sector ───────────────────────────────────────────────
        for title, content in [("Macro Context", state.macro_analysis),
                                ("Sector Dynamics", state.sector_analysis)]:
            if content:
                story.append(_hr(GREY, 0.5))
                story.append(_p(title, "h2"))
                for line in content.split("\n"):
                    if line.strip():
                        story.append(_p(line))

        # ── News ─────────────────────────────────────────────────────────
        if state.news_headlines:
            story.append(_hr(GREY, 0.5))
            story.append(_p("Recent News & Sentiment", "h2"))
            if state.news_sentiment:
                story.append(_p(f"Overall Sentiment: {state.news_sentiment}"))
            for headline in state.news_headlines:
                story.append(_p(f"• {headline}"))

        # ── CIO Reasoning ────────────────────────────────────────────────
        if state.cio_reasoning:
            story.append(_hr(ACCENT, 0.5))
            story.append(_p("CIO Investment Thesis (DeepSeek-R1 Reasoning)", "h2"))
            for line in state.cio_reasoning.split("\n"):
                if line.strip():
                    story.append(_p(line))

        # ── Risk Factors ─────────────────────────────────────────────────
        if state.risk_factors:
            story.append(_hr(GREY, 0.5))
            story.append(_p("Key Risk Factors", "h2"))
            for risk in state.risk_factors:
                story.append(_p(f"⚠  {risk}"))

        # ── Catalysts ────────────────────────────────────────────────────
        if state.catalysts:
            story.append(_p("Catalysts to Watch", "h3"))
            for cat in state.catalysts:
                story.append(_p(f"→  {cat}"))

        # ── Disclaimer ───────────────────────────────────────────────────
        story += [
            Spacer(1, 0.6*cm),
            _hr(GREY, 0.5),
            _p("Disclaimer: This report is generated by an AI multi-agent system "
               "for informational purposes only. It does not constitute financial advice. "
               "Past performance is not indicative of future results. "
               "Always conduct your own due diligence.", "disc"),
        ]

        doc.build(story)
        state.pdf_path = filepath
        logger.info(f"[report_engine] PDF saved: {filepath}")
        return f"✅ PDF report saved: {filepath}"

    # ── Plain-text fallback ──────────────────────────────────────────────

    def _build_txt_fallback(self, state, filename: str) -> str:
        filepath = os.path.join(self.output_dir, filename.replace(".pdf", ".txt"))
        sep = "=" * 65
        lines = [
            sep,
            f"  INVESTMENT ANALYSIS REPORT",
            f"  {state.company_name or state.ticker}  ({state.ticker})",
            f"  Generated: {state.timestamp}",
            sep, "",
            f"RECOMMENDATION: {state.recommendation}",
            f"Target Price  : {state.target_price}",
            f"Conviction    : {state.conviction}",
            f"Time Horizon  : {state.time_horizon}", "",
            "─" * 65,
            "FINANCIAL SUMMARY", "─" * 65,
            state.company_summary or "(no data)", "",
            "─" * 65,
            "NEWS & SENTIMENT", "─" * 65,
            state.news_sentiment,
            *[f"• {h}" for h in state.news_headlines], "",
            "─" * 65,
            "CIO REASONING", "─" * 65,
            state.cio_reasoning or "(no data)", "",
            "─" * 65,
            "RISK FACTORS", "─" * 65,
            *[f"⚠  {r}" for r in state.risk_factors], "",
            sep,
            "Disclaimer: AI-generated. Not financial advice.",
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        state.pdf_path = filepath
        return f"✅ Text report saved (install reportlab for PDF): {filepath}"

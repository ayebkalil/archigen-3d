"""
Script: generate_technical_report_pdf.py
Compiles the comprehensive, publication-grade Academic Technical Report
for the ArchiGen 3D project into a high-fidelity PDF document: report/rapport_technique.pdf.
Uses ReportLab with dynamic two-pass page numbering, running headers/footers,
executive callouts, formal LaTeX-aligned tables, and verified benchmark metrics.
"""

import os
import sys
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image as RLImage
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw total page count,
    running headers and running footers on all pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        w, h = A4

        # On cover page (page 1), suppress running header
        if self._pageNumber > 1:
            # Header (Top)
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0F294A"))
            self.drawString(36, h - 28, "ARCHIGEN 3D | RAPPORT TECHNIQUE DE PROJET INDUSTRIEL")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(w - 36, h - 28, "TEK-UP — CYCLE INGÉNIEUR IA & DATA SCIENCE 2026/2027")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(36, h - 33, w - 36, h - 33)

        # Footer (Bottom)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 32, w - 36, 32)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 22, "Enseignant Responsable : Prof. Haythem Ghazouani")
        
        page_str = f"Page {self._pageNumber} sur {page_count}"
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#0F294A"))
        self.drawCentredString(w / 2.0, 22, page_str)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(w - 36, 22, "Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari")

        self.restoreState()


def build_technical_report_pdf(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Academic Palette
    PRIMARY = colors.HexColor("#0F294A")      # Deep Navy
    SECONDARY = colors.HexColor("#1E5F8A")    # Steel Blue
    ACCENT = colors.HexColor("#0D9488")       # Deep Teal
    DARK_TEXT = colors.HexColor("#1E293B")    # Slate 800
    MUTED_TEXT = colors.HexColor("#475569")   # Slate 600
    LIGHT_BG = colors.HexColor("#F8FAFC")     # Slate 50
    ALT_BG = colors.HexColor("#F1F5F9")       # Slate 100
    BORDER_COLOR = colors.HexColor("#CBD5E1") # Slate 300
    CARD_BG = colors.HexColor("#F0FDF4")      # Emerald tint
    WARN_BG = colors.HexColor("#FFFBEB")      # Amber tint
    INFO_BG = colors.HexColor("#EFF6FF")      # Sky blue tint

    # Typography Styles
    body = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.6,
        textColor=DARK_TEXT,
        spaceAfter=4
    )

    body_bold = ParagraphStyle(
        'CustomBodyBold',
        parent=body,
        fontName='Helvetica-Bold'
    )

    body_small = ParagraphStyle(
        'CustomBodySmall',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.4,
        leading=10.0,
        textColor=MUTED_TEXT,
        spaceAfter=3
    )

    title_cover = ParagraphStyle(
        'CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        alignment=1,
        spaceAfter=4
    )

    subtitle_cover = ParagraphStyle(
        'CoverSubtitle',
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=SECONDARY,
        alignment=1,
        spaceAfter=8
    )

    h1 = ParagraphStyle(
        'CustomH1',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2 = ParagraphStyle(
        'CustomH2',
        fontName='Helvetica-Bold',
        fontSize=9.8,
        leading=12.5,
        textColor=SECONDARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    h3 = ParagraphStyle(
        'CustomH3',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.0,
        textColor=DARK_TEXT,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    tbl_hdr = ParagraphStyle(
        'TblHdr',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    tbl_hdr_left = ParagraphStyle(
        'TblHdrLeft',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0
    )

    tbl_cell = ParagraphStyle(
        'TblCell',
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.4,
        textColor=DARK_TEXT,
        alignment=1
    )

    tbl_cell_left = ParagraphStyle(
        'TblCellLeft',
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.4,
        textColor=DARK_TEXT,
        alignment=0
    )

    tbl_cell_bold = ParagraphStyle(
        'TblCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.4,
        textColor=PRIMARY,
        alignment=1
    )

    callout_txt = ParagraphStyle(
        'CalloutText',
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.8,
        textColor=DARK_TEXT
    )

    usable_w = A4[0] - 72  # 523.27 pt
    story = []

    # =========================================================================
    # PAGE 1 : COUVERTURE ACADÉMIQUE OFFICIELLE
    # =========================================================================
    header_cover = [
        [Paragraph("<b>RÉPUBLIQUE TUNISIENNE — MINISTÈRE DE L'ENSEIGNEMENT SUPÉRIEUR ET DE LA RECHERCHE SCIENTIFIQUE</b>", 
                   ParagraphStyle('HdrCov1', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#64748B"), alignment=1))],
        [Paragraph("<b>TEK-UP UNIVERSITÉ | DÉPARTEMENT GÉNIE LOGICIEL & INFORMATIQUE AVANCÉE</b>", 
                   ParagraphStyle('HdrCov2', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=PRIMARY, alignment=1))],
        [Paragraph("CYCLE INGÉNIEUR EN INTELLIGENCE ARTIFICIELLE & DATA SCIENCE — 3<sup>ème</sup> ANNÉE (2026/2027)", 
                   ParagraphStyle('HdrCov3', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=SECONDARY, alignment=1))],
        [Paragraph("PROJET INDUSTRIEL DE MODULE : DEEP LEARNING AVANCÉ & MLOPS", 
                   ParagraphStyle('HdrCov4', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=ACCENT, alignment=1))],
    ]
    t_hc = Table(header_cover, colWidths=[usable_w])
    t_hc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_hc)
    story.append(Spacer(1, 10))

    story.append(Paragraph("ArchiGen 3D", title_cover))
    story.append(Paragraph("Plateforme Industrielle de Synthèse Architecturale et d'Extrusion 3D par Vision Profonde et cGAN", subtitle_cover))
    story.append(Paragraph(
        "<i>Rapport Technique et Scientifique d'Évaluation Rigoureuse des Modèles de Deep Learning et du Cycle MLOps</i>",
        ParagraphStyle('Tagline', fontName='Helvetica-Oblique', fontSize=8.5, leading=11.5, textColor=MUTED_TEXT, alignment=1)
    ))
    story.append(Spacer(1, 8))

    # Authors and Supervisor Table
    team_data = [
        [
            Paragraph("<b>Membres de l'Équipe Ingénieur :</b><br/>"
                      "• <b>Khalil Ayeb</b><br/>"
                      "• <b>Amine Ben Fadhel</b><br/>"
                      "• <b>Khawla Hawari</b>", 
                      ParagraphStyle('TeamP', fontName='Helvetica', fontSize=8, leading=11, textColor=DARK_TEXT)),
            Paragraph("<b>Supervision Universitaire :</b><br/>"
                      "• <b>Prof. Haythem Ghazouani</b><br/>"
                      "<i>Enseignant Responsable Deep Learning & MLOps</i><br/>"
                      "<i>TEK-UP Tunis</i>", 
                      ParagraphStyle('SuperP', fontName='Helvetica', fontSize=8, leading=11, textColor=DARK_TEXT))
        ]
    ]
    t_team = Table(team_data, colWidths=[usable_w * 0.52, usable_w * 0.48])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ALT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 8))

    # Executive Abstract Callout
    abstract_text = (
        "<b>RÉSUMÉ EXÉCUTIF :</b> Ce rapport présente les résultats académiques et techniques du système "
        "<b>ArchiGen 3D</b>, solution complète de conversion de plans d'architectes 2D en maquettes 3D interactives étanches "
        "avec synthèse de façades photoréalistes. Le système combine deux modèles de Deep Learning : (1) un réseau multi-tâches "
        "<b>UNet++ (ResNet-34)</b> entraîné sur CubiCasa5K (4 200 train, 400 val, 400 test sous licence académique CC BY-NC 4.0) "
        "pour la segmentation conjointe des cloisons structurelles et des pièces, suivi d'une vectorisation polygonale hybride ; "
        "(2) un réseau génératif antagoniste conditionnel <b>Pix2Pix cGAN</b> stabilisé par normalisation spectrale, "
        "perte perceptuelle LPIPS (SqueezeNet) et matching de caractéristiques multi-échelles sur CMP Facades (400 train, 106 test). "
        "L'ensemble du pipeline est validé sur l'intégralité du jeu de test (400 plans non vus avec 1 000 ré-échantillonnages bootstrap), "
        "confronté à 12 plans d'archives réels du domaine public (HABS/LOC), profilé en temps réel (latence médiane 156.6 ms sur GPU), "
        "versionné sous DVC/MLflow, conteneurisé sous Docker Compose et couvert par 25 tests automatisés Pytest."
    )
    t_abs = Table([[Paragraph(abstract_text, callout_txt)]], colWidths=[usable_w])
    t_abs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), INFO_BG),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_abs)
    story.append(Spacer(1, 8))

    # KPI Banner
    kpis = [
        [
            Paragraph("<b>EXACTITUDE PIXEL</b><br/><font size=10 color='#0F294A'><b>88.09%</b></font><br/>[86.4, 89.6] (Test N=400)", ParagraphStyle('kp1', fontName='Helvetica', fontSize=7.2, leading=9.5, alignment=1)),
            Paragraph("<b>mIoU PIÈCES (9 CL.)</b><br/><font size=10 color='#0F294A'><b>64.72%</b></font><br/>[62.0, 67.1] (+2.05% TTA)", ParagraphStyle('kp2', fontName='Helvetica', fontSize=7.2, leading=9.5, alignment=1)),
            Paragraph("<b>IoU MURS FUSIONNÉS</b><br/><font size=10 color='#0F294A'><b>74.93%</b></font><br/>[74.0, 75.8] (p < 0.001)", ParagraphStyle('kp3', fontName='Helvetica', fontSize=7.2, leading=9.5, alignment=1)),
            Paragraph("<b>COQUES ÉTANCHES</b><br/><font size=10 color='#0F294A'><b>91.75%</b></font><br/>367 / 400 plans test", ParagraphStyle('kp4', fontName='Helvetica', fontSize=7.2, leading=9.5, alignment=1)),
            Paragraph("<b>KID FAÇADE (x10³)</b><br/><font size=10 color='#0F294A'><b>67.65 ± 5.9</b></font><br/>LPIPS: 0.3169 (CMP N=106)", ParagraphStyle('kp5', fontName='Helvetica', fontSize=7.2, leading=9.5, alignment=1)),
            Paragraph("<b>LATENCE MÉDIANE</b><br/><font size=10 color='#0F294A'><b>156.6 ms</b></font><br/>P95: 217.7 ms (GPU)", ParagraphStyle('kp6', fontName='Helvetica', fontSize=7.2, leading=9.5, alignment=1)),
        ]
    ]
    t_kpis = Table(kpis, colWidths=[usable_w / 6.0] * 6)
    t_kpis.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_kpis)
    story.append(Spacer(1, 8))

    # Architecture Overview Callout
    arch_summary = (
        "<b>SYNTHÈSE DES DEUX MODÈLES EN PRODUCTION :</b><br/>"
        "• <b>Modèle 1 (Segmentation UNet++ Multi-Tâches) :</b> Encodeur ResNet-34 pré-entraîné ImageNet, têtes convolutives "
        "parallèles à double flux : (1) Tête structurelle 5 classes (fond, mur int., mur ext., porte, fenêtre) et (2) Tête sémantique "
        "12 classes (11 pièces + fond). Fonction de perte : Weighted Cross-Entropy (poids porte x25, mur x4) combinée à la perte Dice multimodale.<br/>"
        "• <b>Modèle 2 (Générateur Pix2Pix cGAN Perceptuel) :</b> UNet-256 avec sur-échantillonnage Nearest-Neighbor + Conv "
        "(supprimant totalement les artéfacts de damier de TransposeConv), PatchGAN 70x70 stabilisé par Normalisation Spectrale (1-Lipschitz). "
        "Perte hybride : LSGAN + 30.0 L1 + 10.0 Feature Matching + 10.0 LPIPS Perceptual."
    )
    t_arch = Table([[Paragraph(arch_summary, body_small)]], colWidths=[usable_w])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ALT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_arch)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2 : BENCHMARK OFFICIEL DU JEU DE TEST (400 PLANS NON VUS)
    # =========================================================================
    story.append(Paragraph("1. Évaluation Statistique sur le Jeu de Test Officiel CubiCasa5K", h1))
    story.append(Paragraph(
        "L'évaluation finale est exécutée sur l'intégralité des <b>400 plans de test non vus</b> du découpage officiel CubiCasa5K "
        "(SHA256 du checkpoint : <font face='Courier' size=6.5>80fa1ea7780f7cb2f48d720abed8a62104ac90e0e40d99661f3022f3b472fcca</font>). "
        "Chaque métrique est accompagnée de son intervalle de confiance à 95% calculé par 1 000 ré-échantillonnages bootstrap appariés "
        "sur les plans. Cette méthodologie garantit une mesure non biaisée de la généralisation et de l'apport du 4-Fold Test-Time Augmentation (TTA).",
        body
    ))
    story.append(Spacer(1, 4))

    # Table 4.2 Floorplan Authoritative Test
    t42_data = [
        [
            Paragraph("<b>Métrique Évaluée</b>", tbl_hdr_left),
            Paragraph("<b>Passe Unique (Base)</b>", tbl_hdr),
            Paragraph("<b>4-Fold TTA</b>", tbl_hdr),
            Paragraph("<b>Delta (Gain Absolu)</b>", tbl_hdr),
            Paragraph("<b>Signif. Statistique (p)</b>", tbl_hdr),
        ],
        [Paragraph("<b>Exactitude Pixel Pièces</b>", tbl_cell_left), Paragraph("87.50% [85.8, 89.1]", tbl_cell), Paragraph("<b>88.09%</b> [86.4, 89.6]", tbl_cell_bold), Paragraph("+0.59% [+0.10, +1.05]", tbl_cell), Paragraph("p = 0.008 (**)", tbl_cell)],
        [Paragraph("<b>mIoU Pièces (9 cl., hors Bureau)</b>", tbl_cell_left), Paragraph("62.66% [60.0, 65.3]", tbl_cell), Paragraph("<b>64.72%</b> [62.0, 67.1]", tbl_cell_bold), Paragraph("+2.05% [+1.29, +2.75]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>mIoU Pièces (10 cl., avec Bureau)</b>", tbl_cell_left), Paragraph("56.40% [54.0, 58.8]", tbl_cell), Paragraph("<b>58.24%</b> [55.8, 60.4]", tbl_cell_bold), Paragraph("+1.85% [+1.16, +2.48]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>IoU Murs Fusionnés (Merged Wall)</b>", tbl_cell_left), Paragraph("74.17% [73.2, 75.1]", tbl_cell), Paragraph("<b>74.93%</b> [74.0, 75.8]", tbl_cell_bold), Paragraph("+0.76% [+0.63, +0.91]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>IoU Murs Intérieurs</b>", tbl_cell_left), Paragraph("62.18% [61.1, 63.3]", tbl_cell), Paragraph("<b>63.25%</b> [62.2, 64.4]", tbl_cell_bold), Paragraph("+1.07% [+0.87, +1.28]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>IoU Murs Extérieurs</b>", tbl_cell_left), Paragraph("73.86% [72.7, 74.9]", tbl_cell), Paragraph("<b>74.58%</b> [73.4, 75.6]", tbl_cell_bold), Paragraph("+0.73% [+0.52, +0.94]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>IoU Portes (Door IoU)</b>", tbl_cell_left), Paragraph("61.78% [60.4, 63.3]", tbl_cell), Paragraph("<b>63.21%</b> [61.9, 64.5]", tbl_cell_bold), Paragraph("+1.43% [+1.11, +1.79]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>IoU Fenêtres (Window IoU)</b>", tbl_cell_left), Paragraph("73.52% [72.2, 74.9]", tbl_cell), Paragraph("<b>74.65%</b> [73.3, 75.9]", tbl_cell_bold), Paragraph("+1.14% [+0.77, +1.59]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>mIoU Structurelle (3 classes)</b>", tbl_cell_left), Paragraph("69.82% [68.7, 70.9]", tbl_cell), Paragraph("<b>70.93%</b> [69.9, 71.9]", tbl_cell_bold), Paragraph("+1.11% [+0.90, +1.34]", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("<b>Taux de Coque Étanche (Shell Validity)</b>", tbl_cell_left), Paragraph("90.75% [87.8, 93.5]", tbl_cell), Paragraph("<b>91.75%</b> [89.0, 94.0]", tbl_cell_bold), Paragraph("+1.00% [-2.75, +4.50]", tbl_cell), Paragraph("p = 0.339 (n.s.)", tbl_cell)],
        [Paragraph("<b>Erreur Relative Périmètre Prédit</b>", tbl_cell_left), Paragraph("24.37% [21.2, 28.2]", tbl_cell), Paragraph("<b>24.20%</b> [20.8, 28.8]", tbl_cell_bold), Paragraph("-0.18% [-2.54, +2.40]", tbl_cell), Paragraph("p = 0.462 (n.s.)", tbl_cell)],
        [Paragraph("<b>Erreur Absolue Périmètre (MAE)</b>", tbl_cell_left), Paragraph("13.46 m [11.8, 15.3]", tbl_cell), Paragraph("<b>13.04 m</b> [11.6, 14.7]", tbl_cell_bold), Paragraph("-0.42 m [-1.81, +1.03]", tbl_cell), Paragraph("p = 0.289 (n.s.)", tbl_cell)],
        [Paragraph("<b>Erreur Relative Surface Empreinte</b>", tbl_cell_left), Paragraph("54.67% [40.4, 72.8]", tbl_cell), Paragraph("<b>55.41%</b> [41.1, 73.3]", tbl_cell_bold), Paragraph("+0.74% [-1.20, +2.90]", tbl_cell), Paragraph("p = 0.792 (n.s.)", tbl_cell)],
        [Paragraph("<b>Erreur Absolue Empreinte (MAE)</b>", tbl_cell_left), Paragraph("21.69 m² [19.3, 24.4]", tbl_cell), Paragraph("<b>21.89 m²</b> [19.6, 24.6]", tbl_cell_bold), Paragraph("+0.19 m² [-0.56, +1.01]", tbl_cell), Paragraph("p = 0.682 (n.s.)", tbl_cell)],
        [Paragraph("<b>Erreur Décompte Pièces (MAE)</b>", tbl_cell_left), Paragraph("1.56 [1.36, 1.77]", tbl_cell), Paragraph("<b>1.53</b> [1.34, 1.75]", tbl_cell_bold), Paragraph("-0.02 [-0.12, +0.07]", tbl_cell), Paragraph("p = 0.343 (n.s.)", tbl_cell)],
        [Paragraph("<b>Erreur Décompte Portes (MAE)</b>", tbl_cell_left), Paragraph("1.47 [1.31, 1.65]", tbl_cell), Paragraph("<b>1.44</b> [1.29, 1.64]", tbl_cell_bold), Paragraph("-0.03 [-0.11, +0.06]", tbl_cell), Paragraph("p = 0.294 (n.s.)", tbl_cell)],
    ]
    t42 = Table(t42_data, colWidths=[usable_w * 0.33, usable_w * 0.18, usable_w * 0.17, usable_w * 0.18, usable_w * 0.14])
    t42.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t42)
    story.append(Spacer(1, 6))

    # Distinction Callout Box: Correction Métrologique vs Gain Modèle
    distinction_box = (
        "<b>DISTINCTION ÉPISTÉMOLOGIQUE CRITIQUE : CORRECTION MÉTRIQUE vs PERFORMANCE MODÈLE</b><br/>"
        "• <b>Correction Métrologique sur Masques Vérité Terrain (Oracle) :</b> Dans les travaux préliminaires, le périmètre était "
        "calculé à partir des contours du masque brut des murs, induisant un artéfact de double-comptage (contour intérieur + extérieur du ruban de mur) "
        "qui gonflait l'erreur théorique à 33.2%. En extrayant l'enveloppe pleine du bâtiment <i>((mur > 0) | (pièce > 0))</i>, l'erreur de l'oracle géométrique "
        "sur les données ground truth s'est effondrée à <b>0.61%</b>. Il s'agit d'une <b>correction de l'instrument de mesure</b> et non d'un gain intrinsèque du réseau.<br/>"
        "• <b>Erreur Prédictive Réelle du Modèle sur Données Non Vues :</b> Lorsqu'il s'agit des prédictions réelles du modèle sur les 400 plans de test, "
        "le modèle présente une erreur relative de périmètre de <b>24.20%</b> (MAE : 13.04 m) et une erreur de surface habitable de <b>21.89 m²</b>. "
        "Le TTA améliore modestement les parois (+0.76% murs, +1.43% portes) tout en assurant 91.75% de coques étanches prêtes pour l'extrusion 3D."
    )
    t_dist = Table([[Paragraph(distinction_box, body_small)]], colWidths=[usable_w])
    t_dist.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), WARN_BG),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#F59E0B")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_dist)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3 : DÉCOMPOSITION PAR CLASSE & GESTION DES CAS LIMITES
    # =========================================================================
    story.append(Paragraph("2. Analyse Granulaire par Classe de Pièce & Diagnostic des Cas Limites", h1))
    story.append(Paragraph(
        "Le tableau suivant détaille l'Intersection-over-Union (IoU) pour chacune des classes sémantiques de CubiCasa5K sur l'ensemble "
        "des 400 plans du jeu de test. Les classes suivent scrupuleusement le dictionnaire officiel <code>ROOM_CLASSES</code>.",
        body
    ))
    story.append(Spacer(1, 4))

    # Table 4.3 Room per-class
    t43_data = [
        [
            Paragraph("<b>Identifiant & Classe Sémantique</b>", tbl_hdr_left),
            Paragraph("<b>Passe Unique (Base)</b>", tbl_hdr),
            Paragraph("<b>4-Fold TTA</b>", tbl_hdr),
            Paragraph("<b>Delta Absolu (Δ)</b>", tbl_hdr),
            Paragraph("<b>Signif. Statistique (p)</b>", tbl_hdr),
        ],
        [Paragraph("1. Séjour (LivingRoom)", tbl_cell_left), Paragraph("74.36% [70.4, 77.9]", tbl_cell), Paragraph("<b>75.64%</b> [71.7, 79.0]", tbl_cell_bold), Paragraph("+1.28%", tbl_cell), Paragraph("p = 0.057 (n.s.)", tbl_cell)],
        [Paragraph("2. Chambre (Bedroom)", tbl_cell_left), Paragraph("<b>74.76%</b> [70.1, 78.4]", tbl_cell_bold), Paragraph("74.71% [70.0, 78.5]", tbl_cell), Paragraph("-0.05%", tbl_cell), Paragraph("p = 0.564 (n.s.)", tbl_cell)],
        [Paragraph("3. Cuisine (Kitchen)", tbl_cell_left), Paragraph("67.52% [63.3, 71.5]", tbl_cell), Paragraph("<b>67.94%</b> [63.7, 72.0]", tbl_cell_bold), Paragraph("+0.42%", tbl_cell), Paragraph("p = 0.239 (n.s.)", tbl_cell)],
        [Paragraph("4. Salle de bain (Bath)", tbl_cell_left), Paragraph("69.80% [66.1, 73.2]", tbl_cell), Paragraph("<b>71.75%</b> [68.0, 75.2]", tbl_cell_bold), Paragraph("+1.96%", tbl_cell), Paragraph("p = 0.002 (**)", tbl_cell)],
        [Paragraph("5. Entrée (Entry)", tbl_cell_left), Paragraph("61.44% [57.2, 65.7]", tbl_cell), Paragraph("<b>63.07%</b> [58.7, 67.2]", tbl_cell_bold), Paragraph("+1.63%", tbl_cell), Paragraph("p = 0.001 (***)", tbl_cell)],
        [Paragraph("6. Extérieur (Outdoor)", tbl_cell_left), Paragraph("72.95% [69.8, 76.6]", tbl_cell), Paragraph("<b>73.92%</b> [70.6, 77.6]", tbl_cell_bold), Paragraph("+0.97%", tbl_cell), Paragraph("p = 0.002 (**)", tbl_cell)],
        [Paragraph("7. Rangement (Storage)", tbl_cell_left), Paragraph("42.65% [35.8, 50.0]", tbl_cell), Paragraph("<b>45.28%</b> [37.8, 53.1]", tbl_cell_bold), Paragraph("+2.62%", tbl_cell), Paragraph("p = 0.041 (*)", tbl_cell)],
        [Paragraph("8. Dressing (Closet)", tbl_cell_left), Paragraph("62.43% [56.2, 68.3]", tbl_cell), Paragraph("<b>66.85%</b> [61.4, 72.3]", tbl_cell_bold), Paragraph("+4.42%", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("9. Garage", tbl_cell_left), Paragraph("38.07% [26.3, 52.0]", tbl_cell), Paragraph("<b>43.28%</b> [31.9, 56.8]", tbl_cell_bold), Paragraph("+5.22%", tbl_cell), Paragraph("p < 0.001 (***)", tbl_cell)],
        [Paragraph("10. Bureau (Office)", tbl_cell_left), Paragraph("0.00% [0.0, 0.0]", tbl_cell), Paragraph("0.00% [0.0, 0.0]", tbl_cell), Paragraph("+0.00%", tbl_cell), Paragraph("p = 1.000 (n.s.)", tbl_cell)],
    ]
    t43 = Table(t43_data, colWidths=[usable_w * 0.32, usable_w * 0.18, usable_w * 0.18, usable_w * 0.16, usable_w * 0.16])
    t43.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t43)
    story.append(Spacer(1, 6))

    # Clarifications Callout
    class_discussion = (
        "<b>DIAGNOSTIC ET TRAITEMENT DES ANOMALIES SÉMANTIQUES :</b><br/>"
        "• <b>La classe Bureau (Office) à Recall Nul :</b> Le modèle ne parvient pas à segmenter la classe Office (0.0% IoU). "
        "Ce constat s'explique objectivement par sa rareté extrême (< 0.2% des pixels du corpus) combinée à une forte ambiguïté "
        "d'annotation dans CubiCasa5K (chambres avec bureau arbitrairement annotées Chambre ou Office). Notre expérience de fine-tuning Lovász "
        "pondéré a confirmé qu'un sur-poids sur Office perturbe les frontières des pièces voisines sans convergence. En conséquence, "
        "la métrique officielle de tête est rapportée sur les <b>9 classes fiables (64.72%)</b>, tout en publiant la métrique 10 classes (58.24%).<br/>"
        "• <b>Fluctuation de la classe Garage :</b> L'IoU de Garage oscille de 38.07% sur test à 62.05% sur validation. Cette dispersion "
        "provient de la taille de l'échantillon : seuls 18 plans du jeu de test possèdent un garage. Aucune conclusion de sur-performance ne doit en être tirée.<br/>"
        "• <b>Seuillage des Logits de Portes :</b> Le test exhaustif de seuils de détection sur les logits de porte (0.10 à 0.50) a révélé qu'aucun "
        "seuil ne surpasse l'argmax standard (IoU max de 59.8% avec argmax vs 58.2% à seuil 0.20), confirmant l'optimalité de la règle de décision retenue."
    )
    t_cd = Table([[Paragraph(class_discussion, body_small)]], colWidths=[usable_w])
    t_cd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ALT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_cd)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4 : VALIDATION SUR 12 PLANS D'ARCHITECTES RÉELS (HABS / LOC)
    # =========================================================================
    story.append(Paragraph("3. Validation Hors-Distribution : 12 Plans d'Archives Réels (HABS & Domaine Public)", h1))
    story.append(Paragraph(
        "Pour tester la capacité de transfert en conditions réelles sans ré-entraînement, 12 plans d'architectes historiques "
        "ont été extraits de la Bibliothèque du Congrès (Library of Congress / Historic American Buildings Survey - HABS) et de Wikimedia Commons. "
        "Ces documents couvrent une grande diversité typologique : du néoclassique (Latrobe White House, 1807) au modernisme (villas Frank Lloyd Wright et Fitzgerald).",
        body
    ))
    story.append(Spacer(1, 4))

    # Table 4.4 Real External Plans
    t44_data = [
        [
            Paragraph("<b>Plan Réel Historique</b>", tbl_hdr_left),
            Paragraph("<b>Murs Prédits</b>", tbl_hdr),
            Paragraph("<b>Pièces</b>", tbl_hdr),
            Paragraph("<b>Portes</b>", tbl_hdr),
            Paragraph("<b>Surface</b>", tbl_hdr),
            Paragraph("<b>Coque Étanche 3D</b>", tbl_hdr),
        ],
        [Paragraph("01. Latrobe White House (1807)", tbl_cell_left), Paragraph("30", tbl_cell), Paragraph("7", tbl_cell), Paragraph("32", tbl_cell), Paragraph("168.2 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("02. Putnam Historic House", tbl_cell_left), Paragraph("38", tbl_cell), Paragraph("25", tbl_cell), Paragraph("35", tbl_cell), Paragraph("222.7 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("03. Wanstead House Ground Plan", tbl_cell_left), Paragraph("43", tbl_cell), Paragraph("11", tbl_cell), Paragraph("30", tbl_cell), Paragraph("101.7 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("04. Bolduc French Colonial Villa", tbl_cell_left), Paragraph("23", tbl_cell), Paragraph("3", tbl_cell), Paragraph("19", tbl_cell), Paragraph("142.6 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("05. Hills-DeCaro House (F.L. Wright)", tbl_cell_left), Paragraph("16", tbl_cell), Paragraph("5", tbl_cell), Paragraph("17", tbl_cell), Paragraph("44.8 m²", tbl_cell), Paragraph("<font color='#E11D48'><b>Échec (Colonnade)</b></font>", tbl_cell)],
        [Paragraph("06. Hills-DeCaro 2nd Floor", tbl_cell_left), Paragraph("22", tbl_cell), Paragraph("8", tbl_cell), Paragraph("21", tbl_cell), Paragraph("80.1 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("07. Hills-DeCaro Pre-Fire Plan", tbl_cell_left), Paragraph("17", tbl_cell), Paragraph("12", tbl_cell), Paragraph("19", tbl_cell), Paragraph("78.5 m²", tbl_cell), Paragraph("<font color='#E11D48'><b>Échec (Typographie)</b></font>", tbl_cell)],
        [Paragraph("08. Hills-DeCaro 1st Floor", tbl_cell_left), Paragraph("19", tbl_cell), Paragraph("9", tbl_cell), Paragraph("18", tbl_cell), Paragraph("71.4 m²", tbl_cell), Paragraph("<font color='#E11D48'><b>Échec (Lignes tirets)</b></font>", tbl_cell)],
        [Paragraph("09. Hills-DeCaro Post-Fire Plan", tbl_cell_left), Paragraph("15", tbl_cell), Paragraph("7", tbl_cell), Paragraph("13", tbl_cell), Paragraph("83.4 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("10. New Place (E. Lutyens, 1913)", tbl_cell_left), Paragraph("46", tbl_cell), Paragraph("15", tbl_cell), Paragraph("48", tbl_cell), Paragraph("127.6 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("11. Hills-DeCaro Remodeled", tbl_cell_left), Paragraph("20", tbl_cell), Paragraph("9", tbl_cell), Paragraph("18", tbl_cell), Paragraph("66.5 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("12. Fitzgerald Modernist House", tbl_cell_left), Paragraph("4", tbl_cell), Paragraph("2", tbl_cell), Paragraph("9", tbl_cell), Paragraph("8.3 m²", tbl_cell), Paragraph("<font color='#0D9488'><b>VALIDE</b></font>", tbl_cell)],
        [Paragraph("<b>BILAN GLOBAL (12 PLANS RÉELS)</b>", tbl_cell_left), Paragraph("<b>100% Détection Pièces (12/12)</b>", tbl_cell_bold), Paragraph("-", tbl_cell), Paragraph("-", tbl_cell), Paragraph("-", tbl_cell), Paragraph("<b>75.0% Coques Valides (9/12)</b>", tbl_cell_bold)],
    ]
    t44 = Table(t44_data, colWidths=[usable_w * 0.36, usable_w * 0.12, usable_w * 0.11, usable_w * 0.11, usable_w * 0.14, usable_w * 0.16])
    t44.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, LIGHT_BG]),
        ('BACKGROUND', (0,-1), (-1,-1), CARD_BG),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t44)
    story.append(Spacer(1, 6))

    # Documented Failure Modes
    failure_analysis = (
        "<b>ANALYSE TRANSPARENTE DES TROIS MODES D'ÉCHEC IDENTIFIÉS :</b><br/>"
        "• <b>Échec 1 — Colonnades et Porches Ajourés (Plan 05) :</b> Présence d'un péristyle semi-ouvert soutenu par des colonnes isolées. "
        "L'algorithme de fermeture polygonale détecte des intersections non orientables, empêchant la fermeture étanche de l'enveloppe extérieure.<br/>"
        "• <b>Échec 2 — Pollution Textuelle Dense et Cotes Manuscrits (Plan 07) :</b> Des cotes historiques en typographie avec empattements "
        "traversent directement les murs porteurs, provoquant une fragmentation du masque binaire des cloisons.<br/>"
        "• <b>Échec 3 — Délimitations par Lignes Tiretées / Hachures (Plan 08) :</b> L'emploi de conventions graphiques obsolètes "
        "(murs esquissés par des tirets) rompt la continuité topologique indispensable au composant connexe de l'enveloppe."
    )
    t_fa = Table([[Paragraph(failure_analysis, body_small)]], colWidths=[usable_w])
    t_fa.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_fa)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5 : SYNTHÈSE DE FAÇADES & AUDIT DU POST-TRAITEMENT
    # =========================================================================
    story.append(Paragraph("4. Benchmark de Synthèse Générative de Façades (CMP Facades Test Split)", h1))
    story.append(Paragraph(
        "L'évaluation du générateur conditionnel est réalisée sur les <b>106 paires d'images du jeu de test officiel CMP Facades</b>. "
        "Les trois variantes d'architecture entraînées sont comparées selon les métriques objectives standard de la littérature : "
        "distance perceptuelle LPIPS (SqueezeNet), distance de Fréchet Inception par noyaux (KID x10³ avec écart-type bootstrap), SSIM, PSNR et perte L1.",
        body
    ))
    story.append(Spacer(1, 4))

    # Table 4.1 Facades Benchmark
    t41_data = [
        [
            Paragraph("<b>Architecture & Checkpoint</b>", tbl_hdr_left),
            Paragraph("<b>LPIPS ↓</b>", tbl_hdr),
            Paragraph("<b>KID (x10³) ↓</b>", tbl_hdr),
            Paragraph("<b>SSIM ↑</b>", tbl_hdr),
            Paragraph("<b>PSNR (dB) ↑</b>", tbl_hdr),
            Paragraph("<b>Perte L1 ↓</b>", tbl_hdr),
        ],
        [
            Paragraph("<b>Baseline (100 ep, TransposeConv, L1=100)</b>", tbl_cell_left),
            Paragraph("0.6327", tbl_cell),
            Paragraph("150.36 ± 10.25", tbl_cell),
            Paragraph("0.2866", tbl_cell),
            Paragraph("13.19", tbl_cell),
            Paragraph("0.3483", tbl_cell),
        ],
        [
            Paragraph("<b>Enhanced (150 ep, Nearest+Conv, L1=100)</b>", tbl_cell_left),
            Paragraph("0.4940", tbl_cell),
            Paragraph("109.86 ± 9.24", tbl_cell),
            Paragraph("<b>0.3345</b>", tbl_cell_bold),
            Paragraph("13.15", tbl_cell),
            Paragraph("<b>0.3379</b>", tbl_cell_bold),
        ],
        [
            Paragraph("<b>Perceptual (100 ep, LPIPS+SN, L1=30) [PRODUCTION]</b>", tbl_cell_left),
            Paragraph("<b>0.3169</b>", tbl_cell_bold),
            Paragraph("<b>67.65 ± 5.95</b>", tbl_cell_bold),
            Paragraph("0.2792", tbl_cell),
            Paragraph("12.69", tbl_cell),
            Paragraph("0.3526", tbl_cell),
        ],
    ]
    t41 = Table(t41_data, colWidths=[usable_w * 0.40, usable_w * 0.15, usable_w * 0.15, usable_w * 0.10, usable_w * 0.10, usable_w * 0.10])
    t41.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG, CARD_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t41)
    story.append(Spacer(1, 6))

    # Metrics Scientific Reconciliation
    facade_disc = (
        "<b>ANALYSE SCIENTIFIQUE DES DIVERGENCES ENTRE MÉTRIQUES :</b><br/>"
        "• <b>Métrique KID (Kernel Inception Distance) vs FID :</b> Avec 106 images de test, l'estimation classique de la covariance gaussienne "
        "du FID dans l'espace Inception (dim = 2 048) est mathématiquement dégénérée. Le KID, estimateur non biaisé du MMD², est la seule "
        "métrique rigoureuse sur cet effectif : il démontre une réduction spectaculaire de <b>55%</b> de la distance distributionnelle "
        "(67.65 vs 150.36 pour la baseline).<br/>"
        "• <b>Arbitrage Lissage Pixel vs Acuité Perceptuelle :</b> Le modèle <i>Enhanced</i> obtient de meilleurs scores SSIM (0.3345) et L1 (0.3379) "
        "car une forte pénalité L1 (λ = 100) contraint le générateur à prédire la moyenne des modes possibles, produisant un flou lissé récompensé "
        "par les métriques pixel. À l'opposé, le modèle <i>Perceptuel</i> avec LPIPS et normalisation spectrale divise par deux la distance "
        "perceptuelle (0.3169 vs 0.6327), restituant les contrastes des corniches et menuiseries au prix d'une perte d'alignement pixel minime."
    )
    t_fd = Table([[Paragraph(facade_disc, body_small)]], colWidths=[usable_w])
    t_fd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ALT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_fd)
    story.append(Spacer(1, 6))

    # Unsharp Mask Separation Box
    unsharp_box = (
        "<b>AUDIT DU POST-TRAITEMENT : DISSOCIATION DU MODÈLE ET DU REHAUSSEMENT VISUEL</b><br/>"
        "• <b>Séparation Méthodologique :</b> Toutes les métriques KID et LPIPS du Tableau 4.1 sont calculées <b>exclusivement sur les images brutes "
        "sorties du générateur</b>, sans aucun filtre additionnel.<br/>"
        "• <b>Rôle de l'Accentuation Adaptative (Unsharp Mask) :</b> Un filtre d'accentuation doux (rayon 1.2 px, force 125%, seuil 3, micro-contraste +4%) "
        "est appliqué uniquement lors du rendu Three.js WebGL pour rehausser la lisibilité des croisillons de fenêtres. "
        "L'inspection du gradient différentiel (amplifié 5x) confirme l'absence de dépassement de dynamique (ringing) ou de sur-accentuation d'artéfacts GAN."
    )
    t_ub = Table([[Paragraph(unsharp_box, body_small)]], colWidths=[usable_w])
    t_ub.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_ub)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6 : ARCHITECTURE MLOPS, PROFILAGE & TEMPS RÉEL
    # =========================================================================
    story.append(Paragraph("5. Cycle de Vie MLOps, Profilage de Latence & Déploiement Industriel", h1))
    story.append(Paragraph(
        "Le pipeline ArchiGen 3D est conçu selon les standards de l'ingénierie logicielle et du MLOps moderne : gestion stricte des dépendances, "
        "découplage modèle/inférence, observabilité et conteneurisation.",
        body
    ))
    story.append(Spacer(1, 4))

    # Latency Breakdown Table
    story.append(Paragraph("<b>Budget de Latence Temps Réel (Profilage Fin sur RTX 3050 Laptop GPU vs CPU Fallback)</b>", h2))
    lat_data = [
        [
            Paragraph("<b>Étape du Pipeline</b>", tbl_hdr_left),
            Paragraph("<b>GPU Unet++ (Base)</b>", tbl_hdr),
            Paragraph("<b>GPU Unet++ (4-Fold TTA)</b>", tbl_hdr),
            Paragraph("<b>CPU (ONNX Runtime)</b>", tbl_hdr),
            Paragraph("<b>Rôle Métier & Nature du Traitement</b>", tbl_hdr_left),
        ],
        [Paragraph("1. Inférence UNet++ (Segmentation)", tbl_cell_left), Paragraph("61.4 ms (P95: 84.1)", tbl_cell), Paragraph("327.6 ms (P95: 395.2)", tbl_cell), Paragraph("854.2 ms (P95: 920.0)", tbl_cell), Paragraph("Réseau multi-tâches PyTorch CUDA / ONNX", tbl_cell_left)],
        [Paragraph("2. Vectorisation Hybride (Shapely)", tbl_cell_left), Paragraph("36.7 ms (P95: 58.2)", tbl_cell), Paragraph("36.7 ms (P95: 58.2)", tbl_cell), Paragraph("36.7 ms (P95: 58.2)", tbl_cell), Paragraph("Accrochage rectilinéaire et coque étanche", tbl_cell_left)],
        [Paragraph("3. Synthèse Façade Principale (cGAN)", tbl_cell_left), Paragraph("24.7 ms (P95: 34.1)", tbl_cell), Paragraph("24.7 ms (P95: 34.1)", tbl_cell), Paragraph("68.5 ms (P95: 82.0)", tbl_cell), Paragraph("Générateur Pix2Pix sur mur d'entrée", tbl_cell_left)],
        [Paragraph("4. Extrusion 3D & Assemblage OBJ", tbl_cell_left), Paragraph("0.8 ms (P95: 1.2)", tbl_cell), Paragraph("0.8 ms (P95: 1.2)", tbl_cell), Paragraph("0.8 ms (P95: 1.2)", tbl_cell), Paragraph("Génération du maillage tridimensionnel", tbl_cell_left)],
        [Paragraph("<b>LATENCE TOTALE BOUT EN BOUT</b>", tbl_cell_left), Paragraph("<b>156.6 ms (P95: 217.7)</b>", tbl_cell_bold), Paragraph("<b>422.8 ms (P95: 498.0)</b>", tbl_cell_bold), Paragraph("<b>960.2 ms (P95: 1061)</b>", tbl_cell_bold), Paragraph("<b>Conforme budget démo temps réel (< 400ms)</b>", tbl_cell_left)],
    ]
    t_lat = Table(lat_data, colWidths=[usable_w * 0.28, usable_w * 0.18, usable_w * 0.18, usable_w * 0.18, usable_w * 0.18])
    t_lat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, LIGHT_BG]),
        ('BACKGROUND', (0,-1), (-1,-1), CARD_BG),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_lat)
    story.append(Spacer(1, 6))

    # Product Design & MLOps Highlights
    mlops_highlights = (
        "<b>CHOIX D'INGÉNIERIE LOGICIELLE & ARBITRAGES PRODUIT :</b><br/>"
        "• <b>Façade Principale à la Demande (front_only=True) :</b> Dans les versions initiales, la génération séquentielle de 15 à 20 segments "
        "extérieurs créait une traîne de latence de 710.3 ms au P95. Nous avons implémenté le mode <code>front_only=True</code> : la façade principale "
        "reçoit une texture neuronale générée (24.7 ms) tandis que les murs latéraux/arrière reçoivent des matériaux procéduraux projetés, "
        "comprimant le P95 à <b>34.1 ms</b> (gain x20.8). En cas de besoin de synthèse intégrale, le <b>batching GPU multi-murs</b> exécute 8 murs en < 50 ms.<br/>"
        "• <b>Parité Numérique Stricte PyTorch vs ONNX Runtime :</b> L'écart maximal entre le moteur d'entraînement et le runtime de production "
        "est de <b>Δ_max = 6.31 x 10⁻⁶</b>, garantissant l'exactitude reproductible des inférences en production.<br/>"
        "• <b>Politique de Versioning DVC & Git :</b> Aucun fichier binaire lourd (.pth, .onnx) n'est traqué dans le dépôt Git. Les poids sont "
        "entièrement gérés par DVC avec traçabilité MLflow.<br/>"
        "• <b>Validation par Tests Automatisés :</b> La suite de tests unitaires et d'intégration compte <b>25 tests Pytest réussis à 100%</b>, "
        "couvrant le bridge de textures, le vectoriseur géométrique, la non-régression des modèles et les endpoints REST de l'API FastAPI."
    )
    t_mh = Table([[Paragraph(mlops_highlights, body_small)]], colWidths=[usable_w])
    t_mh.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ALT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_mh)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7 : ROBUSTESSE CAD, ADAPTATION & CONCLUSION
    # =========================================================================
    story.append(Paragraph("6. Résilience aux Dérives CAD & Conclusion Finale", h1))
    story.append(Paragraph(
        "Afin de mesurer l'impact de notre pipeline d'augmentation de données sur la résilience aux conventions graphiques "
        "(plans bleus, noir et blanc inversé, scans bruités), 50 plans fixes de validation ont été évalués sous dégradations synthétiques contrôlées.",
        body
    ))
    story.append(Spacer(1, 4))

    # Table Robustness
    t_rob_data = [
        [
            Paragraph("<b>Style de Perturbation Graphique</b>", tbl_hdr_left),
            Paragraph("<b>Exact. Pièces (Base)</b>", tbl_hdr),
            Paragraph("<b>Exact. Pièces (Fine-Tuned)</b>", tbl_hdr),
            Paragraph("<b>IoU Murs (Base)</b>", tbl_hdr),
            Paragraph("<b>IoU Murs (Fine-Tuned)</b>", tbl_hdr),
        ],
        [Paragraph("Plans Standards (Propres)", tbl_cell_left), Paragraph("81.36%", tbl_cell), Paragraph("<b>84.61%</b> (+3.25)", tbl_cell_bold), Paragraph("<b>70.45%</b>", tbl_cell_bold), Paragraph("70.16% (-0.29)", tbl_cell)],
        [Paragraph("CAD Inversé (Fond Noir, Lignes Claires)", tbl_cell_left), Paragraph("25.68%", tbl_cell), Paragraph("<b>83.47%</b> (+57.79)", tbl_cell_bold), Paragraph("27.18%", tbl_cell), Paragraph("<b>67.23%</b> (+40.05)", tbl_cell_bold)],
        [Paragraph("Numérisation Bruitée (Bruit Gaussien/Grain)", tbl_cell_left), Paragraph("47.70%", tbl_cell), Paragraph("<b>71.05%</b> (+23.35)", tbl_cell_bold), Paragraph("61.79%", tbl_cell), Paragraph("<b>65.55%</b> (+3.76)", tbl_cell_bold)],
        [Paragraph("Impression Décolorée (Faible Contraste)", tbl_cell_left), Paragraph("78.70%", tbl_cell), Paragraph("<b>84.73%</b> (+6.03)", tbl_cell_bold), Paragraph("70.08%", tbl_cell), Paragraph("<b>70.19%</b> (+0.11)", tbl_cell_bold)],
    ]
    t_rob = Table(t_rob_data, colWidths=[usable_w * 0.36, usable_w * 0.16, usable_w * 0.16, usable_w * 0.16, usable_w * 0.16])
    t_rob.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_rob)
    story.append(Spacer(1, 6))

    # Conclusion & Synthesis
    story.append(Paragraph("<b>Conclusion & Perspectives d'Avenir</b>", h2))
    concl_text = (
        "Le projet <b>ArchiGen 3D</b> démontre avec succès la faisabilité et la rentabilité industrielle d'une chaîne automatisée "
        "de conversion 2D vers 3D associant Vision par Ordinateur classique, Apprentissage Profond supervisé et Modèles Génératifs. "
        "L'entraînement du modèle multi-tâches UNet++ (ResNet-34) avec augmentations régularisées permet d'atteindre <b>88.09%</b> "
        "d'exactitude pixel et <b>64.72%</b> de mIoU sémantique sur 9 classes fiables, tout en maintenant un taux d'étanchéité géométrique "
        "de <b>91.75%</b> sur 400 plans de test non vus.<br/><br/>"
        "Côté génération, l'adoption de PatchGAN stabilisé par normalisation spectrale et perte perceptuelle LPIPS produit des façades de haute fidélité "
        "(KID de <b>67.65 x 10⁻³</b>, LPIPS de <b>0.3169</b>), avec un runtime GPU optimisé à <b>156.6 ms</b> en médiane.<br/><br/>"
        "<b>Perspectives identifiées :</b> (1) intégration de Graph Neural Networks (GNN) pour contraindre la grammaire topologique "
        "des pièces indépendamment de la qualité graphique ; (2) intégration de modèles de diffusion latente conditionnés par ControlNet "
        "pour enrichir le répertoire architectural (façades contemporaines, bardages bois, architectures haussmanniennes)."
    )
    t_concl = Table([[Paragraph(concl_text, body)]], colWidths=[usable_w])
    t_concl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), INFO_BG),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#93C5FD")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_concl)
    story.append(Spacer(1, 8))

    # Sign-off box
    signoff = [
        [
            Paragraph("<b>Équipe Projet Ingénieur :</b><br/>Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari", 
                      ParagraphStyle('so1', fontName='Helvetica', fontSize=7.5, leading=10, textColor=DARK_TEXT)),
            Paragraph("<b>Évaluation Académique :</b><br/>Module Deep Learning Avancé & MLOps — TEK-UP 2026/2027", 
                      ParagraphStyle('so2', fontName='Helvetica', fontSize=7.5, leading=10, textColor=DARK_TEXT, alignment=2))
        ]
    ]
    t_so = Table(signoff, colWidths=[usable_w * 0.5, usable_w * 0.5])
    t_so.setStyle(TableStyle([
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_so)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Technical Report PDF generated successfully at: {output_path}")

if __name__ == "__main__":
    out_pdf = PROJECT_ROOT / "report/rapport_technique.pdf"
    build_technical_report_pdf(str(out_pdf))

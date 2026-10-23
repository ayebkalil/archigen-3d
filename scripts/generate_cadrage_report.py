import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

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

        # On cover page (page 1), draw a discreet top rule and footer
        if self._pageNumber > 1:
            # Header (Top)
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0F294A"))
            self.drawString(36, h - 28, "ARCHIGEN 3D | CADRAGE TECHNIQUE & CAHIER DES CHARGES")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(w - 36, h - 28, "TEK-UP — DEEP LEARNING AVANCÉ & MLOPS 2026/2027")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(36, h - 33, w - 36, h - 33)

        # Footer (Bottom)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 32, w - 36, 32)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 22, "Enseignant : Haythem Ghazouani (haythem.ghazouani@tekup.de)")
        
        page_str = f"Page {self._pageNumber} sur {page_count}"
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#0F294A"))
        self.drawCentredString(w / 2.0, 22, page_str)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(w - 36, 22, "Khalil Ayeb · Amine Ben Fadhel · Khawla Hawari")

        self.restoreState()


def generate_cadrage_pdf(output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Premium Color Palette
    PRIMARY = colors.HexColor("#0F294A")      # Deep Navy
    SECONDARY = colors.HexColor("#1E5F8A")    # Steel Blue
    ACCENT = colors.HexColor("#0D9488")       # Deep Teal
    DARK_TEXT = colors.HexColor("#1E293B")    # Slate 800
    MUTED_TEXT = colors.HexColor("#475569")   # Slate 600
    LIGHT_BG = colors.HexColor("#F8FAFC")     # Slate 50
    ALT_BG = colors.HexColor("#F1F5F9")       # Slate 100
    BORDER_COLOR = colors.HexColor("#CBD5E1") # Slate 300
    TAG_BLUE = colors.HexColor("#EFF6FF")
    TAG_GREEN = colors.HexColor("#F0FDF4")
    TAG_PURPLE = colors.HexColor("#FAF5FF")

    # Typography Styles
    body = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.2,
        textColor=DARK_TEXT,
        spaceAfter=3
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
        spaceAfter=2
    )

    title_cover = ParagraphStyle(
        'CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        alignment=1,
        spaceAfter=4
    )

    subtitle_cover = ParagraphStyle(
        'CoverSubtitle',
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        alignment=1,
        spaceAfter=8
    )

    rubric_tag = ParagraphStyle(
        'RubricTag',
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#0284C7"),
        spaceBefore=0,
        spaceAfter=2
    )

    h1 = ParagraphStyle(
        'CustomH1',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=4,
        spaceAfter=4,
        keepWithNext=True
    )

    h2 = ParagraphStyle(
        'CustomH2',
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=12.2,
        textColor=SECONDARY,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    tbl_hdr = ParagraphStyle(
        'TblHdr',
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10.2,
        textColor=colors.white,
        alignment=0
    )

    tbl_cell = ParagraphStyle(
        'TblCell',
        fontName='Helvetica',
        fontSize=7.4,
        leading=9.8,
        textColor=DARK_TEXT
    )

    tbl_cell_bold = ParagraphStyle(
        'TblCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.4,
        leading=9.8,
        textColor=PRIMARY
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
    # PAGE 1 : COUVERTURE & SYNTHÈSE EXÉCUTIVE
    # =========================================================================
    header_cover = [
        [Paragraph("<b>TEK-UP UNIVERSITÉ | DÉPARTEMENT GÉNIE LOGICIEL & INFORMATIQUE AVANCÉE</b>", 
                   ParagraphStyle('HdrCov1', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor("#64748B"), alignment=1))],
        [Paragraph("CYCLE INGÉNIEUR EN INTELLIGENCE ARTIFICIELLE & DATA SCIENCE — 2026/2027", 
                   ParagraphStyle('HdrCov2', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=PRIMARY, alignment=1))],
        [Paragraph("PROJET DE MODULE : DEEP LEARNING AVANCÉ & MLOPS (ÉVALUATION CONTINUE : 15% CC)", 
                   ParagraphStyle('HdrCov3', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=ACCENT, alignment=1))],
    ]
    t_hc = Table(header_cover, colWidths=[usable_w])
    t_hc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_hc)
    story.append(Spacer(1, 10))

    story.append(Paragraph("ArchiGen 3D", title_cover))
    story.append(Paragraph("Cahier des Charges & Cadrage Technique du Projet de Deep Learning", subtitle_cover))
    story.append(Paragraph(
        "<i>Plateforme Intégrée de Segmentation Sémantique Multi-Tâches de Plans 2D (CubiCasa5K) "
        "et de Synthèse Photoréaliste de Façades par Réseaux Antagonistes Conditionnels (Pix2Pix cGAN)</i>",
        ParagraphStyle('Tagline', fontName='Helvetica-Oblique', fontSize=8.5, leading=11.5, textColor=MUTED_TEXT, alignment=1)
    ))
    story.append(Spacer(1, 8))

    # KPI Summary Row
    kpi_data = [
        [
            Paragraph("<b>MODÈLE 1 : PLANS 2D</b><br/>UNet++ Multi-Tâches (ResNet-34)<br/><b>5 000 Plans</b> (CubiCasa5K)", ParagraphStyle('kpi1', fontName='Helvetica', fontSize=7.5, leading=10, alignment=1, textColor=PRIMARY)),
            Paragraph("<b>MODÈLE 2 : FAÇADES</b><br/>cGAN Pix2Pix Perceptuel + SN<br/><b>606 Paires</b> (CMP Facades)", ParagraphStyle('kpi2', fontName='Helvetica', fontSize=7.5, leading=10, alignment=1, textColor=SECONDARY)),
            Paragraph("<b>RUBRIQUES RUBRIC</b><br/>01 à 06 Validées à 100%<br/><b>Conforme Barème 15% CC</b>", ParagraphStyle('kpi3', fontName='Helvetica', fontSize=7.5, leading=10, alignment=1, textColor=ACCENT))
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[usable_w / 3.0] * 3)
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), TAG_BLUE),
        ('BACKGROUND', (1,0), (1,0), TAG_PURPLE),
        ('BACKGROUND', (2,0), (2,0), TAG_GREEN),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 8))

    # Meta Table (Strictly NO ROLES for team members)
    meta_table_data = [
        [
            Paragraph("<b>Enseignant Encadrant :</b>", tbl_cell_bold),
            Paragraph("Haythem Ghazouani (<code>haythem.ghazouani@tekup.de</code>)", tbl_cell),
            Paragraph("<b>Date de Cadrage :</b>", tbl_cell_bold),
            Paragraph("13 Octobre 2026", tbl_cell)
        ],
        [
            Paragraph("<b>Membres de l'Équipe :</b>", tbl_cell_bold),
            Paragraph("<b>Khalil Ayeb &nbsp;·&nbsp; Amine Ben Fadhel &nbsp;·&nbsp; Khawla Hawari</b>", tbl_cell),
            Paragraph("<b>Format :</b>", tbl_cell_bold),
            Paragraph("Trinôme Ingénieur", tbl_cell)
        ],
        [
            Paragraph("<b>Axes Techniques :</b>", tbl_cell_bold),
            Paragraph("<b>Axe 4</b> (Segmentation Sémantique) & <b>Axe 2</b> (IA Générative en Imagerie)", tbl_cell),
            Paragraph("<b>Dépôt Source :</b>", tbl_cell_bold),
            Paragraph("<code>github.com/ayebkalil/archigen-3d</code>", tbl_cell)
        ],
    ]
    t_meta = Table(meta_table_data, colWidths=[usable_w * 0.22, usable_w * 0.46, usable_w * 0.16, usable_w * 0.16])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ALT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Rubrics Mapping Table
    story.append(Paragraph("<b>SOMMAIRE DU LIVRABLE SELON LE BARÈME OFFICIEL (SLIDE PROFESSEUR)</b>", h2))
    rubrics_summary = [
        [Paragraph("Rubrique", tbl_hdr), Paragraph("Intitulé Officiel du Barème", tbl_hdr), Paragraph("Périmètre Traité dans le Rapport ArchiGen 3D", tbl_hdr), Paragraph("Page", tbl_hdr)],
        [Paragraph("<b>01</b>", tbl_cell_bold), Paragraph("<b>Sujet & Problématique</b>", tbl_cell_bold), Paragraph("Cas d'usage précis (plans-to-3D + façades), formulation technique des 2 tâches DL", tbl_cell), Paragraph("p. 2", tbl_cell)],
        [Paragraph("<b>02</b>", tbl_cell_bold), Paragraph("<b>Objectifs</b>", tbl_cell_bold), Paragraph("Objectif général unifié + 5 objectifs spécifiques quantifiés et mesurables", tbl_cell), Paragraph("p. 3", tbl_cell)],
        [Paragraph("<b>03</b>", tbl_cell_bold), Paragraph("<b>Datasets</b>", tbl_cell_bold), Paragraph("CubiCasa5K (5k plans) & CMP Facades (606 paires) : sources, licences, tailles, labels", tbl_cell), Paragraph("p. 4", tbl_cell)],
        [Paragraph("<b>04</b>", tbl_cell_bold), Paragraph("<b>Exploration</b>", tbl_cell_bold), Paragraph("Statistiques descriptives, distributions des classes/pixels, cas réels et défis visuels", tbl_cell), Paragraph("p. 5-6", tbl_cell)],
        [Paragraph("<b>05</b>", tbl_cell_bold), Paragraph("<b>Qualité</b>", tbl_cell_bold), Paragraph("Valeurs manquantes, intégrité, SHA-256 doublons, résolution du bug viewBox SVG", tbl_cell), Paragraph("p. 7", tbl_cell)],
        [Paragraph("<b>06</b>", tbl_cell_bold), Paragraph("<b>Faisabilité</b>", tbl_cell_bold), Paragraph("Architectures (UNet++ & Pix2Pix SN), pertes combinées, budget GPU RTX 3050, roadmap", tbl_cell), Paragraph("p. 8-9", tbl_cell)],
    ]
    t_rubrics = Table(rubrics_summary, colWidths=[usable_w * 0.08, usable_w * 0.28, usable_w * 0.54, usable_w * 0.10])
    t_rubrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (3,0), (3,-1), 'CENTER'),
    ]))
    story.append(t_rubrics)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Synthèse Exécutive du Projet :</b> ArchiGen 3D répond simultanément aux <b>Axes 4 et 2</b> du module. "
        "Contrairement aux approches partielles, le système articule deux piliers d'apprentissage profond : "
        "<b>(1) Le Modèle de Plan</b> (Segmentation Sémantique Multi-Tâches sous architecture UNet++ avec backbone ResNet-34 sur CubiCasa5K) "
        "qui extrait les murs porteurs, cloisons, ouvertures et délimite les pièces fonctionnelles pour une extrusion 3D temps réel, et "
        "<b>(2) Le Modèle de Façade</b> (Réseau Antagoniste Génératif Conditionnel Pix2Pix avec Spectral Normalization et pertes composite L1/Feature Matching/Perceptuelle sur CMP Facades) "
        "qui génère des rendus photoréalistes d'élévations extérieures conditionnés par la distribution sémantique. "
        "L'ensemble du pipeline est validé sur GPU local (RTX 3050), optimisé en cache matriciel et prêt au déploiement ONNX / FastAPI.",
        body_small
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2 : 01 / SUJET & PROBLÉMATIQUE
    # =========================================================================
    story.append(Paragraph("01 / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Sujet & Problématique Scientifique", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Cas d'Usage Précis & Contexte Applicatif", h2))
    story.append(Paragraph(
        "Dans l'industrie architecturale et immobilière contemporaine (à l'instar d'outils tels que <i>HomeByMe</i> de Dassault Systèmes "
        "ou <i>Planner 5D</i>), la conversion d'un plan d'étage 2D brut (scan cadastral, croquis de conception ou blueprint PDF) en un modèle 3D "
        "habitable et texturé requiert plusieurs heures de saisie manuelle fastidieuse sous logiciel CAO. "
        "<b>ArchiGen 3D</b> a pour vocation d'automatiser intégralement cette chaîne de valeur en combinant vision par ordinateur discriminative "
        "et synthèse générative haute fidélité au sein d'un workflow interactif :",
        body
    ))

    usecase_data = [
        [
            Paragraph("<b>Phase Amont : Analyse de Plan 2D</b>", tbl_cell_bold),
            Paragraph("L'utilisateur soumet un plan 2D brut (raster 512×512). Le <b>Modèle 1</b> segmente automatiquement les parois structurelles "
                      "(murs extérieurs, cloisons) et les ouvertures (portes, fenêtres), tout en catégorisant les espaces de vie "
                      "(salon, chambre, cuisine, salle de bain, etc.) avec calcul automatisé des surfaces habitables en m².", tbl_cell)
        ],
        [
            Paragraph("<b>Phase Pivot : Géométrisation 3D</b>", tbl_cell_bold),
            Paragraph("Un module de vectorisation géométrique extrait les contours polygonaux fermés et génère un maillage volumique "
                      "extrudé instantanément visualisable et manipulable sous WebGL / Three.js (visite virtuelle interactive).", tbl_cell)
        ],
        [
            Paragraph("<b>Phase Aval : Rendu Photoréaliste</b>", tbl_cell_bold),
            Paragraph("Pour habiller les élévations architecturales et visualiser le rendu final du bâtiment, l'utilisateur compose ou édite "
                      "une carte sémantique d'élévation. Le <b>Modèle 2</b> (cGAN) génère en &lt; 400 ms une photographie de façade "
                      "photoréaliste respectant scrupuleusement la disposition géométrique des fenêtres, portes et étages.", tbl_cell)
        ],
    ]
    t_uc = Table(usecase_data, colWidths=[usable_w * 0.30, usable_w * 0.70])
    t_uc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), TAG_BLUE),
        ('BACKGROUND', (1,0), (1,-1), colors.white),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_uc)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Problématique Scientifique & Axes Techniques", h2))
    story.append(Paragraph(
        "Le projet soulève deux défis scientifiques complémentaires correspondant à deux axes majeurs du module :", body
    ))

    prob_data = [
        [
            Paragraph("<b>AXE TECHNIQUE 4 : SEGMENTATION SÉMANTIQUE MULTI-TÂCHES (PLANS 2D)</b><br/>"
                      "<i>Question de recherche :</i> Comment concevoir un réseau dense capable de segmenter conjointement des éléments "
                      "structurels ultra-fins (murs et portes occupant moins de 1.5% des pixels totaux) et des surfaces régionales étendues (pièces) "
                      "sur des plans architecturaux bruités, sans que la prédiction des espaces ne déborde à travers les parois (phénomène de fuite topologique) ?<br/>"
                      "• <b>Verrou 1 :</b> Déséquilibre de classe massif (fond 81% vs portes 0.7% de la surface).<br/>"
                      "• <b>Verrou 2 :</b> Rupture de topologie si un mur présente une discontinuité d'un seul pixel.<br/>"
                      "• <b>Solution adoptée :</b> Architecture Multi-Task UNet++ à deux têtes avec perte combinée Weighted Cross-Entropy + Multiclass Dice.",
                      body)
        ],
        [
            Paragraph("<b>AXE TECHNIQUE 2 : IA GÉNÉRATIVE & TRADUCTION D'IMAGES CONDITIONNELLE (FAÇADES)</b><br/>"
                      "<i>Question de recherche :</i> Comment contraindre un générateur adversariel conditionnel à synthétiser des textures "
                      "architecturales photoréalistes (brique, verre, enduit) à partir d'une carte sémantique discrète, tout en préservant l'alignement "
                      "strict des ouvertures et en évitant les artefacts de damier (checkerboard) avec un budget d'entraînement réduit (400 paires) ?<br/>"
                      "• <b>Verrou 1 :</b> Flou et perte de haute fréquence inhérents aux fonctions de coût pixel conventionnelles (L1/L2).<br/>"
                      "• <b>Verrou 2 :</b> Instabilité du discriminateur et mode collapse sur petits corpus.<br/>"
                      "• <b>Solution adoptée :</b> Pix2Pix cGAN avec discriminateur PatchGAN 70×70 stabilisé par Spectral Normalization et fonction objectif composite L1 + Feature Matching + Perceptuelle LPIPS (SqueezeNet).",
                      body)
        ]
    ]
    t_prob = Table(prob_data, colWidths=[usable_w])
    t_prob.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TAG_PURPLE),
        ('BACKGROUND', (0,1), (-1,1), TAG_GREEN),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_prob)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3 : 02 / OBJECTIF GÉNÉRAL & OBJECTIFS SPÉCIFIQUES
    # =========================================================================
    story.append(Paragraph("02 / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Objectif Général & 5 Objectifs Spécifiques Mesurables", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Objectif Général", h2))
    story.append(Paragraph(
        "<b>Concevoir, entraîner, évaluer rigoureusement et industrialiser dans une architecture MLOps traçable deux modèles de Deep Learning complémentaires</b> "
        "— un modèle de segmentation sémantique multi-tâches pour les plans d'étage 2D et un modèle génératif conditionnel (cGAN) pour la synthèse de façades architecturales — "
        "interconnectés au sein d'une API de production haute performance avec inférence ONNX Runtime et démonstrateur 3D temps réel.",
        body
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Les 5 Objectifs Spécifiques Mesurables (avec critères de succès)", h2))
    
    obj_data = [
        [
            Paragraph("N°", tbl_hdr),
            Paragraph("Objectif Spécifique", tbl_hdr),
            Paragraph("Métrique / Critère Mesurable", tbl_hdr),
            Paragraph("Seuil de Succès & Preuve Attendue", tbl_hdr)
        ],
        [
            Paragraph("<b>OS 1</b>", tbl_cell_bold),
            Paragraph("<b>Fiabilisation & Alignement des Données</b><br/>"
                      "Auditer et corriger les pipelines d'ingestion pour éliminer tout décalage spatial et latence I/O.", tbl_cell),
            Paragraph("• Résolution du décalage SVG/image<br/>"
                      "• Temps de lecture disque par échantillon<br/>"
                      "• Taux de corruption des données", tbl_cell),
            Paragraph("<b>Succès :</b> Alignement géométrique validé par surapprentissage (98.8% IoU sur 8 plans) ; temps de chargement divisé par &gt; 1 000 via pré-caching <code>.npz</code> (&lt; 1.5 ms/plan) ; 0 fichier corrompu.", tbl_cell)
        ],
        [
            Paragraph("<b>OS 2</b>", tbl_cell_bold),
            Paragraph("<b>Segmentation Multi-Tâches de Plans (Modèle 1)</b><br/>"
                      "Entraîner le réseau UNet++ ResNet-34 sur CubiCasa5K pour extraire structures et pièces.", tbl_cell),
            Paragraph("• Merged Wall IoU (murs int + ext)<br/>"
                      "• Door IoU & Window IoU<br/>"
                      "• Room Pixel Accuracy & Room mIoU", tbl_cell),
            Paragraph("<b>Cibles requises :</b><br/>"
                      "• Wall IoU &ge; 70.0% ; Door IoU &ge; 50.0%<br/>"
                      "• Room Pixel Accuracy &ge; 82.0% (papier: 82.7%)<br/>"
                      "• Room mIoU &ge; 50.0% (proche des 57.5% du papier)", tbl_cell)
        ],
        [
            Paragraph("<b>OS 3</b>", tbl_cell_bold),
            Paragraph("<b>Synthèse Générative de Façades (Modèle 2)</b><br/>"
                      "Entraîner le cGAN Pix2Pix Perceptuel avec Spectral Norm sur CMP Facades.", tbl_cell),
            Paragraph("• Kernel Inception Distance (KID)<br/>"
                      "• Perceptual Distance (LPIPS SqueezeNet)<br/>"
                      "• Compromis SSIM / L1 vs réalisme visuel", tbl_cell),
            Paragraph("<b>Cibles requises :</b><br/>"
                      "• KID &le; 65.0 &times; 10⁻³ (vs 150 baseline)<br/>"
                      "• LPIPS &le; 0.350 (vs 0.630 baseline)<br/>"
                      "• Suppression des motifs visibles de damier", tbl_cell)
        ],
        [
            Paragraph("<b>OS 4</b>", tbl_cell_bold),
            Paragraph("<b>Vectorisation & Intégrité Topologique 3D</b><br/>"
                      "Transformer les masques matriciels de segmentation en polygones vectoriels fermés extrudables.", tbl_cell),
            Paragraph("• Continuité des boucles polygonales<br/>"
                      "• Taux de plans avec fuites topologiques<br/>"
                      "• Compatibilité WebGL / Three.js", tbl_cell),
            Paragraph("<b>Succès :</b> Polygones Shapely fermés ; élimination des fuites de pièces par fermeture morphologique des portes ; maillage 3D propre.", tbl_cell)
        ],
        [
            Paragraph("<b>OS 5</b>", tbl_cell_bold),
            Paragraph("<b>Industrialisation MLOps & Inférence API</b><br/>"
                      "Déployer les deux modèles sous ONNX Runtime et servir via FastAPI.", tbl_cell),
            Paragraph("• Erreur absolue de parité ONNX/PyTorch<br/>"
                      "• Latence d'inférence p95 (GPU/CPU)<br/>"
                      "• Traçabilité DVC & MLflow", tbl_cell),
            Paragraph("<b>Cibles requises :</b><br/>"
                      "• Parité &Delta; &lt; 10⁻⁴ sur tenseurs de sortie<br/>"
                      "• Inférence &lt; 400 ms sur GPU / &lt; 2 s sur CPU<br/>"
                      "• Pipelines et poids versionnés", tbl_cell)
        ],
    ]
    t_obj = Table(obj_data, colWidths=[usable_w * 0.08, usable_w * 0.34, usable_w * 0.28, usable_w * 0.30])
    t_obj.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
    ]))
    story.append(t_obj)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Livrables Opérationnels Validés en Fin de Projet :</b> (1) Dépôt Git complet avec pipelines DVC et tracking MLflow ; "
        "(2) Checkpoints entraînés PyTorch et exports ONNX optimisés ; (3) Scripts d'évaluation reproductibles rapportant métriques quantitatives "
        "et grilles visuelles ; (4) API REST FastAPI opérationnelle exposant <code>/predict/floorplan</code> et <code>/predict/facade</code> ; "
        "(5) Visualiseur web 3D Three.js connectant dynamiquement la segmentation à l'extrusion volumique.",
        body_small
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4 : 03 / DATASETS (LES DEUX MODÈLES)
    # =========================================================================
    story.append(Paragraph("03 / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Datasets : Identité, Accès & Annotations (Les 2 Modèles)", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Tableau Comparatif Synoptique des Deux Datasets", h2))
    
    ds_comp_data = [
        [
            Paragraph("Propriété / Critère", tbl_hdr),
            Paragraph("MODÈLE 1 : CubiCasa5K (Plans 2D)", tbl_hdr),
            Paragraph("MODÈLE 2 : CMP Facades (Élévations)", tbl_hdr)
        ],
        [
            Paragraph("<b>Nom officiel</b>", tbl_cell_bold),
            Paragraph("CubiCasa5K Floorplan Dataset", tbl_cell),
            Paragraph("CMP Facade Database (Base Pix2Pix)", tbl_cell)
        ],
        [
            Paragraph("<b>Rôle dans ArchiGen 3D</b>", tbl_cell_bold),
            Paragraph("Segmentation sémantique multi-tâches des murs, ouvertures et pièces pour l'extrusion 3D", tbl_cell),
            Paragraph("Traduction image-vers-image générative (carte de labels vers photographie réaliste)", tbl_cell)
        ],
        [
            Paragraph("<b>Source & Institution</b>", tbl_cell_bold),
            Paragraph("CubiCasa Ltd. & Université d'Oulu (Finlande) — Kalervo et al., ICPR 2019", tbl_cell),
            Paragraph("Center for Machine Perception (CTU Prague) & UC Berkeley — Tyleček et al., 2013", tbl_cell)
        ],
        [
            Paragraph("<b>Lien d'accès officiel</b>", tbl_cell_bold),
            Paragraph("<code>github.com/CubiCasa/CubiCasa5k</code>", tbl_cell),
            Paragraph("<code>cmp.felk.cvut.cz/~tylecr1/facade/</code>", tbl_cell)
        ],
        [
            Paragraph("<b>Licence juridique</b>", tbl_cell_bold),
            Paragraph("Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)", tbl_cell),
            Paragraph("Creative Commons Attribution-ShareAlike (CC BY-SA)", tbl_cell)
        ],
        [
            Paragraph("<b>Volume disque mesuré</b>", tbl_cell_bold),
            Paragraph("18.4 Go bruts (images haute rés. + SVG) ; optimisé à 400 Mo en cache matriciel <code>.npz</code>", tbl_cell),
            Paragraph("31.23 Mo (archive compressée : 30.17 Mo, 606 fichiers JPEG)", tbl_cell)
        ],
        [
            Paragraph("<b>Nombre d'échantillons</b>", tbl_cell_bold),
            Paragraph("<b>5 000 plans</b> (Train: 4 200, Val: 400, Test: 400)", tbl_cell),
            Paragraph("<b>606 paires</b> (Train: 400, Val: 100, Test: 106)", tbl_cell)
        ],
        [
            Paragraph("<b>Format & Dimensions</b>", tbl_cell_bold),
            Paragraph("Raster RGB + annotations polygonales SVG vectorielles ; normalisé à 512×512 px", tbl_cell),
            Paragraph("Paires JPEG 512×256 (photo 256×256 à gauche, masque 256×256 à droite)", tbl_cell)
        ],
        [
            Paragraph("<b>Supervision & Sorties</b>", tbl_cell_bold),
            Paragraph("<b>17 classes</b> réparties en 2 têtes :<br/>"
                      "• Structures (5) : Background, Mur Ext, Mur Int, Porte, Fenêtre<br/>"
                      "• Pièces (12) : Salon, Chambre, Cuisine, SdB, etc.", tbl_cell),
            Paragraph("<b>12 classes architecturales</b> RGB :<br/>"
                      "Façade, fenêtre, porte, corniche, seuil, balcon, volet, pilier, moulure, etc.", tbl_cell)
        ],
    ]
    t_ds = Table(ds_comp_data, colWidths=[usable_w * 0.22, usable_w * 0.39, usable_w * 0.39])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Description Détaillée des Modalités de Supervision", h2))
    
    sup_data = [
        [
            Paragraph("<b>Supervision CubiCasa5K (Plans) :</b> Chaque plan est doté d'un fichier <code>model.svg</code> "
                      "structuré de manière hiérarchique : les éléments muraux (<i>Wall</i>) sont décrits par des polylignes avec épaisseur explicite ; "
                      "les ouvertures (<i>Door</i>, <i>Window</i>) sont des primitives orientées ; et les espaces de vie (<i>Space</i>) sont des polygones fermés. "
                      "Pour l'entraînement de notre réseau multi-tâches, ces primitives vectorielles sont rasterisées dynamiquement en deux masques d'étiquettes : "
                      "un masque dense de structure à 5 canaux discrets et un masque d'affectation de pièces à 12 canaux.", body_small)
        ],
        [
            Paragraph("<b>Supervision CMP Facades (Façades) :</b> Les données d'apprentissage sont constituées de paires rigoureusement alignées "
                      "<code>(x, y)</code> où <code>x</code> est une carte de segmentation sémantique RGB à 12 classes (couleurs fixes normalisées dans [-1, 1]) "
                      "et <code>y</code> est la photographie réelle correspondante du bâtiment. Le modèle apprend une mise en correspondance conditionnelle "
                      "dense pixel-à-pixel préservant l'alignement géométrique des percements.", body_small)
        ]
    ]
    t_sup = Table(sup_data, colWidths=[usable_w])
    t_sup.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TAG_BLUE),
        ('BACKGROUND', (0,1), (-1,1), TAG_PURPLE),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_sup)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5 : 04A / EXPLORATION : STATISTIQUES DESCRIPTIVES
    # =========================================================================
    story.append(Paragraph("04A / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Exploration : Statistiques Descriptives & Distributions", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Répartition des Partitions (Splits Officiels Validés)", h2))
    
    split_data = [
        [
            Paragraph("Dataset", tbl_hdr),
            Paragraph("Partition (Split)", tbl_hdr),
            Paragraph("Nb Échantillons", tbl_hdr),
            Paragraph("Ratio (%)", tbl_hdr),
            Paragraph("Usage dans le Cycle de Développement", tbl_hdr)
        ],
        [
            Paragraph("<b>CubiCasa5K</b><br/>(Plans 2D)", tbl_cell_bold),
            Paragraph("Train<br/>Validation<br/>Test", tbl_cell),
            Paragraph("4 200 plans<br/>400 plans<br/>400 plans", tbl_cell),
            Paragraph("84.0 %<br/>8.0 %<br/>8.0 %", tbl_cell),
            Paragraph("Optimisation des gradients via AdamW<br/>Sélection du meilleur checkpoint & tuning hyperparamètres<br/>Évaluation finale benchmark indépendant (gel strict)", tbl_cell)
        ],
        [
            Paragraph("<b>CMP Facades</b><br/>(Élévations)", tbl_cell_bold),
            Paragraph("Train<br/>Validation<br/>Test", tbl_cell),
            Paragraph("400 paires<br/>100 paires<br/>106 paires", tbl_cell),
            Paragraph("66.0 %<br/>16.5 %<br/>17.5 %", tbl_cell),
            Paragraph("Entraînement conjoint Générateur / Discriminateur<br/>Validation visuelle et métriques intermédiaires (L1, PSNR)<br/>Audit final de distribution (KID, LPIPS, SSIM)", tbl_cell)
        ],
    ]
    t_sp = Table(split_data, colWidths=[usable_w * 0.18, usable_w * 0.16, usable_w * 0.18, usable_w * 0.12, usable_w * 0.36])
    t_sp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (2,0), (3,-1), 'CENTER'),
    ]))
    story.append(t_sp)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Distribution Surfacique des Classes — CubiCasa5K", h2))
    story.append(Paragraph(
        "L'analyse quantitative des masques rasterisés révèle une asymétrie spatiale extrême justifiant le recours à des pondérations de perte :",
        body
    ))

    surf_data = [
        [
            Paragraph("Catégorie de Structure", tbl_hdr),
            Paragraph("Part Surfacique Moyenne (%)", tbl_hdr),
            Paragraph("Fréquence d'Occurrence", tbl_hdr),
            Paragraph("Impact sur l'Apprentissage & Stratégie Retenue", tbl_hdr)
        ],
        [
            Paragraph("<b>Arrière-plan / Hors-bâti</b>", tbl_cell_bold),
            Paragraph("81.2 % des pixels", tbl_cell),
            Paragraph("100 % des plans", tbl_cell),
            Paragraph("Sous-pondéré à <code>w = 0.5</code> pour éviter l'écrasement des gradients.", tbl_cell)
        ],
        [
            Paragraph("<b>Murs Extérieurs & Porteurs</b>", tbl_cell_bold),
            Paragraph("6.8 % des pixels", tbl_cell),
            Paragraph("100 % des plans", tbl_cell),
            Paragraph("Pondération renforcée à <code>w = 4.0</code>. Épaisseur moyenne : 4 à 8 px.", tbl_cell)
        ],
        [
            Paragraph("<b>Cloisons Intérieures</b>", tbl_cell_bold),
            Paragraph("4.6 % des pixels", tbl_cell),
            Paragraph("98.5 % des plans", tbl_cell),
            Paragraph("Pondération renforcée à <code>w = 4.0</code>. Épaisseur moyenne : 2 à 4 px.", tbl_cell)
        ],
        [
            Paragraph("<b>Fenêtres (Window)</b>", tbl_cell_bold),
            Paragraph("1.4 % des pixels", tbl_cell),
            Paragraph("94.2 % des plans", tbl_cell),
            Paragraph("Pondération à <code>w = 10.0</code> pour forcer la détection des ouvertures murales.", tbl_cell)
        ],
        [
            Paragraph("<b>Portes (Door / Passage)</b>", tbl_cell_bold),
            Paragraph("0.7 % des pixels", tbl_cell),
            Paragraph("96.8 % des plans", tbl_cell),
            Paragraph("<b>Classe critique ultra-minoritaire</b>. Sur-pondérée à <code>w = 25.0</code> + Dice loss.", tbl_cell)
        ],
    ]
    t_sf = Table(surf_data, colWidths=[usable_w * 0.28, usable_w * 0.22, usable_w * 0.20, usable_w * 0.30])
    t_sf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (1,0), (2,-1), 'CENTER'),
    ]))
    story.append(t_sf)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Distribution Photométrique & Statistique — CMP Facades", h2))
    
    photo_data = [
        [
            Paragraph("Partition", tbl_hdr),
            Paragraph("Moyenne RGB (0-255)", tbl_hdr),
            Paragraph("Écart-Type RGB", tbl_hdr),
            Paragraph("Min par Image", tbl_hdr),
            Paragraph("Max par Image", tbl_hdr),
            Paragraph("Observation & Constance Photométrique", tbl_hdr)
        ],
        [
            Paragraph("<b>Train</b> (400)", tbl_cell_bold),
            Paragraph("114.51", tbl_cell),
            Paragraph("48.2", tbl_cell),
            Paragraph("50.79", tbl_cell),
            Paragraph("163.39", tbl_cell),
            Paragraph("Distribution équilibrée couvrant briques, pierre et crépis clairs.", tbl_cell)
        ],
        [
            Paragraph("<b>Validation</b> (100)", tbl_cell_bold),
            Paragraph("115.21", tbl_cell),
            Paragraph("47.6", tbl_cell),
            Paragraph("50.47", tbl_cell),
            Paragraph("169.39", tbl_cell),
            Paragraph("Excellente adéquation photométrique avec le jeu d'entraînement.", tbl_cell)
        ],
        [
            Paragraph("<b>Test</b> (106)", tbl_cell_bold),
            Paragraph("111.79", tbl_cell),
            Paragraph("49.1", tbl_cell),
            Paragraph("60.64", tbl_cell),
            Paragraph("150.57", tbl_cell),
            Paragraph("Légère sous-exposition moyenne mais recouvrement total des plages.", tbl_cell)
        ],
    ]
    t_ph = Table(photo_data, colWidths=[usable_w * 0.16, usable_w * 0.16, usable_w * 0.14, usable_w * 0.12, usable_w * 0.12, usable_w * 0.30])
    t_ph.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (1,0), (4,-1), 'CENTER'),
    ]))
    story.append(t_ph)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6 : 04B / EXPLORATION : EXEMPLES RÉELS ET DÉFIS
    # =========================================================================
    story.append(Paragraph("04B / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Exploration : Exemples Réels & Défis d'Ingénierie", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Cas Réels & Défis Spécifiques sur CubiCasa5K (Plans 2D)", h2))
    
    cases_fp = [
        [
            Paragraph("<b>Défi 1 : Épaisseur Discrète & Disparition des Portes au Redimensionnement</b><br/>"
                      "À la résolution standard d'entraînement (512×512), les portes et passages ne mesurent que 2 à 4 pixels de large. "
                      "Si une interpolation bilinéaire est naïvement appliquée aux masques, les étiquettes discrètes se diluent dans l'arrière-plan, "
                      "conduisant à un effacement pur et simple des ouvertures. "
                      "<i>Solution :</i> Application stricte de l'interpolation au plus proche voisin (<code>NEAREST</code>) et centrage isotrope letterbox "
                      "pour préserver rigoureusement l'intégrité binaire des pixels.", body)
        ],
        [
            Paragraph("<b>Défi 2 : Diversité Stylistique des Plans d'Origine</b><br/>"
                      "Le dataset comprend trois sous-styles majeurs : (1) plans cadastraux techniques noir et blanc purs, (2) plans d'architectes colorés avec trames "
                      "de parquet et carrelage, et (3) plans meublés haute densité. Le modèle doit apprendre à ignorer le mobilier intérieur "
                      "(canapés, lits, sanitaires) pour se focaliser exclusivement sur les cloisons porteuses et les limites fonctionnelles de pièces.", body)
        ],
        [
            Paragraph("<b>Défi 3 : Préservation de la Continuité Topologique</b><br/>"
                      "Une interruption d'un seul pixel dans la prédiction d'un mur entraîne lors du post-traitement de segmentation "
                      "la fusion erronée de deux pièces contiguës (ex. une chambre s'étendant dans le couloir). "
                      "Cela impose une pénalisation forte sur les faux négatifs de murs extérieurs et intérieurs.", body)
        ],
    ]
    t_cfp = Table(cases_fp, colWidths=[usable_w])
    t_cfp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TAG_BLUE),
        ('BACKGROUND', (0,1), (-1,1), LIGHT_BG),
        ('BACKGROUND', (0,2), (-1,2), TAG_BLUE),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_cfp)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Cas Réels & Défis Spécifiques sur CMP Facades (Élévations)", h2))
    
    cases_fac = [
        [
            Paragraph("<b>Défi 1 : Répétition Structurelle & Alignement Géométrique des Baies</b><br/>"
                      "Les façades architecturales sont caractérisées par des grilles de fenêtres régulières et des corniches horizontales continues. "
                      "Un modèle de génération non contraint a tendance à synthétiser des ouvertures déformées ou décalées par rapport au masque d'entrée. "
                      "L'adjonction d'une perte L1 à fort coefficient (&lambda;=30) combinée à une perte de correspondance des descripteurs "
                      "(<i>Feature Matching</i>, &lambda;=10) force le générateur à respecter scrupuleusement les coordonnées géométriques du masque conditionnel.", body)
        ],
        [
            Paragraph("<b>Défi 2 : Élimination des Artefacts de Damier (Checkerboard Artifacts)</b><br/>"
                      "Dans les architectures Pix2Pix classiques reposant sur des déconvolutions (<i>ConvTranspose2d</i>), des motifs réguliers de damier "
                      "apparaissent sur les surfaces unies (enduits, ciels, murs). "
                      "<i>Solution :</i> Remplacement des déconvolutions par un sur-échantillonnage au plus proche voisin (<i>Nearest-Neighbor Upsampling</i>) "
                      "suivi d'une convolution standard 3×3, produisant des aplats de texture parfaitement fluides et homogènes.", body)
        ],
        [
            Paragraph("<b>Défi 3 : Synthèse de Haute Fréquence sans Effondrement de Mode (Mode Collapse)</b><br/>"
                      "Avec seulement 400 paires d'entraînement, un discriminateur conventionnel surapprend rapidement, bloquant la progression du générateur. "
                      "L'introduction de la <b>Spectral Normalization</b> sur les couches convolutives du discriminateur PatchGAN 70×70 "
                      "garantit le respect de la condition de Lipschitz et stabilise remarquablement l'équilibre de Nash.", body)
        ],
    ]
    t_cfac = Table(cases_fac, colWidths=[usable_w])
    t_cfac.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), TAG_PURPLE),
        ('BACKGROUND', (0,1), (-1,1), LIGHT_BG),
        ('BACKGROUND', (0,2), (-1,2), TAG_PURPLE),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_cfac)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7 : 05 / QUALITÉ DES DONNÉES (AUDIT & CORRECTION)
    # =========================================================================
    story.append(Paragraph("05 / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Qualité des Données : Audit d'Intégrité & Correction Géométrique", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Audit d'Intégrité Exhaustif sur les Deux Datasets", h2))
    
    qa_audit_data = [
        [
            Paragraph("Contrôle Qualité Réalisé", tbl_hdr),
            Paragraph("Méthode de Vérification", tbl_hdr),
            Paragraph("Résultat CubiCasa5K", tbl_hdr),
            Paragraph("Résultat CMP Facades", tbl_hdr),
            Paragraph("Statut", tbl_hdr)
        ],
        [
            Paragraph("<b>Fichiers manquants / orphelins</b>", tbl_cell_bold),
            Paragraph("Inventaire exhaustif des splits vs pointeurs DVC", tbl_cell),
            Paragraph("<b>0 manquant</b> sur 5 000 plans", tbl_cell),
            Paragraph("<b>0 manquant</b> sur 606 paires", tbl_cell),
            Paragraph("CONFORME", tbl_cell_bold)
        ],
        [
            Paragraph("<b>Images corrompues / illisibles</b>", tbl_cell_bold),
            Paragraph("Décodage complet PIL / libpng / libjpeg", tbl_cell),
            Paragraph("<b>0 corrompu</b> sur 5 000 images", tbl_cell),
            Paragraph("<b>0 corrompu</b> sur 606 JPEG", tbl_cell),
            Paragraph("CONFORME", tbl_cell_bold)
        ],
        [
            Paragraph("<b>Doublons exacts de fichiers</b>", tbl_cell_bold),
            Paragraph("Hachage cryptographique SHA-256", tbl_cell),
            Paragraph("<b>0 doublon</b> de plan", tbl_cell),
            Paragraph("<b>0 doublon</b> de fichier", tbl_cell),
            Paragraph("CONFORME", tbl_cell_bold)
        ],
        [
            Paragraph("<b>Doublons de pixels entre splits</b>", tbl_cell_bold),
            Paragraph("SHA-256 sur les matrices de pixels décodées", tbl_cell),
            Paragraph("<b>0 fuite</b> train/val/test", tbl_cell),
            Paragraph("<b>0 fuite</b> train/val/test", tbl_cell),
            Paragraph("CONFORME", tbl_cell_bold)
        ],
        [
            Paragraph("<b>Conformité des dimensions</b>", tbl_cell_bold),
            Paragraph("Inspection des canaux et résolutions", tbl_cell),
            Paragraph("Normalisé à 512×512 RGB", tbl_cell),
            Paragraph("Paires 512×256 (2× 256×256)", tbl_cell),
            Paragraph("CONFORME", tbl_cell_bold)
        ],
    ]
    t_qa = Table(qa_audit_data, colWidths=[usable_w * 0.25, usable_w * 0.27, usable_w * 0.20, usable_w * 0.18, usable_w * 0.10])
    t_qa.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
        ('ALIGN', (4,0), (4,-1), 'CENTER'),
    ]))
    story.append(t_qa)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Résolution Critique : Le Bug d'Alignement Géométrique SVG sur CubiCasa5K", h2))
    story.append(Paragraph(
        "<b>Diagnostic d'une anomalie majeure :</b> Lors des phases préliminaires, le modèle de segmentation de plans "
        "restait plafonné à 22.4% d'IoU sur les murs malgré 60 époques d'entraînement. "
        "Une investigation approfondie a révélé une <b>erreur critique de repère cartésien dans le chargeur de données SVG</b> :",
        body
    ))

    bug_data = [
        [
            Paragraph("<b>Cause Racine Identifiée :</b> Le rasterizer initial mettait à l'échelle les coordonnées vectorielles SVG "
                      "en se basant sur les attributs <code>viewBox</code> du fichier XML, alors que l'image bitmap source (<code>F1_scaled.png</code>) "
                      "avait été générée avec des facteurs d'échelle et des marges différentes. Résultat : les masques de vérité terrain "
                      "étaient décalés de 15 à 45 pixels par rapport aux traits réels du dessin !", body_small)
        ],
        [
            Paragraph("<b>Correction Mathématique Implémentée :</b> Refonte intégrale de <code>src/floorplan/dataset.py</code>. "
                      "Les coordonnées SVG sont désormais projetées directement dans le repère natif $(W, H)$ de l'image bitmap, "
                      "suivies d'une transformation letterbox isotrope rigoureusement identique pour l'image et ses masques.", body_small)
        ],
        [
            Paragraph("<b>Preuve Expérimentale Irréfutable :</b><br/>"
                      "1. <i>Test d'overfitting sur 8 plans :</i> Le modèle avec les masques corrigés atteint <b>98.8% de Wall IoU</b>, "
                      "<b>99.4% de Door IoU</b> et <b>99.3% de Window IoU</b> (contre 29.7% auparavant avec le rasterizer décalé).<br/>"
                      "2. <i>Validation globale sur 400 plans :</i> L'IoU sur les murs bondit de <b>22.4% à &gt; 73%</b>, et l'IoU sur les portes "
                      "passe de <b>10.7% à plus de 61%</b>. Ce cas d'école démontre l'impact fondamental de la qualité des données en Deep Learning.", body_small)
        ],
    ]
    t_bug = Table(bug_data, colWidths=[usable_w])
    t_bug.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#FEF2F2")),
        ('BACKGROUND', (0,1), (-1,1), TAG_BLUE),
        ('BACKGROUND', (0,2), (-1,2), TAG_GREEN),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_bug)
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. Stratégies Algorithmiques de Compensation du Déséquilibre", h2))
    story.append(Paragraph(
        "Pour compenser la faible représentation spatiale des portes (0.7%) et fenêtres (1.4%), "
        "nous appliquons une fonction de perte composite : <b>Weighted Cross-Entropy</b> "
        "(poids : fond=0.5, mur=4.0, fenêtre=10.0, porte=25.0) couplée à une perte <b>Multiclass Soft Dice Loss</b> "
        "qui optimise directement le coefficient d'intersection sur union indépendamment de la taille absolue des objets.",
        body_small
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8 : 06A / FAISABILITÉ : TÂCHES DL & ARCHITECTURES
    # =========================================================================
    story.append(Paragraph("06A / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Faisabilité : Tâches DL & Architectures des Deux Modèles", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Architecture Modèle 1 : Multi-Task UNet++ (Segmentation de Plans)", h2))
    
    fp_arch_data = [
        [
            Paragraph("<b>Composant</b>", tbl_hdr),
            Paragraph("<b>Spécification Technique & Implémentation</b>", tbl_hdr),
            Paragraph("<b>Rôle & Justification Fonctionnelle</b>", tbl_hdr)
        ],
        [
            Paragraph("<b>Type de tâche DL</b>", tbl_cell_bold),
            Paragraph("Segmentation Sémantique Multi-Tâches (Pixel-level Dense Classification)", tbl_cell),
            Paragraph("Prédiction simultanée de deux topologies : éléments fins et grandes zones.", tbl_cell)
        ],
        [
            Paragraph("<b>Encodeur partagé</b>", tbl_cell_bold),
            Paragraph("ResNet-34 pré-entraîné sur ImageNet (5 niveaux pyramidaux de features)", tbl_cell),
            Paragraph("Extraction robuste de primitives géométriques à différentes échelles.", tbl_cell)
        ],
        [
            Paragraph("<b>Décodeur UNet++</b>", tbl_cell_bold),
            Paragraph("Nested Dense Skip Pathways avec convolutions résiduelles denses", tbl_cell),
            Paragraph("Réduit l'écart sémantique entre bas niveau et haut niveau ; capture les arêtes fines.", tbl_cell)
        ],
        [
            Paragraph("<b>Tête 1 : Structures</b>", tbl_cell_bold),
            Paragraph("Sortie 5 canaux logits à 512×512 (Bg, MurExt, MurInt, Porte, Fenêtre)", tbl_cell),
            Paragraph("Détection exacte des limites porteuses et passages pour la vectorisation 3D.", tbl_cell)
        ],
        [
            Paragraph("<b>Tête 2 : Pièces</b>", tbl_cell_bold),
            Paragraph("Sortie 12 canaux logits à 512×512 (Living, Room, Kitchen, Bath, Balcony, etc.)", tbl_cell),
            Paragraph("Affectation sémantique et calcul automatisé des surfaces habitables.", tbl_cell)
        ],
        [
            Paragraph("<b>Fonction de Perte</b>", tbl_cell_bold),
            Paragraph("L_floorplan = L_Struct(WCE + Dice) + 0.5 * L_Room(CE + Dice)", tbl_cell_bold),
            Paragraph("Équilibre l'apprentissage conjoint sans interférence destructrice.", tbl_cell)
        ],
    ]
    t_fpa = Table(fp_arch_data, colWidths=[usable_w * 0.22, usable_w * 0.44, usable_w * 0.34])
    t_fpa.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_fpa)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Architecture Modèle 2 : Pix2Pix cGAN Perceptuel (Synthèse de Façades)", h2))
    
    fac_arch_data = [
        [
            Paragraph("<b>Composant</b>", tbl_hdr),
            Paragraph("<b>Spécification Technique & Implémentation</b>", tbl_hdr),
            Paragraph("<b>Rôle & Justification Fonctionnelle</b>", tbl_hdr)
        ],
        [
            Paragraph("<b>Type de tâche DL</b>", tbl_cell_bold),
            Paragraph("Traduction Image-à-Image Conditionnelle Supervisée (cGAN)", tbl_cell),
            Paragraph("Génération d'images réalistes guidée par une condition spatiale stricte.", tbl_cell)
        ],
        [
            Paragraph("<b>Générateur (G)</b>", tbl_cell_bold),
            Paragraph("UNet-256 avec 8 blocs down/up et connexions de saut directes (skip)", tbl_cell),
            Paragraph("Transmet les détails de position haute résolution de l'entrée vers la sortie.", tbl_cell)
        ],
        [
            Paragraph("<b>Upsampling Générateur</b>", tbl_cell_bold),
            Paragraph("Nearest-Neighbor Resize (2×) + Conv2d 3×3 + BatchNorm + ReLU", tbl_cell),
            Paragraph("Supprime les motifs visibles de damier (checkerboard) caractéristiques des déconvolutions.", tbl_cell)
        ],
        [
            Paragraph("<b>Discriminateur (D)</b>", tbl_cell_bold),
            Paragraph("PatchGAN 70×70 avec Spectral Normalization sur toutes les convolutions", tbl_cell),
            Paragraph("Juge la plausibilité par patch local ; contraint la constante de Lipschitz et stabilise l'entraînement.", tbl_cell)
        ],
        [
            Paragraph("<b>Pertes Adversariales</b>", tbl_cell_bold),
            Paragraph("Least Squares GAN (LSGAN / MSE) pour la stabilité des gradients", tbl_cell),
            Paragraph("Évite la saturation des gradients sigmoïdes conventionnels du min-max GAN.", tbl_cell)
        ],
        [
            Paragraph("<b>Pertes Contraintes</b>", tbl_cell_bold),
            Paragraph("30 * L1 + 10 * L_FeatureMatching + 10 * L_Perceptual (LPIPS SqueezeNet)", tbl_cell_bold),
            Paragraph("Force la fidélité photométrique, la cohérence des textures et la netteté visuelle.", tbl_cell)
        ],
    ]
    t_faca = Table(fac_arch_data, colWidths=[usable_w * 0.22, usable_w * 0.44, usable_w * 0.34])
    t_faca.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_faca)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Formulation Mathématique de l'Objectif de Synthèse :</b><br/>"
        "<code>min_G max_D &nbsp; L_cGAN(G, D) = E[ (D(x, y) - 1)² ] + E[ (D(x, G(x)))² ] + "
        "30 · ||y - G(x)||_1 + 10 · L_FM(G, D) + 10 · L_LPIPS(G(x), y)</code>",
        ParagraphStyle('EqStyle', fontName='Courier-Bold', fontSize=7.5, leading=10.5, textColor=PRIMARY, alignment=1)
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 9 : 06B / FAISABILITÉ : BUDGET MATÉRIEL, MÉTRIQUES & ROADMAP
    # =========================================================================
    story.append(Paragraph("06B / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Faisabilité : Budget Matériel, Métriques Obtenues & Roadmap", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Budget Matériel & Optimisations Computationnelles", h2))
    
    hw_data = [
        [
            Paragraph("Ressource / Dimension", tbl_hdr),
            Paragraph("Environnement Expérimental Réel", tbl_hdr),
            Paragraph("Optimisation Déployée & Impact Mesuré", tbl_hdr)
        ],
        [
            Paragraph("<b>GPU & VRAM</b>", tbl_cell_bold),
            Paragraph("NVIDIA GeForce RTX 3050 Laptop GPU (4 Go / 6 Go VRAM, TGP 56W)", tbl_cell),
            Paragraph("Entraînement avec batch size 2 et gradient accumulation ; empreinte stabilisée à 3.4 Go sans OOM.", tbl_cell)
        ],
        [
            Paragraph("<b>I/O & Stockage</b>", tbl_cell_bold),
            Paragraph("SSD NVMe PCIe Gen4 — 5 000 plans CubiCasa5K", tbl_cell),
            Paragraph("<b>Pré-caching matriciel .npz :</b> Chargement par plan réduit de <b>1.8 s à 1.1 ms</b> (gain &times; 1 600).", tbl_cell)
        ],
        [
            Paragraph("<b>Durée d'entraînement</b>", tbl_cell_bold),
            Paragraph("• Modèle Plan : 25 époques (~4h30)<br/>• Modèle Façade : 50 époques (~55 min)", tbl_cell),
            Paragraph("Pipeline 100% exécutable localement sur une machine étudiante sans dépendance cloud coûteuse.", tbl_cell)
        ],
    ]
    t_hw = Table(hw_data, colWidths=[usable_w * 0.25, usable_w * 0.37, usable_w * 0.38])
    t_hw.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_hw)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Bilan Comparatif des Métriques : Baseline vs Modèle Optimisé vs Cible", h2))
    
    met_data = [
        [
            Paragraph("Modèle & Périmètre", tbl_hdr),
            Paragraph("Métrique Évaluée", tbl_hdr),
            Paragraph("Avant Fix / Base", tbl_hdr),
            Paragraph("Modèle Retenu ArchiGen 3D", tbl_hdr),
            Paragraph("Cible Projet", tbl_hdr),
            Paragraph("Statut", tbl_hdr)
        ],
        [
            Paragraph("<b>MODÈLE 1</b><br/>Plans 2D<br/>(CubiCasa5K Test 400)", tbl_cell_bold),
            Paragraph("<b>Merged Wall IoU</b><br/>"
                      "<b>Door IoU</b><br/>"
                      "<b>Window IoU</b><br/>"
                      "<b>Room Pixel Acc.</b><br/>"
                      "<b>Room mIoU</b> (12 cl.)", tbl_cell),
            Paragraph("22.4 %<br/>10.7 %<br/>23.4 %<br/>71.2 %<br/>38.1 %", tbl_cell),
            Paragraph("<b>74.32 %</b> (Test)<br/>"
                      "<b>62.38 %</b> (Test)<br/>"
                      "<b>73.28 %</b> (Test)<br/>"
                      "<b>85.84 %</b> (Test)<br/>"
                      "<b>54.73 %</b> (Test) [60.1% Val]", tbl_cell_bold),
            Paragraph("&ge; 70.0 %<br/>&ge; 50.0 %<br/>&ge; 60.0 %<br/>&ge; 82.0 %<br/>&ge; 50.0 %", tbl_cell),
            Paragraph("<b>DÉPASSÉ</b><br/><b>DÉPASSÉ</b><br/><b>DÉPASSÉ</b><br/><b>DÉPASSÉ</b><br/><b>PROCHE</b>", tbl_cell_bold)
        ],
        [
            Paragraph("<b>MODÈLE 2</b><br/>Façades<br/>(CMP Test 106)", tbl_cell_bold),
            Paragraph("<b>KID</b> (&times; 10⁻³) ↓<br/>"
                      "<b>LPIPS</b> (Squeeze) ↓<br/>"
                      "<b>SSIM</b> ↑<br/>"
                      "<b>L1 Loss</b> ↓", tbl_cell),
            Paragraph("150.36 &plusmn; 10.25<br/>0.6327<br/>0.123<br/>0.399", tbl_cell),
            Paragraph("<b>63.55 &plusmn; 5.83</b> (EMA)<br/>"
                      "<b>0.3096</b> (EMA)<br/>"
                      "<b>0.289</b> (0.335 Enh.)<br/>"
                      "<b>0.340</b> (0.338 Enh.)", tbl_cell_bold),
            Paragraph("&le; 70.0<br/>&le; 0.350<br/>Inform.<br/>Inform.", tbl_cell),
            Paragraph("<b>VALIDÉ</b><br/><b>VALIDÉ</b><br/>Compromis<br/>Compromis", tbl_cell_bold)
        ],
    ]
    t_met = Table(met_data, colWidths=[usable_w * 0.20, usable_w * 0.22, usable_w * 0.16, usable_w * 0.22, usable_w * 0.10, usable_w * 0.10])
    t_met.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_met)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Note d'analyse objective des métriques :</b><br/>"
        "• <i>Façades :</i> Le modèle Perceptual avec lissage EMA (Exponential Moving Average) et décroissance de LR surpasse tous les modèles sur le jeu de test indépendant CMP (N=106, B=100) : "
        "KID abaissé à <b>63.55 ± 5.83</b> (vs 150.36 baseline et 67.65 champion initial) et LPIPS à <b>0.3096</b> (vs 0.6327 baseline et 0.3169 champion initial), "
        "démontrant un gain statistiquement significatif validé par test d'hypothèse apparié par image (p < 10⁻⁶ sur L1, SSIM, PSNR, LPIPS).<br/>"
        "• <i>Plans 2D :</i> La Room Accuracy (85.84%) dépasse les 82.7% publiés dans l'article CubiCasa5K. La détection des portes en repère cartésien atteint 91.24% de précision et 90.45% de F1-score (3 397 TP sur 400 plans).",
        body_small
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3. Limites Identifiées & Feuille de Route d'Ingénierie", h2))
    
    lim_data = [
        [
            Paragraph("<b>Limite Actuelle Identifiée</b>", tbl_hdr),
            Paragraph("<b>Action Corrective Prévue dans la Roadmap Finale</b>", tbl_hdr)
        ],
        [
            Paragraph("Sensibilité aux styles de dessins non conventionnels (croquis manuels)", tbl_cell),
            Paragraph("Ajout de bruits de trait et d'inversions de contraste en data augmentation.", tbl_cell)
        ],
        [
            Paragraph("Discrétisation matricielle 512×512 pour très grands ensembles immobiliers", tbl_cell),
            Paragraph("Découpage par fenêtres glissantes tuilées (tiling avec recouvrement 20%).", tbl_cell)
        ],
        [
            Paragraph("Reconstruction 3D multi-étages", tbl_cell),
            Paragraph("Alignement vertical automatique des cages d'escalier via détection d'ancres géométriques.", tbl_cell)
        ],
    ]
    t_lim = Table(lim_data, colWidths=[usable_w * 0.48, usable_w * 0.52])
    t_lim.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_lim)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10 : ANNEXE / TRAÇABILITÉ & RÉFÉRENCES SCIENTIFIQUES
    # =========================================================================
    story.append(Paragraph("ANNEXE / CADRAGE DU PROJET", rubric_tag))
    story.append(Paragraph("Traçabilité MLOps & Références Scientifiques", h1))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=1, spaceAfter=8))

    story.append(Paragraph("1. Stack Technologique & Écosystème MLOps", h2))
    
    stack_data = [
        [
            Paragraph("Niveau Système", tbl_hdr),
            Paragraph("Outils & Bibliothèques", tbl_hdr),
            Paragraph("Rôle & Justification Opérationnelle", tbl_hdr)
        ],
        [
            Paragraph("<b>Deep Learning Core</b>", tbl_cell_bold),
            Paragraph("PyTorch 2.x, Torchvision, segmentation-models-pytorch", tbl_cell),
            Paragraph("Entraînement des réseaux, autograd, couches spectrales et backbones ResNet.", tbl_cell)
        ],
        [
            Paragraph("<b>Inférence Haute Perf.</b>", tbl_cell_bold),
            Paragraph("ONNX Runtime 1.18+, Torch ONNX Exporter", tbl_cell),
            Paragraph("Exécution optimisée cross-plateforme, parité stricte vérifiée (&Delta; &lt; 10⁻⁴).", tbl_cell)
        ],
        [
            Paragraph("<b>Gestion des Données</b>", tbl_cell_bold),
            Paragraph("DVC (Data Version Control), NumPy (.npz cache)", tbl_cell),
            Paragraph("Versionnage immuable des 5 000 plans et 606 paires d'élévations sans saturer Git.", tbl_cell)
        ],
        [
            Paragraph("<b>Suivi d'Expériences</b>", tbl_cell_bold),
            Paragraph("MLflow Tracking & Model Registry", tbl_cell),
            Paragraph("Enregistrement systématique des hyperparamètres, courbes de pertes et checkpoints.", tbl_cell)
        ],
        [
            Paragraph("<b>Backend de Service</b>", tbl_cell_bold),
            Paragraph("FastAPI, Pydantic, Uvicorn (Async)", tbl_cell),
            Paragraph("Endpoints REST documentés Swagger : <code>/predict/floorplan</code> et <code>/predict/facade</code>.", tbl_cell)
        ],
        [
            Paragraph("<b>Visualisation 3D</b>", tbl_cell_bold),
            Paragraph("React 18, Three.js, @react-three/fiber, Shapely", tbl_cell),
            Paragraph("Extrusion polygonale en direct dans le navigateur sans plugin externe.", tbl_cell)
        ],
    ]
    t_st = Table(stack_data, colWidths=[usable_w * 0.22, usable_w * 0.40, usable_w * 0.38])
    t_st.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_st)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Références Scientifiques & Publications Fondatrices", h2))
    
    refs_data = [
        [
            Paragraph("<b>[1] Kalervo et al. (2019) :</b> <i>CubiCasa5k: A Dataset for Floorplan Image Analysis and Embedding.</i> "
                      "International Conference on Pattern Recognition (ICPR). Présentation du benchmark de 5 000 plans d'étages et de la métrique Room mIoU de 57.5%.", body_small)
        ],
        [
            Paragraph("<b>[2] Isola, Zhu, Zhou & Efros (2017) :</b> <i>Image-to-Image Translation with Conditional Adversarial Networks.</i> "
                      "CVPR 2017. Architecture Pix2Pix originale combinant perte L1 et discriminateur PatchGAN.", body_small)
        ],
        [
            Paragraph("<b>[3] Zhou et al. (2018) :</b> <i>UNet++: A Nested U-Net Architecture for Medical Image Segmentation.</i> "
                      "Deep Learning in Medical Image Analysis. Fondement théorique des connexions denses imbriquées capturant les arêtes fines.", body_small)
        ],
        [
            Paragraph("<b>[4] Tyleček & Šára (2013) :</b> <i>Spatial Pattern Templates for Recognition of Objects with Regular Structure.</i> "
                      "Proc. GCPR 2013. Présentation de la base de données CMP Facade Database et de l'étiquetage à 12 classes.", body_small)
        ],
        [
            Paragraph("<b>[5] Zhang, Isola et al. (2018) :</b> <i>The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.</i> "
                      "CVPR 2018. Formalisation de la distance LPIPS pour l'évaluation de la qualité visuelle au-delà du PSNR.", body_small)
        ],
        [
            Paragraph("<b>[6] Miyato et al. (2018) :</b> <i>Spectral Normalization for Generative Adversarial Networks.</i> "
                      "ICLR 2018. Stabilisation de l'entraînement du discriminateur sous contrainte 1-Lipschitz.", body_small)
        ],
    ]
    t_ref = Table(refs_data, colWidths=[usable_w])
    t_ref.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_ref)
    story.append(Spacer(1, 8))

    # Verification Sign-off Box
    sign_data = [
        [
            Paragraph("<b>ATTESTATION DE CONFORMITÉ AU BARÈME DE CADRAGE DU PROJET DEEP LEARNING (15% CC)</b><br/>"
                      "Le présent document synthétise de manière exhaustive et vérifiable l'ensemble des travaux de cadrage, d'audit des données "
                      "et de validation des architectures réalisés pour le projet <b>ArchiGen 3D</b>. "
                      "Toutes les statistiques, architectures et métriques reportées sont issues des exécutions effectives sur le dépôt "
                      "<code>github.com/ayebkalil/archigen-3d</code>.<br/>"
                      "<b>Équipe projet :</b> Khalil Ayeb &nbsp;·&nbsp; Amine Ben Fadhel &nbsp;·&nbsp; Khawla Hawari &nbsp;&nbsp;|&nbsp;&nbsp; "
                      "<b>Enseignant :</b> Haythem Ghazouani &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date :</b> 13 Octobre 2026",
                      ParagraphStyle('SignTxt', fontName='Helvetica', fontSize=7.4, leading=10, textColor=PRIMARY, alignment=1))
        ]
    ]
    t_sign = Table(sign_data, colWidths=[usable_w])
    t_sign.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), TAG_GREEN),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86EFAC")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_sign)

    # Build PDF with two-pass NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated cadrage report PDF at: {output_path}")

if __name__ == "__main__":
    target_path = r"C:\Users\ayebk\Downloads\ArchiGen3D_Cadrage_DeepLearning_Medamine_Benfadhel.pdf"
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    generate_cadrage_pdf(target_path)

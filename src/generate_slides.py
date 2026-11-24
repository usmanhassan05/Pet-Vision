"""
Generate Academic Presentation Slides for PetVision AI
======================================================
Generates exactly 3 slides matching Dr. Rao Muhammad Adeel Nawab's required template:
- Slide 1: Title Slide (Semester Project, Title, Presenter, Instructor, Department)
- Slide 2: Technical Specification (Dataset, Technique, Models, Metric, Top Performing Model)
- Slide 3: Course Feedback (What I Learned, Strengths, Suggestions - 3x3 layout)
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette matching the slide template in the prompt
    NAVY_DARK = RGBColor(15, 23, 42)       # #0f172a
    NAVY_CARD = RGBColor(30, 41, 59)       # #1e293b
    BLUE_ACCENT = RGBColor(59, 130, 246)   # #3b82f6
    CYAN_ACCENT = RGBColor(6, 182, 212)    # #06b6d4
    TEXT_WHITE = RGBColor(255, 255, 255)
    TEXT_MUTED = RGBColor(148, 163, 184)   # #94a3b8
    CARD_BG = RGBColor(248, 250, 252)      # #f8fafc
    CARD_BORDER = RGBColor(226, 232, 240)  # #e2e8f0
    TEXT_DARK = RGBColor(15, 23, 42)
    GREEN_ACCENT = RGBColor(16, 185, 129)  # #10b981
    AMBER_ACCENT = RGBColor(245, 158, 11)  # #f59e0b

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Navy Modern Layout)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY_DARK
    bg1.line.color.rgb = NAVY_DARK

    # Badge Pill: SEMESTER PROJECT
    badge = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.0), Inches(2.8), Inches(0.45))
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(30, 58, 138)
    badge.line.color.rgb = BLUE_ACCENT
    btf = badge.text_frame
    btf.word_wrap = True
    bp = btf.paragraphs[0]
    bp.alignment = PP_ALIGN.CENTER
    bp.text = "🎓  SEMESTER PROJECT"
    bp.font.size = Pt(12)
    bp.font.bold = True
    bp.font.color.rgb = RGBColor(191, 219, 254)

    # Main Title
    tb_title = s1.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(1.5))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = "PetVision AI: Breed Identification System"
    p_title.font.size = Pt(40)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE

    # Subtitle
    p_sub = tf_title.add_paragraph()
    p_sub.text = "Fine-Grained Pet Recognition using Deep Transfer Learning, Vision Transformers, and Cloud Web Deployment"
    p_sub.font.size = Pt(17)
    p_sub.font.color.rgb = TEXT_MUTED

    # Divider Line
    line = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(5.2), Inches(11.333), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(51, 65, 85)
    line.line.color.rgb = RGBColor(51, 65, 85)

    # Footer Left: Presented By
    tb_f1 = s1.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(3.5), Inches(1.4))
    tf_f1 = tb_f1.text_frame
    p_lbl1 = tf_f1.paragraphs[0]
    p_lbl1.text = "PRESENTED BY"
    p_lbl1.font.size = Pt(11)
    p_lbl1.font.bold = True
    p_lbl1.font.color.rgb = TEXT_MUTED
    p_val1 = tf_f1.add_paragraph()
    p_val1.text = "Mr. Usman Hassan\nBS Computer Science / Software Engineering"
    p_val1.font.size = Pt(14)
    p_val1.font.bold = True
    p_val1.font.color.rgb = TEXT_WHITE

    # Footer Middle: Instructor
    tb_f2 = s1.shapes.add_textbox(Inches(4.8), Inches(5.5), Inches(4.0), Inches(1.4))
    tf_f2 = tb_f2.text_frame
    p_lbl2 = tf_f2.paragraphs[0]
    p_lbl2.text = "COURSE INSTRUCTOR"
    p_lbl2.font.size = Pt(11)
    p_lbl2.font.bold = True
    p_lbl2.font.color.rgb = TEXT_MUTED
    p_val2 = tf_f2.add_paragraph()
    p_val2.text = "Dr. Rao Muhammad Adeel Nawab\nAssociate Professor"
    p_val2.font.size = Pt(14)
    p_val2.font.bold = True
    p_val2.font.color.rgb = TEXT_WHITE

    # Footer Right: Department
    tb_f3 = s1.shapes.add_textbox(Inches(9.0), Inches(5.5), Inches(3.8), Inches(1.4))
    tf_f3 = tb_f3.text_frame
    p_lbl3 = tf_f3.paragraphs[0]
    p_lbl3.text = "DEPARTMENT & INSTITUTION"
    p_lbl3.font.size = Pt(11)
    p_lbl3.font.bold = True
    p_lbl3.font.color.rgb = TEXT_MUTED
    p_val3 = tf_f3.add_paragraph()
    p_val3.text = "Department of Computer Science\nCOMSATS University Islamabad, Lahore Campus"
    p_val3.font.size = Pt(13)
    p_val3.font.bold = True
    p_val3.font.color.rgb = CYAN_ACCENT

    # =========================================================================
    # SLIDE 2: SYSTEM ARCHITECTURE & EVALUATION (Split Card Layout)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    bg2 = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg2.fill.solid()
    bg2.fill.fore_color.rgb = RGBColor(241, 245, 249)
    bg2.line.color.rgb = RGBColor(241, 245, 249)

    # Slide Header
    tb_h2 = s2.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.9))
    tf_h2 = tb_h2.text_frame
    p_sub2 = tf_h2.paragraphs[0]
    p_sub2.text = "Technical Specification & Benchmark Evaluation"
    p_sub2.font.size = Pt(12)
    p_sub2.font.bold = True
    p_sub2.font.color.rgb = BLUE_ACCENT
    p_h2 = tf_h2.add_paragraph()
    p_h2.text = "PetVision AI: System Architecture & Results"
    p_h2.font.size = Pt(28)
    p_h2.font.bold = True
    p_h2.font.color.rgb = TEXT_DARK

    # Left Column: Detailed Specifications (Cards)
    # Card 1: Dataset
    c1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(6.8), Inches(1.5))
    c1.fill.solid()
    c1.fill.fore_color.rgb = TEXT_WHITE
    c1.line.color.rgb = CARD_BORDER
    tf_c1 = c1.text_frame
    tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]
    p.text = "📊  Oxford-IIIT Pet Dataset (7,390 Instances)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p2 = tf_c1.add_paragraph()
    p2.text = "• 37 Fine-grained classes: 12 Cat breeds (2,400 imgs) & 25 Dog breeds (4,990 imgs).\n• 100% full dataset utilized with zero class downsampling. Verified RGB standardization."
    p2.font.size = Pt(11)
    p2.font.color.rgb = RGBColor(71, 85, 105)

    # Card 2: Feature Extraction & Validation
    c2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.05), Inches(6.8), Inches(1.6))
    c2.fill.solid()
    c2.fill.fore_color.rgb = TEXT_WHITE
    c2.line.color.rgb = CARD_BORDER
    tf_c2 = c2.text_frame
    tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]
    p.text = "⚙️  Technique & Validation Methodology"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p2 = tf_c2.add_paragraph()
    p2.text = "• Feature Extraction: Data Augmentations (ColorJitter, Flips, Rotations) & Deep Transfer Heads.\n• Validation Scheme: Stratified 70/15/15 Train-Val-Test Split (5,173 train / 1,108 val / 1,109 test).\n• Evaluation Metrics: Top-1 & Top-3 Accuracy, Macro & Weighted F1-Score, Latency (ms)."
    p2.font.size = Pt(11)
    p2.font.color.rgb = RGBColor(71, 85, 105)

    # Card 3: Models Evaluated
    c3 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.8), Inches(6.8), Inches(2.1))
    c3.fill.solid()
    c3.fill.fore_color.rgb = TEXT_WHITE
    c3.line.color.rgb = CARD_BORDER
    tf_c3 = c3.text_frame
    tf_c3.word_wrap = True
    p = tf_c3.paragraphs[0]
    p.text = "🔬  Machine Learning & Deep Architectures Evaluated"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p2 = tf_c3.add_paragraph()
    p2.text = "1. Custom CNN (From Scratch): 4-Stage Conv-BatchNorm-Dropout (1.4M params) → 58.4% Acc.\n2. MobileNetV2 (Transfer Learning): 2-Stage frozen backbone + top fine-tuning → 92.4% Acc.\n3. Vision Transformer (ViT-B/16): 12 Multi-head self-attention blocks (86.2M params) → 93.1% Acc.\n• Live Cloud Deployment: Streamlit Community Cloud (https://petvision-ai.streamlit.app/)"
    p2.font.size = Pt(11)
    p2.font.color.rgb = RGBColor(71, 85, 105)

    # Right Column: Big Highlight Card (Navy Trophy Card)
    c_right = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.0), Inches(1.4), Inches(4.5), Inches(5.5))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = NAVY_DARK
    c_right.line.color.rgb = BLUE_ACCENT
    tf_r = c_right.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "🏆\nTOP PERFORMING MODEL"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT

    p_acc = tf_r.add_paragraph()
    p_acc.alignment = PP_ALIGN.CENTER
    p_acc.text = "92.4%"
    p_acc.font.size = Pt(54)
    p_acc.font.bold = True
    p_acc.font.color.rgb = TEXT_WHITE

    p_subacc = tf_r.add_paragraph()
    p_subacc.alignment = PP_ALIGN.CENTER
    p_subacc.text = "Top-1 Test Accuracy (Top-3: 98.6%)"
    p_subacc.font.size = Pt(13)
    p_subacc.font.color.rgb = RGBColor(147, 197, 253)

    p_div = tf_r.add_paragraph()
    p_div.text = "\nArchitecture: MobileNetV2 (Fine-Tuned)\nMacro F1-Score: 0.923 | Weighted F1: 0.924\nInference Latency: 14.1 ms (~70 FPS)\nModel Weight Size: 11.3 MB (Ultra Lightweight)\nParameters: 2.89 Million (vs 86M for ViT)\nOptimal Efficiency / Accuracy Trade-off"
    p_div.font.size = Pt(12)
    p_div.font.color.rgb = RGBColor(226, 232, 240)

    # =========================================================================
    # SLIDE 3: COURSE FEEDBACK (3-Column Clean Card Layout)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    bg3 = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg3.fill.solid()
    bg3.fill.fore_color.rgb = RGBColor(241, 245, 249)
    bg3.line.color.rgb = RGBColor(241, 245, 249)

    # Header
    tb_h3 = s3.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.9))
    tf_h3 = tb_h3.text_frame
    p_sub3 = tf_h3.paragraphs[0]
    p_sub3.text = "Reflections & Insights"
    p_sub3.font.size = Pt(12)
    p_sub3.font.bold = True
    p_sub3.font.color.rgb = BLUE_ACCENT
    p_h3 = tf_h3.add_paragraph()
    p_h3.text = "Course Feedback: Machine Learning Fundamentals (AIC354)"
    p_h3.font.size = Pt(28)
    p_h3.font.bold = True
    p_h3.font.color.rgb = TEXT_DARK

    card_width = Inches(3.64)
    card_height = Inches(5.4)
    gap = Inches(0.3)
    top_pos = Inches(1.5)

    # Card 1: What I Learned
    col1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_pos, card_width, card_height)
    col1.fill.solid()
    col1.fill.fore_color.rgb = TEXT_WHITE
    col1.line.color.rgb = CARD_BORDER
    tf_col1 = col1.text_frame
    tf_col1.word_wrap = True
    p = tf_col1.paragraphs[0]
    p.text = "💡  What I Learned"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p_pts = tf_col1.add_paragraph()
    p_pts.text = (
        "\n1. End-to-End Deep Learning Lifecycle:\n"
        "Mastered raw unstructured image processing, stratified splits, augmentation pipelines, and reproducible training loops.\n\n"
        "2. Transfer Learning vs. Scratch:\n"
        "Observed empirical proof that pre-trained deep backbones (MobileNetV2, ViT) dramatically surpass scratch models on fine-grained data (92.4% vs 58.4%).\n\n"
        "3. Rigorous Evaluation & Deployment:\n"
        "Learned that Top-k accuracy, confusion matrices, and inference latency are vital for production ML beyond overall accuracy."
    )
    p_pts.font.size = Pt(11.5)
    p_pts.font.color.rgb = RGBColor(51, 65, 85)

    # Card 2: Strengths of Course
    col2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8) + card_width + gap, top_pos, card_width, card_height)
    col2.fill.solid()
    col2.fill.fore_color.rgb = TEXT_WHITE
    col2.line.color.rgb = CARD_BORDER
    tf_col2 = col2.text_frame
    tf_col2.word_wrap = True
    p = tf_col2.paragraphs[0]
    p.text = "⭐  Strengths of Course"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p_pts = tf_col2.add_paragraph()
    p_pts.text = (
        "\n1. Practical, Dataset-Driven Focus:\n"
        "Direct emphasis on solving non-trivial, real-world unstructured data challenges rather than purely toy tabular datasets.\n\n"
        "2. Pedagogical Rigor & High Standards:\n"
        "Exceptional curriculum structure connecting theoretical foundations (loss functions, optimization) to functional PyTorch code.\n\n"
        "3. Industry-Aligned Cloud Deployment:\n"
        "Requiring end-to-end web deployment bridges the gap between academic notebook prototypes and actual software engineering."
    )
    p_pts.font.size = Pt(11.5)
    p_pts.font.color.rgb = RGBColor(51, 65, 85)

    # Card 3: Humble Suggestions
    col3 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8) + (card_width + gap) * 2, top_pos, card_width, card_height)
    col3.fill.solid()
    col3.fill.fore_color.rgb = TEXT_WHITE
    col3.line.color.rgb = CARD_BORDER
    tf_col3 = col3.text_frame
    tf_col3.word_wrap = True
    p = tf_col3.paragraphs[0]
    p.text = "🛠️  Suggestions"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = AMBER_ACCENT
    p_pts = tf_col3.add_paragraph()
    p_pts.text = (
        "\n1. Institutional Cloud GPU Resources:\n"
        "Providing shared institutional Colab Pro or GPU access for students training heavy Vision Transformers or LLMs.\n\n"
        "2. Modern Transformer Workshops:\n"
        "Adding a dedicated mid-semester hands-on lab specifically focusing on Hugging Face transformers and attention visualization.\n\n"
        "3. Milestone Check-ins:\n"
        "Introducing an early 1-page milestone checkpoint to offer early feedback before final end-to-end submission."
    )
    p_pts.font.size = Pt(11.5)
    p_pts.font.color.rgb = RGBColor(51, 65, 85)

    # Output paths
    output_path = "PetVision_AI_Presentation.pptx"
    prs.save(output_path)
    print(f"Successfully generated presentation: {output_path}")

if __name__ == "__main__":
    create_presentation()

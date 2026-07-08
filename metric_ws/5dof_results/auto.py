import os
import json
import csv
import glob
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class RobotCertificateGenerator:
    def __init__(self, results_dir="5dof_results"):
        """Initializes the generator and points to the directory containing module results."""
        self.results_dir = results_dir
        self.styles = getSampleStyleSheet()
        self.elements =[]
        
        # Professional Styles
        self.styles.add(ParagraphStyle(name='CertTitle', fontSize=22, leading=26, alignment=1, spaceAfter=20, textColor=colors.HexColor("#1A365D"), fontName="Helvetica-Bold"))
        self.styles.add(ParagraphStyle(name='SectionHeader', fontSize=14, leading=18, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor("#2B6CB0"), fontName="Helvetica-Bold"))
        self.styles.add(ParagraphStyle(name='LaymanText', fontSize=10, leading=14, spaceAfter=10, fontName="Helvetica-Oblique", textColor=colors.HexColor("#4A5568")))
        self.styles.add(ParagraphStyle(name='StandardText', fontSize=11, leading=15, spaceAfter=8, fontName="Helvetica"))

    def get_latest_file(self, search_path):
        """Dynamically finds the latest file matching a path/pattern."""
        files = glob.glob(search_path)
        if not files:
            return None
        return max(files, key=os.path.getctime)

    def add_title(self):
        self.elements.append(Paragraph("ROBOTIC ARM QUALITY & PERFORMANCE CERTIFICATE", self.styles['CertTitle']))
        self.elements.append(Paragraph("This document certifies the evaluated metrics, kinematics, and operational reach of the robotic manipulator based on automated benchmark testing.", self.styles['StandardText']))
        self.elements.append(Spacer(1, 0.2 * inch))

    def process_kinematics_and_workspace(self):
        """Processes Module 1 and 2 (Kinematics & Reach)"""
        metadata_file = self.get_latest_file(os.path.join(self.results_dir, "workspace_fk_dataset_*_metadata.json"))
        csv_file = self.get_latest_file(os.path.join(self.results_dir, "workspace_fk_dataset_*.csv"))

        # --- SECTION 1: KINEMATICS ---
        self.elements.append(Paragraph("1. Kinematic Profile & Physical Specifications", self.styles['SectionHeader']))
        explanation_1 = ("<b>Metric Definition:</b> The Kinematic Profile defines the physical 'skeleton' of the robot.<br/>"
                         "<b>Implication:</b> Degrees of Freedom (DOF) indicates maneuverability. Joint limits define the safe physical boundaries the robot can move within.")
        self.elements.append(Paragraph(explanation_1, self.styles['LaymanText']))

        if metadata_file:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            self.elements.append(Paragraph(f"<b>Degrees of Freedom (DOF):</b> {data.get('robot_dof', 'Unknown')}-DOF", self.styles['StandardText']))
            self.elements.append(Paragraph(f"<b>Base Anchor:</b> {data.get('base_link', 'Unknown')} | <b>Tool Tip:</b> {data.get('end_link', 'Unknown')}", self.styles['StandardText']))
        else:
            self.elements.append(Paragraph("<i>⚠️ Kinematic metadata not found.</i>", self.styles['StandardText']))
        self.elements.append(Spacer(1, 0.2 * inch))

        # --- SECTION 2: WORKSPACE REACH ---
        self.elements.append(Paragraph("2. Operational Workspace Bounds", self.styles['SectionHeader']))
        explanation_2 = ("<b>Metric Definition:</b> A 3D spatial bounding map of reachable coordinates.<br/>"
                         "<b>Implication:</b> Defines the absolute maximum stretch of the arm, dictating workstation size requirements.")
        self.elements.append(Paragraph(explanation_2, self.styles['LaymanText']))

        if csv_file:
            x_vals, y_vals, z_vals = [], [],[]
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['x'] and row['x'] != 'None':
                        x_vals.append(float(row['x']))
                        y_vals.append(float(row['y']))
                        z_vals.append(float(row['z']))
            
            if x_vals:
                bbox_data =[
                    ["Axis", "Min Reach (m)", "Max Reach (m)", "Total Span (m)"],["X (Forward)", f"{min(x_vals):.4f}", f"{max(x_vals):.4f}", f"{(max(x_vals) - min(x_vals)):.4f}"],["Y (Lateral)", f"{min(y_vals):.4f}", f"{max(y_vals):.4f}", f"{(max(y_vals) - min(y_vals)):.4f}"],["Z (Vertical)", f"{min(z_vals):.4f}", f"{max(z_vals):.4f}", f"{(max(z_vals) - min(z_vals)):.4f}"]
                ]
                bbox_table = Table(bbox_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                bbox_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0"))
                ]))
                self.elements.append(bbox_table)
        else:
            self.elements.append(Paragraph("<i>⚠️ Workspace CSV dataset not found.</i>", self.styles['StandardText']))
        self.elements.append(Spacer(1, 0.2 * inch))

    def process_workspace_metrics(self):
        """Processes Module 3: WorkspaceAnalyzer (Volumes, Entropy, Anisotropy)"""
        metrics_file = self.get_latest_file(os.path.join(self.results_dir, "workspace_metrics", "workspace_metrics_summary.csv"))
        plot_file = self.get_latest_file(os.path.join(self.results_dir, "workspace_metrics", "plots", "workspace_3d_scatter.png"))

        self.elements.append(Paragraph("3. Advanced Volumetric & Spatial Analysis", self.styles['SectionHeader']))
        explanation_3 = ("<b>Metric Definition:</b> Mathematical modeling of the robot's physical operating volume, point distribution, and shape bias.<br/>"
                         "<b>Implication:</b> 'Volume' (m³) defines total usable space. 'Anisotropy' determines if the robot favors reaching in specific directions (shape bias). "
                         "'Entropy' measures reach uniformity—higher entropy means smoother coverage with fewer dead-zones.")
        self.elements.append(Paragraph(explanation_3, self.styles['LaymanText']))

        if metrics_file:
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                try:
                    data = next(reader)
                    
                    # Formatting the complex metrics into a clean table
                    metrics_data =[
                        ["Metric Category", "Calculated Value", "Unit"],["Max Reach Radius", f"{float(data.get('max_reach_radius', 0)):.4f}", "Meters (m)"],["Total Workspace Volume", f"{float(data.get('workspace_volume', 0)):.4f}", "Cubic Meters (m³)"],["Reachable Surface Area", f"{float(data.get('reachable_surface_area', 0)):.4f}", "Square Meters (m²)"],["Workspace Coverage Ratio", f"{float(data.get('workspace_coverage', 0))*100:.2f}", "Percentage (%)"],["Workspace Anisotropy", f"{float(data.get('workspace_anisotropy', 0)):.4f}", "Ratio (Bias)"],["Reachability Entropy", f"{float(data.get('reachability_entropy', 0)):.4f}", "Bits (Uniformity)"]
                    ]

                    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch, 2*inch])
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0"))
                    ]))
                    self.elements.append(Spacer(1, 0.1 * inch))
                    self.elements.append(metrics_table)

                except StopIteration:
                    self.elements.append(Paragraph("<i>⚠️ Metrics summary file is empty.</i>", self.styles['StandardText']))
        else:
            self.elements.append(Paragraph("<i>⚠️ Advanced Metrics CSV not found. Ensure Module 3 has executed.</i>", self.styles['StandardText']))
        
        self.elements.append(Spacer(1, 0.2 * inch))

        # --- EMBEDDING THE PLOT ---
        if plot_file and os.path.exists(plot_file):
            self.elements.append(Paragraph("<b>Visual Representation: 3D Workspace Scatter</b>", self.styles['StandardText']))
            # Insert the plot image generated by Module 3! 
            img = Image(plot_file, width=5*inch, height=3.5*inch)
            self.elements.append(img)
            self.elements.append(Spacer(1, 0.2 * inch))

    def generate(self, output_filename="Robot_Performance_Certificate.pdf"):
        """Compiles all sections and generates the PDF."""
        doc = SimpleDocTemplate(output_filename, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        self.add_title()
        self.process_kinematics_and_workspace()
        self.process_workspace_metrics()
        doc.build(self.elements)
        print(f"✅ Professional Certificate generated successfully: {output_filename}")

if __name__ == "__main__":
    generator = RobotCertificateGenerator(results_dir="5dof_results")
    generator.generate()
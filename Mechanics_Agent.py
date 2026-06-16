"""
Integrated Engineering Problem (Collaborative Robot Arm) 
Driven "Mechanics of Materials" Teaching (XDU)
Streamlit Web Version - Full Interactive Learning Mode

Complete features:
1. 6-axis robot 3D visualization
2. Model simplification (robot → cantilever beam)
3. Basic deformations: Axial, Torsion, Bending (each with interactive learning)
4. Combined deformation + stress contours
5. Principal stress + Mohr's Circle
6. Four strength theories comparison
7. Stiffness check + deflection curve
8. Buckling stability + mode shape
9. Interactive guided learning (think → answer → feedback → results)
10. Comprehensive knowledge summary
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle
import pandas as pd
import time

# Page config
st.set_page_config(
    page_title="Mechanics of Materials - Collaborative Robot Arm Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use English fonts only
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


# ==================== Calculation Functions ====================

def calculate_mechanics(L, d, F_x, T, F_z, F_y, sigma_y, n):
    """Material mechanics core calculation - full version"""
    
    # Circular section properties
    A = np.pi * (d/2)**2
    I = np.pi * d**4 / 64
    W = np.pi * d**3 / 32
    I_p = np.pi * d**4 / 32
    W_p = np.pi * d**3 / 16
    i = d / 4
    
    # Internal forces
    M = np.sqrt(F_z**2 + F_y**2) * L
    M_z = F_z * L
    M_y = F_y * L
    
    # Basic deformation stresses
    sigma_axial = F_x / A
    tau_torsion = T / W_p
    sigma_bending = M / W
    sigma_bending_z = M_z / W
    sigma_bending_y = M_y / W
    
    # Combined deformation
    sigma_x = sigma_axial + sigma_bending
    tau_xy = tau_torsion
    
    # Principal stresses
    sigma_1 = sigma_x/2 + np.sqrt((sigma_x/2)**2 + tau_xy**2)
    sigma_2 = 0
    sigma_3 = sigma_x/2 - np.sqrt((sigma_x/2)**2 + tau_xy**2)
    
    # Four strength theories
    mu = 0.3
    sigma_r1 = sigma_1
    sigma_r2 = sigma_1 - mu * (sigma_2 + sigma_3)
    sigma_r3 = sigma_1 - sigma_3
    sigma_r4 = np.sqrt(sigma_x**2 + 3*tau_xy**2)
    sigma_allow = sigma_y / n
    
    # Safety factors
    safety_factors = {
        "Maximum Normal Stress Theory": sigma_y / sigma_r1 if sigma_r1 > 0 else 999,
        "Maximum Normal Strain Theory": sigma_y / sigma_r2 if sigma_r2 > 0 else 999,
        "Maximum Shear Stress Theory": sigma_y / sigma_r3 if sigma_r3 > 0 else 999,
        "Distortion Energy Theory": sigma_y / sigma_r4 if sigma_r4 > 0 else 999,
    }
    
    # Stiffness check
    F = np.sqrt(F_z**2 + F_y**2)
    delta = F * L**3 / (3 * 71e9 * I) if I > 0 else 0
    delta_allow = L / 200
    stiffness_safe = delta <= delta_allow
    
    # Buckling stability
    lambda_val = 2 * L / i if i > 0 else 0
    lambda_p = np.sqrt(71e9 / sigma_y) * np.pi
    if lambda_val < lambda_p:
        stability_safe = True
        stability_msg = "Short/stocky column - no elastic buckling"
    else:
        stability_safe = False
        P_cr = np.pi**2 * 71e9 * I / (2 * L)**2
        stability_msg = f"Slender column - Euler critical load P_cr = {P_cr/1000:.2f} kN"
    
    strength_safe = sigma_r4 <= sigma_allow
    
    return {
        "geometry": {"A": A, "I": I, "W": W, "I_p": I_p, "W_p": W_p, "i": i},
        "stresses": {"axial": sigma_axial, "torsion": tau_torsion, "bending": sigma_bending, 
                     "bending_z": sigma_bending_z, "bending_y": sigma_bending_y,
                     "combined": sigma_x, "shear": tau_xy},
        "principal": {"sigma1": sigma_1, "sigma2": sigma_2, "sigma3": sigma_3},
        "strength_theories": {"r1": sigma_r1, "r2": sigma_r2, "r3": sigma_r3, 
                              "r4": sigma_r4, "allow": sigma_allow},
        "safety_factors": safety_factors,
        "stiffness": {"delta": delta, "delta_allow": delta_allow, "safe": stiffness_safe},
        "stability": {"lambda": lambda_val, "lambda_p": lambda_p, "safe": stability_safe, "msg": stability_msg},
        "strength_safe": strength_safe,
        "L": L, "d": d,
        "F_x": F_x, "T": T, "F_z": F_z, "F_y": F_y, "M": M,
        "sigma_y": sigma_y, "n": n,
    }


# ==================== Plotting Functions ====================

def draw_robot_3d():
    """Draw 3D robot schematic"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('6-Axis Collaborative Robot', fontsize=14)
    
    joints = [(0,0,0), (0,0,0.3), (0.4,0,0.6), (0.7,0,0.7), (0.85,0.15,0.7), (0.95,0.25,0.7)]
    for i in range(len(joints)-1):
        ax.plot([joints[i][0], joints[i+1][0]],
               [joints[i][1], joints[i+1][1]],
               [joints[i][2], joints[i+1][2]], 'b-', linewidth=3)
    for j in joints:
        ax.scatter(*j, color='red', s=50)
    
    base_x = [-0.2, 0.2, 0.2, -0.2, -0.2]
    base_y = [-0.2, -0.2, 0.2, 0.2, -0.2]
    base_z = [0, 0, 0, 0, 0]
    ax.plot(base_x, base_y, base_z, 'k-', linewidth=2)
    return fig


def draw_robot_animation_frames():
    """Generate robot animation frames (static frames for Streamlit)"""
    frames = []
    for angle in np.linspace(0, 2*np.pi, 8):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('6-Axis Robot Animation', fontsize=14)
        
        end_x = 0.95 + 0.1 * np.cos(angle)
        end_y = 0.25 + 0.1 * np.sin(angle)
        end_z = 0.7 + 0.05 * np.sin(angle*2)
        
        joints = [(0,0,0), (0,0,0.3), (0.4,0,0.6), (0.7,0,0.7), (0.85,0.15,0.7), (end_x, end_y, end_z)]
        for i in range(len(joints)-1):
            ax.plot([joints[i][0], joints[i+1][0]],
                   [joints[i][1], joints[i+1][1]],
                   [joints[i][2], joints[i+1][2]], 'b-', linewidth=3)
        for j in joints:
            ax.scatter(*j, color='red', s=50)
        
        base_x = [-0.2, 0.2, 0.2, -0.2, -0.2]
        base_y = [-0.2, -0.2, 0.2, 0.2, -0.2]
        base_z = [0, 0, 0, 0, 0]
        ax.plot(base_x, base_y, base_z, 'k-', linewidth=2)
        
        frames.append(fig)
    return frames


def draw_model_simplification(L, d):
    """Draw model simplification diagram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: Robot schematic
    joints_x = [0, 0, 0.4, 0.7, 0.85, 0.95]
    joints_y = [0, 0.3, 0.6, 0.7, 0.7, 0.7]
    for i in range(len(joints_x)-1):
        ax1.plot([joints_x[i], joints_x[i+1]], [joints_y[i], joints_y[i+1]], 'b-', linewidth=3)
    ax1.scatter(joints_x, joints_y, color='red', s=50)
    ax1.set_title('6-Axis Robot (Actual Structure)', fontsize=12)
    ax1.axis('off')
    
    # Right: Simplified model
    ax2.plot([0, L], [0, 0], 'b-', linewidth=4)
    ax2.fill_between([0, 0.02], [-d/2, -d/2], [d/2, d/2], color='gray', alpha=0.7)
    ax2.plot([L], [0], 'ro', markersize=8, label='Free End')
    ax2.annotate('Fixed End', xy=(0, 0), xytext=(-0.08, -0.06))
    ax2.annotate('Free End', xy=(L, 0), xytext=(L+0.05, 0.03))
    ax2.set_xlim(-0.15, L+0.15)
    ax2.set_ylim(-0.12, 0.12)
    ax2.set_xlabel('Length [m]')
    ax2.set_title('Simplified: Cantilever Beam (Circular Section)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def draw_axial_diagram(L, F_x, sigma_axial):
    """Draw axial tension/compression diagram"""
    fig, ax = plt.subplots(figsize=(6, 3))
    
    ax.plot([0, L], [0, 0], 'b-', linewidth=3)
    ax.fill_between([0, 0.02], [-0.03, -0.03], [0.03, 0.03], color='gray', alpha=0.7)
    
    if F_x > 0:
        ax.arrow(L, 0, 0.1, 0, head_width=0.02, head_length=0.02, color='red', linewidth=2)
        ax.text(L+0.12, 0, f'F_x = {F_x} N', fontsize=10)
        ax.set_title(f'Axial Tension (σ = {sigma_axial/1e6:.3f} MPa)', fontsize=12)
    else:
        ax.set_title(f'Axial Compression (σ = {sigma_axial/1e6:.3f} MPa)', fontsize=12)
    
    ax.set_xlim(-0.2, L+0.2)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('Length [m]')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    return fig


def draw_torsion_diagram(L, T, tau_torsion):
    """Draw torsion diagram"""
    fig, ax = plt.subplots(figsize=(6, 3))
    
    ax.plot([0, L], [0, 0], 'b-', linewidth=3)
    ax.fill_between([0, 0.02], [-0.03, -0.03], [0.03, 0.03], color='gray', alpha=0.7)
    
    ax.annotate('', xy=(L, 0.04), xytext=(L, 0.01), 
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=(L, -0.04), xytext=(L, -0.01), 
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(L+0.05, 0, f'T = {T} N·m', fontsize=10)
    
    ax.set_title(f'Torsion (τ_max = {tau_torsion/1e6:.3f} MPa)', fontsize=12)
    ax.set_xlim(-0.2, L+0.2)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('Length [m]')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    return fig


def draw_bending_diagram(L, M, sigma_bending, delta):
    """Draw bending diagram with deflection curve"""
    fig, ax = plt.subplots(figsize=(6, 3))
    
    ax.plot([0, L], [0, 0], 'b-', linewidth=3)
    ax.fill_between([0, 0.02], [-0.03, -0.03], [0.03, 0.03], color='gray', alpha=0.7)
    
    x_curve = np.linspace(0, L, 30)
    y_curve = -delta * (1 - np.cos(np.pi * x_curve / (2 * L))) * 0.5 / (delta + 1e-10) * 0.03
    ax.plot(x_curve, y_curve, 'r--', linewidth=1.5, alpha=0.7, label='Deflection Curve')
    
    ax.set_title(f'Bending (σ_max = {sigma_bending/1e6:.3f} MPa)', fontsize=12)
    ax.set_xlim(-0.2, L+0.2)
    ax.set_ylim(-0.1, 0.08)
    ax.set_xlabel('Length [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    return fig


def draw_axial_contour(d, sigma_axial):
    """Draw axial stress contour"""
    r = np.linspace(0, d/2, 30)
    theta = np.linspace(0, 2*np.pi, 60)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    sigma = np.ones_like(X) * sigma_axial
    
    fig, ax = plt.subplots(figsize=(5, 4))
    contour = ax.contourf(X*1000, Y*1000, sigma/1e6, levels=15, cmap='RdBu_r')
    ax.set_aspect('equal')
    ax.set_title(f'Axial Stress Contour\nσ = {sigma_axial/1e6:.3f} MPa', fontsize=11)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    plt.colorbar(contour, ax=ax, label='Stress [MPa]')
    return fig


def draw_torsion_contour(d, tau_torsion):
    """Draw torsion shear stress contour"""
    r = np.linspace(0, d/2, 30)
    theta = np.linspace(0, 2*np.pi, 60)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    tau = tau_torsion * R / (d/2)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    contour = ax.contourf(X*1000, Y*1000, tau/1e6, levels=15, cmap='hot')
    ax.set_aspect('equal')
    ax.set_title(f'Torsional Stress Contour\nτ_max = {tau_torsion/1e6:.3f} MPa', fontsize=11)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    plt.colorbar(contour, ax=ax, label='Shear Stress [MPa]')
    return fig


def draw_bending_contour(d, M, I):
    """Draw bending stress contour"""
    r = np.linspace(0, d/2, 30)
    theta = np.linspace(0, 2*np.pi, 60)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    sigma_bending = M * Y / I if I > 0 else 0
    
    fig, ax = plt.subplots(figsize=(5, 4))
    contour = ax.contourf(X*1000, Y*1000, sigma_bending/1e6, levels=15, cmap='RdBu_r')
    ax.set_aspect('equal')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Neutral Axis')
    ax.set_title(f'Bending Stress Contour\nσ_max = {np.max(np.abs(sigma_bending))/1e6:.3f} MPa', fontsize=11)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    plt.colorbar(contour, ax=ax, label='Stress [MPa]')
    ax.legend()
    return fig


def draw_combined_contour(d, sigma_axial, M, I):
    """Draw combined deformation stress contour"""
    r = np.linspace(0, d/2, 30)
    theta = np.linspace(0, 2*np.pi, 60)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    sigma_bending = M * Y / I if I > 0 else 0
    sigma_combined = sigma_axial + sigma_bending
    
    fig, ax = plt.subplots(figsize=(5, 4))
    contour = ax.contourf(X*1000, Y*1000, sigma_combined/1e6, levels=15, cmap='RdBu_r')
    ax.set_aspect('equal')
    y_neutral = -sigma_axial * I / M if M != 0 else 0
    ax.axhline(y=y_neutral*1000, color='black', linestyle='--', linewidth=1, label='Neutral Axis')
    ax.set_title(f'Combined Stress Contour\nσ_max = {np.max(np.abs(sigma_combined))/1e6:.3f} MPa', fontsize=11)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    plt.colorbar(contour, ax=ax, label='Stress [MPa]')
    ax.legend()
    return fig


def draw_stress_contour(d, sigma_x, tau_xy):
    """Draw combined stress contour (two plots)"""
    r = np.linspace(0, d/2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    sigma_dist = sigma_x * Y / (d/2)
    sigma_von_mises = np.sqrt(sigma_dist**2 + 3*tau_xy**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    contour1 = ax1.contourf(X*1000, Y*1000, sigma_dist/1e6, levels=20, cmap='RdBu_r')
    ax1.set_aspect('equal')
    ax1.set_title('Normal Stress σ_x (MPa)')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    plt.colorbar(contour1, ax=ax1)
    
    contour2 = ax2.contourf(X*1000, Y*1000, sigma_von_mises/1e6, levels=20, cmap='jet')
    ax2.set_aspect('equal')
    ax2.set_title('von Mises Stress σ_r4 (MPa)')
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    plt.colorbar(contour2, ax=ax2)
    
    plt.tight_layout()
    return fig


def draw_mohr_circle(sigma_x, tau_xy, sigma_allow):
    """Draw Mohr's circle"""
    center = sigma_x / 2
    radius = np.sqrt((sigma_x/2)**2 + tau_xy**2)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center + radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(circle_x/1e6, circle_y/1e6, 'b-', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.plot(sigma_x/1e6, tau_xy/1e6, 'ro', markersize=8, label='Stress Point')
    ax.plot(center/1e6, 0, 'go', markersize=8, label='Center')
    sigma_1 = center + radius
    sigma_3 = center - radius
    ax.plot(sigma_1/1e6, 0, 'r*', markersize=12, label=f'σ₁={sigma_1/1e6:.2f}MPa')
    ax.plot(sigma_3/1e6, 0, 'r*', markersize=12, label=f'σ₃={sigma_3/1e6:.2f}MPa')
    ax.axvline(x=sigma_allow/1e6, color='g', linestyle='--', linewidth=2, label=f'Allowable {sigma_allow/1e6:.0f}MPa')
    ax.set_xlabel('σ [MPa]')
    ax.set_ylabel('τ [MPa]')
    ax.set_title("Mohr's Circle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    return fig


def draw_deflection_curve(L, delta, delta_allow):
    """Draw deflection curve"""
    x = np.linspace(0, L, 50)
    y_deflect = delta * (3*(x/L)**2 - (x/L)**3) / 2
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x*1000, y_deflect*1000, 'b-', linewidth=2)
    ax.fill_between(x*1000, 0, y_deflect*1000, alpha=0.3, color='blue')
    ax.axhline(y=delta_allow*1000, color='r', linestyle='--', label=f'Allowable {delta_allow*1000:.2f}mm')
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Deflection δ [mm]')
    ax.set_title(f'Cantilever Beam Deflection (End: {delta*1000:.3f} mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def draw_buckling_mode(L):
    """Draw buckling mode"""
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(0, L, 100)
    mode_shape = 1 - np.cos(np.pi * x / (2*L))
    ax.plot(x*1000, mode_shape, 'b-', linewidth=2)
    ax.fill_between(x*1000, 0, mode_shape, alpha=0.3, color='blue')
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Lateral Displacement')
    ax.set_title('1st Buckling Mode (μ=2, Fixed-Free)')
    ax.grid(True, alpha=0.3)
    return fig


def draw_bending_stress_distribution(d, M, I):
    """Draw bending stress distribution across section"""
    y = np.linspace(-d/2, d/2, 50)
    sigma = M * y / I if I > 0 else 0
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sigma/1e6, y*1000, 'b-', linewidth=2)
    ax.fill_betweenx(y*1000, 0, sigma/1e6, alpha=0.3, color='blue')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Stress σ [MPa]')
    ax.set_ylabel('Height Y [mm]')
    ax.set_title('Bending Stress Linear Distribution', fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig


def draw_euler_curve(L, E, I, sigma_y, lambda_val, lambda_p):
    """Draw Euler critical stress curve"""
    fig, ax = plt.subplots(figsize=(7, 5))
    lambda_range = np.linspace(0, 200, 200)
    sigma_cr_range = np.pi**2 * E / lambda_range**2
    
    ax.plot(lambda_range, sigma_cr_range/1e6, 'b-', linewidth=2)
    ax.axvline(x=lambda_p, color='r', linestyle='--', label=f'λ_p = {lambda_p:.1f}')
    ax.axhline(y=sigma_y/1e6, color='g', linestyle='--', label=f'σ_y = {sigma_y/1e6:.0f}MPa')
    
    current_sigma = sigma_y/1e6 if lambda_val < lambda_p else np.pi**2 * E / lambda_val**2 / 1e6
    ax.scatter([lambda_val], [current_sigma], color='red', s=100, zorder=5, label='Working Point')
    
    ax.set_xlabel('Slenderness Ratio λ')
    ax.set_ylabel('Critical Stress σ_cr [MPa]')
    ax.set_title('Euler Critical Stress Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# ==================== Interactive Learning Functions ====================

def show_learning_step(step_key, title, question, hint, answer, correct_key):
    """Display interactive learning step"""
    
    # Initialize session state
    if f"{step_key}_answered" not in st.session_state:
        st.session_state[f"{step_key}_answered"] = False
        st.session_state[f"{step_key}_correct"] = False
        st.session_state[f"{step_key}_attempts"] = 0
    
    st.markdown(f"### 💡 {title}")
    st.markdown(f"**Question:** {question}")
    
    if hint:
        with st.expander("💭 Need a hint?"):
            st.markdown(hint)
    
    user_answer = st.text_input("Your answer:", key=f"{step_key}_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Answer", key=f"{step_key}_submit"):
            st.session_state[f"{step_key}_attempts"] += 1
            if user_answer.strip():
                if correct_key.lower() in user_answer.lower():
                    st.session_state[f"{step_key}_answered"] = True
                    st.session_state[f"{step_key}_correct"] = True
                else:
                    st.session_state[f"{step_key}_answered"] = True
                    st.session_state[f"{step_key}_correct"] = False
            else:
                st.warning("Please enter your answer first!")
                st.session_state[f"{step_key}_answered"] = False
    
    with col2:
        if st.button("Show Answer", key=f"{step_key}_show"):
            st.session_state[f"{step_key}_answered"] = True
            st.session_state[f"{step_key}_correct"] = False
    
    # Show feedback
    if st.session_state[f"{step_key}_answered"]:
        if st.session_state[f"{step_key}_correct"]:
            st.success("✅ Correct! Great thinking!")
            st.markdown(f"**Answer:** {answer}")
            return True
        else:
            attempts = st.session_state[f"{step_key}_attempts"]
            if attempts > 1:
                st.warning(f"💡 Hint: {hint}")
            st.markdown(f"**Correct Answer:** {answer}")
            return False
    
    return None


def show_learned_content(module_name, results):
    """Display the learned content after student has answered"""
    
    if module_name == "axial":
        st.subheader("📊 Axial Tension/Compression Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Axial Stress σ", f"{results['stresses']['axial']/1e6:.4f} MPa")
        with col2:
            st.metric("Deformation ΔL", f"{results['F_x'] * results['L'] / (71e9 * results['geometry']['A']) * 1000:.4f} mm")
        with col3:
            st.metric("Cross-section A", f"{results['geometry']['A']*1e6:.2f} mm²")
        st.caption("Formula: σ = N/A, ΔL = N·L/(E·A)")
        st.info("💡 Axial stress is uniformly distributed across the cross-section.")
        
    elif module_name == "torsion":
        st.subheader("📊 Torsion Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Shear Stress τ", f"{results['stresses']['torsion']/1e6:.4f} MPa")
        with col2:
            st.metric("Twist Angle θ", f"{results['T'] * results['L'] / (27e9 * results['geometry']['I_p']) * 180/np.pi:.2f}°")
        with col3:
            st.metric("Polar Moment I_p", f"{results['geometry']['I_p']*1e8:.2f} cm⁴")
        st.caption("Formula: τ = T·r/I_p, θ = T·L/(G·I_p)")
        st.info("💡 Shear stress varies linearly with radius, zero at center, maximum at surface.")
        
    elif module_name == "bending":
        st.subheader("📊 Bending Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Bending Stress σ", f"{results['stresses']['bending']/1e6:.4f} MPa")
        with col2:
            st.metric("End Deflection δ", f"{results['stiffness']['delta']*1000:.4f} mm")
        with col3:
            st.metric("Resultant Moment M", f"{results['M']:.2f} N·m")
        st.caption("Formula: σ = M·y/I, δ = F·L³/(3EI)")
        st.info("💡 Bending stress varies linearly across the section, zero at the neutral axis.")
        
    elif module_name == "combined":
        st.subheader("📊 Combined Deformation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Normal Stress σ_x", f"{results['stresses']['combined']/1e6:.4f} MPa")
        with col2:
            st.metric("Shear Stress τ_xy", f"{results['stresses']['shear']/1e6:.4f} MPa")
        st.caption("Formula: σ_x = σ_axial + σ_bending, τ_xy = τ_torsion")
        st.info("💡 Combined stress uses superposition principle (linear elastic, small deformation).")
        
    elif module_name == "principal":
        st.subheader("📊 Principal Stresses")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("σ₁", f"{results['principal']['sigma1']/1e6:.4f} MPa")
        with col2:
            st.metric("σ₂", f"{results['principal']['sigma2']/1e6:.4f} MPa")
        with col3:
            st.metric("σ₃", f"{results['principal']['sigma3']/1e6:.4f} MPa")
        st.caption("Formula: σ₁,₃ = (σ_x ± √(σ_x² + 4τ_xy²))/2")
        
    elif module_name == "strength":
        st.subheader("📊 Four Strength Theories Comparison")
        df = pd.DataFrame({
            "Theory": ["Max Normal Stress", "Max Normal Strain", "Max Shear Stress", "Distortion Energy"],
            "Equivalent Stress (MPa)": [
                f"{results['strength_theories']['r1']/1e6:.4f}",
                f"{results['strength_theories']['r2']/1e6:.4f}",
                f"{results['strength_theories']['r3']/1e6:.4f}",
                f"{results['strength_theories']['r4']/1e6:.4f}",
            ],
            "Safety Factor": [
                f"{results['safety_factors']['Maximum Normal Stress Theory']:.2f}",
                f"{results['safety_factors']['Maximum Normal Strain Theory']:.2f}",
                f"{results['safety_factors']['Maximum Shear Stress Theory']:.2f}",
                f"{results['safety_factors']['Distortion Energy Theory']:.2f}",
            ],
            "Status": [
                "✅ Safe" if results['safety_factors']['Maximum Normal Stress Theory'] >= results['n'] else "❌ Unsafe",
                "✅ Safe" if results['safety_factors']['Maximum Normal Strain Theory'] >= results['n'] else "❌ Unsafe",
                "✅ Safe" if results['safety_factors']['Maximum Shear Stress Theory'] >= results['n'] else "❌ Unsafe",
                "✅ Safe" if results['safety_factors']['Distortion Energy Theory'] >= results['n'] else "❌ Unsafe",
            ]
        })
        st.dataframe(df, use_container_width=True)
        st.info("💡 Distortion Energy Theory (von Mises) is most suitable for ductile materials like aluminum.")


# ==================== Main Program ====================

def main():
    st.title("🤖 Integrated Engineering Problem")
    st.markdown("### Collaborative Robot Arm - Mechanics of Materials Analysis")
    st.markdown("Interactive Guided Learning | Circular Section | 4 Strength Theories | Stiffness | Buckling")
    st.markdown("---")
    
    # ========== Sidebar: Parameters ==========
    with st.sidebar:
        st.header("📐 Engineering Parameters")
        
        st.subheader("Geometry")
        L = st.number_input("Arm Length L (m)", value=0.5, step=0.1, min_value=0.1, max_value=3.0)
        d = st.number_input("Diameter d (mm)", value=60, step=5, min_value=20, max_value=200) / 1000
        
        st.subheader("Material")
        sigma_y = st.number_input("Yield Strength σ_y (MPa)", value=505, step=10, min_value=200, max_value=1000)
        n = st.number_input("Safety Factor n", value=2.0, step=0.1, min_value=1.0, max_value=5.0)
        
        st.subheader("Loads")
        F_x = st.number_input("Axial Force F_x (N)", value=200, step=50, min_value=-1000, max_value=1000)
        T = st.number_input("Torque T (N·m)", value=50, step=10, min_value=0, max_value=500)
        F_z = st.number_input("Vertical Force F_z (N)", value=800, step=100, min_value=0, max_value=5000)
        F_y = st.number_input("Lateral Force F_y (N)", value=300, step=50, min_value=0, max_value=2000)
        
        st.subheader("Student")
        student_id = st.text_input("Student ID", value="20240001")
        
        st.markdown("---")
        calculate_btn = st.button("🚀 Start Interactive Learning", type="primary", use_container_width=True)
    
    # ========== Main Content ==========
    if calculate_btn:
        # Execute calculation
        results = calculate_mechanics(L, d, F_x, T, F_z, F_y, sigma_y, n)
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🤖 Robot Model", "📐 Guided Learning", "📊 Basic Deformations", 
            "🎨 Stress Contours", "💪 Strength Check", "📏 Stiffness", 
            "⚖️ Stability", "📚 Knowledge"
        ])
        
        # ===== Tab1: Robot Model =====
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("6-Axis Collaborative Robot")
                robot_fig = draw_robot_3d()
                st.pyplot(robot_fig)
                plt.close(robot_fig)
                st.info("6-Axis Collaborative Robot: 6 rotational joints, the arm is the key load-bearing component")
                
                st.subheader("Robot Joint Structure")
                st.markdown("""
                - **J1:** Base rotation (waist)
                - **J2:** Shoulder joint
                - **J3:** Elbow joint  
                - **J4:** Wrist 1 (roll)
                - **J5:** Wrist 2 (pitch)
                - **J6:** Wrist 3 (yaw)
                """)
            
            with col2:
                st.subheader("Model Simplification")
                simplify_fig = draw_model_simplification(L, d)
                st.pyplot(simplify_fig)
                plt.close(simplify_fig)
                st.success("""
                **Simplification Steps:**
                1. Robot Arm → Cantilever Beam
                2. Fixed End → Shoulder joint
                3. Free End → Elbow joint
                4. Loads → F_x, F_y, F_z, T
                5. Circular section → d = 60mm
                """)
        
        # ===== Tab2: Guided Learning =====
        with tab2:
            st.subheader("📐 Interactive Guided Learning")
            st.markdown("""
            **Instructions:** For each section, read the question, think about your answer, 
            then type it in and submit. You'll get immediate feedback!
            """)
            st.markdown("---")
            
            # Learning step 1: Model Simplification
            st.markdown("### 🔧 Step 1: Model Simplification")
            st.markdown("""
            **Background:** The robot arm is a complex structure with joints, motors, and sensors. 
            For mechanics analysis, we need to simplify it to a basic model.
            """)
            
            simplified = show_learning_step(
                "simplification",
                "Model Simplification Question",
                "What is the simplest mechanics model that represents the robot arm?",
                "Think about: the arm's shape (long and slender), boundary conditions (fixed at one end), loading",
                "A cantilever beam (fixed at one end, free at the other)",
                "cantilever"
            )
            
            if simplified is True:
                st.markdown("---")
                st.markdown("""
                **📚 Why Cantilever Beam?**
                - The arm is fixed at the shoulder joint → **Fixed End**
                - The arm is free at the elbow joint → **Free End**
                - Loads are applied at the free end → **Concentrated loads**
                - The arm is long and slender → **Beam theory applies**
                """)
            
            st.markdown("---")
            
            # Learning step 2: Axial
            st.markdown("### 📊 Step 2: Axial Tension/Compression")
            st.markdown("""
            **Concept:** When a force acts along the axis of the member, it creates axial stress.
            The stress distribution is uniform across the cross-section.
            """)
            
            axial_learned = show_learning_step(
                "axial",
                "Axial Stress Question",
                "What is the formula for axial stress? What is the stress distribution?",
                "Think about the definition of stress (force per unit area)",
                "σ = N/A, uniformly distributed",
                "uniform"
            )
            
            if axial_learned is True:
                show_learned_content("axial", results)
                st.subheader("📈 Axial Stress Distribution")
                axial_contour = draw_axial_contour(d, results['stresses']['axial'])
                st.pyplot(axial_contour)
                plt.close(axial_contour)
            
            st.markdown("---")
            
            # Learning step 3: Torsion
            st.markdown("### 🔄 Step 3: Torsion")
            st.markdown("""
            **Concept:** When a torque is applied to a circular shaft, it creates shear stress.
            The shear stress varies with radius.
            """)
            
            torsion_learned = show_learning_step(
                "torsion",
                "Torsion Question",
                "How does the torsional shear stress vary across the cross-section?",
                "Think about the relationship between radius and shear stress (τ ∝ r)",
                "τ = T·r/I_p, varies linearly with radius",
                "linear"
            )
            
            if torsion_learned is True:
                show_learned_content("torsion", results)
                st.subheader("📈 Torsional Stress Distribution")
                torsion_contour = draw_torsion_contour(d, results['stresses']['torsion'])
                st.pyplot(torsion_contour)
                plt.close(torsion_contour)
            
            st.markdown("---")
            
            # Learning step 4: Bending
            st.markdown("### 📈 Step 4: Bending")
            st.markdown("""
            **Concept:** When a beam bends, it creates bending stress that varies linearly across the section.
            The stress is zero at the neutral axis and maximum at the outer fibers.
            """)
            
            bending_learned = show_learning_step(
                "bending",
                "Bending Question",
                "Where is the bending stress zero in a beam cross-section?",
                "Think about the neutral axis concept",
                "At the neutral axis (center of the section)",
                "neutral"
            )
            
            if bending_learned is True:
                show_learned_content("bending", results)
                st.subheader("📈 Bending Stress Distribution")
                bending_contour = draw_bending_contour(d, results['M'], results['geometry']['I'])
                st.pyplot(bending_contour)
                plt.close(bending_contour)
            
            st.markdown("---")
            
            # Learning step 5: Combined Deformation
            st.markdown("### 🔗 Step 5: Combined Deformation")
            st.markdown("""
            **Concept:** In real engineering, multiple loads act simultaneously on a structure.
            The total stress is the superposition of individual effects.
            """)
            
            combined_learned = show_learning_step(
                "combined",
                "Combined Deformation Question",
                "How do we calculate the total stress when multiple loads act together?",
                "Think about superposition principle",
                "Superposition: σ_total = σ_axial + σ_bending, τ_total = τ_torsion",
                "superposition"
            )
            
            if combined_learned is True:
                show_learned_content("combined", results)
                st.subheader("📈 Combined Stress Contour")
                combined_contour = draw_combined_contour(d, results['stresses']['axial'], results['M'], results['geometry']['I'])
                st.pyplot(combined_contour)
                plt.close(combined_contour)
            
            st.markdown("---")
            
            # Learning step 6: Principal Stress
            st.markdown("### 🎯 Step 6: Principal Stress")
            st.markdown("""
            **Concept:** Principal stresses are the normal stresses acting on planes with zero shear stress.
            They represent the maximum and minimum normal stresses at a point.
            """)
            
            principal_learned = show_learning_step(
                "principal",
                "Principal Stress Question",
                "What do principal stresses represent?",
                "Think about the meaning of σ₁ and σ₃",
                "The maximum and minimum normal stresses at a point",
                "maximum"
            )
            
            if principal_learned is True:
                show_learned_content("principal", results)
                st.subheader("📈 Mohr's Circle")
                mohr_fig = draw_mohr_circle(
                    results['stresses']['combined'],
                    results['stresses']['shear'],
                    results['strength_theories']['allow']
                )
                st.pyplot(mohr_fig)
                plt.close(mohr_fig)
            
            st.markdown("---")
            
            # Learning step 7: Strength Theory
            st.markdown("### 💪 Step 7: Strength Theory")
            st.markdown("""
            **Concept:** Different materials fail in different ways. We need a theory to predict failure.
            For ductile materials like aluminum, we use the Distortion Energy Theory (von Mises).
            """)
            
            strength_learned = show_learning_step(
                "strength",
                "Strength Theory Question",
                "Which strength theory is most suitable for ductile materials like aluminum?",
                "Think about distortion energy",
                "Fourth strength theory (von Mises / Distortion Energy)",
                "fourth"
            )
            
            if strength_learned is True:
                show_learned_content("strength", results)
            
            st.markdown("---")
            
            # Learning step 8: Stiffness
            st.markdown("### 📏 Step 8: Stiffness Check")
            st.markdown("""
            **Concept:** Stiffness is the resistance to deformation. Excessive deflection can affect 
            the robot's positioning accuracy.
            """)
            
            stiffness_learned = show_learning_step(
                "stiffness",
                "Stiffness Question",
                "What is the deflection formula for a cantilever beam with end load?",
                "Think about beam deflection formula",
                "δ = F·L³/(3EI)",
                "L³"
            )
            
            if stiffness_learned is True:
                st.subheader("📏 Stiffness Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("End Deflection δ", f"{results['stiffness']['delta']*1000:.4f} mm")
                with col2:
                    st.metric("Allowable [δ]", f"{results['stiffness']['delta_allow']*1000:.3f} mm")
                if results['stiffness']['safe']:
                    st.success("✅ Stiffness condition satisfied!")
                else:
                    st.error("❌ Stiffness condition NOT satisfied!")
                st.subheader("Deflection Curve")
                deflect_fig = draw_deflection_curve(L, results['stiffness']['delta'], results['stiffness']['delta_allow'])
                st.pyplot(deflect_fig)
                plt.close(deflect_fig)
            
            st.markdown("---")
            
            # Learning step 9: Buckling
            st.markdown("### ⚖️ Step 9: Buckling Stability")
            st.markdown("""
            **Concept:** Slender columns can fail by buckling before reaching the yield strength.
            Euler's formula calculates the critical load.
            """)
            
            buckling_learned = show_learning_step(
                "buckling",
                "Buckling Question",
                "What is the Euler formula for critical buckling load?",
                "Think about the factors affecting buckling",
                "P_cr = π²EI/(μL)²",
                "π²"
            )
            
            if buckling_learned is True:
                st.subheader("⚖️ Buckling Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Slenderness Ratio λ", f"{results['stability']['lambda']:.2f}")
                with col2:
                    st.metric("Critical λ_p", f"{results['stability']['lambda_p']:.2f}")
                st.info(results['stability']['msg'])
                if results['stability']['safe']:
                    st.success("✅ No elastic buckling")
                else:
                    st.warning("⚠️ Slender column - check stability!")
                st.subheader("Buckling Mode")
                buckling_fig = draw_buckling_mode(L)
                st.pyplot(buckling_fig)
                plt.close(buckling_fig)
        
        # ===== Tab3: Basic Deformations =====
        with tab3:
            st.subheader("📊 Basic Deformations Analysis")
            
            st.markdown("### 1. Axial Tension/Compression")
            col1, col2 = st.columns(2)
            with col1:
                axial_diag = draw_axial_diagram(L, F_x, results['stresses']['axial'])
                st.pyplot(axial_diag)
                plt.close(axial_diag)
            with col2:
                st.metric("Axial Stress σ", f"{results['stresses']['axial']/1e6:.4f} MPa")
                st.metric("Deformation ΔL", f"{results['F_x'] * L / (71e9 * results['geometry']['A']) * 1000:.4f} mm")
                st.caption("σ = N/A, ΔL = N·L/(E·A)")
            
            st.divider()
            
            st.markdown("### 2. Torsion")
            col1, col2 = st.columns(2)
            with col1:
                torsion_diag = draw_torsion_diagram(L, T, results['stresses']['torsion'])
                st.pyplot(torsion_diag)
                plt.close(torsion_diag)
            with col2:
                st.metric("Max Shear Stress τ", f"{results['stresses']['torsion']/1e6:.4f} MPa")
                st.metric("Twist Angle θ", f"{T * L / (27e9 * results['geometry']['I_p']) * 180/np.pi:.2f}°")
                st.caption("τ = T·r/I_p, θ = T·L/(G·I_p)")
            
            st.divider()
            
            st.markdown("### 3. Bending")
            col1, col2 = st.columns(2)
            with col1:
                bending_diag = draw_bending_diagram(L, results['M'], results['stresses']['bending'], results['stiffness']['delta'])
                st.pyplot(bending_diag)
                plt.close(bending_diag)
            with col2:
                st.metric("Max Bending Stress σ", f"{results['stresses']['bending']/1e6:.4f} MPa")
                st.metric("End Deflection δ", f"{results['stiffness']['delta']*1000:.4f} mm")
                st.caption("σ = M·y/I, δ = F·L³/(3EI)")
        
        # ===== Tab4: Stress Contours =====
        with tab4:
            st.subheader("🎨 Stress Distribution Contours")
            st.markdown("### Combined Stress Contours")
            
            contour_fig = draw_stress_contour(
                results['d'], 
                results['stresses']['combined'], 
                results['stresses']['shear']
            )
            st.pyplot(contour_fig)
            plt.close(contour_fig)
            st.caption("Left: Normal Stress σ_x | Right: von Mises Stress σ_r4")
            
            st.divider()
            
            st.markdown("### Individual Stress Contours")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Axial Stress**")
                axial_contour = draw_axial_contour(d, results['stresses']['axial'])
                st.pyplot(axial_contour)
                plt.close(axial_contour)
                
                st.markdown("**Bending Stress**")
                bending_contour = draw_bending_contour(d, results['M'], results['geometry']['I'])
                st.pyplot(bending_contour)
                plt.close(bending_contour)
            
            with col2:
                st.markdown("**Torsional Shear Stress**")
                torsion_contour = draw_torsion_contour(d, results['stresses']['torsion'])
                st.pyplot(torsion_contour)
                plt.close(torsion_contour)
                
                st.markdown("**Combined Stress**")
                combined_contour = draw_combined_contour(d, results['stresses']['axial'], results['M'], results['geometry']['I'])
                st.pyplot(combined_contour)
                plt.close(combined_contour)
        
        # ===== Tab5: Strength Check =====
        with tab5:
            st.subheader("💪 Four Strength Theories Comparison")
            
            theories = ["Theory 1", "Theory 2", "Theory 3", "Theory 4"]
            sigma_r_values = [
                results['strength_theories']['r1']/1e6,
                results['strength_theories']['r2']/1e6,
                results['strength_theories']['r3']/1e6,
                results['strength_theories']['r4']/1e6,
            ]
            safety_vals = [
                results['safety_factors']['Maximum Normal Stress Theory'],
                results['safety_factors']['Maximum Normal Strain Theory'],
                results['safety_factors']['Maximum Shear Stress Theory'],
                results['safety_factors']['Distortion Energy Theory'],
            ]
            
            df = pd.DataFrame({
                "Theory": ["Max Normal Stress", "Max Normal Strain", "Max Shear Stress", "Distortion Energy"],
                "Equivalent Stress (MPa)": [f"{v:.4f}" for v in sigma_r_values],
                "Safety Factor": [f"{v:.2f}" for v in safety_vals],
                "Status": ["✅ Safe" if v >= n else "❌ Unsafe" for v in safety_vals]
            })
            st.dataframe(df, use_container_width=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Yield Strength σ_y", f"{sigma_y:.0f} MPa")
                st.metric("Allowable [σ]", f"{results['strength_theories']['allow']/1e6:.1f} MPa")
                st.metric("Required Safety Factor n", f"{n}")
            with col2:
                st.metric("von Mises Stress", f"{results['strength_theories']['r4']/1e6:.4f} MPa")
                if results['strength_safe']:
                    st.success("✅ Strength: PASS")
                    st.metric("Actual Safety Factor", f"{results['safety_factors']['Distortion Energy Theory']:.2f}")
                else:
                    st.error("❌ Strength: FAIL")
                    st.metric("Actual Safety Factor", f"{results['safety_factors']['Distortion Energy Theory']:.2f}")
            
            st.divider()
            
            st.subheader("Mohr's Circle")
            col1, col2 = st.columns(2)
            with col1:
                mohr_fig = draw_mohr_circle(
                    results['stresses']['combined'],
                    results['stresses']['shear'],
                    results['strength_theories']['allow']
                )
                st.pyplot(mohr_fig)
                plt.close(mohr_fig)
            with col2:
                st.markdown("**Mohr's Circle Interpretation:**")
                st.markdown(f"""
                - **Center:** σ_avg = {results['stresses']['combined']/1e6:.2f} MPa
                - **Radius:** R = {np.sqrt((results['stresses']['combined']/2)**2 + results['stresses']['shear']**2)/1e6:.2f} MPa
                - **Principal Stress σ₁:** {results['principal']['sigma1']/1e6:.2f} MPa
                - **Principal Stress σ₃:** {results['principal']['sigma3']/1e6:.2f} MPa
                - **Max Shear Stress:** {np.sqrt((results['stresses']['combined']/2)**2 + results['stresses']['shear']**2)/1e6:.2f} MPa
                """)
            
            st.divider()
            
            st.subheader("Stress-Strain Behavior")
            fig, ax = plt.subplots(figsize=(8, 4))
            strain = np.linspace(0, 0.01, 100)
            stress = 71e9 * strain
            ax.plot(strain*100, stress/1e6, 'b-', linewidth=2)
            ax.axhline(y=sigma_y/1e6, color='r', linestyle='--', label=f'Yield Strength {sigma_y}MPa')
            ax.axhline(y=results['strength_theories']['r4']/1e6, color='g', linestyle='--', label=f'von Mises {results["strength_theories"]["r4"]/1e6:.2f}MPa')
            ax.set_xlabel('Strain ε [%]')
            ax.set_ylabel('Stress σ [MPa]')
            ax.set_title('Stress-Strain Diagram (Hooke\'s Law)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        # ===== Tab6: Stiffness =====
        with tab6:
            st.subheader("📏 Stiffness Check")
            
            st.markdown("### Deflection Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("End Deflection δ", f"{results['stiffness']['delta']*1000:.4f} mm")
            with col2:
                st.metric("Allowable [δ]", f"{results['stiffness']['delta_allow']*1000:.3f} mm")
            with col3:
                if results['stiffness']['safe']:
                    st.metric("Status", "✅ PASS")
                else:
                    st.metric("Status", "❌ FAIL")
            
            st.divider()
            
            st.subheader("Deflection Curve")
            deflect_fig = draw_deflection_curve(
                results['L'],
                results['stiffness']['delta'],
                results['stiffness']['delta_allow']
            )
            st.pyplot(deflect_fig)
            plt.close(deflect_fig)
            
            st.divider()
            
            st.markdown("### Parameters Affecting Stiffness")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Sensitivity Analysis:**
                - δ ∝ L³ (length is most influential!)
                - δ ∝ 1/I (inertia)
                - δ ∝ 1/E (material)
                """)
            with col2:
                st.markdown("""
                **Ways to Increase Stiffness:**
                1. ✅ Decrease length (most effective)
                2. ✅ Increase cross-section height (I ∝ h³)
                3. ✅ Use higher modulus material
                4. ✅ Change boundary conditions
                """)
            
            st.divider()
            
            st.markdown("### Comparison with Allowable Values")
            fig, ax = plt.subplots(figsize=(8, 4))
            x_labels = ['End Deflection', 'Allowable']
            values = [results['stiffness']['delta']*1000, results['stiffness']['delta_allow']*1000]
            colors = ['blue' if results['stiffness']['safe'] else 'red', 'green']
            ax.bar(x_labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Deflection [mm]')
            ax.set_title('Deflection Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.3f}mm', ha='center')
            st.pyplot(fig)
            plt.close(fig)
        
        # ===== Tab7: Stability =====
        with tab7:
            st.subheader("⚖️ Buckling Stability")
            
            st.markdown("### Buckling Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Slenderness Ratio λ", f"{results['stability']['lambda']:.2f}")
            with col2:
                st.metric("Critical λ_p", f"{results['stability']['lambda_p']:.2f}")
            with col3:
                if results['stability']['safe']:
                    st.metric("Type", "Short Column ✅")
                else:
                    st.metric("Type", "Slender Column ⚠️")
            
            st.info(results['stability']['msg'])
            
            if not results['stability']['safe']:
                P_cr = np.pi**2 * 71e9 * results['geometry']['I'] / (2 * L)**2
                st.warning(f"⚠️ Critical load P_cr = {P_cr/1000:.2f} kN (compared to actual load)")
            
            st.divider()
            
            st.subheader("Buckling Mode Shape")
            buckling_fig = draw_buckling_mode(results['L'])
            st.pyplot(buckling_fig)
            plt.close(buckling_fig)
            
            st.divider()
            
            st.markdown("### Euler Critical Stress Curve")
            euler_fig = draw_euler_curve(
                results['L'],
                71e9,
                results['geometry']['I'],
                results['sigma_y'],
                results['stability']['lambda'],
                results['stability']['lambda_p']
            )
            st.pyplot(euler_fig)
            plt.close(euler_fig)
            
            st.divider()
            
            st.markdown("### Buckling Classification")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Classification by Slenderness Ratio:**
                - λ < 40: Short column (strength failure)
                - 40 < λ < 100: Intermediate column (inelastic buckling)
                - λ > 100: Slender column (elastic buckling, Euler formula)
                """)
            with col2:
                st.markdown("""
                **Current Status:**
                - λ = {:.1f}
                - λ_p = {:.1f}
                - **Type:** {}
                """.format(
                    results['stability']['lambda'],
                    results['stability']['lambda_p'],
                    "Short column" if results['stability']['safe'] else "Slender column"
                ))
        
        # ===== Tab8: Knowledge =====
        with tab8:
            st.markdown(f"""
            ## 📚 Mechanics of Materials Knowledge Summary
            
            ### 1. Basic Deformation Formulas
            
            | Deformation Type | Stress Formula | Deformation Formula |
            |------------------|----------------|---------------------|
            | Axial Tension/Compression | σ = N/A | ΔL = N·L/(E·A) |
            | Torsion (Circular) | τ = T·r/I_p | θ = T·L/(G·I_p) |
            | Bending | σ = M·y/I | δ = F·L³/(3EI) |
            
            ### 2. Combined Deformation
            
            | Quantity | Formula |
            |----------|---------|
            | Total Normal Stress | σ_x = σ_axial + σ_bending |
            | Shear Stress | τ_xy = τ_torsion |
            | Principal Stresses | σ₁,₃ = (σ_x ± √(σ_x² + 4τ_xy²))/2 |
            
            ### 3. Four Strength Theories
            
            | Theory | Name | Equivalent Stress Formula | Application |
            |--------|------|---------------------------|-------------|
            | Theory 1 | Maximum Normal Stress | σ_r1 = σ₁ | Brittle materials |
            | Theory 2 | Maximum Normal Strain | σ_r2 = σ₁ - μ(σ₂+σ₃) | Some brittle materials |
            | Theory 3 | Maximum Shear Stress | σ_r3 = σ₁ - σ₃ | Ductile materials |
            | Theory 4 | Distortion Energy | σ_r4 = √(σ_x² + 3τ_xy²) | Ductile materials (best) |
            
            ### 4. Stiffness
            
            | Parameter | Formula |
            |-----------|---------|
            | End Deflection | δ = F·L³/(3EI) |
            | Allowable Deflection | [δ] = L/200 |
            | Stiffness Condition | δ ≤ [δ] |
            
            ### 5. Buckling Stability
            
            | Parameter | Formula |
            |-----------|---------|
            | Euler Critical Load | P_cr = π²EI/(μL)² |
            | Slenderness Ratio | λ = μL/i |
            | Critical λ | λ_p = π√(E/σ_y) |
            | Stability Condition | λ < λ_p (short column) |
            
            ### 6. This Analysis Results
            
            | Item | Value | Status |
            |------|-------|--------|
            | von Mises Stress | {results['strength_theories']['r4']/1e6:.4f} MPa | {'✅ Safe' if results['strength_safe'] else '❌ Unsafe'} |
            | Allowable Stress | {results['strength_theories']['allow']/1e6:.1f} MPa | |
            | Safety Factor | {results['safety_factors']['Distortion Energy Theory']:.2f} | {'✅' if results['safety_factors']['Distortion Energy Theory'] >= n else '❌'} |
            | End Deflection | {results['stiffness']['delta']*1000:.4f} mm | {'✅' if results['stiffness']['safe'] else '❌'} |
            | Slenderness Ratio | {results['stability']['lambda']:.2f} | {'✅ Short' if results['stability']['safe'] else '⚠️ Slender'} |
            """)
        
        st.sidebar.success(f"👤 Student: {student_id} | Analysis Complete")
    
    else:
        # Welcome screen
        st.info("👈 Set parameters in the sidebar, then click **Start Interactive Learning**")
        
        col1, col2 = st.columns(2)
        with col1:
            robot_fig = draw_robot_3d()
            st.pyplot(robot_fig)
            plt.close(robot_fig)
        with col2:
            simplify_fig = draw_model_simplification(0.5, 0.06)
            st.pyplot(simplify_fig)
            plt.close(simplify_fig)
        
        st.markdown("""
        ---
        ### 🎯 Interactive Learning Features
        
        | Feature | Description |
        |---------|-------------|
        | 🤖 Robot Model | 6-axis collaborative robot visualization |
        | 📐 Guided Learning | 9 learning steps: think → answer → feedback → results |
        | 📊 Basic Deformations | Axial, Torsion, Bending with diagrams |
        | 🎨 Stress Contours | 4 types of stress distribution visualization |
        | 💪 Strength Check | 4 strength theories + Mohr's Circle |
        | 📏 Stiffness Check | Deflection analysis + sensitivity |
        | ⚖️ Stability Check | Buckling + Euler curve |
        | 📚 Knowledge | Comprehensive formula summary |
        
        **How it works:**
        1. Set parameters in the sidebar
        2. Click **Start Interactive Learning**
        3. Go to the **Guided Learning** tab
        4. Answer 9 questions step by step
        5. Get immediate feedback and see results!
        """)
    
    st.markdown("---")
    st.caption("Integrated Engineering Problem (Collaborative Robot Arm) | Mechanics of Materials Teaching (XDU)")


if __name__ == "__main__":
    main()
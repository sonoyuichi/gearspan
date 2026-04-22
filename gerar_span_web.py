"""
GearMesh Viewer - 平歯車バックラッシ計算＋かみ合い描画 (Streamlit版)

起動: streamlit run gerar_span_web.py
"""
import math
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

# =========================
# JIS寄せ inv
# =========================
def inv(a_rad: float) -> float:
    return math.tan(a_rad) - a_rad


# =========================
# 基本幾何（中心距離）
# =========================
def center_distance_ref(m, z1, z2):
    return m * (z1 + z2) / 2.0


def center_distance_design(m, z1, z2, x1, x2):
    return center_distance_ref(m, z1, z2) + m * (x1 + x2)


def operating_pressure_angle(alpha_deg, a_ref, a):
    if a <= 0:
        raise ValueError("軸間距離 a は正の値が必要です")
    alpha = math.radians(alpha_deg)
    cos_aw = (a_ref * math.cos(alpha)) / a
    cos_aw = max(min(cos_aw, 1.0), -1.0)
    return math.acos(cos_aw)


# =========================
# 跨ぎ歯厚（平歯式）
# =========================
def span_from_xeff(x_eff, k, m, z, alpha_deg):
    a = math.radians(alpha_deg)
    return m * math.cos(a) * ((k - 0.5) * math.pi + z * inv(a) + 2.0 * x_eff * math.tan(a))


def x_eff_from_span(W_mm, k, m, z, alpha_deg):
    a = math.radians(alpha_deg)
    return ((W_mm / (m * math.cos(a))) - (k - 0.5) * math.pi - z * inv(a)) / (2 * math.tan(a))


# =========================
# バックラッシ計算（a入力）
# =========================
def calc_backlash_from_W_and_a(m, alpha_deg, z1, z2, x1_design, x2_design,
                                a, W1, k1, W2, k2):
    a_ref = center_distance_ref(m, z1, z2)
    a_des = center_distance_design(m, z1, z2, x1_design, x2_design)
    aw = operating_pressure_angle(alpha_deg, a_ref, a)

    rw1 = a * z1 / (z1 + z2)
    rw2 = a * z2 / (z1 + z2)

    x1_eff = x_eff_from_span(W1, k1, m, z1, alpha_deg)
    x2_eff = x_eff_from_span(W2, k2, m, z2, alpha_deg)

    alpha = math.radians(alpha_deg)

    tau1_ref = math.pi / z1 + (4.0 * x1_eff * math.tan(alpha)) / z1
    tau2_ref = math.pi / z2 + (4.0 * x2_eff * math.tan(alpha)) / z2

    corr = 2.0 * (inv(alpha) - inv(aw))
    tau1_w = tau1_ref + corr
    tau2_w = tau2_ref + corr

    pw = 2.0 * math.pi * rw1 / z1

    sw1 = rw1 * tau1_w
    sw2 = rw2 * tau2_w

    jt = pw - (sw1 + sw2)
    bn = jt / max(math.cos(aw), 1e-12)

    theta1_rad = jt / rw1
    theta2_rad = jt / rw2

    # かみ合い率
    r1 = m * z1 / 2.0
    r2 = m * z2 / 2.0
    rb1 = r1 * math.cos(alpha)
    rb2 = r2 * math.cos(alpha)
    ra1 = r1 + m * (1.0 + x1_design)
    ra2 = r2 + m * (1.0 + x2_design)
    pb = math.pi * m * math.cos(alpha)
    ga = (math.sqrt(max(ra1**2 - rb1**2, 0.0))
          + math.sqrt(max(ra2**2 - rb2**2, 0.0))
          - a * math.sin(aw))
    epsilon_alpha = ga / pb

    return {
        "a_ref_mm": a_ref,
        "a_des_mm": a_des,
        "a_mm": a,
        "da_from_des_mm": a - a_des,
        "alpha_w_deg": math.degrees(aw),
        "rw1_mm": rw1,
        "rw2_mm": rw2,
        "x1_eff": x1_eff,
        "x2_eff": x2_eff,
        "jt_um": jt * 1000.0,
        "bn_um": bn * 1000.0,
        "theta1_deg": math.degrees(theta1_rad),
        "theta1_arcmin": math.degrees(theta1_rad) * 60.0,
        "theta2_deg": math.degrees(theta2_rad),
        "theta2_arcmin": math.degrees(theta2_rad) * 60.0,
        "epsilon_alpha": epsilon_alpha,
    }


# =========================
# ギアかみ合い描画 (matplotlib版)
# =========================
def involute_point(rb, theta):
    x = rb * (math.cos(theta) + theta * math.sin(theta))
    y = rb * (math.sin(theta) - theta * math.cos(theta))
    return x, y


def draw_gear_mesh_mpl(m, z1, z2, x1, x2, alpha_deg, jt_um,
                       a_actual=None, g2_extra_rot_deg=0.0,
                       x1_design=None, x2_design=None):
    """
    matplotlib版ギアかみ合い描画。
    Returns: (fig, has_interference)
    """
    alpha = math.radians(alpha_deg)

    xd1 = x1_design if x1_design is not None else x1
    xd2 = x2_design if x2_design is not None else x2

    r1  = m * z1 / 2.0
    r2  = m * z2 / 2.0
    rb1 = r1 * math.cos(alpha)
    rb2 = r2 * math.cos(alpha)
    ra1 = r1 + m * (1.0 + xd1)
    ra2 = r2 + m * (1.0 + xd2)
    rf1 = r1 - m * (1.25 - xd1)
    rf2 = r2 - m * (1.25 - xd2)

    a_des = r1 + r2 + m * (xd1 + xd2)
    a_center = a_actual if a_actual is not None else a_des

    rw1 = a_center * z1 / (z1 + z2)
    rw2 = a_center * z2 / (z1 + z2)

    max_tooth = max(ra1 - rf1, ra2 - rf2)
    view_w = m * math.pi * 3.5
    view_h = max(m * 5.0, max_tooth * 2.8)

    g1x, g1y = -rw1, 0.0
    g2x, g2y =  rw2, 0.0

    jt_mm = jt_um / 1000.0
    bl_angle = jt_mm / rw2 if rw2 > 0 else 0

    pa1 = 2.0 * math.pi / z1
    pa2 = 2.0 * math.pi / z2

    a_ref = m * (z1 + z2) / 2.0
    cos_aw = a_ref * math.cos(alpha) / a_center
    cos_aw = max(min(cos_aw, 1.0), -1.0)
    aw = math.acos(cos_aw)

    g2_extra_rot = math.radians(g2_extra_rot_deg)
    base_rot1 = 0.0
    base_rot2 = math.pi - pa2 / 2.0 + bl_angle + g2_extra_rot

    N_INV  = 40
    N_DRAW = 3

    def build_outline_and_teeth(cx, cy, rb, ra, rf, z, xc, rot0, nd):
        pa   = 2.0 * math.pi / z
        half = math.pi / (2.0 * z) + 2.0 * xc * math.tan(alpha) / z + inv(alpha)
        thm  = math.sqrt(max((ra / rb)**2 - 1.0, 0.01))

        if rf >= rb:
            th0 = math.sqrt(max((rf / rb)**2 - 1.0, 0.0))
        else:
            th0 = 0.0
        ia0 = th0 - math.atan(th0) if th0 > 1e-12 else 0.0

        pts = []
        teeth = []

        for ti in range(-nd, nd + 1):
            rot = rot0 + ti * pa
            tooth = []

            if ti > -nd:
                a_s = rot - pa + half - ia0
                a_e = rot - half + ia0
                for i in range(11):
                    a_ = a_s + (a_e - a_s) * i / 10
                    pts.append((cx + rf * math.cos(a_),
                                cy + rf * math.sin(a_)))

            if rb > rf:
                ang_lf = rot - half
                p1 = (cx + rf * math.cos(ang_lf), cy + rf * math.sin(ang_lf))
                p2 = (cx + rb * math.cos(ang_lf), cy + rb * math.sin(ang_lf))
                pts.extend([p1, p2])
                tooth.extend([p1, p2])

            for i in range(N_INV):
                th = th0 + (thm - th0) * i / (N_INV - 1)
                ix = rb * (math.cos(th) + th * math.sin(th))
                iy = rb * (math.sin(th) - th * math.cos(th))
                r_  = math.sqrt(ix*ix + iy*iy)
                ia  = math.atan2(iy, ix)
                if r_ > ra:
                    a_new = rot - half + ia
                    p = (cx + ra * math.cos(a_new), cy + ra * math.sin(a_new))
                    pts.append(p)
                    tooth.append(p)
                    break
                a_new = rot - half + ia
                p = (cx + r_ * math.cos(a_new), cy + r_ * math.sin(a_new))
                pts.append(p)
                tooth.append(p)

            if len(pts) >= 1:
                a_lt = math.atan2(pts[-1][1] - cy, pts[-1][0] - cx)
                a_rt = None
                for i in range(N_INV - 1, -1, -1):
                    th = th0 + (thm - th0) * i / (N_INV - 1)
                    ix = rb * (math.cos(th) + th * math.sin(th))
                    iy = rb * (math.sin(th) - th * math.cos(th))
                    r_ = math.sqrt(ix*ix + iy*iy)
                    ia = math.atan2(iy, ix)
                    if r_ <= ra:
                        a_new = rot + half - ia
                        a_rt = math.atan2(
                            min(r_, ra) * math.sin(a_new),
                            min(r_, ra) * math.cos(a_new))
                        break
                if a_rt is None:
                    a_rt = a_lt
                for i in range(1, 8):
                    a_ = a_lt + (a_rt - a_lt) * i / 8
                    p = (cx + ra * math.cos(a_), cy + ra * math.sin(a_))
                    pts.append(p)
                    tooth.append(p)

            right = []
            for i in range(N_INV):
                th = th0 + (thm - th0) * i / (N_INV - 1)
                ix = rb * (math.cos(th) + th * math.sin(th))
                iy = rb * (math.sin(th) - th * math.cos(th))
                r_ = math.sqrt(ix*ix + iy*iy)
                ia = math.atan2(iy, ix)
                a_new = rot + half - ia
                if r_ > ra:
                    right.append((cx + ra * math.cos(a_new), cy + ra * math.sin(a_new)))
                else:
                    right.append((cx + r_ * math.cos(a_new), cy + r_ * math.sin(a_new)))
            for p in reversed(right):
                pts.append(p)
                tooth.append(p)

            if rb > rf:
                ang_rf = rot + half
                p1 = (cx + rb * math.cos(ang_rf), cy + rb * math.sin(ang_rf))
                p2 = (cx + rf * math.cos(ang_rf), cy + rf * math.sin(ang_rf))
                pts.extend([p1, p2])
                tooth.extend([p1, p2])

            if len(tooth) >= 3:
                a_end = math.atan2(tooth[-1][1] - cy, tooth[-1][0] - cx)
                a_beg = math.atan2(tooth[0][1] - cy, tooth[0][0] - cx)
                da_t = a_beg - a_end
                while da_t >  math.pi: da_t -= 2 * math.pi
                while da_t < -math.pi: da_t += 2 * math.pi
                for i in range(1, 6):
                    a_ = a_end + da_t * i / 6
                    tooth.append((cx + rf * math.cos(a_), cy + rf * math.sin(a_)))
                teeth.append(tooth)

        last_rot = rot0 + nd * pa
        a_s = last_rot + half - ia0
        a_e = a_s + pa * 0.4
        for i in range(6):
            a_ = a_s + (a_e - a_s) * i / 5
            pts.append((cx + rf * math.cos(a_), cy + rf * math.sin(a_)))

        return pts, teeth

    def build_fill_shape(cx, cy, rf, outline, facing_angle):
        if len(outline) < 3:
            return []
        shape = []
        a_first = math.atan2(outline[0][1] - cy, outline[0][0] - cx)
        a_last  = math.atan2(outline[-1][1] - cy, outline[-1][0] - cx)
        da = a_first - a_last
        while da < 0:
            da += 2 * math.pi
        while da >= 2 * math.pi:
            da -= 2 * math.pi
        n_back = 20
        back_arc = []
        for i in range(n_back + 1):
            a_ = a_last + da * i / n_back
            back_arc.append((cx + rf * math.cos(a_), cy + rf * math.sin(a_)))
        shape.extend(outline)
        shape.extend(back_arc)
        return shape

    def point_in_polygon(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    def polygons_overlap_region(poly_a, poly_b):
        pts = []
        for px, py in poly_a:
            if point_in_polygon(px, py, poly_b):
                pts.append((px, py))
        for px, py in poly_b:
            if point_in_polygon(px, py, poly_a):
                pts.append((px, py))
        if len(pts) < 3:
            return None
        cx_m = sum(p[0] for p in pts) / len(pts)
        cy_m = sum(p[1] for p in pts) / len(pts)
        pts.sort(key=lambda p: math.atan2(p[1] - cy_m, p[0] - cx_m))
        return pts

    def build_overlap_polygons(teeth_a, teeth_b):
        overlaps = []
        for ta in teeth_a:
            if len(ta) < 3:
                continue
            ax_min = min(p[0] for p in ta)
            ax_max = max(p[0] for p in ta)
            ay_min = min(p[1] for p in ta)
            ay_max = max(p[1] for p in ta)
            for tb in teeth_b:
                if len(tb) < 3:
                    continue
                bx_min = min(p[0] for p in tb)
                bx_max = max(p[0] for p in tb)
                by_min = min(p[1] for p in tb)
                by_max = max(p[1] for p in tb)
                if ax_max < bx_min or bx_max < ax_min:
                    continue
                if ay_max < by_min or by_max < ay_min:
                    continue
                region = polygons_overlap_region(ta, tb)
                if region and len(region) >= 3:
                    overlaps.append(region)
        return overlaps

    # ---- Build outlines ----
    outline1, teeth1 = build_outline_and_teeth(
        g1x, g1y, rb1, ra1, rf1, z1, x1, base_rot1, N_DRAW)
    outline2, teeth2 = build_outline_and_teeth(
        g2x, g2y, rb2, ra2, rf2, z2, x2, base_rot2, N_DRAW)

    shape2 = build_fill_shape(g2x, g2y, rf2, outline2, math.pi)
    shape1 = build_fill_shape(g1x, g1y, rf1, outline1, 0.0)

    overlaps = build_overlap_polygons(teeth1, teeth2)
    has_interference = len(overlaps) > 0

    # ---- matplotlib描画 ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    # G2 塗りつぶし（背面）
    if len(shape2) >= 3:
        poly2 = MplPolygon(shape2, closed=True, facecolor='#FFDDDD',
                           edgecolor='none', zorder=1)
        ax.add_patch(poly2)

    # G2 輪郭線
    if len(outline2) >= 2:
        ox2, oy2 = zip(*outline2)
        ax.plot(ox2, oy2, color='#AA2222', linewidth=1.5, zorder=2)

    # G1 塗りつぶし（前面）
    if len(shape1) >= 3:
        poly1 = MplPolygon(shape1, closed=True, facecolor='#DDEEFF',
                           edgecolor='none', zorder=3)
        ax.add_patch(poly1)

    # G1 輪郭線
    if len(outline1) >= 2:
        ox1, oy1 = zip(*outline1)
        ax.plot(ox1, oy1, color='#2266AA', linewidth=1.5, zorder=4)

    # 重なり部分
    for ovlp in overlaps:
        if len(ovlp) >= 3:
            opoly = MplPolygon(ovlp, closed=True, facecolor='#CC88FF',
                               edgecolor='#8844CC', linewidth=1, zorder=5)
            ax.add_patch(opoly)

    # 補助円弧
    sp1 = pa1 * (N_DRAW + 1.5)
    sp2 = pa2 * (N_DRAW + 1.5)

    def draw_arc(cx, cy, radius, color, ls, lw, ac, span):
        angles = np.linspace(ac - span/2, ac + span/2, 80)
        xs = cx + radius * np.cos(angles)
        ys = cy + radius * np.sin(angles)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, zorder=0)

    # ピッチ円
    draw_arc(g1x, g1y, r1, '#4488CC', 'dashdot', 0.8, 0, sp1)
    draw_arc(g2x, g2y, r2, '#4488CC', 'dashdot', 0.8, math.pi, sp2)
    # 基礎円
    draw_arc(g1x, g1y, rb1, '#AAAAAA', 'dotted', 0.8, 0, sp1)
    draw_arc(g2x, g2y, rb2, '#AAAAAA', 'dotted', 0.8, math.pi, sp2)

    # ピッチ点
    ax.plot(0, 0, 'o', color='#FF6600', markersize=6, zorder=10)
    ax.annotate('P', (0, 0), textcoords='offset points', xytext=(6, 6),
                fontsize=10, fontweight='bold', color='#FF6600', zorder=10)

    # 作用線
    line_len = m * 4.0
    ax.plot([-line_len * math.cos(alpha), line_len * math.cos(alpha)],
            [-line_len * math.sin(alpha), line_len * math.sin(alpha)],
            color='#00AA00', linestyle='dashed', linewidth=0.8, zorder=0)

    # ラベル
    ax.text(g1x, g1y, f'G1(z={int(z1)})', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#2266AA', zorder=10)
    ax.text(g2x, g2y, f'G2(z={int(z2)})', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#AA2222', zorder=10)

    # 情報テキスト
    da_info = f"  Δa={((a_center - a_des)*1000):+.1f}µm" if a_actual is not None else ""
    g2rot_info = f"  G2回転: {g2_extra_rot_deg:+.3f}°" if abs(g2_extra_rot_deg) > 0.0001 else ""
    info_text = (f"G1: z={int(z1)}, x={x1:.3f}  |  G2: z={int(z2)}, x={x2:.3f}  |  "
                 f"m={m}mm  α={alpha_deg}°\n"
                 f"a={a_center:.4f}mm{da_info}  |  jt={jt_um:.1f}µm{g2rot_info}")
    ax.set_title(info_text, fontsize=8, color='#333333', loc='left', pad=10,
                 fontfamily='monospace')

    # 干渉警告
    if has_interference:
        ax.text(0.5, 0.02, '⚠ 歯形干渉が発生しています',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='red',
                          edgecolor='darkred', alpha=0.9),
                zorder=20)

    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#4488CC', linestyle='dashdot', lw=1, label='ピッチ円'),
        Line2D([0], [0], color='#AAAAAA', linestyle='dotted', lw=1, label='基礎円'),
        Line2D([0], [0], color='#00AA00', linestyle='dashed', lw=1, label='作用線'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6600',
               markersize=6, label='ピッチ点 P'),
        mpatches.Patch(facecolor='#CC88FF', edgecolor='#8844CC', label='歯形重なり（干渉）'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
              framealpha=0.8)

    # 表示範囲
    margin = view_w * 0.08
    ax.set_xlim(-view_w / 2 - margin, view_w / 2 + margin)
    ax.set_ylim(-view_h / 2 - margin, view_h / 2 + margin)
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    return fig, has_interference


# =====================================================
# Streamlit GUI
# =====================================================
st.set_page_config(page_title="GearMesh Viewer", layout="wide")
st.title("⚙ GearMesh Viewer — 平歯車バックラッシ計算＋かみ合い描画")

# --- サイドバー: 入力パラメータ ---
with st.sidebar:
    st.header("入力パラメータ")

    st.subheader("基本諸元")
    m_val     = st.number_input("モジュール m [mm]", value=2.0, step=0.1, format="%.2f")
    alpha_val = st.number_input("圧力角 α [deg]", value=20.0, step=0.5, format="%.1f")
    z1_val    = st.number_input("歯数 z1", value=20, step=1, min_value=5)
    z2_val    = st.number_input("歯数 z2", value=40, step=1, min_value=5)
    x1_val    = st.number_input("転位係数 x1（設計）", value=0.0, step=0.01, format="%.4f")
    x2_val    = st.number_input("転位係数 x2（設計）", value=0.0, step=0.01, format="%.4f")

    st.divider()
    st.subheader("跨ぎ歯厚")
    k1_val = st.number_input("跨ぎ歯数 k1", value=3, step=1, min_value=1)
    k2_val = st.number_input("跨ぎ歯数 k2", value=4, step=1, min_value=1)

    # 自動計算のデフォルト値
    W1_default = span_from_xeff(x1_val, k1_val, m_val, z1_val, alpha_val)
    W2_default = span_from_xeff(x2_val, k2_val, m_val, z2_val, alpha_val)

    W1_val = st.number_input("跨ぎ歯厚 W1 [mm]", value=W1_default, step=0.001, format="%.4f")
    W2_val = st.number_input("跨ぎ歯厚 W2 [mm]", value=W2_default, step=0.001, format="%.4f")

    st.divider()
    st.subheader("軸間距離")
    a_default = center_distance_design(m_val, z1_val, z2_val, x1_val, x2_val)
    a_val = st.number_input("軸間距離 a [mm]", value=a_default, step=0.001, format="%.6f")

    st.divider()
    g2rot_val = st.number_input("G2追加回転 [deg]", value=0.0, step=0.01, format="%.4f")

# --- メイン: 計算＋表示 ---
try:
    out = calc_backlash_from_W_and_a(
        m=m_val, alpha_deg=alpha_val,
        z1=z1_val, z2=z2_val,
        x1_design=x1_val, x2_design=x2_val,
        a=a_val,
        W1=W1_val, k1=k1_val,
        W2=W2_val, k2=k2_val,
    )

    col_chart, col_result = st.columns([3, 2])

    with col_chart:
        fig, has_interference = draw_gear_mesh_mpl(
            m_val, z1_val, z2_val,
            out['x1_eff'], out['x2_eff'],
            alpha_val, out['jt_um'],
            a_actual=a_val, g2_extra_rot_deg=g2rot_val,
            x1_design=x1_val, x2_design=x2_val,
        )
        st.pyplot(fig)
        plt.close(fig)

    with col_result:
        st.subheader("計算結果")

        if has_interference:
            st.error("⚠ 歯形干渉が発生しています")

        st.markdown("#### 中心距離の整理")
        st.code(
            f"a_ref（幾何学基準） = {out['a_ref_mm']:.6f} mm\n"
            f"a_des（転位込み）   = {out['a_des_mm']:.6f} mm\n"
            f"a（入力値）         = {out['a_mm']:.6f} mm\n"
            f"Δa（a - a_des）     = {out['da_from_des_mm']*1000.0:+.2f} µm",
            language=None
        )

        st.markdown("#### 運転条件")
        st.code(
            f"αw（作用角）        = {out['alpha_w_deg']:.5f} deg\n"
            f"rw1 = {out['rw1_mm']:.5f} mm,  rw2 = {out['rw2_mm']:.5f} mm",
            language=None
        )

        st.markdown("#### 実効転位係数（W逆算）")
        st.code(
            f"x1_eff = {out['x1_eff']:.6f}  (設計 x1 = {x1_val:.4f})\n"
            f"x2_eff = {out['x2_eff']:.6f}  (設計 x2 = {x2_val:.4f})",
            language=None
        )

        st.markdown("#### バックラッシ")
        st.code(
            f"jt（円周）          = {out['jt_um']:.2f} µm\n"
            f"bn（法線）          = {out['bn_um']:.2f} µm",
            language=None
        )

        st.markdown("#### 角度バックラッシ")
        st.code(
            f"θ1 = {out['theta1_deg']:.10f} deg  ({out['theta1_arcmin']:.5f} arcmin)\n"
            f"θ2 = {out['theta2_deg']:.10f} deg  ({out['theta2_arcmin']:.5f} arcmin)",
            language=None
        )

        st.markdown("#### かみ合い率")
        st.code(f"εα = {out['epsilon_alpha']:.4f}", language=None)

    # 用語説明
    with st.expander("📖 用語説明"):
        st.markdown("""
**x_eff（実効転位係数）**
跨ぎ歯厚 W の実測値から逆算した転位係数。設計転位 x と異なる場合、加工誤差や歯面仕上げの影響で実際の歯厚が設計値とずれていることを意味します。

**a_ref（幾何学基準中心距離）**
転位を考慮しない、歯数とモジュールだけで決まる基準中心距離。 `a_ref = m·(z1 + z2) / 2`

**a_des（設計中心距離）**
転位係数を含めた設計上の中心距離。 `a_des = a_ref + m·(x1 + x2)`

**αw（作用圧力角）**
実際の軸間距離 a における作用圧力角。 `cos αw = a_ref · cos α / a`

**jt（円周バックラッシ）**
運転ピッチ円上での歯面間の隙間（円周方向）。 `jt = pw - (sw1 + sw2)`

**bn（法線バックラッシ）**
歯面の法線方向のバックラッシ。 `bn = jt / cos αw`

**θ（角度バックラッシ）**
一方のギアを固定したとき、もう一方がガタつく角度。 `θ = jt / rw`

**εα（かみ合い率）**
同時にかみ合う歯の平均枚数。εα > 1 で常に1枚以上がかみ合い、一般に εα ≥ 1.2 が推奨。
""")

except Exception as ex:
    st.error(f"計算エラー: {ex}")

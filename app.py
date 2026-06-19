import os
import sys
import json
import streamlit as st
import streamlit.components.v1 as components

# Add current directory to path to import stages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import stage1_parser
import stage2_ranker
from stage2b_india_signals import apply_india_signals
from stage2c_trajectory import apply_trajectory
from stage2d_clustering import cluster_candidates
from stage2e_bias_audit import run_fairness_audit
import stage3_explainer
import stage4_reverse_jd

# Configure streamlit page layout
st.set_page_config(
    page_title="TalentLens Bharat",
    page_icon="TL",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_visual_design():
    st.markdown(
        """
        <style>
            :root {
                --tl-ink: #102027;
                --tl-muted: #50616b;
                --tl-panel: rgba(255, 255, 255, 0.66);
                --tl-panel-strong: rgba(255, 255, 255, 0.82);
                --tl-line: rgba(255, 255, 255, 0.55);
                --tl-shadow: 0 24px 80px rgba(16, 32, 39, 0.13);
                --tl-cyan: #10b8c4;
                --tl-coral: #e86f61;
                --tl-gold: #d8a11d;
                --tl-green: #0c9868;
            }

            html {
                scroll-behavior: smooth;
            }

            body,
            .stApp {
                color: var(--tl-ink);
                background:
                    radial-gradient(circle at 12% 9%, rgba(16, 184, 196, 0.20), transparent 30vw),
                    radial-gradient(circle at 84% 18%, rgba(232, 111, 97, 0.18), transparent 32vw),
                    radial-gradient(circle at 48% 90%, rgba(216, 161, 29, 0.16), transparent 34vw),
                    linear-gradient(135deg, #f7fbfb 0%, #edf7f4 42%, #fff9ef 100%);
                background-attachment: fixed;
            }

            .stApp::before {
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                z-index: 0;
                background-image:
                    linear-gradient(rgba(16, 32, 39, 0.045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(16, 32, 39, 0.045) 1px, transparent 1px);
                background-size: 54px 54px;
                mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.72), transparent 78%);
                animation: grid-drift 24s linear infinite;
            }

            .stApp::after {
                content: "";
                position: fixed;
                inset: auto 0 0 0;
                height: 42vh;
                pointer-events: none;
                z-index: 0;
                background: linear-gradient(to top, rgba(255, 255, 255, 0.88), transparent);
            }

            @keyframes grid-drift {
                from { transform: translate3d(0, 0, 0); }
                to { transform: translate3d(-54px, -54px, 0); }
            }

            @keyframes rise-in {
                from { opacity: 0; transform: translateY(18px) scale(0.985); }
                to { opacity: 1; transform: translateY(0) scale(1); }
            }

            @keyframes sheen {
                from { transform: translateX(-120%) rotate(18deg); }
                to { transform: translateX(120%) rotate(18deg); }
            }

            .block-container {
                position: relative;
                z-index: 1;
                max-width: 1240px;
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }

            .tl-hero {
                position: relative;
                overflow: hidden;
                min-height: 560px;
                padding: clamp(28px, 5vw, 56px);
                border: 1px solid rgba(255, 255, 255, 0.78);
                border-radius: 28px;
                background:
                    linear-gradient(135deg, rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.38)),
                    linear-gradient(140deg, rgba(16, 184, 196, 0.10), rgba(232, 111, 97, 0.08));
                box-shadow: var(--tl-shadow);
                backdrop-filter: blur(26px) saturate(1.32);
                -webkit-backdrop-filter: blur(26px) saturate(1.32);
                animation: rise-in 720ms ease both;
            }

            .tl-hero::before {
                content: "";
                position: absolute;
                inset: -40% auto auto -20%;
                width: 70%;
                height: 140%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.42), transparent);
                animation: sheen 8s ease-in-out infinite;
            }

            .tl-hero-grid {
                position: relative;
                z-index: 1;
                display: grid;
                grid-template-columns: minmax(0, 0.95fr) minmax(420px, 1.05fr);
                gap: clamp(22px, 5vw, 56px);
                align-items: center;
            }

            .tl-eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                width: fit-content;
                padding: 8px 12px;
                border: 1px solid rgba(16, 32, 39, 0.10);
                border-radius: 999px;
                color: #0a6b73;
                background: rgba(255, 255, 255, 0.62);
                font-size: 0.78rem;
                font-weight: 750;
                text-transform: uppercase;
                letter-spacing: 0;
            }

            .tl-pulse {
                width: 9px;
                height: 9px;
                border-radius: 999px;
                background: var(--tl-green);
                box-shadow: 0 0 0 7px rgba(12, 152, 104, 0.12);
            }

            .tl-hero h1 {
                margin: 20px 0 14px;
                color: var(--tl-ink);
                font-size: clamp(3.2rem, 8vw, 6.8rem);
                line-height: 0.88;
                letter-spacing: 0;
            }

            .tl-hero p {
                max-width: 650px;
                color: var(--tl-muted);
                font-size: clamp(1rem, 1.4vw, 1.22rem);
                line-height: 1.75;
            }

            .tl-stat-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 28px;
            }

            .tl-stat {
                min-height: 112px;
                padding: 16px;
                border: 1px solid rgba(255, 255, 255, 0.62);
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.58);
                box-shadow: 0 14px 35px rgba(16, 32, 39, 0.09);
                backdrop-filter: blur(18px);
            }

            .tl-stat strong {
                display: block;
                color: var(--tl-ink);
                font-size: 1.55rem;
                line-height: 1.2;
            }

            .tl-stat span {
                display: block;
                margin-top: 8px;
                color: var(--tl-muted);
                font-size: 0.82rem;
                line-height: 1.35;
            }

            .tl-section-label {
                margin: 34px 0 12px;
                color: #0a6b73;
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0;
            }

            div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stExpander"],
            div[data-testid="stForm"],
            div[data-testid="stAlert"] {
                border-color: rgba(255, 255, 255, 0.62) !important;
                border-radius: 18px !important;
                background: var(--tl-panel) !important;
                box-shadow: 0 18px 60px rgba(16, 32, 39, 0.10);
                backdrop-filter: blur(22px) saturate(1.24);
                -webkit-backdrop-filter: blur(22px) saturate(1.24);
            }

            div[data-testid="stMetric"] {
                min-height: 112px;
                padding: 14px 14px 12px;
                border: 1px solid rgba(255, 255, 255, 0.66);
                border-radius: 16px;
                background: linear-gradient(160deg, rgba(255, 255, 255, 0.78), rgba(255, 255, 255, 0.46));
                box-shadow: 0 14px 34px rgba(16, 32, 39, 0.08);
            }

            div[data-testid="stMetricLabel"] p {
                color: #50616b;
                font-size: 0.78rem;
                font-weight: 700;
            }

            div[data-testid="stMetricValue"] {
                color: var(--tl-ink);
                font-weight: 850;
            }

            div[data-testid="stTextArea"] textarea,
            div[data-testid="stTextInput"] input,
            div[data-baseweb="select"] > div,
            div[data-testid="stFileUploader"] section {
                border: 1px solid rgba(16, 32, 39, 0.10) !important;
                border-radius: 16px !important;
                background: rgba(255, 255, 255, 0.72) !important;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 12px 34px rgba(16, 32, 39, 0.06);
                transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            }

            div[data-testid="stTextInput"] input,
            div[data-testid="stTextArea"] textarea,
            div[data-baseweb="select"] *,
            div[data-baseweb="select"] span,
            div[data-baseweb="select"] input {
                color: var(--tl-ink) !important;
                -webkit-text-fill-color: var(--tl-ink) !important;
                caret-color: var(--tl-cyan) !important;
            }

            div[data-testid="stTextInput"] input::placeholder,
            div[data-testid="stTextArea"] textarea::placeholder {
                color: rgba(80, 97, 107, 0.66) !important;
                -webkit-text-fill-color: rgba(80, 97, 107, 0.66) !important;
            }

            div[data-testid="stTextInput"] div,
            div[data-testid="stTextArea"] div,
            div[data-baseweb="select"] div {
                background-color: transparent !important;
            }

            div[data-testid="stTextArea"] textarea:focus,
            div[data-testid="stTextInput"] input:focus {
                border-color: rgba(16, 184, 196, 0.52) !important;
                box-shadow: 0 0 0 4px rgba(16, 184, 196, 0.13), 0 16px 42px rgba(16, 32, 39, 0.08) !important;
            }

            .stButton > button,
            div[data-testid="stDownloadButton"] button {
                min-height: 48px;
                border: 0;
                border-radius: 16px;
                color: white;
                background: linear-gradient(135deg, #102027, #0b7f87 54%, #e86f61);
                box-shadow: 0 18px 38px rgba(16, 32, 39, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.22);
                transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
            }

            .stButton > button:hover,
            div[data-testid="stDownloadButton"] button:hover {
                transform: translateY(-2px);
                filter: saturate(1.08);
                box-shadow: 0 24px 52px rgba(16, 32, 39, 0.23), inset 0 1px 0 rgba(255, 255, 255, 0.24);
            }

            .stSlider [data-baseweb="slider"] > div {
                color: var(--tl-cyan);
            }

            hr {
                margin: 2rem 0;
                border-color: rgba(16, 32, 39, 0.08);
            }

            h2, h3, h4 {
                letter-spacing: 0;
                color: var(--tl-ink);
            }

            .element-container {
                animation: rise-in 520ms ease both;
            }

            @media (max-width: 920px) {
                .tl-hero {
                    min-height: auto;
                    padding: 26px;
                    border-radius: 22px;
                }

                .tl-hero-grid {
                    grid-template-columns: 1fr;
                }

                .tl-stat-row {
                    grid-template-columns: 1fr;
                }
            }

            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 1ms !important;
                    animation-iteration-count: 1 !important;
                    scroll-behavior: auto !important;
                    transition-duration: 1ms !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_neural_canvas():
    components.html(
        """
        <canvas id="talent-orbit" aria-label="3D talent intelligence visualization"></canvas>
        <style>
            html, body {
                margin: 0;
                overflow: hidden;
                background: transparent;
            }

            #talent-orbit {
                width: 100%;
                height: 410px;
                display: block;
                border-radius: 26px;
                background:
                    radial-gradient(circle at 50% 40%, rgba(255, 255, 255, 0.55), rgba(255, 255, 255, 0.10) 42%, transparent 68%),
                    linear-gradient(145deg, rgba(16, 184, 196, 0.14), rgba(232, 111, 97, 0.10));
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.62), 0 24px 70px rgba(16,32,39,0.14);
                backdrop-filter: blur(20px);
            }
        </style>
        <script>
            const canvas = document.getElementById("talent-orbit");
            const ctx = canvas.getContext("2d");
            let width = 0;
            let height = 0;
            let px = 0;
            let py = 0;
            const nodes = Array.from({ length: 72 }, (_, i) => {
                const band = i % 3;
                return {
                    a: Math.random() * Math.PI * 2,
                    b: Math.random() * Math.PI * 2,
                    r: 86 + band * 44 + Math.random() * 42,
                    speed: 0.002 + Math.random() * 0.004,
                    size: 1.8 + Math.random() * 3.2,
                    hue: band === 0 ? 185 : band === 1 ? 12 : 42
                };
            });

            function resize() {
                const ratio = Math.min(window.devicePixelRatio || 1, 2);
                width = canvas.clientWidth;
                height = canvas.clientHeight;
                canvas.width = width * ratio;
                canvas.height = height * ratio;
                ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
            }

            function project(node, t) {
                const spin = node.a + t * node.speed;
                const tilt = node.b + Math.sin(t * 0.001 + node.a) * 0.45;
                const z = Math.sin(spin) * Math.cos(tilt);
                const depth = 0.64 + (z + 1) * 0.32;
                return {
                    x: width / 2 + Math.cos(spin) * node.r * depth + px * 28,
                    y: height / 2 + Math.sin(tilt) * node.r * 0.58 * depth + py * 20,
                    z,
                    depth
                };
            }

            function frame(t) {
                ctx.clearRect(0, 0, width, height);

                const grad = ctx.createRadialGradient(width / 2, height / 2, 20, width / 2, height / 2, Math.min(width, height) * 0.54);
                grad.addColorStop(0, "rgba(255,255,255,0.74)");
                grad.addColorStop(0.38, "rgba(16,184,196,0.16)");
                grad.addColorStop(0.72, "rgba(232,111,97,0.08)");
                grad.addColorStop(1, "rgba(255,255,255,0)");
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(width / 2 + px * 22, height / 2 + py * 14, Math.min(width, height) * 0.45, 0, Math.PI * 2);
                ctx.fill();

                const pts = nodes.map(n => ({ node: n, p: project(n, t) })).sort((a, b) => a.p.z - b.p.z);

                for (let i = 0; i < pts.length; i++) {
                    for (let j = i + 1; j < pts.length; j++) {
                        const dx = pts[i].p.x - pts[j].p.x;
                        const dy = pts[i].p.y - pts[j].p.y;
                        const dist = Math.hypot(dx, dy);
                        if (dist < 92) {
                            const alpha = (1 - dist / 92) * 0.19 * pts[i].p.depth;
                            ctx.strokeStyle = `rgba(16, 32, 39, ${alpha})`;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(pts[i].p.x, pts[i].p.y);
                            ctx.lineTo(pts[j].p.x, pts[j].p.y);
                            ctx.stroke();
                        }
                    }
                }

                pts.forEach(({ node, p }) => {
                    const radius = node.size * p.depth;
                    ctx.beginPath();
                    ctx.fillStyle = `hsla(${node.hue}, 78%, ${42 + p.depth * 18}%, ${0.58 + p.depth * 0.28})`;
                    ctx.shadowColor = `hsla(${node.hue}, 85%, 48%, 0.38)`;
                    ctx.shadowBlur = 18 * p.depth;
                    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                });

                ctx.save();
                ctx.translate(width / 2 + px * 34, height / 2 + py * 18);
                ctx.rotate(Math.sin(t * 0.0007) * 0.25);
                ctx.strokeStyle = "rgba(16, 32, 39, 0.14)";
                ctx.lineWidth = 1.2;
                for (let r of [92, 142, 190]) {
                    ctx.beginPath();
                    ctx.ellipse(0, 0, r, r * 0.38, Math.PI * 0.1, 0, Math.PI * 2);
                    ctx.stroke();
                }
                ctx.restore();

                requestAnimationFrame(frame);
            }

            window.addEventListener("resize", resize);
            window.addEventListener("mousemove", event => {
                px += ((event.clientX / window.innerWidth) - 0.5 - px) * 0.08;
                py += ((event.clientY / window.innerHeight) - 0.5 - py) * 0.08;
            });
            resize();
            requestAnimationFrame(frame);
        </script>
        """,
        height=430,
    )


def render_hero():
    components.html(
        """
        <section class="tl-hero" aria-label="TalentLens Bharat hero">
            <div class="tl-hero-grid">
                <div>
                    <div class="tl-eyebrow"><span class="tl-pulse"></span> Live recruiter intelligence</div>
                    <h1>TalentLens Bharat</h1>
                    <p>
                        Intelligent candidate discovery built for the way India hires. Standardize raw job descriptions,
                        score local semantic fit, surface India-specific talent signals, audit fairness, and generate
                        recruiter-ready justifications from one refined workspace.
                    </p>
                    <div class="tl-stat-row">
                        <div class="tl-stat"><strong>4-stage</strong><span>JD parsing, semantic ranking, explainability, and reverse-JD analysis.</span></div>
                        <div class="tl-stat"><strong>Glass UI</strong><span>Focused controls with liquid depth, soft motion, and readable data density.</span></div>
                        <div class="tl-stat"><strong>Local signals</strong><span>Activity, city tier, trajectory, behavior, and fairness in one scoring lens.</span></div>
                    </div>
                </div>
                <div>
                    <canvas id="talent-orbit" aria-label="3D talent intelligence visualization"></canvas>
                </div>
            </div>
        </section>
        <style>
            :root {
                --tl-ink: #102027;
                --tl-muted: #50616b;
                --tl-cyan: #10b8c4;
                --tl-coral: #e86f61;
                --tl-gold: #d8a11d;
                --tl-green: #0c9868;
            }

            html, body {
                margin: 0;
                overflow: hidden;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            .tl-hero {
                position: relative;
                overflow: hidden;
                box-sizing: border-box;
                min-height: 520px;
                padding: clamp(28px, 5vw, 56px);
                border: 1px solid rgba(255, 255, 255, 0.78);
                border-radius: 28px;
                background:
                    linear-gradient(135deg, rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.38)),
                    linear-gradient(140deg, rgba(16, 184, 196, 0.10), rgba(232, 111, 97, 0.08));
                box-shadow: 0 24px 80px rgba(16, 32, 39, 0.13);
                backdrop-filter: blur(26px) saturate(1.32);
                -webkit-backdrop-filter: blur(26px) saturate(1.32);
            }

            .tl-hero::before {
                content: "";
                position: absolute;
                inset: -40% auto auto -20%;
                width: 70%;
                height: 140%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.42), transparent);
                animation: sheen 8s ease-in-out infinite;
            }

            .tl-hero-grid {
                position: relative;
                z-index: 1;
                display: grid;
                grid-template-columns: minmax(0, 0.95fr) minmax(420px, 1.05fr);
                gap: clamp(22px, 5vw, 56px);
                align-items: center;
            }

            .tl-eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                width: fit-content;
                padding: 8px 12px;
                border: 1px solid rgba(16, 32, 39, 0.10);
                border-radius: 999px;
                color: #0a6b73;
                background: rgba(255, 255, 255, 0.62);
                font-size: 0.78rem;
                font-weight: 750;
                text-transform: uppercase;
                letter-spacing: 0;
            }

            .tl-pulse {
                width: 9px;
                height: 9px;
                border-radius: 999px;
                background: var(--tl-green);
                box-shadow: 0 0 0 7px rgba(12, 152, 104, 0.12);
            }

            h1 {
                margin: 20px 0 14px;
                color: var(--tl-ink);
                font-size: clamp(3.2rem, 8vw, 6.8rem);
                line-height: 0.88;
                letter-spacing: 0;
            }

            p {
                max-width: 650px;
                color: var(--tl-muted);
                font-size: clamp(1rem, 1.4vw, 1.22rem);
                line-height: 1.75;
            }

            .tl-stat-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 12px;
                margin-top: 28px;
            }

            .tl-stat {
                min-height: 112px;
                padding: 16px;
                border: 1px solid rgba(255, 255, 255, 0.62);
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.58);
                box-shadow: 0 14px 35px rgba(16, 32, 39, 0.09);
                backdrop-filter: blur(18px);
            }

            .tl-stat strong {
                display: block;
                color: var(--tl-ink);
                font-size: 1.55rem;
                line-height: 1.2;
            }

            .tl-stat span {
                display: block;
                margin-top: 8px;
                color: var(--tl-muted);
                font-size: 0.82rem;
                line-height: 1.35;
            }

            #talent-orbit {
                width: 100%;
                height: 410px;
                display: block;
                border-radius: 26px;
                background:
                    radial-gradient(circle at 50% 40%, rgba(255, 255, 255, 0.55), rgba(255, 255, 255, 0.10) 42%, transparent 68%),
                    linear-gradient(145deg, rgba(16, 184, 196, 0.14), rgba(232, 111, 97, 0.10));
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.62), 0 24px 70px rgba(16,32,39,0.14);
                backdrop-filter: blur(20px);
            }

            @keyframes sheen {
                from { transform: translateX(-120%) rotate(18deg); }
                to { transform: translateX(120%) rotate(18deg); }
            }

            @media (max-width: 920px) {
                html, body { overflow: auto; }
                .tl-hero {
                    min-height: auto;
                    padding: 26px;
                    border-radius: 22px;
                }

                .tl-hero-grid {
                    grid-template-columns: 1fr;
                }

                .tl-stat-row {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        <script>
            const canvas = document.getElementById("talent-orbit");
            const ctx = canvas.getContext("2d");
            let width = 0;
            let height = 0;
            let px = 0;
            let py = 0;
            const nodes = Array.from({ length: 72 }, (_, i) => {
                const band = i % 3;
                return {
                    a: Math.random() * Math.PI * 2,
                    b: Math.random() * Math.PI * 2,
                    r: 86 + band * 44 + Math.random() * 42,
                    speed: 0.002 + Math.random() * 0.004,
                    size: 1.8 + Math.random() * 3.2,
                    hue: band === 0 ? 185 : band === 1 ? 12 : 42
                };
            });

            function resize() {
                const ratio = Math.min(window.devicePixelRatio || 1, 2);
                width = canvas.clientWidth;
                height = canvas.clientHeight;
                canvas.width = width * ratio;
                canvas.height = height * ratio;
                ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
            }

            function project(node, t) {
                const spin = node.a + t * node.speed;
                const tilt = node.b + Math.sin(t * 0.001 + node.a) * 0.45;
                const z = Math.sin(spin) * Math.cos(tilt);
                const depth = 0.64 + (z + 1) * 0.32;
                return {
                    x: width / 2 + Math.cos(spin) * node.r * depth + px * 28,
                    y: height / 2 + Math.sin(tilt) * node.r * 0.58 * depth + py * 20,
                    z,
                    depth
                };
            }

            function frame(t) {
                ctx.clearRect(0, 0, width, height);

                const grad = ctx.createRadialGradient(width / 2, height / 2, 20, width / 2, height / 2, Math.min(width, height) * 0.54);
                grad.addColorStop(0, "rgba(255,255,255,0.74)");
                grad.addColorStop(0.38, "rgba(16,184,196,0.16)");
                grad.addColorStop(0.72, "rgba(232,111,97,0.08)");
                grad.addColorStop(1, "rgba(255,255,255,0)");
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(width / 2 + px * 22, height / 2 + py * 14, Math.min(width, height) * 0.45, 0, Math.PI * 2);
                ctx.fill();

                const pts = nodes.map(n => ({ node: n, p: project(n, t) })).sort((a, b) => a.p.z - b.p.z);

                for (let i = 0; i < pts.length; i++) {
                    for (let j = i + 1; j < pts.length; j++) {
                        const dx = pts[i].p.x - pts[j].p.x;
                        const dy = pts[i].p.y - pts[j].p.y;
                        const dist = Math.hypot(dx, dy);
                        if (dist < 92) {
                            const alpha = (1 - dist / 92) * 0.19 * pts[i].p.depth;
                            ctx.strokeStyle = `rgba(16, 32, 39, ${alpha})`;
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(pts[i].p.x, pts[i].p.y);
                            ctx.lineTo(pts[j].p.x, pts[j].p.y);
                            ctx.stroke();
                        }
                    }
                }

                pts.forEach(({ node, p }) => {
                    const radius = node.size * p.depth;
                    ctx.beginPath();
                    ctx.fillStyle = `hsla(${node.hue}, 78%, ${42 + p.depth * 18}%, ${0.58 + p.depth * 0.28})`;
                    ctx.shadowColor = `hsla(${node.hue}, 85%, 48%, 0.38)`;
                    ctx.shadowBlur = 18 * p.depth;
                    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                });

                ctx.save();
                ctx.translate(width / 2 + px * 34, height / 2 + py * 18);
                ctx.rotate(Math.sin(t * 0.0007) * 0.25);
                ctx.strokeStyle = "rgba(16, 32, 39, 0.14)";
                ctx.lineWidth = 1.2;
                for (let r of [92, 142, 190]) {
                    ctx.beginPath();
                    ctx.ellipse(0, 0, r, r * 0.38, Math.PI * 0.1, 0, Math.PI * 2);
                    ctx.stroke();
                }
                ctx.restore();

                requestAnimationFrame(frame);
            }

            window.addEventListener("resize", resize);
            window.addEventListener("mousemove", event => {
                px += ((event.clientX / window.innerWidth) - 0.5 - px) * 0.08;
                py += ((event.clientY / window.innerHeight) - 0.5 - py) * 0.08;
            });
            resize();
            requestAnimationFrame(frame);
        </script>
        """,
        height=650,
    )


inject_visual_design()
render_hero()
st.markdown('<div class="tl-section-label">Configure analysis</div>', unsafe_allow_html=True)
# Select Role Configuration
role_config = st.selectbox(
    "Select a pre-configured sample role to test:",
    ["Fintech AI/ML Backend Engineer (Default)", "B2B SaaS Product Manager"]
)

# Set file paths based on role configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
if role_config == "B2B SaaS Product Manager":
    jd_filename = "job_description_2.txt"
    default_cand_filename = "candidates.json"
else:
    jd_filename = "job_description.txt"
    default_cand_filename = "candidates.json"

default_jd_path = os.path.join(base_dir, "data", jd_filename)
default_cand_path = os.path.join(base_dir, "data", default_cand_filename)

# Create two-column layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Job Description")
    # Provide a sample job description by default for easy testing
    sample_jd = ""
    try:
        if os.path.exists(default_jd_path):
            with open(default_jd_path, "r", encoding="utf-8") as f:
                sample_jd = f.read()
    except Exception:
        pass
        
    jd_input = st.text_area(
        "Paste the raw job description text here:",
        value=sample_jd,
        height=300,
        placeholder="e.g. Seeking a Senior Backend Engineer with 5+ years experience in Python and FastAPI...",
        key="jd_text_area"
    )

with col2:
    st.markdown("### 👥 Candidate Pool")
    uploaded_file = st.file_uploader(
        "Upload your candidate pool (JSON format):",
        type=["json"],
        help="Upload a candidate JSON file matching the TalentLens schema."
    )
    
    # Check if sample candidate file exists to offer as a default
    if uploaded_file is None:
        if os.path.exists(default_cand_path):
            st.info(f"💡 No file uploaded. Using the default mock candidate pool (`{default_cand_filename}`).")
        else:
            st.warning("⚠️ Please upload a candidate pool JSON file to proceed.")

st.divider()

# Controls Section
explain_limit = st.slider(
    "Number of top candidates to explain:",
    min_value=1,
    max_value=10,
    value=5,
    help="We will call the LLM to generate two-sentence match justifications for this many top profiles."
)

def extract_disqualifier_features_for_cand(cand: dict) -> dict:
    profile = cand.get("profile") or {}
    signals = cand.get("redrob_signals") or {}
    skills_raw = cand.get("skills") or []
    career_raw = cand.get("career_history") or []

    # Skill names
    skill_names = []
    skill_proficiencies = {}
    total_endorsements = 0
    for s in skills_raw:
        if isinstance(s, dict):
            name = (s.get("name") or "").strip()
            if name:
                skill_names.append(name)
                skill_proficiencies[name.lower()] = s.get("proficiency", "intermediate")
                total_endorsements += s.get("endorsements", 0) or 0
        elif isinstance(s, str):
            skill_names.append(s)

    # Career history
    career_companies = []
    career_durations = []
    for role in career_raw:
        if isinstance(role, dict):
            company = role.get("company", "")
            duration = role.get("duration_months", 0) or 0
            if company:
                career_companies.append(company)
            career_durations.append(duration)

    assessment_scores = signals.get("skill_assessment_scores") or {}

    return {
        "career_durations": career_durations,
        "num_career_roles": len(career_durations),
        "career_companies": career_companies,
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1) or cand.get("offer_acceptance_rate", -1),
        "interview_completion_rate": signals.get("interview_completion_rate", 0) or cand.get("interview_completion_rate", 0),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0) or cand.get("recruiter_response_rate", 0),
        "num_skills": len(skill_names),
        "total_endorsements": total_endorsements,
        "years_of_experience": profile.get("years_of_experience", 0) or cand.get("experience_years", 0) or 0,
        "skill_proficiencies": skill_proficiencies,
        "assessment_scores": assessment_scores,
    }

def compute_behavioral_signal_score_for_cand(cand: dict, today=None) -> float:
    import datetime
    if today is None:
        today = datetime.date.today()

    score = 0.0

    # Open to work
    signals = cand.get("redrob_signals") or {}
    open_to_work = signals.get("open_to_work_flag") or cand.get("open_to_work", False)
    if open_to_work:
        score += 0.30

    # Last active date
    last_active_str = signals.get("last_active_date") or cand.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.datetime.strptime(str(last_active_str).strip(), "%Y-%m-%d").date()
            days_inactive = max(0, (today - last_active).days)
            if days_inactive < 7:
                score += 0.30
            elif days_inactive < 14:
                score += 0.25
            elif days_inactive < 30:
                score += 0.20
            elif days_inactive < 60:
                score += 0.12
            elif days_inactive < 90:
                score += 0.06
            elif days_inactive < 180:
                score += 0.03
        except (ValueError, TypeError):
            pass

    # Recruiter response rate
    rr = signals.get("recruiter_response_rate") or cand.get("recruiter_response_rate", 0)
    score += min(0.25, float(rr) * 0.25)

    # GitHub activity
    github = signals.get("github_activity_score")
    if github is None:
        github = cand.get("github_activity_score", -1)
    if github is not None and github >= 0:
        if github >= 70:
            score += 0.15
        elif github >= 50:
            score += 0.12
        elif github >= 30:
            score += 0.08
        elif github >= 10:
            score += 0.04

    return max(0.0, min(1.0, score))

def recalculate_scores_for_app(ranked_candidates, parsed_jd):
    import rank
    import datetime

    today = datetime.date.today()
    jd_seniority = parsed_jd.get("seniority", "Senior")

    W_SEMANTIC   = 0.55
    W_INDIA      = 0.20
    W_TRAJECTORY = 0.10
    W_BEHAVIORAL = 0.15

    updated_candidates = []

    for cand in ranked_candidates:
        # Honeypot check
        if rank.is_honeypot(cand):
            cand["final_score"] = 0.0500
            cand["match_score"] = 0.0500
            cand["behavioral_score"] = 0.0
            cand["recruiter_flag"] = "Profile flagged: skill proficiency claims inconsistent with duration data."
            updated_candidates.append(cand)
            continue

        # Extract features for disqualifier/stuffer checks
        feat = extract_disqualifier_features_for_cand(cand)

        # Component scores
        semantic_score = cand.get("semantic_score", 0.0)
        india_score = cand.get("india_signal_score", 0.50)
        trajectory_score = cand.get("trajectory_score", 0.0)

        # Behavioral score
        behavioral_score = compute_behavioral_signal_score_for_cand(cand, today)
        cand["behavioral_score"] = round(behavioral_score, 4)

        # Seniority penalty
        current_title = cand.get("current_title") or (cand.get("profile") or {}).get("current_title", "")
        seniority_mult, seniority_match = stage2_ranker.calculate_seniority_penalty(jd_seniority, current_title)
        cand["seniority_match"] = seniority_match

        # Disqualifier and stuffer penalties
        disqualifier_mult = rank.compute_disqualifier_penalty(feat)
        stuffer_mult = rank.compute_honeypot_penalty(feat)

        # Final blended score
        final_score = disqualifier_mult * stuffer_mult * (
            W_SEMANTIC * semantic_score * seniority_mult
            + W_INDIA * india_score
            + W_TRAJECTORY * trajectory_score
            + W_BEHAVIORAL * behavioral_score
        )
        final_score = max(0.0, min(1.0, final_score))

        # Flags
        flags = []
        if cand.get("recruiter_flag"):
            flags.append(cand.get("recruiter_flag"))
        if disqualifier_mult < 1.0:
            flags.append("Disqualifier penalty applied.")
        if stuffer_mult < 1.0:
            flags.append("Skill-stuffer penalty applied.")
        if flags:
            cand["recruiter_flag"] = " | ".join(flags)

        cand["final_score"] = round(final_score, 4)
        cand["match_score"] = cand["final_score"]
        updated_candidates.append(cand)

    # Stable sort: tie-breaker ID ascending, score descending
    updated_candidates.sort(key=lambda x: x.get("candidate_id", "") or x.get("id", ""))
    updated_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # Re-assign ranks
    for idx, cand in enumerate(updated_candidates):
        cand["rank"] = idx + 1

    return updated_candidates

run_button = st.button("🚀 Run TalentLens", type="primary", use_container_width=True)

# Main Processing Logic
if run_button:
    # 1. Validation checks
    if not jd_input.strip():
        st.error("Error: Job description text cannot be empty.")
    elif uploaded_file is None and not os.path.exists(default_cand_path):
        st.error("Error: Candidate pool JSON file is missing.")
    else:
        # Define paths for temporary scratch files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        output_dir = os.path.join(base_dir, "output")
        utils.ensure_directories_exist([data_dir, output_dir])

        temp_jd_file = os.path.join(data_dir, "temp_jd.txt")
        temp_candidates_file = os.path.join(data_dir, "temp_candidates.json")
        temp_parsed_jd_file = os.path.join(data_dir, "temp_parsed_jd.json")
        temp_ranked_output = os.path.join(output_dir, "temp_ranked_candidates.json")

        try:
            # Write inputs to temp files
            with open(temp_jd_file, "w", encoding="utf-8") as f:
                f.write(jd_input.strip())

            if uploaded_file is not None:
                candidates_data = json.load(uploaded_file)
                utils.save_json(candidates_data, temp_candidates_file)
            else:
                # Fall back to default
                candidates_data = utils.load_json(default_cand_path)
                utils.save_json(candidates_data, temp_candidates_file)

            # 2. Execute pipeline stages with spinners
            # Stage 1
            with st.spinner("Stage 1: Standardizing and parsing Job Description using LLM..."):
                parsed_jd = stage1_parser.parse_job_description(temp_jd_file, temp_parsed_jd_file)
            st.success("✅ Stage 1 complete: Job description standardized.")

            # Stage 2
            with st.spinner("Stage 2: Computing semantic cosine similarities locally..."):
                ranked_candidates = stage2_ranker.rank_candidates(
                    temp_parsed_jd_file, 
                    temp_candidates_file, 
                    temp_ranked_output
                )
            st.success("✅ Stage 2 complete: Semantic scores computed.")

            # Stage 2b
            with st.spinner("Stage 2b: Applying India-Specific Talent Signals and Flags..."):
                ranked_candidates = apply_india_signals(ranked_candidates)
                utils.save_json(ranked_candidates, temp_ranked_output)
            st.success("✅ Stage 2b complete: India boosts and activity date metrics applied.")

            # Stage 2c
            with st.spinner("Stage 2c: Computing Career Trajectory Predictions..."):
                ranked_candidates = apply_trajectory(ranked_candidates)
                # Recalculate using the new 4-component formula and penalties
                ranked_candidates = recalculate_scores_for_app(ranked_candidates, parsed_jd)
                utils.save_json(ranked_candidates, temp_ranked_output)
            st.success("✅ Stage 2c complete: Trajectory & behavioral scoring applied.")
            
            # Stage 2d
            with st.spinner("Stage 2d: Running Unsupervised Persona Clustering (KMeans)..."):
                clustering_res = cluster_candidates(parsed_jd, ranked_candidates)
                cluster_assignments = clustering_res.get("candidate_clusters", {})
                cluster_rec_text = clustering_res.get("cluster_recommendation", "")
                
                # Enrich candidates with persona cluster labels
                for cand in ranked_candidates:
                    cand["persona_cluster"] = cluster_assignments.get(cand["id"], "The Generalist Builder")
                utils.save_json(ranked_candidates, temp_ranked_output)
            st.success("✅ Stage 2d complete: Unsupervised candidate personas clustered.")
            
            # Stage 2e
            with st.spinner("Stage 2e: Running Bias Detection & Fairness Audit..."):
                fairness_report = run_fairness_audit(ranked_candidates)
            st.success("✅ Stage 2e complete: Fairness audit generated.")
            
            # Stage 3
            with st.spinner("Stage 3: Generating explainable match justifications..."):
                final_candidates = stage3_explainer.explain_top_candidates(
                    temp_parsed_jd_file, 
                    temp_ranked_output, 
                    limit=explain_limit
                )
            st.success("✅ Stage 3 complete: Justifications generated.")
            
            # Enrich candidates with confidence reasoning and select Dark Horse
            for cand in final_candidates:
                cand["confidence_reasoning"] = utils.compute_confidence_reasoning(cand, parsed_jd)
            utils.save_json(final_candidates, temp_ranked_output)
            dark_horse_candidate = utils.select_dark_horse(final_candidates, limit=explain_limit)
            
            # Stage 4
            with st.spinner("Stage 4: Running Reverse JD & Talent Alignment Analysis..."):
                temp_reverse_jd_output = temp_ranked_output.replace(".json", "_reverse_jd.json")
                reverse_jd_report = stage4_reverse_jd.analyze_alignment(
                    temp_parsed_jd_file,
                    temp_ranked_output,
                    temp_reverse_jd_output
                )
            st.success("✅ Stage 4 complete: Reverse JD alignment analyzed.")
            
            st.divider()
            
            # 3. Render Results Display
            st.markdown("## 📊 Candidates Discovery Results")
            
            # Expandable Parsed JD JSON
            with st.expander("🔍 View Parsed Job Description Attributes (JSON)"):
                st.json(parsed_jd)
                
            # Display Recruiter Clustering Recommendation
            if 'cluster_rec_text' in locals() and cluster_rec_text:
                st.info(f"🎯 **Recruiter Persona Recommendation:** {cluster_rec_text}")
                
            # Display Dark Horse Spotlight
            if 'dark_horse_candidate' in locals() and dark_horse_candidate:
                st.info(
                    f"🏇 **Dark Horse Spotlight:** **{dark_horse_candidate.get('name')}** "
                    f"(Rank {dark_horse_candidate.get('rank')}, Score: {dark_horse_candidate.get('final_score'):.4f}) — *{dark_horse_candidate.get('current_title')}*\n\n"
                    f"💡 **Why they are a Dark Horse:** {dark_horse_candidate.get('reason')}"
                )
                
            st.markdown(f"Displaying ranked matches (top {len(final_candidates)} shown):")
            
            # Display Fairness Report Section
            if 'fairness_report' in locals() and fairness_report:
                f_score = fairness_report.get('fairness_score', 0)
                f_grade = fairness_report.get('fairness_grade', 'N/A')
                
                # Grade color mapping
                grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
                grade_icon = grade_colors.get(f_grade, "⚪")
                
                with st.expander(f"🛡️ **Fairness Report** — Score: {f_score}/100 {grade_icon} Grade {f_grade}", expanded=False):
                    st.caption(fairness_report.get('methodology_note', ''))
                    
                    # Population distribution
                    pop = fairness_report.get('population_distribution', {})
                    dist_col1, dist_col2 = st.columns(2)
                    with dist_col1:
                        st.markdown("**Gender Distribution:**")
                        for g, count in pop.get('by_gender', {}).items():
                            st.markdown(f"- {g}: **{count}** candidates")
                    with dist_col2:
                        st.markdown("**City Tier Distribution:**")
                        for t, count in pop.get('by_city_tier', {}).items():
                            st.markdown(f"- {t}: **{count}** candidates")
                    
                    st.markdown("---")
                    
                    # Per-dimension reports
                    for dim in fairness_report.get('dimensions', []):
                        verdict = dim.get('verdict', {})
                        level = verdict.get('level', 'PASS')
                        
                        # Verdict display
                        if level == 'PASS':
                            st.success(verdict.get('summary', ''))
                        elif level == 'WATCH':
                            st.warning(verdict.get('summary', ''))
                        else:
                            st.error(verdict.get('summary', ''))
                        
                        # Group statistics table
                        group_stats = dim.get('group_statistics', {})
                        if group_stats:
                            stats_md = "| Group | Count | Avg Score | Median | Std Dev | Avg Rank |\n"
                            stats_md += "|-------|-------|-----------|--------|---------|----------|\n"
                            for gname, gdata in group_stats.items():
                                stats_md += (
                                    f"| {gname} | {gdata['count']} | "
                                    f"{gdata['avg_final_score']:.4f} | "
                                    f"{gdata['median_final_score']:.4f} | "
                                    f"{gdata['stdev_final_score']:.4f} | "
                                    f"{gdata['avg_rank']:.1f} |\n"
                                )
                            st.markdown(stats_md)
                        
                        # Pairwise comparisons
                        for comp in dim.get('pairwise_comparisons', []):
                            effect = comp.get('effect_size', 'negligible')
                            st.markdown(
                                f"**{comp['group_a']} vs {comp['group_b']}**: "
                                f"Gap = {comp['score_gap']:.4f} ({comp['gap_direction']}), "
                                f"Cohen's d = {comp['cohens_d']:.3f} ({effect})"
                            )
                            if comp.get('sample_size_warning'):
                                st.caption(comp['sample_size_warning'])
                        
                        # Underranked candidates
                        underranked = dim.get('potentially_underranked', [])
                        if underranked:
                            st.markdown("**⚡ Potentially Underranked Candidates:**")
                            for flag in underranked:
                                st.markdown(
                                    f"- **{flag['candidate_name']}** (Rank {flag['rank']}, "
                                    f"Semantic: {flag['semantic_score']:.4f}, "
                                    f"Final: {flag['final_score']:.4f}) — "
                                    f"outscored {flag['outperformed_by']['candidate_name']} "
                                    f"(Rank {flag['outperformed_by']['rank']}, "
                                    f"Semantic: {flag['outperformed_by']['semantic_score']:.4f}) "
                                    f"on raw skills but ranked lower"
                                )
                        
                        # Recommendations
                        recs = verdict.get('recommendations', [])
                        if recs:
                            st.markdown("**Recommendations:**")
                            for rec in recs:
                                st.markdown(f"- {rec}")
                        
                        st.markdown("---")
            
            # Display Reverse JD Section
            if 'reverse_jd_report' in locals() and reverse_jd_report:
                alignment_score = reverse_jd_report.get('alignment_score', 0)
                alignment_icon = "🟢" if alignment_score >= 85 else "🟡" if alignment_score >= 70 else "🔴"
                
                with st.expander(f"🔄 **Reverse JD & Talent Alignment** — Score: {alignment_score}/100 {alignment_icon}", expanded=False):
                    st.markdown(f"**Talent Pool Alignment Score:** `{alignment_score}%` — {reverse_jd_report.get('alignment_explanation', '')}")
                    
                    # Side-by-side columns
                    jd_col1, jd_col2 = st.columns(2)
                    with jd_col1:
                        st.markdown("#### 📄 Original Job Description")
                        st.markdown(f"**Seniority:** {parsed_jd.get('seniority', 'N/A')}")
                        st.markdown(f"**Domain:** {parsed_jd.get('domain', 'N/A')}")
                        st.markdown("**Core Required Skills:**")
                        for s in parsed_jd.get('required_skills', []):
                            st.markdown(f"- {s}")
                        if parsed_jd.get('nice_to_have'):
                            st.markdown("**Nice-to-Have Skills:**")
                            for s in parsed_jd.get('nice_to_have', []):
                                st.markdown(f"- {s}")
                    
                    with jd_col2:
                        ideal = reverse_jd_report.get('ideal_jd', {})
                        st.markdown("#### 🎯 Reconstructed Ideal JD")
                        st.markdown(f"**Suggested Title:** {ideal.get('suggested_title', 'N/A')}")
                        st.markdown(f"**Suggested Seniority:** {ideal.get('suggested_seniority', 'N/A')}")
                        st.markdown("**Ideal Core Skills:**")
                        for s in ideal.get('core_skills', []):
                            st.markdown(f"- {s}")
                        if ideal.get('nice_to_have_skills'):
                            st.markdown("**Ideal Nice-to-Have Skills:**")
                            for s in ideal.get('nice_to_have_skills', []):
                                st.markdown(f"- {s}")
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Skill Comparison Grid")
                    
                    # Create markdown table
                    table_md = "| Skill | In Original JD? | In Top Candidates? | Status | Recommendation |\n"
                    table_md += "| :--- | :---: | :---: | :--- | :--- |\n"
                    for item in reverse_jd_report.get("skill_comparisons", []):
                        orig = "✅ Yes" if item.get("in_original_jd") else "❌ No"
                        cand = "✅ Yes" if item.get("in_top_candidates") else "❌ No"
                        status = item.get("status", "Unknown")
                        status_emoji = "🟩 Aligned" if status == "Aligned" else "🟥 Missing in Candidates" if status == "Missing in Candidates" else "🟦 Bonus in Candidates"
                        table_md += f"| {item.get('skill_name')} | {orig} | {cand} | {status_emoji} | {item.get('recommendation')} |\n"
                    st.markdown(table_md)
                    
                    # Rewrite Suggestions
                    rewrites = reverse_jd_report.get("suggested_jd_rewrites", [])
                    if rewrites:
                        st.markdown("---")
                        st.markdown("### 💡 Actionable JD Rewrite Suggestions")
                        for rw in rewrites:
                            st.markdown(f"- ✍️ **Suggestion:** {rw}")
            
            st.markdown("---")
            
            for idx, cand in enumerate(final_candidates):
                rank = cand.get("rank", idx + 1)
                name = cand.get("name", "Unknown Candidate")
                title = cand.get("current_title", "Software Engineer")
                
                # Determine trajectory badge
                traj_label = cand.get("trajectory_label", "")
                badge_map = {
                    "High momentum": "🚀",
                    "Steady growth": "📈",
                    "Early stage": "🌱",
                    "Plateau": "⏸️"
                }
                trajectory_badge = f"{badge_map.get(traj_label, '')} {traj_label}" if traj_label else ""
                
                # Card Container
                with st.container():
                    persona_cluster = cand.get("persona_cluster", "The Generalist Builder")
                    st.markdown(f"#### **Rank {rank}: {name}** — *{title}*  `{persona_cluster}`  {trajectory_badge}".strip())
                    
                    # Confidence Score with Reasoning (Structured Uncertainty Communication)
                    confidence_reasoning = cand.get("confidence_reasoning", "")
                    if confidence_reasoning:
                        if "High confidence" in confidence_reasoning:
                            st.success(f"🛡️ **Structured Confidence:** {confidence_reasoning}")
                        elif "Medium confidence" in confidence_reasoning:
                            st.warning(f"🛡️ **Structured Confidence:** {confidence_reasoning}")
                        else:
                            st.error(f"🛡️ **Structured Confidence:** {confidence_reasoning}")
                            
                    # Columns for scores and indicators
                    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
                    with m_col1:
                        st.metric(
                            label="Final Score", 
                            value=f"{cand.get('final_score', 0.0):.4f}", 
                            help="Combined metric: 55% semantic * seniority + 20% India signals + 10% trajectory + 15% behavioral."
                        )
                    with m_col2:
                        st.metric(
                            label="Semantic Match", 
                            value=f"{cand.get('semantic_score', 0.0):.4f}",
                            help="Raw semantic cosine similarity score using sentence embeddings."
                        )
                    with m_col3:
                        st.metric(
                            label="India Signal Score", 
                            value=f"{cand.get('india_signal_score', 0.50):.2f}",
                            help="Calculated rating representing activity dates, India tech skills, and Tier-2/3 city alignment."
                        )
                    with m_col4:
                        st.metric(
                            label="Trajectory",
                            value=f"{cand.get('trajectory_score', 0.0):.2f}",
                            help="Career trajectory prediction score based on skill velocity, title progression, and advanced recency."
                        )
                    with m_col5:
                        st.metric(
                            label="Behavioral Score",
                            value=f"{cand.get('behavioral_score', 0.0):.2f}",
                            help="Engagement score representing open_to_work status, recent activity, response rates, and GitHub presence."
                        )
                    with m_col6:
                        match_status = "Exact Match" if cand.get("seniority_match") else "Mismatch"
                        st.metric(
                            label="Seniority Status", 
                            value=match_status,
                            help="Indicates whether the candidate's seniority level matches the target JD requirements exactly."
                        )
                    
                    # Display India signals
                    india_signals = cand.get("india_signals_detected", [])
                    if india_signals:
                        st.write(f"💼 **Signals Detected:** {', '.join(india_signals)}")
                        
                    # Recruiter Warning Flags
                    recruiter_flag = cand.get("recruiter_flag")
                    if recruiter_flag:
                        st.warning(f"⚠️ **Recruiter Action Required:** {recruiter_flag}")
                    
                    # Trajectory Signals Expander
                    traj_signals = cand.get("trajectory_signals", [])
                    if traj_signals:
                        with st.expander(f"📊 Trajectory Signals — {traj_label}"):
                            for sig in traj_signals:
                                st.markdown(f"- {sig}")
                        
                    # Match Justification Caption
                    justification = cand.get("justification")
                    if justification:
                        st.info(f"💡 **Why they match:** {justification}")
                    else:
                        st.caption("No justification generated for this candidate.")
                        
                    # Skill Gap Report & Learning Bridge
                    gap_report = cand.get("skill_gap_report")
                    if gap_report:
                        missing = gap_report.get("missing_skills", [])
                        pace = gap_report.get("learning_time_inferred", "")
                        resources = gap_report.get("suggested_resources", [])
                        
                        if missing or resources:
                            with st.expander("🛠️ **Skill Gap & Learning Bridge**", expanded=True):
                                if missing:
                                    st.markdown(f"**Missing Skills:** {', '.join(f'`{m}`' for m in missing)}")
                                if pace:
                                    st.markdown(f"**Inferred Learning Pace:** {pace}")
                                if resources:
                                    st.markdown("**Suggested Resources to Bridge the Gap:**")
                                    for r in resources:
                                        st.markdown(f"- **{r.get('name')}** ({r.get('url_or_platform')}) — *{r.get('description')}*")
                                        
                    # Targeted Interview Questions
                    questions = cand.get("targeted_interview_questions", [])
                    if questions:
                        with st.expander("❓ **Targeted Interview Questions**", expanded=False):
                            st.caption("3 auto-generated interview questions probing this candidate's specific skill gaps:")
                            for q_idx, q in enumerate(questions):
                                st.markdown(f"**{q_idx+1}.** {q}")
                                        
                    st.divider()
                    
        except Exception as e:
            st.error(f"An error occurred while executing the pipeline: {str(e)}")
            st.exception(e)

# ============================================================================
# Production Hackathon Submission Generator Section
# ============================================================================
st.divider()
st.markdown("## 🏆 Production Submission Generator & Validator")
st.markdown(
    "Generate and validate the final hackathon `submission.csv` file using the official 4-component scoring logic, honeypot filters, and disqualifier penalties."
)

with st.expander("🛠️ **Submission Settings**", expanded=True):
    # Dropdown to select Job Description
    selected_role_generator = st.selectbox(
        "Select Role / Job Description for Generator:",
        ["Fintech AI/ML Backend Engineer (Default)", "B2B SaaS Product Manager"],
        key="gen_role_select"
    )

    # Auto-resolve defaults based on selection
    if "Product Manager" in selected_role_generator:
        default_candidates = "candidates.jsonl"
        default_jd = "data/parsed_jd_2.json"
        default_cache = "data/reasoning_cache_pm.json"
    else:
        default_candidates = "candidates.jsonl"
        default_jd = "data/parsed_jd.json"
        default_cache = "data/reasoning_cache.json"

    col_input, col_output = st.columns(2)
    with col_input:
        candidates_input_path = st.text_input(
            "Input Candidates Path:", 
            value=default_candidates, 
            help="Path to the gzipped or raw candidate JSONL/JSON file (e.g. candidates.jsonl or data/candidates_pm.json)"
        )
    with col_output:
        submission_output_path = st.text_input(
            "Output CSV Path:", 
            value="submission.csv", 
            help="Path where the final 100-row CSV will be written."
        )

    cache_col, parsed_jd_col = st.columns(2)
    with cache_col:
        reasoning_cache_path_ui = st.text_input(
            "Reasoning Cache Path:", 
            value=default_cache, 
            help="Path to the pre-generated LLM reasoning cache."
        )
    with parsed_jd_col:
        parsed_jd_path_ui = st.text_input(
            "Parsed JD Path:",
            value=default_jd,
            help="Path to the pre-parsed job description JSON."
        )

generate_sub_button = st.button("🚀 Generate & Validate Submission", type="secondary", use_container_width=True)

if generate_sub_button:
    if not os.path.exists(candidates_input_path):
        st.error(f"Error: Candidate file not found at `{candidates_input_path}`. Please verify the path.")
    else:
        with st.spinner("Streaming candidate dataset and executing offline ranker (this takes ~6 seconds)..."):
            try:
                import rank
                import validate_submission
                import importlib
                importlib.reload(rank)
                importlib.reload(validate_submission)

                # Execute ranking
                rank.run_ranking(
                    candidates_path=candidates_input_path,
                    output_path=submission_output_path,
                    jd_path=parsed_jd_path_ui,
                    reasoning_cache_path=reasoning_cache_path_ui
                )

                st.success("✅ Ranking execution complete! Final CSV file generated.")

                # Execute validation
                is_strict = (candidates_input_path.strip() in ("candidates.jsonl", "candidates.jsonl.gz"))
                with st.spinner("Running Python validator rules on the output CSV..."):
                    errors = validate_submission.validate_submission(submission_output_path, strict=is_strict)

                if not is_strict:
                    st.info("💡 **Mock/Custom Mode**: Validation was run with relaxed rules (allowing < 100 rows and custom candidate ID formats) suitable for mock/custom datasets.")

                if errors:
                    st.error("❌ Submission Validation FAILED with the following issues:")
                    for err in errors:
                        st.markdown(f"- {err}")
                else:
                    if is_strict:
                        st.success("🎉 **SUBMISSION IS VALID!** The CSV passed all strict formatting, sorting, and constraint checks for the official hackathon submission.")
                    else:
                        st.success("🎉 **SUBMISSION IS VALID!** The CSV passed all custom sorting, formatting, and non-increasing score checks.")

                    # Provide download button
                    with open(submission_output_path, "r", encoding="utf-8") as f:
                        csv_data = f.read()
                    st.download_button(
                        label="📥 Download Submission CSV",
                        data=csv_data,
                        file_name=os.path.basename(submission_output_path),
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as ex:
                st.error(f"An error occurred during submission generation: {ex}")
                st.exception(ex)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings, base64, io, math
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IPL Analytics", page_icon="🏏", layout="wide", initial_sidebar_state="collapsed")

TEAM_INFO = {
    "Chennai Super Kings":         {"abbr":"CSK",  "color":"#D4AF37","bg":"#1A1200","text":"#1a1200","shape":"star"},
    "Mumbai Indians":              {"abbr":"MI",   "color":"#1E4E91","bg":"#001628","text":"#ffffff","shape":"wave"},
    "Royal Challengers Bengaluru": {"abbr":"RCB",  "color":"#EC1C24","bg":"#2B0000","text":"#ffffff","shape":"diamond"},
    "Kolkata Knight Riders":       {"abbr":"KKR",  "color":"#C8A951","bg":"#1a0a2e","text":"#C8A951","shape":"crown"},
    "Rajasthan Royals":            {"abbr":"RR",   "color":"#E91E8C","bg":"#1a0012","text":"#ffffff","shape":"circle"},
    "Sunrisers Hyderabad":         {"abbr":"SRH",  "color":"#C75B00","bg":"#1a1000","text":"#1a1000","shape":"sun"},
    "Delhi Capitals":              {"abbr":"DC",   "color":"#4169E1","bg":"#000d1a","text":"#ffffff","shape":"shield"},
    "Punjab Kings":                {"abbr":"PBKS", "color":"#B11226","bg":"#1a0000","text":"#ffffff","shape":"lion"},
    "Gujarat Titans":              {"abbr":"GT",   "color":"#36495E","bg":"#000d1a","text":"#ffffff","shape":"titan"},
    "Lucknow Super Giants":        {"abbr":"LSG",  "color":"#00C8FF","bg":"#001824","text":"#000d1a","shape":"giant"},
}
ACTIVE_TEAMS = list(TEAM_INFO.keys())

VENUES = sorted(['Wankhede Stadium','Eden Gardens','M Chinnaswamy Stadium',
    'MA Chidambaram Stadium, Chepauk','Arun Jaitley Stadium',
    'Rajiv Gandhi International Stadium, Uppal',
    'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Sawai Mansingh Stadium','Narendra Modi Stadium, Ahmedabad',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Barsapara Cricket Stadium, Guwahati',
    'Maharashtra Cricket Association Stadium, Pune',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala',
    'Dr DY Patil Sports Academy','Brabourne Stadium',
    'Holkar Cricket Stadium','JSCA International Stadium Complex',
    'Saurashtra Cricket Association Stadium','Sharjah Cricket Stadium',
    'Dubai International Cricket Stadium',
    'M Chinnaswamy Stadium, Bengaluru','Wankhede Stadium, Mumbai'])

def make_logo_svg(info, size=80):
    abbr=info["abbr"]; color=info["color"]; bg=info["bg"]; tc=info["text"]; shape=info["shape"]
    s=size; cx=cy=s//2; r=s//2-3
    parts=[f'<svg width="{s}" height="{s}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{bg}" stroke="{color}" stroke-width="3"/>')
    if shape=="star":
        pts=[]
        for i in range(10):
            ang=(i*36-90)*math.pi/180; rr=(r-8) if i%2==0 else (r-22)
            pts.append(f"{cx+rr*math.cos(ang):.1f},{cy+rr*math.sin(ang):.1f}")
        parts.append(f'<polygon points="{" ".join(pts)}" fill="{color}" opacity="0.28"/>')
    elif shape=="wave":
        parts.append(f'<path d="M {cx-r+4} {cy+10} Q {cx} {cy-12} {cx+r-4} {cy+10}" stroke="{color}" stroke-width="3.5" fill="none" opacity="0.5"/>')
        parts.append(f'<path d="M {cx-r+4} {cy+18} Q {cx} {cy-4} {cx+r-4} {cy+18}" stroke="{color}" stroke-width="2" fill="none" opacity="0.25"/>')
    elif shape=="diamond":
        mid=s//2
        parts.append(f'<polygon points="{mid},{8} {s-8},{mid} {mid},{s-8} {8},{mid}" fill="{color}" opacity="0.2"/>')
        parts.append(f'<polygon points="{mid},{18} {s-18},{mid} {mid},{s-18} {18},{mid}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.4"/>')
    elif shape=="crown":
        parts.append(f'<path d="M {cx-22},{cy+12} L {cx-22},{cy-10} L {cx-10},{cy+2} L {cx},{cy-16} L {cx+10},{cy+2} L {cx+22},{cy-10} L {cx+22},{cy+12} Z" fill="{color}" opacity="0.3"/>')
    elif shape=="sun":
        for i in range(8):
            ang=i*45*math.pi/180
            x1=cx+(r-18)*math.cos(ang); y1=cy+(r-18)*math.sin(ang)
            x2=cx+(r-6)*math.cos(ang); y2=cy+(r-6)*math.sin(ang)
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="3" opacity="0.5"/>')
    elif shape=="shield":
        parts.append(f'<path d="M {cx},{cy-20} L {cx+18},{cy-10} L {cx+18},{cy+8} Q {cx+18},{cy+22} {cx},{cy+26} Q {cx-18},{cy+22} {cx-18},{cy+8} L {cx-18},{cy-10} Z" fill="{color}" opacity="0.25"/>')
    elif shape in ("lion","titan","giant","circle"):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r-14}" fill="{color}" opacity="0.18"/>')
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r-8}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.35"/>')
    fs=22 if len(abbr)<=2 else (17 if len(abbr)==3 else 13)
    parts.append(f'<text x="{cx}" y="{cy+fs//3+1}" text-anchor="middle" dominant-baseline="middle" font-family="Arial Black,Arial" font-size="{fs}" font-weight="900" fill="{tc}" letter-spacing="1">{abbr}</text>')
    parts.append('</svg>')
    b64=base64.b64encode("".join(parts).encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"

for name,info in TEAM_INFO.items():
    info["logo"]=make_logo_svg(info,80)
    info["logo_sm"]=make_logo_svg(info,52)

if 'page' not in st.session_state: st.session_state.page='predict'

IPL_LOGO = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1NiIgaGVpZ2h0PSI1NiIgdmlld0JveD0iMCAwIDU2IDU2Ij4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iYmciIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzAwMzA5OSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwMDFhNjAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImdvbGQiIHgxPSIwIiB5MT0iMCIgeDI9IjAiIHkyPSIxIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iI0ZGRTg1MCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiNGRkE1MDAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxjaXJjbGUgY3g9IjI4IiBjeT0iMjgiIHI9IjI3IiBmaWxsPSJ1cmwoI2JnKSIgc3Ryb2tlPSIjRkZENzAwIiBzdHJva2Utd2lkdGg9IjIuNSIvPgogIDwhLS0gQ3Jvd24gLS0+CiAgPHBhdGggZD0iTTE2LDIyIEwyMCwxMyBMMjQsMjAgTDI4LDExIEwzMiwyMCBMMzYsMTMgTDQwLDIyIFoiIGZpbGw9InVybCgjZ29sZCkiLz4KICA8cmVjdCB4PSIxNiIgeT0iMjIiIHdpZHRoPSIyNCIgaGVpZ2h0PSI0IiByeD0iMiIgZmlsbD0idXJsKCNnb2xkKSIvPgogIDwhLS0gSVBMIHRleHQgLS0+CiAgPHRleHQgeD0iMjgiIHk9IjQxIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iQXJpYWwgQmxhY2siIGZvbnQtc2l6ZT0iMTMiIGZvbnQtd2VpZ2h0PSI5MDAiIGZpbGw9InVybCgjZ29sZCkiIGxldHRlci1zcGFjaW5nPSIyIj5JUEw8L3RleHQ+CiAgPCEtLSBTaGluZSAtLT4KICA8ZWxsaXBzZSBjeD0iMjAiIGN5PSIxNiIgcng9IjYiIHJ5PSIzIiBmaWxsPSJ3aGl0ZSIgb3BhY2l0eT0iMC4xNSIgdHJhbnNmb3JtPSJyb3RhdGUoLTI1LDIwLDE2KSIvPgo8L3N2Zz4="
BG_SVG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjEwODAiIHZpZXdCb3g9IjAgMCAxOTIwIDEwODAiPgogIDxkZWZzPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjAiIHkyPSIxIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzAxMDQwOCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjM1JSIgc3RvcC1jb2xvcj0iIzAzMDkxYSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwNTBkMjAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8cmFkaWFsR3JhZGllbnQgaWQ9Im91dGZpZWxkIiBjeD0iNTAlIiBjeT0iOTAlIiByPSI3MCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMGE2MDI1IiBzdG9wLW9wYWNpdHk9IjAuNTUiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI0MCUiIHN0b3AtY29sb3I9IiMwNTJmMTAiIHN0b3Atb3BhY2l0eT0iMC4yNSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwMDAiIHN0b3Atb3BhY2l0eT0iMCIvPgogICAgPC9yYWRpYWxHcmFkaWVudD4KICAgIDxyYWRpYWxHcmFkaWVudCBpZD0icGl0Y2hnbG93IiBjeD0iNTAlIiBjeT0iODUlIiByPSIyMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjYzhhMDQwIiBzdG9wLW9wYWNpdHk9IjAuMTgiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjYzhhMDQwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgICA8cmFkaWFsR3JhZGllbnQgaWQ9InNwb3RsaWdodCIgY3g9IjUwJSIgY3k9IjAlIiByPSI2MCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMWEzYTZhIiBzdG9wLW9wYWNpdHk9IjAuMjIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDAwIiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxyZWN0IHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjEwODAiIGZpbGw9InVybCgjc2t5KSIvPgogIDxyZWN0IHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjEwODAiIGZpbGw9InVybCgjc3BvdGxpZ2h0KSIvPgogIDxyZWN0IHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjEwODAiIGZpbGw9InVybCgjb3V0ZmllbGQpIi8+CiAgPHJlY3Qgd2lkdGg9IjE5MjAiIGhlaWdodD0iMTA4MCIgZmlsbD0idXJsKCNwaXRjaGdsb3cpIi8+CiAgPGVsbGlwc2UgY3g9Ijk2MCIgY3k9IjExMDAiIHJ4PSI4MjAiIHJ5PSIyODAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzBhNjAyNSIgc3Ryb2tlLXdpZHRoPSIxLjUiIG9wYWNpdHk9IjAuMTgiLz4KICA8ZWxsaXBzZSBjeD0iOTYwIiBjeT0iMTEwMCIgcng9IjYyMCIgcnk9IjIxMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMGE2MDI1IiBzdHJva2Utd2lkdGg9IjEiIG9wYWNpdHk9IjAuMTIiLz4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg5NDAsODYwKSIgb3BhY2l0eT0iMC4yMiI+CiAgICA8cmVjdCB4PSIwIiAgeT0iMCIgd2lkdGg9IjUiIGhlaWdodD0iNjAiIHJ4PSIyIiBmaWxsPSIjZDRhZjM3Ii8+CiAgICA8cmVjdCB4PSIxNCIgeT0iMCIgd2lkdGg9IjUiIGhlaWdodD0iNjAiIHJ4PSIyIiBmaWxsPSIjZDRhZjM3Ii8+CiAgICA8cmVjdCB4PSIyOCIgeT0iMCIgd2lkdGg9IjUiIGhlaWdodD0iNjAiIHJ4PSIyIiBmaWxsPSIjZDRhZjM3Ii8+CiAgICA8cmVjdCB4PSItMiIgeT0iMCIgd2lkdGg9IjM2IiBoZWlnaHQ9IjUiIHJ4PSIyIiBmaWxsPSIjZDRhZjM3Ii8+CiAgICA8cmVjdCB4PSIyIiAgeT0iLTYiIHdpZHRoPSIxMCIgaGVpZ2h0PSI3IiByeD0iMS41IiBmaWxsPSIjZDRhZjM3Ii8+CiAgICA8cmVjdCB4PSIyMSIgeT0iLTYiIHdpZHRoPSIxMCIgaGVpZ2h0PSI3IiByeD0iMS41IiBmaWxsPSIjZDRhZjM3Ii8+CiAgPC9nPgogIDxyZWN0IHg9IjkxMCIgeT0iNzgwIiB3aWR0aD0iMTAwIiBoZWlnaHQ9IjIyMCIgcng9IjQiIGZpbGw9IiNiODk2MmEiIG9wYWNpdHk9IjAuMDciLz4KICA8ZWxsaXBzZSBjeD0iOTYwIiBjeT0iOTIwIiByeD0iNTAwIiByeT0iMTQwIiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMSIgc3Ryb2tlLWRhc2hhcnJheT0iOCwxOCIgb3BhY2l0eT0iMC4wNCIvPgo8L3N2Zz4="

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
.stApp{{background-color:#02040a!important;background-image:url("{BG_SVG}")!important;background-size:cover!important;background-position:center top!important;background-repeat:no-repeat!important;background-attachment:fixed!important;min-height:100vh;}}
section[data-testid="stSidebar"]{{display:none!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding-top:0!important;padding-bottom:1rem;max-width:1200px;position:relative;z-index:1;}}
.top-header{{display:flex;align-items:center;justify-content:space-between;padding:10px 20px 10px 16px;background:linear-gradient(90deg,#07111e 0%,#0a1628 50%,#07111e 100%);border-bottom:1.5px solid #1a2a40;margin-bottom:0;position:relative;overflow:hidden;}}
.top-header::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#f4820a88,transparent);}}
.brand-wrap{{display:flex;align-items:center;gap:12px;}}
.brand-text-ipl{{font-size:20px;font-weight:900;color:#f4820a;line-height:1;letter-spacing:1px;}}
.brand-text-sub{{font-size:10px;font-weight:700;color:#607080;letter-spacing:2px;margin-top:3px;text-transform:uppercase;}}
.trophy-badge{{display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,#0c1828,#111e30);border:1px solid #f4820a55;border-radius:14px;padding:8px 16px 8px 12px;box-shadow:0 0 24px rgba(244,130,10,0.18);}}
.trophy-icon{{font-size:28px;animation:trophy-glow 2.5s ease-in-out infinite;}}
@keyframes trophy-glow{{0%,100%{{filter:drop-shadow(0 0 5px rgba(244,130,10,0.5));}}50%{{filter:drop-shadow(0 0 14px rgba(244,130,10,1.0));}}}}
.trophy-text-year{{font-size:15px;font-weight:900;color:#f4820a;line-height:1;}}
.trophy-text-tagline{{font-size:9px;color:#7090a0;font-weight:600;letter-spacing:1px;margin-top:3px;}}
.stButton>button{{background:linear-gradient(135deg,#0a1628 0%,#0d1e35 100%)!important;color:#e0eaf5!important;border:1px solid #2a4060!important;border-radius:10px!important;font-size:12px!important;font-weight:700!important;padding:9px 6px!important;width:100%!important;letter-spacing:0.3px!important;transition:all 0.2s ease!important;box-shadow:0 2px 8px rgba(0,0,0,0.4)!important;}}
.stButton>button:hover{{background:linear-gradient(135deg,#f4820a 0%,#d4620a 100%)!important;color:#ffffff!important;border:1px solid #f4820a!important;box-shadow:0 4px 18px rgba(244,130,10,0.45)!important;transform:translateY(-1px)!important;}}
.nav-bar{{background:linear-gradient(90deg,#05101e,#07152a,#05101e);border-bottom:1px solid #1a2a40;padding:8px 16px;margin-bottom:1rem;}}
.nav-label{{font-size:10px;color:#8899aa;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;display:block;}}
.page-title{{font-size:26px;font-weight:900;color:#fff;letter-spacing:-.5px;}}
.page-title span{{color:#f4820a;}}
.page-sub{{font-size:13px;color:#9aaabb;margin-top:3px;margin-bottom:1.25rem;}}
.divider{{display:flex;align-items:center;gap:10px;margin:0.75rem 0 1rem;color:#6080a0;font-size:10px;font-weight:700;letter-spacing:2px;}}
.divider::before,.divider::after{{content:'';flex:1;height:.5px;background:#1a2a40;}}
.card{{background:rgba(12,24,40,0.92);border-radius:14px;padding:1.25rem;margin-bottom:1rem;border:.5px solid #1a2a40;backdrop-filter:blur(4px);}}
.metric-card{{background:#07111e;border-radius:12px;padding:1rem 1.25rem;border:.5px solid #1a2a40;text-align:center;}}
.metric-val{{font-size:28px;font-weight:900;margin-bottom:2px;}}
.metric-lbl{{font-size:10px;color:#8899aa;font-weight:600;letter-spacing:.5px;}}
.metric-sub{{font-size:10px;color:#5a7080;margin-top:3px;}}
.team-card{{border-radius:14px;padding:1.25rem .75rem;text-align:center;border:2px solid transparent;}}
.vs-badge{{display:flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:50%;background:#0c1828;border:.5px solid #1a2a40;font-size:12px;font-weight:800;color:#fff;margin:0 auto;}}
.result-winner{{font-size:28px;font-weight:900;text-align:center;letter-spacing:-.5px;}}
.result-sub{{font-size:11px;color:#607080;text-align:center;margin-top:2px;margin-bottom:1rem;letter-spacing:1px;}}
.prob-bar-outer{{width:100%;height:10px;background:#0a1428;border-radius:5px;overflow:hidden;display:flex;}}
.prob-labels{{display:flex;justify-content:space-between;font-size:10px;color:#8899aa;margin-top:4px;}}
.factors-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;}}
.factor-item{{background:#07111e;border-radius:10px;padding:10px 6px;text-align:center;border:.5px solid #1a2a40;}}
.factor-icon{{font-size:18px;margin-bottom:4px;}}
.factor-name{{font-size:10px;font-weight:700;color:#fff;}}
.factor-sub{{font-size:9px;color:#607080;margin-top:2px;line-height:1.4;}}
.ai-badge{{background:#1a6fd4;color:#fff;font-size:9px;font-weight:700;padding:3px 10px;border-radius:20px;display:inline-block;letter-spacing:.5px;}}
.stat-table{{width:100%;border-collapse:collapse;}}
.stat-table th{{font-size:11px;color:#607080;font-weight:700;padding:8px 12px;border-bottom:.5px solid #1a2a40;text-align:left;}}
.stat-table td{{font-size:12px;color:#dde;padding:9px 12px;border-bottom:.5px solid #0c1828;}}
.stat-table tr:hover td{{background:#0c1828;}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700;}}
.player-row{{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:.5px solid #1a2a40;}}
.player-rank{{font-size:12px;font-weight:700;color:#5a7080;min-width:22px;}}
.player-name{{font-size:13px;font-weight:700;color:#fff;flex:1;}}
.player-val{{font-size:13px;font-weight:800;}}
.insight-box{{background:#07111e;border-radius:10px;padding:1rem 1.2rem;border:.5px solid #1a2a40;margin-bottom:1rem;}}
.insight-title{{font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;}}
.insight-body{{font-size:12px;color:#94a3b8;line-height:1.85;}}
.footer{{text-align:center;color:#4a6070;font-size:10px;margin-top:1rem;padding-bottom:1rem;letter-spacing:.5px;}}
div[data-testid="stRadio"] label,div[data-testid="stRadio"] span,div[data-testid="stRadio"] p{{color:#ffffff!important;font-weight:600!important;font-size:13px!important;text-shadow:0 1px 4px rgba(0,0,0,0.9)!important;}}
div[data-testid="stRadio"] label:hover{{color:#f4820a!important;}}
.stSelectbox>div>div{{background:#0c1828!important;color:#fff!important;border-color:#1a2a40!important;}}
label[data-baseweb="label"]{{color:#8899aa!important;}}
</style>
""", unsafe_allow_html=True)

# ── TOP HEADER ──
st.markdown(f"""
<div class="top-header">
  <div class="brand-wrap">
    <img src="{IPL_LOGO}" width="52" height="52" style="border-radius:50%;box-shadow:0 0 20px rgba(255,215,0,0.7);flex-shrink:0;border:2px solid rgba(255,215,0,0.5);">
    <div>
      <div class="brand-text-ipl">IPL ANALYTICS</div>
      <div class="brand-text-sub">Cricket Intelligence 2008–2026</div>
    </div>
  </div>
  <div class="trophy-badge">
    <div class="trophy-icon">🏆</div>
    <div>
      <div class="trophy-text-year">IPL 2026</div>
      <div class="trophy-text-tagline">THE ULTIMATE<br>CRICKET SHOWDOWN</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DATA LOAD & TRAIN ──
@st.cache_resource
def load_and_train():
    MATCHES_URL    = "https://huggingface.co/datasets/shibangmaity/ipl-analytics-data/resolve/main/matches_2008_2026.csv"
    DELIVERIES_URL = "https://huggingface.co/datasets/shibangmaity/ipl-analytics-data/resolve/main/deliveries_enriched_2008_2026.csv"
    PLAYERS_URL    = "https://huggingface.co/datasets/shibangmaity/ipl-analytics-data/resolve/main/players_clean.csv"
    matches    = pd.read_csv(MATCHES_URL)
    deliveries = pd.read_csv(DELIVERIES_URL)
    players    = pd.read_csv(PLAYERS_URL)

    sm={'2007/08':'2008','2009/10':'2010','2020/21':'2021'}
    tm={'Rising Pune Supergiant':'Rising Pune Supergiants',
        'Royal Challengers Bangalore':'Royal Challengers Bengaluru',
        'Delhi Daredevils':'Delhi Capitals','Kings XI Punjab':'Punjab Kings'}
    matches['season']=matches['season'].replace(sm).astype(str)
    for col in ['team1','team2','toss_winner','winner']: matches[col]=matches[col].replace(tm)
    for col in ['batting_team','bowling_team']: deliveries[col]=deliveries[col].replace(tm)
    matches=matches.dropna(subset=['winner']).sort_values('date').reset_index(drop=True)

    matches['team1_won_toss']=(matches['toss_winner']==matches['team1']).astype(int)
    matches['toss_win_bat']=((matches['toss_winner']==matches['team1'])&(matches['toss_decision']=='bat')|(matches['toss_winner']==matches['team2'])&(matches['toss_decision']=='bat')).astype(int)
    matches['team1_won']=(matches['winner']==matches['team1']).astype(int)
    matches['season_num']=matches['season'].astype(int)

    def h2h_winrate(df):
        h2h,rates={},[]
        for _,row in df.iterrows():
            key=tuple(sorted([row['team1'],row['team2']]))
            if key not in h2h: h2h[key]={'w':0,'t':0}
            rates.append(h2h[key]['w']/h2h[key]['t'] if h2h[key]['t']>0 else 0.5)
            h2h[key]['t']+=1
            if row['winner']==row['team1']: h2h[key]['w']+=1
        return rates
    matches['h2h_winrate']=h2h_winrate(matches)

    def rolling_wr(df,col,w=15):
        rates,hist={},[]
        for _,row in df.iterrows():
            t=row[col]
            if t not in rates: rates[t]=[]
            h=rates[t]; hist.append(np.mean(h[-w:]) if len(h)>=5 else 0.5)
            rates[t].append(1 if row['winner']==t else 0)
        return hist
    matches['team1_form']=rolling_wr(matches,'team1',15)
    matches['team2_form']=rolling_wr(matches,'team2',15)
    matches['form_diff']=matches['team1_form']-matches['team2_form']

    def rolling_wr5(df,col,w=5):
        rates,hist={},[]
        for _,row in df.iterrows():
            t=row[col]
            if t not in rates: rates[t]=[]
            h=rates[t]; hist.append(np.mean(h[-w:]) if len(h)>=3 else 0.5)
            rates[t].append(1 if row['winner']==t else 0)
        return hist
    matches['team1_form5']=rolling_wr5(matches,'team1',5)
    matches['team2_form5']=rolling_wr5(matches,'team2',5)
    matches['form5_diff']=matches['team1_form5']-matches['team2_form5']

    tw=matches['winner'].value_counts().to_dict()
    tm2=pd.concat([matches['team1'],matches['team2']]).value_counts().to_dict()
    matches['team1_overall_wr']=matches['team1'].map(lambda t:tw.get(t,0)/tm2.get(t,1))
    matches['team2_overall_wr']=matches['team2'].map(lambda t:tw.get(t,0)/tm2.get(t,1))
    matches['wr_diff']=matches['team1_overall_wr']-matches['team2_overall_wr']

    def venue_team_wr(df):
        vt,rates={},[]
        for _,row in df.iterrows():
            k1=(row['venue'],row['team1']); k2=(row['venue'],row['team2'])
            if k1 not in vt: vt[k1]={'w':0,'t':0}
            if k2 not in vt: vt[k2]={'w':0,'t':0}
            t1vwr=vt[k1]['w']/vt[k1]['t'] if vt[k1]['t']>=3 else 0.5
            t2vwr=vt[k2]['w']/vt[k2]['t'] if vt[k2]['t']>=3 else 0.5
            rates.append(t1vwr-t2vwr)
            vt[k1]['t']+=1; vt[k2]['t']+=1
            if row['winner']==row['team1']: vt[k1]['w']+=1
            else: vt[k2]['w']+=1
        return rates
    matches['venue_wr_diff']=venue_team_wr(matches)
    matches['match_num_season']=matches.groupby('season').cumcount()+1
    tot=matches.groupby('season')['match_num_season'].transform('max')
    matches['season_stage']=matches['match_num_season']/tot

    le=LabelEncoder(); le.fit(pd.concat([matches['team1'],matches['team2']]).unique())
    matches['team1_enc']=le.transform(matches['team1']); matches['team2_enc']=le.transform(matches['team2'])
    ve=LabelEncoder(); matches['venue_enc']=ve.fit_transform(matches['venue'])

    feats=['team1_won_toss','toss_win_bat','h2h_winrate','form_diff','form5_diff',
           'venue_wr_diff','wr_diff','season_num','season_stage',
           'team1_form','team2_form','team1_form5','team2_form5',
           'team1_overall_wr','team2_overall_wr']

    matches_flip=matches.copy()
    matches_flip['team1_won_toss']=1-matches['team1_won_toss']
    matches_flip['h2h_winrate']=1-matches['h2h_winrate']
    matches_flip['form_diff']=-matches['form_diff']
    matches_flip['form5_diff']=-matches['form5_diff']
    matches_flip['venue_wr_diff']=-matches['venue_wr_diff']
    matches_flip['wr_diff']=-matches['wr_diff']
    matches_flip['team1_form']=matches['team2_form']; matches_flip['team2_form']=matches['team1_form']
    matches_flip['team1_form5']=matches['team2_form5']; matches_flip['team2_form5']=matches['team1_form5']
    matches_flip['team1_overall_wr']=matches['team2_overall_wr']; matches_flip['team2_overall_wr']=matches['team1_overall_wr']
    matches_flip['team1_won']=1-matches['team1_won']

    matches_aug=pd.concat([matches,matches_flip],ignore_index=True).sample(frac=1,random_state=42).reset_index(drop=True)
    X,y=matches_aug[feats],matches_aug['team1_won']
    split=int(len(matches)*0.75)
    X_train,y_train=X[:split*2],y[:split*2]
    X_test,y_test=matches[feats][split:],matches['team1_won'][split:]
    rf=RandomForestClassifier(n_estimators=1000,max_depth=6,min_samples_leaf=8,max_features='sqrt',random_state=42)
    rf.fit(X_train,y_train); acc=rf.score(X_test,y_test)

    # Batting stats
    bat=deliveries.groupby('batter').agg(runs=('batsman_runs','sum'),balls=('ball','count'),innings=('match_id','nunique')).reset_index()
    bat['sr']=(bat['runs']/bat['balls']*100).round(1)
    bat['avg']=(bat['runs']/bat['innings']).round(1)

    # Bowling stats
    wk_df=deliveries[deliveries['is_wicket']==True]
    wk_df=wk_df[~wk_df['dismissal_kind'].isin(['run out','retired hurt','obstructing the field'])]
    bowl=deliveries.groupby('bowler').agg(runs=('total_runs','sum'),balls=('ball','count'),matches=('match_id','nunique')).reset_index()
    wk_grp=wk_df.groupby('bowler')['is_wicket'].count().reset_index(); wk_grp.columns=['bowler','wickets']
    bowl=bowl.merge(wk_grp,on='bowler',how='left').fillna(0)
    bowl['wickets']=bowl['wickets'].astype(int)
    bowl['economy']=(bowl['runs']/(bowl['balls']/6)).round(2)
    bowl['avg']=(bowl['runs']/bowl['wickets'].replace(0,np.nan)).round(1)

    # Boundaries
    sixes=deliveries[deliveries['batsman_runs']==6].groupby('batter').size().reset_index(); sixes.columns=['batter','sixes']
    fours=deliveries[deliveries['batsman_runs']==4].groupby('batter').size().reset_index(); fours.columns=['batter','fours']
    bat=bat.merge(sixes,on='batter',how='left').merge(fours,on='batter',how='left').fillna(0)
    bat['sixes']=bat['sixes'].astype(int); bat['fours']=bat['fours'].astype(int)

    # Season awards
    bat2=deliveries.merge(matches[['id','season']],left_on='match_id',right_on='id')
    oc=bat2.groupby(['season','batter'])['batsman_runs'].sum().reset_index()
    oc=oc.loc[oc.groupby('season')['batsman_runs'].idxmax()].sort_values('season')
    wk2=wk_df.merge(matches[['id','season']],left_on='match_id',right_on='id')
    pc=wk2.groupby(['season','bowler'])['is_wicket'].count().reset_index()
    pc=pc.loc[pc.groupby('season')['is_wicket'].idxmax()].sort_values('season')

    # Season avg scores
    ss=deliveries.merge(matches[['id','season']],left_on='match_id',right_on='id')
    season_avg=ss.groupby(['season','match_id'])['total_runs'].sum().reset_index().groupby('season')['total_runs'].mean().round(1)

    # Toss decisions
    td=matches.groupby(['season','toss_decision']).size().unstack(fill_value=0)

    # Team boundaries
    t_fours=deliveries[deliveries['batsman_runs']==4].groupby('batting_team').size().reset_index(); t_fours.columns=['team','fours']
    t_sixes=deliveries[deliveries['batsman_runs']==6].groupby('batting_team').size().reset_index(); t_sixes.columns=['team','sixes']
    team_bnd=t_fours.merge(t_sixes,on='team')

    # Bat hand stats
    bat_hand_stats=deliveries.groupby('batter_hand').agg(
        total_runs=('batsman_runs','sum'),avg_runs=('batsman_runs','mean'),total_balls=('batsman_runs','count')).reset_index()

    # Bowl type stats
    bowl_type_stats=deliveries[deliveries['bowler_type']!='Unknown'].groupby('bowler_type').agg(
        total_wickets=('is_wicket','sum'),total_balls=('is_wicket','count'),runs_conceded=('batsman_runs','sum')).reset_index()
    bowl_type_stats['wicket_rate']=(bowl_type_stats['total_wickets']/bowl_type_stats['total_balls']*100).round(2)
    bowl_type_stats['economy']=(bowl_type_stats['runs_conceded']/(bowl_type_stats['total_balls']/6)).round(2)

    # Phase stats
    deliveries['phase']=pd.cut(deliveries['over'],bins=[-1,5,14,19],labels=['Powerplay (0-5)','Middle (6-14)','Death (15-19)'])
    phase_stats=deliveries.groupby('phase').agg(runs=('batsman_runs','sum'),balls=('batsman_runs','count'),wickets=('is_wicket','sum')).reset_index()
    phase_stats['run_rate']=(phase_stats['runs']/(phase_stats['balls']/6)).round(2)
    phase_stats['wicket_rate']=(phase_stats['wickets']/phase_stats['balls']*100).round(2)

    # Dismissals
    wk_df2=deliveries[deliveries['is_wicket']==True]
    dismissal_stats=wk_df2['dismissal_kind'].value_counts().reset_index()
    dismissal_stats.columns=['dismissal_kind','count']
    dismissal_stats=dismissal_stats[~dismissal_stats['dismissal_kind'].isin(['retired hurt','obstructing the field','retired out'])]

    # Bowl style — wickets + wicket rate
    bowl_style_merge=deliveries.merge(players[['player_name','bowl_style']],left_on='bowler',right_on='player_name',how='left')
    wk_style=bowl_style_merge[bowl_style_merge['is_wicket']==True]
    bsw=wk_style.groupby('bowl_style')['is_wicket'].count().reset_index(); bsw.columns=['bowl_style','wickets']
    bsb=bowl_style_merge.groupby('bowl_style').size().reset_index(); bsb.columns=['bowl_style','balls']
    bowl_style_stats=bsw.merge(bsb,on='bowl_style')
    bowl_style_stats['wicket_rate']=(bowl_style_stats['wickets']/bowl_style_stats['balls']*100).round(2)
    bowl_style_stats=bowl_style_stats[bowl_style_stats['bowl_style'].str.strip()!=''].sort_values('wickets',ascending=False).head(10)

    # City stats
    city_stats=matches.groupby('city').agg(matches_played=('id','count')).reset_index().sort_values('matches_played',ascending=False).head(12)

    # City-wise team win rates
    city_team_wins={}
    for team in ACTIVE_TEAMS:
        tm_m=matches[(matches['team1']==team)|(matches['team2']==team)]
        cp=tm_m.groupby('city')['id'].count().reset_index(); cp.columns=['city','played']
        cw=tm_m[tm_m['winner']==team].groupby('city')['id'].count().reset_index(); cw.columns=['city','wins']
        mg=cp.merge(cw,on='city',how='left').fillna(0)
        mg['win_rate']=(mg['wins']/mg['played']*100).round(1)
        city_team_wins[team]=mg.sort_values('played',ascending=False).head(10)

    # Allrounders
    bat_r=deliveries.groupby('batter')['batsman_runs'].sum().reset_index(); bat_r.columns=['player','runs']
    bowl_w=deliveries[deliveries['is_wicket']==True].groupby('bowler')['is_wicket'].count().reset_index(); bowl_w.columns=['player','wickets']
    allrounders_df=bat_r.merge(bowl_w,on='player')
    allrounders_df=allrounders_df[(allrounders_df['runs']>=500)&(allrounders_df['wickets']>=30)].sort_values('wickets',ascending=False).head(10)

    # Arm matchup
    arm_matchup=deliveries[deliveries['bowler_type']!='Unknown'].groupby(['bowler_arm','bowler_type','batter_hand']).agg(
        avg_runs=('batsman_runs','mean'),wicket_rate=('is_wicket','mean'),balls=('batsman_runs','count')).reset_index()
    arm_matchup['wicket_pct']=(arm_matchup['wicket_rate']*100).round(2)
    arm_matchup['sr']=(arm_matchup['avg_runs']*100).round(1)

    # ── NEW ANALYTICS ──
    # 1. Death bowlers (overs 15-19)
    death_del=deliveries[deliveries['over']>=15]
    d_wk=death_del[(death_del['is_wicket']==True)&(~death_del['dismissal_kind'].isin(['run out','retired hurt']))]
    d_wk_cnt=d_wk.groupby('bowler')['is_wicket'].count().reset_index(); d_wk_cnt.columns=['bowler','wickets']
    d_balls=death_del.groupby('bowler')['ball'].count().reset_index(); d_balls.columns=['bowler','balls']
    d_runs=death_del.groupby('bowler')['total_runs'].sum().reset_index(); d_runs.columns=['bowler','runs']
    death_bowlers=d_wk_cnt.merge(d_balls,on='bowler').merge(d_runs,on='bowler')
    death_bowlers=death_bowlers[death_bowlers['balls']>=120]
    death_bowlers['economy']=(death_bowlers['runs']/(death_bowlers['balls']/6)).round(2)
    death_bowlers['wicket_rate']=(death_bowlers['wickets']/death_bowlers['balls']*100).round(2)
    death_bowlers=death_bowlers.sort_values('wickets',ascending=False).head(10).reset_index(drop=True)

    # 2. Powerplay batters (overs 0-5)
    pp_del=deliveries[deliveries['over']<=5]
    pp_bat=pp_del.groupby('batter').agg(runs=('batsman_runs','sum'),balls=('ball','count')).reset_index()
    pp_bat=pp_bat[pp_bat['balls']>=100]
    pp_bat['sr']=(pp_bat['runs']/pp_bat['balls']*100).round(1)
    pp_bat=pp_bat.sort_values('runs',ascending=False).head(10).reset_index(drop=True)

    # 3. Caught dismissals by bowler
    caught_df=deliveries[(deliveries['is_wicket']==True)&(deliveries['dismissal_kind']=='caught')]
    caught_bowl=caught_df.groupby('bowler')['is_wicket'].count().reset_index()
    caught_bowl.columns=['bowler','caught_wickets']
    caught_bowl=caught_bowl.sort_values('caught_wickets',ascending=False).head(10).reset_index(drop=True)

    return (rf,le,ve,matches,tw,tm2,acc,bat,bowl,oc,pc,season_avg,td,team_bnd,
            deliveries,players,bat_hand_stats,bowl_type_stats,phase_stats,
            dismissal_stats,bowl_style_stats,city_stats,allrounders_df,arm_matchup,
            death_bowlers,pp_bat,caught_bowl,city_team_wins)

def predict_match(rf,le,ve,matches,tw,tm2,t1,t2,venue,t1_toss,dec):
    if t1 not in le.classes_ or t2 not in le.classes_: return None,None,None,0,0,0,0
    wr1=tw.get(t1,0)/tm2.get(t1,1); wr2=tw.get(t2,0)/tm2.get(t2,1)
    t1m=matches[(matches['team1']==t1)|(matches['team2']==t1)].tail(15)
    t2m=matches[(matches['team1']==t2)|(matches['team2']==t2)].tail(15)
    f1=(t1m['winner']==t1).mean(); f2=(t2m['winner']==t2).mean()
    t1m5=matches[(matches['team1']==t1)|(matches['team2']==t1)].tail(5)
    t2m5=matches[(matches['team1']==t2)|(matches['team2']==t2)].tail(5)
    f1_5=(t1m5['winner']==t1).mean(); f2_5=(t2m5['winner']==t2).mean()
    h2h=matches[((matches['team1']==t1)&(matches['team2']==t2))|((matches['team1']==t2)&(matches['team2']==t1))]
    h2hwr=(h2h['winner']==t1).sum()/len(h2h) if len(h2h)>0 else 0.5
    h1=int((h2h['winner']==t1).sum()); h2=int((h2h['winner']==t2).sum())
    v_t1=matches[(matches['venue']==venue)&((matches['team1']==t1)|(matches['team2']==t1))]
    v_t2=matches[(matches['venue']==venue)&((matches['team1']==t2)|(matches['team2']==t2))]
    vwr1=(v_t1['winner']==t1).mean() if len(v_t1)>=3 else 0.5
    vwr2=(v_t2['winner']==t2).mean() if len(v_t2)>=3 else 0.5
    toss_win_bat=1 if dec=='bat' else 0
    X=[[int(t1_toss),toss_win_bat,h2hwr,f1-f2,f1_5-f2_5,vwr1-vwr2,wr1-wr2,2026,1.0,f1,f2,f1_5,f2_5,wr1,wr2]]
    prob=rf.predict_proba(X)[0]; winner=t1 if prob[1]>prob[0] else t2
    return winner,round(prob[1]*100,1),round(prob[0]*100,1),h1,h2,round(f1*100),round(f2*100)

def fig_to_b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor=fig.get_facecolor()); buf.seek(0)
    return "data:image/png;base64,"+base64.b64encode(buf.read()).decode()

BG='#080f1e'; CARD='#0c1828'; GRID='#1a2a40'; TEXT='#ccd'

with st.spinner("🏏 Loading model & data (2008–2026)..."):
    (rf,le,ve,matches,tw,tm2,acc,bat,bowl,oc,pc,season_avg,td,team_bnd,
     deliveries,players,bat_hand_stats,bowl_type_stats,phase_stats,
     dismissal_stats,bowl_style_stats,city_stats,allrounders_df,arm_matchup,
     death_bowlers,pp_bat,caught_bowl,city_team_wins)=load_and_train()

# ── NAV BAR ──
st.markdown('<div class="nav-bar"><span class="nav-label">⚡ Navigate</span></div>',unsafe_allow_html=True)
nav_cols=st.columns(8)
nav_items=[("predict","⚡ Predict"),("teams","👥 Teams"),("players","🏏 Players"),
           ("style","🎯 Style"),("deep","🔬 Deep"),("charts","📊 Charts"),("h2h","⚔️ H2H"),("about","ℹ️ About")]
for col,(key,label) in zip(nav_cols,nav_items):
    with col:
        if st.button(label,key=f"nav_{key}",use_container_width=True):
            st.session_state.page=key; st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════
if st.session_state.page=='predict':
    st.markdown('<div class="page-title">IPL MATCH <span>PREDICTOR</span> ⚡</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Predict. Analyze. Win. — (2008–2026)</div>',unsafe_allow_html=True)
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown('<div class="divider">SELECT TEAMS</div>',unsafe_allow_html=True)
    c1,cv,c2=st.columns([5,1,5])
    with c1: team1=st.selectbox("Team 1",ACTIVE_TEAMS,index=0,key="t1",label_visibility="collapsed")
    with c2:
        t2_opts=[t for t in ACTIVE_TEAMS if t!=team1]
        team2=st.selectbox("Team 2",t2_opts,index=1,key="t2",label_visibility="collapsed")
    t1i,t2i=TEAM_INFO[team1],TEAM_INFO[team2]
    c1,cv,c2=st.columns([5,1,5])
    with c1:
        st.markdown(f'<div class="team-card" style="border:2px solid {t1i["color"]};background:{t1i["bg"]}"><img src="{t1i["logo"]}" width="72" height="72" style="border-radius:50%;box-shadow:0 0 16px {t1i["color"]}55"><div style="font-size:12px;font-weight:800;color:#fff;margin-top:8px">{team1.upper()}</div><div style="font-size:10px;color:#607080;margin-top:3px">{t1i["abbr"]}</div></div>',unsafe_allow_html=True)
    with cv:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;padding-top:1.5rem">
          <svg width="60" height="110" viewBox="0 0 60 110" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#FFE066"/>
                <stop offset="100%" stop-color="#C8861A"/>
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2.5" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>
            <!-- boundary arc -->
            <ellipse cx="30" cy="98" rx="28" ry="8" fill="none" stroke="#2a5a2a" stroke-width="1.2" stroke-dasharray="4,4" opacity="0.6"/>
            <!-- pitch strip -->
            <rect x="22" y="62" width="16" height="36" rx="2" fill="#8a6a1a" opacity="0.18"/>
            <!-- stump 1 -->
            <rect x="14" y="35" width="5" height="56" rx="2" fill="url(#sg)" filter="url(#glow)"/>
            <!-- stump 2 -->
            <rect x="27.5" y="35" width="5" height="56" rx="2" fill="url(#sg)" filter="url(#glow)"/>
            <!-- stump 3 -->
            <rect x="41" y="35" width="5" height="56" rx="2" fill="url(#sg)" filter="url(#glow)"/>
            <!-- bail 1 -->
            <rect x="12" y="33" width="16" height="5" rx="2.5" fill="url(#sg)" filter="url(#glow)"/>
            <!-- bail 2 -->
            <rect x="32" y="33" width="16" height="5" rx="2.5" fill="url(#sg)" filter="url(#glow)"/>
            <!-- VS text -->
            <text x="30" y="22" text-anchor="middle" font-family="Arial Black" font-size="13" font-weight="900" fill="#ffffff" opacity="0.9" letter-spacing="2">VS</text>
          </svg>
        </div>""",unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="team-card" style="border:2px solid {t2i["color"]};background:{t2i["bg"]}"><img src="{t2i["logo"]}" width="72" height="72" style="border-radius:50%;box-shadow:0 0 16px {t2i["color"]}55"><div style="font-size:12px;font-weight:800;color:#fff;margin-top:8px">{team2.upper()}</div><div style="font-size:10px;color:#607080;margin-top:3px">{t2i["abbr"]}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    d1,d2,d3,d4=st.columns(4)
    with d1: venue=st.selectbox("🏟️ Venue",VENUES,key="venue")
    with d2: st.date_input("📅 Date",key="mdate")
    with d3: toss_winner=st.radio("🪙 Toss winner",[team1,team2],key="tw")
    with d4: toss_decision=st.radio("🏏 Decision",["bat","field"],key="td_val")
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("⚡  Predict Match Outcome",use_container_width=True):
        t1_toss=1 if toss_winner==team1 else 0
        winner,p1,p2,h1,h2,f1p,f2p=predict_match(rf,le,ve,matches,tw,tm2,team1,team2,venue,t1_toss,toss_decision)
        if winner:
            wc=t1i['color'] if winner==team1 else t2i['color']
            h_lbl=f"{t1i['abbr']} leads {h1}–{h2}" if h1>=h2 else f"{t2i['abbr']} leads {h2}–{h1}"
            vs=venue.split(',')[0]
            st.markdown(f"""<div class="card">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
                    <span style="font-size:11px;font-weight:700;color:#607080;letter-spacing:1px">📈 PREDICTION RESULT</span>
                    <span class="ai-badge">AI POWERED · 2026 DATA</span>
                </div>
                <div class="result-winner" style="color:{wc}">{winner.upper()}</div>
                <div class="result-sub">LIKELY TO WIN</div>
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
                    <span style="font-size:16px;font-weight:800;color:{t1i['color']};min-width:50px">{p1}%</span>
                    <div class="prob-bar-outer" style="flex:1">
                        <div style="width:{p1}%;background:{t1i['color']};height:100%;border-radius:5px 0 0 5px"></div>
                        <div style="width:{p2}%;background:{t2i['color']};height:100%;border-radius:0 5px 5px 0"></div>
                    </div>
                    <span style="font-size:16px;font-weight:800;color:{t2i['color']};min-width:50px;text-align:right">{p2}%</span>
                </div>
                <div class="prob-labels"><span style="color:{t1i['color']}">{t1i['abbr']}</span><span>WIN PROBABILITY</span><span style="color:{t2i['color']}">{t2i['abbr']}</span></div>
            </div>
            <div class="card">
                <div class="divider">FACTORS CONSIDERED</div>
                <div class="factors-grid">
                    <div class="factor-item"><div class="factor-icon">📈</div><div class="factor-name">Team form</div><div class="factor-sub">{t1i['abbr']}: {f1p}%<br>{t2i['abbr']}: {f2p}%</div></div>
                    <div class="factor-item"><div class="factor-icon">⚔️</div><div class="factor-name">Head to head</div><div class="factor-sub">{h_lbl}</div></div>
                    <div class="factor-item"><div class="factor-icon">🏟️</div><div class="factor-name">Venue stats</div><div class="factor-sub">{vs}</div></div>
                    <div class="factor-item"><div class="factor-icon">📊</div><div class="factor-name">Win rate</div><div class="factor-sub">Historical record</div></div>
                    <div class="factor-item"><div class="factor-icon">🪙</div><div class="factor-name">Toss impact</div><div class="factor-sub">Low ~50.8%</div></div>
                </div>
                <p style="font-size:9px;color:#304050;margin-top:.875rem;text-align:center">Disclaimer: Based on statistical analysis. Does not guarantee actual results.</p>
            </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: TEAMS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=='teams':
    st.markdown('<div class="page-title">👥 <span>Teams</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">All IPL franchises — updated 2008–2026 performance</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    for col,lbl,val,sub,color in [
        (m1,"Total matches",len(matches),"2008–2026","#1a6fd4"),
        (m2,"Active teams",10,"IPL 2026","#f4820a"),
        (m3,"Best model acc","54.7%","RF+GB Ensemble","#1c8b6e"),
        (m4,"Toss effect","50.8%","wins match","#a855f7")]:
        with col: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{val}</div><div class="metric-lbl">{lbl}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    cols=st.columns(5)
    for i,(name,info) in enumerate(TEAM_INFO.items()):
        w=tw.get(name,0); t=tm2.get(name,1); wr=round(w/t*100)
        bnd=team_bnd[team_bnd['team']==name]
        sx=int(bnd['sixes'].values[0]) if len(bnd)>0 else 0
        with cols[i%5]:
            st.markdown(f"""<div class="card" style="border:1.5px solid {info['color']}44;text-align:center;padding:1rem .5rem">
                <img src="{info['logo_sm']}" width="52" height="52" style="border-radius:50%;box-shadow:0 0 12px {info['color']}44">
                <div style="font-size:11px;font-weight:800;color:#fff;line-height:1.3;margin-top:8px">{name}</div>
                <div style="font-size:10px;color:#607080;margin-top:2px">{info['abbr']}</div>
                <div style="margin-top:10px"><div style="font-size:18px;font-weight:900;color:{info['color']}">{wr}%</div><div style="font-size:9px;color:#607080">win rate</div></div>
                <div style="display:flex;justify-content:space-around;margin-top:10px;padding-top:8px;border-top:.5px solid #1a2a40">
                    <div><div style="font-size:12px;font-weight:700;color:#fff">{w}</div><div style="font-size:9px;color:#607080">wins</div></div>
                    <div><div style="font-size:12px;font-weight:700;color:#f4820a">{sx}</div><div style="font-size:9px;color:#607080">sixes</div></div>
                    <div><div style="font-size:12px;font-weight:700;color:#fff">{t}</div><div style="font-size:9px;color:#607080">played</div></div>
                </div></div>""",unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:.5rem">',unsafe_allow_html=True)
    st.markdown('<div class="divider">ALL-TIME WIN RANKINGS (2008–2026)</div>',unsafe_allow_html=True)
    rows=sorted([{"_n":n,"abbr":TEAM_INFO[n]["abbr"],"color":TEAM_INFO[n]["color"],"w":tw.get(n,0),"t":tm2.get(n,1)} for n in ACTIVE_TEAMS],key=lambda x:-x["w"])
    th='<table class="stat-table"><thead><tr><th>#</th><th>Team</th><th>Wins</th><th>Matches</th><th>Win rate</th><th>Loss</th></tr></thead><tbody>'
    for i,r in enumerate(rows,1):
        wr2=round(r['w']/r['t']*100); loss=r['t']-r['w']
        th+=f'<tr><td style="color:#607080">{i}</td><td><span style="color:{r["color"]};font-weight:700">{r["abbr"]}</span>  {r["_n"]}</td><td style="color:{r["color"]};font-weight:700">{r["w"]}</td><td style="color:#607080">{r["t"]}</td><td><span class="badge" style="background:{r["color"]}22;color:{r["color"]}">{wr2}%</span></td><td style="color:#607080">{loss}</td></tr>'
    th+='</tbody></table>'; st.markdown(th,unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: PLAYERS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=='players':
    st.markdown('<div class="page-title">🏏 <span>Player Stats</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">IPL 2008–2026 — batting, bowling, and milestone records</div>',unsafe_allow_html=True)
    tab=st.radio("View",["🏏 Batting","🎯 Bowling","🏆 Season Awards","🌟 Milestones"],horizontal=True,key="ptab",label_visibility="collapsed")
    if tab=="🏏 Batting":
        sort_by=st.radio("",["Total runs","Strike rate","Avg per innings","Sixes","Fours"],horizontal=True,key="bsort",label_visibility="collapsed")
        sort_map={"Total runs":"runs","Strike rate":"sr","Avg per innings":"avg","Sixes":"sixes","Fours":"fours"}
        min_balls={"Total runs":500,"Strike rate":400,"Avg per innings":500,"Sixes":300,"Fours":300}
        col_key=sort_map[sort_by]; min_b=min_balls[sort_by]
        top=bat[bat['balls']>=min_b].sort_values(col_key,ascending=False).head(15).reset_index(drop=True)
        max_val=top[col_key].max()
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown(f'<div class="divider">TOP 15 — {sort_by.upper()}</div>',unsafe_allow_html=True)
        for i,row in top.iterrows():
            val=row[col_key]; pct=val/max_val*100
            disp=f"{val:.1f}" if col_key in ('sr','avg') else str(int(val))
            st.markdown(f"""<div class="player-row"><div class="player-rank">#{i+1}</div>
                <div style="flex:1"><div class="player-name">{row['batter']}</div>
                <div style="background:#1a2a40;height:4px;border-radius:2px;margin-top:5px;width:{pct:.0f}%"><div style="height:100%;border-radius:2px;background:#1a6fd4;width:100%"></div></div></div>
                <div style="text-align:right"><div class="player-val" style="color:#1a6fd4">{disp}</div>
                <div style="font-size:9px;color:#607080">{int(row['runs'])} runs · {int(row['balls'])} balls</div></div></div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    elif tab=="🎯 Bowling":
        sort_by=st.radio("",["Wickets","Economy","Bowling avg"],horizontal=True,key="bwsort",label_visibility="collapsed")
        sort_map={"Wickets":"wickets","Economy":"economy","Bowling avg":"avg"}
        asc_map={"Wickets":False,"Economy":True,"Bowling avg":True}
        col_key=sort_map[sort_by]; asc=asc_map[sort_by]
        top_b=bowl[bowl['balls']>=300].dropna(subset=['avg']).sort_values(col_key,ascending=asc).head(15).reset_index(drop=True)
        max_val=top_b[col_key].max()
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown(f'<div class="divider">TOP 15 — {sort_by.upper()}</div>',unsafe_allow_html=True)
        for i,row in top_b.iterrows():
            val=row[col_key]; disp=f"{val:.2f}" if col_key in ('economy','avg') else str(int(val))
            bar_pct=(val/max_val*100) if not asc else ((max_val-val)/max_val*100+10)
            st.markdown(f"""<div class="player-row"><div class="player-rank">#{i+1}</div>
                <div style="flex:1"><div class="player-name">{row['bowler']}</div>
                <div style="background:#1a2a40;height:4px;border-radius:2px;margin-top:5px;width:{bar_pct:.0f}%"><div style="height:100%;border-radius:2px;background:#e63946;width:100%"></div></div></div>
                <div style="text-align:right"><div class="player-val" style="color:#e63946">{disp}</div>
                <div style="font-size:9px;color:#607080">{int(row['wickets'])} wkts · econ {row['economy']:.2f}</div></div></div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    elif tab=="🏆 Season Awards":
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="divider">🟠 ORANGE CAP — MOST RUNS</div>',unsafe_allow_html=True)
            th='<table class="stat-table"><thead><tr><th>Season</th><th>Batter</th><th>Runs</th></tr></thead><tbody>'
            for _,r in oc.sort_values('season',ascending=False).iterrows():
                th+=f'<tr><td style="color:#f4820a;font-weight:700">{r["season"]}</td><td style="color:#fff">{r["batter"]}</td><td style="color:#f4820a;font-weight:700">{int(r["batsman_runs"])}</td></tr>'
            st.markdown(th+'</tbody></table>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><div class="divider">💜 PURPLE CAP — MOST WICKETS</div>',unsafe_allow_html=True)
            th2='<table class="stat-table"><thead><tr><th>Season</th><th>Bowler</th><th>Wickets</th></tr></thead><tbody>'
            for _,r in pc.sort_values('season',ascending=False).iterrows():
                th2+=f'<tr><td style="color:#a855f7;font-weight:700">{r["season"]}</td><td style="color:#fff">{r["bowler"]}</td><td style="color:#a855f7;font-weight:700">{int(r["is_wicket"])}</td></tr>'
            st.markdown(th2+'</tbody></table>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
    elif tab=="🌟 Milestones":
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="divider">💥 MOST SIXES</div>',unsafe_allow_html=True)
            for i,r in bat.sort_values('sixes',ascending=False).head(10).reset_index(drop=True).iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["batter"]}</div><div class="player-val" style="color:#f4820a">{int(r["sixes"])} 6s</div></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><div class="divider">🔵 MOST FOURS</div>',unsafe_allow_html=True)
            for i,r in bat.sort_values('fours',ascending=False).head(10).reset_index(drop=True).iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["batter"]}</div><div class="player-val" style="color:#1a6fd4">{int(r["fours"])} 4s</div></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: STYLE ANALYTICS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=='style':
    st.markdown('<div class="page-title">🎯 <span>Style Analytics</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Left vs Right batters · Pace vs Spin · Full matchup matrix (2008–2026)</div>',unsafe_allow_html=True)

    total_p=len(players)
    left_n=int((players['bat_hand']=='Left').sum()); right_n=total_p-left_n
    pace_n=int((players['bowl_type']=='Pace').sum()); spin_n=int((players['bowl_type']=='Spin').sum())
    left_pct=round(left_n/total_p*100); right_pct=100-left_pct
    pace_pct=round(pace_n/total_p*100); spin_pct=round(spin_n/total_p*100)

    m1,m2,m3,m4=st.columns(4)
    for col,lbl,val,sub,color in [
        (m1,"Left-hand batters",f"{left_pct}%",f"{left_n} of {total_p}","#1a6fd4"),
        (m2,"Right-hand batters",f"{right_pct}%",f"{right_n} of {total_p}","#f4820a"),
        (m3,"Pace bowlers",f"{pace_pct}%",f"{pace_n} of {total_p}","#e63946"),
        (m4,"Spin bowlers",f"{spin_pct}%",f"{spin_n} of {total_p}","#1c8b6e")]:
        with col: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{val}</div><div class="metric-lbl">{lbl}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    stab=st.radio("View",["🤚 Bat Hand","🏏 Bowl Type","🔥 Bat vs Bowl Matchup","💪 Arm + Type Matchup","👤 Player Profile"],horizontal=True,key="stab",label_visibility="collapsed")

    if stab=="🤚 Bat Hand":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">LEFT-HAND vs RIGHT-HAND BATTER PERFORMANCE (ALL IPL DELIVERIES)</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #1a6fd4">
            <div class="insight-title" style="color:#1a6fd4">📊 What does this tell us?</div>
            <div class="insight-body">
            Compares how <b style="color:#fff">left-handed</b> and <b style="color:#fff">right-handed</b> batters perform across all 288K+ deliveries (2008–2026).<br>
            • <b style="color:#1a6fd4">Left-handers</b> score a slightly <b style="color:#fff">higher average per ball (1.297)</b> — they tend to be aggressive openers who take on pace early.<br>
            • <b style="color:#f4820a">Right-handers</b> face far more total deliveries (190K+ vs 98K) simply because <b style="color:#fff">73% of IPL players bat right-handed</b>.<br>
            • The strike rate edge for left-handers vs pace is a well-known T20 tactical advantage — left-arm pace angles differently, creating scoring gaps on the off-side.
            </div></div>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            for _,row in bat_hand_stats.iterrows():
                color='#1a6fd4' if row['batter_hand']=='Left' else '#f4820a'
                label='Left-Hand Batter' if row['batter_hand']=='Left' else 'Right-Hand Batter'
                st.markdown(f"""<div style="background:#07111e;border-radius:12px;padding:1.2rem;margin-bottom:10px;border-left:4px solid {color}">
                    <div style="font-size:14px;font-weight:800;color:{color}">{label}</div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px">
                        <div style="text-align:center"><div style="font-size:18px;font-weight:900;color:#fff">{int(row['total_runs']):,}</div><div style="font-size:9px;color:#607080">Total Runs</div></div>
                        <div style="text-align:center"><div style="font-size:18px;font-weight:900;color:{color}">{row['avg_runs']:.3f}</div><div style="font-size:9px;color:#607080">Avg / Ball</div></div>
                        <div style="text-align:center"><div style="font-size:18px;font-weight:900;color:#fff">{int(row['total_balls']):,}</div><div style="font-size:9px;color:#607080">Balls Faced</div></div>
                    </div></div>""",unsafe_allow_html=True)
        with c2:
            lhb=deliveries[deliveries['batter_hand']=='Left'].groupby('batter')['batsman_runs'].sum().reset_index().sort_values('batsman_runs',ascending=False).head(6).reset_index(drop=True)
            st.markdown('<div class="divider">🔵 TOP LEFT-HAND BATTERS</div>',unsafe_allow_html=True)
            for i,r in lhb.iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["batter"]}</div><div class="player-val" style="color:#1a6fd4">{int(r["batsman_runs"]):,}</div></div>',unsafe_allow_html=True)
            rhb=deliveries[deliveries['batter_hand']=='Right'].groupby('batter')['batsman_runs'].sum().reset_index().sort_values('batsman_runs',ascending=False).head(6).reset_index(drop=True)
            st.markdown('<div class="divider" style="margin-top:.75rem">🟠 TOP RIGHT-HAND BATTERS</div>',unsafe_allow_html=True)
            for i,r in rhb.iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["batter"]}</div><div class="player-val" style="color:#f4820a">{int(r["batsman_runs"]):,}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif stab=="🏏 Bowl Type":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">PACE vs SPIN BOWLING — EFFECTIVENESS COMPARISON</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #e63946">
            <div class="insight-title" style="color:#e63946">🎯 Pace vs Spin — What the data says</div>
            <div class="insight-body">
            • <b style="color:#e63946">Pace bowling</b> takes ~9,700 wickets total with a <b style="color:#fff">5.21% wicket rate</b> — faster throughput and more breakthroughs.<br>
            • <b style="color:#1c8b6e">Spin bowling</b> takes ~4,400 wickets but with <b style="color:#fff">lower economy (7.40 vs 7.87)</b> — spinners contain runs and work best in middle overs.<br>
            • <b style="color:#fff">Key takeaway:</b> Use pace for wickets, use spin to choke scoring. 
            Both types are needed for a balanced T20 attack.
            </div></div>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            for _,row in bowl_type_stats.iterrows():
                color='#e63946' if row['bowler_type']=='Pace' else '#1c8b6e'
                st.markdown(f"""<div style="background:#07111e;border-radius:12px;padding:1.2rem;margin-bottom:10px;border-left:4px solid {color}">
                    <div style="font-size:14px;font-weight:800;color:{color}">{row['bowler_type']} Bowling</div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px">
                        <div style="text-align:center"><div style="font-size:18px;font-weight:900;color:#fff">{int(row['total_wickets']):,}</div><div style="font-size:9px;color:#607080">Wickets</div></div>
                        <div style="text-align:center"><div style="font-size:18px;font-weight:900;color:{color}">{row['wicket_rate']:.2f}%</div><div style="font-size:9px;color:#607080">Wicket Rate</div></div>
                        <div style="text-align:center"><div style="font-size:18px;font-weight:900;color:#fff">{row['economy']:.2f}</div><div style="font-size:9px;color:#607080">Economy</div></div>
                    </div></div>""",unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="divider">🔴 TOP PACE WICKET TAKERS</div>',unsafe_allow_html=True)
            pw=deliveries[(deliveries['bowler_type']=='Pace')&(deliveries['is_wicket']==True)&(~deliveries['dismissal_kind'].isin(['run out','retired hurt']))].groupby('bowler')['is_wicket'].count().reset_index().sort_values('is_wicket',ascending=False).head(5).reset_index(drop=True)
            for i,r in pw.iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["bowler"]}</div><div class="player-val" style="color:#e63946">{int(r["is_wicket"])} wkts</div></div>',unsafe_allow_html=True)
            st.markdown('<div class="divider" style="margin-top:.75rem">🟢 TOP SPIN WICKET TAKERS</div>',unsafe_allow_html=True)
            sw=deliveries[(deliveries['bowler_type']=='Spin')&(deliveries['is_wicket']==True)&(~deliveries['dismissal_kind'].isin(['run out','retired hurt']))].groupby('bowler')['is_wicket'].count().reset_index().sort_values('is_wicket',ascending=False).head(5).reset_index(drop=True)
            for i,r in sw.iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["bowler"]}</div><div class="player-val" style="color:#1c8b6e">{int(r["is_wicket"])} wkts</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif stab=="🔥 Bat vs Bowl Matchup":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">BATTER HAND vs BOWLER TYPE — HEAD-TO-HEAD MATCHUP</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #f4820a">
            <div class="insight-title" style="color:#f4820a">🔥 Reading the Matchup Table</div>
            <div class="insight-body">
            This shows how each batter-hand vs bowler-type combination performs across all IPL history:<br><br>
            📌 <b style="color:#1a6fd4">Left vs Pace:</b> SR <b style="color:#fff">131.7</b> — left-handers are most aggressive vs pace, creating off-side scoring gaps.<br>
            📌 <b style="color:#f4820a">Right vs Pace:</b> SR <b style="color:#fff">130.8</b> but highest wicket% <b style="color:#e63946">5.35%</b> — right-handers face more pace and lose more wickets.<br>
            📌 <b style="color:#1a6fd4">Left vs Spin:</b> SR <b style="color:#fff">125.7</b> — spinners contain left-handers better; angle difficult to score freely.<br>
            📌 <b style="color:#f4820a">Right vs Spin:</b> SR <b style="color:#fff">122.0</b>, wicket% <b style="color:#e63946">4.6%</b> — off-spin to right-handers is most economical for spinners.<br><br>
            <b style="color:#fff">Why no spinner on right-hand top scorer list?</b> Right-handers face more deliveries overall, but spinners concede fewer runs per ball to them — making spinners the preferred weapon vs right-hand bats in middle overs.
            </div></div>""",unsafe_allow_html=True)
        matchup=deliveries[deliveries['bowler_type']!='Unknown'].groupby(['batter_hand','bowler_type']).agg(
            avg_runs=('batsman_runs','mean'),wicket_rate=('is_wicket','mean'),total_balls=('batsman_runs','count')).reset_index()
        matchup['sr']=(matchup['avg_runs']*100).round(1)
        matchup['wicket_pct']=(matchup['wicket_rate']*100).round(2)
        th='<table class="stat-table"><thead><tr><th>Batter Hand</th><th>Bowler Type</th><th>Avg/Ball</th><th>Strike Rate</th><th>Wicket %</th><th>Total Balls</th></tr></thead><tbody>'
        for _,r in matchup.iterrows():
            hc='#1a6fd4' if r['batter_hand']=='Left' else '#f4820a'
            bc='#e63946' if r['bowler_type']=='Pace' else '#1c8b6e'
            th+=f'<tr><td><span class="badge" style="background:{hc}22;color:{hc}">{r["batter_hand"]}</span></td><td><span class="badge" style="background:{bc}22;color:{bc}">{r["bowler_type"]}</span></td><td style="color:#fff;font-weight:700">{r["avg_runs"]:.3f}</td><td style="color:#f4820a;font-weight:700">{r["sr"]}</td><td style="color:#e63946">{r["wicket_pct"]}%</td><td style="color:#607080">{int(r["total_balls"]):,}</td></tr>'
        st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif stab=="💪 Arm + Type Matchup":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">BOWLING ARM + TYPE vs BATTER HAND — DEEP MATCHUP MATRIX</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #a855f7">
            <div class="insight-title" style="color:#a855f7">💪 Deep Matchup — Arm × Type × Hand</div>
            <div class="insight-body">
            The deepest matchup table — combining <b style="color:#fff">bowling arm</b> (Left/Right), <b style="color:#fff">bowling type</b> (Pace/Spin) and <b style="color:#fff">batter handedness</b>:<br><br>
            📌 <b style="color:#e63946">Left arm Pace vs Right hand:</b> highest wicket% <b style="color:#fff">5.41%</b> — the classic "round the wicket" angle is hardest to play.<br>
            📌 <b style="color:#1a6fd4">Left arm Spin vs Left hand:</b> SR <b style="color:#fff">137.6</b> — left-handers love left-arm spin; angle opens up the leg side.<br>
            📌 <b style="color:#1c8b6e">Left arm Spin vs Right hand:</b> SR only <b style="color:#fff">119.0</b> — left-arm spin closes the angle for right-handers; most restrictive combo.<br>
            📌 <b style="color:#f4820a">Right arm Spin vs Right hand:</b> SR <b style="color:#fff">123.6</b>, lowest wicket% — off-spin to right-handers is easiest to play (natural angle).
            </div></div>""",unsafe_allow_html=True)
        th='<table class="stat-table"><thead><tr><th>Bowl Arm</th><th>Bowl Type</th><th>Bat Hand</th><th>Avg/Ball</th><th>SR</th><th>Wicket %</th><th>Balls</th></tr></thead><tbody>'
        for _,row in arm_matchup.sort_values('wicket_pct',ascending=False).iterrows():
            ac='#1a6fd4' if row['bowler_arm']=='Left' else '#f4820a'
            tc='#e63946' if row['bowler_type']=='Pace' else '#1c8b6e'
            hc='#a855f7' if row['batter_hand']=='Left' else '#00b4d8'
            th+=f'<tr><td><span class="badge" style="background:{ac}22;color:{ac}">{row["bowler_arm"]}</span></td><td><span class="badge" style="background:{tc}22;color:{tc}">{row["bowler_type"]}</span></td><td><span class="badge" style="background:{hc}22;color:{hc}">{row["batter_hand"]}</span></td><td style="color:#fff;font-weight:700">{row["avg_runs"]:.3f}</td><td style="color:#f4820a;font-weight:700">{row["sr"]}</td><td style="color:#e63946">{row["wicket_pct"]}%</td><td style="color:#607080">{int(row["balls"]):,}</td></tr>'
        st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif stab=="👤 Player Profile":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">🔍 SEARCH PLAYER</div>',unsafe_allow_html=True)
        search=st.text_input("",placeholder="e.g. V Kohli, MS Dhoni, JJ Bumrah",label_visibility="collapsed")
        if search:
            p_match=players[players['player_name'].str.contains(search,case=False,na=False)]
            if len(p_match)>0:
                for _,p in p_match.head(3).iterrows():
                    b_s=bat[bat['batter']==p['player_name']]; bw_s=bowl[bowl['bowler']==p['player_name']]
                    bc='#1a6fd4' if p['bat_hand']=='Left' else '#f4820a'
                    bwc='#e63946' if p['bowl_type']=='Pace' else '#1c8b6e'
                    st.markdown(f"""<div style="background:#07111e;border-radius:12px;padding:1.2rem;margin-bottom:10px;border:.5px solid #1a2a40">
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
                            <div style="width:44px;height:44px;border-radius:50%;background:#1a2a40;display:flex;align-items:center;justify-content:center;font-size:20px">🏏</div>
                            <div><div style="font-size:15px;font-weight:800;color:#fff">{p['player_full_name']}</div>
                            <div style="font-size:11px;color:#607080">{p['player_name']}</div></div>
                        </div>
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
                            <span class="badge" style="background:{bc}22;color:{bc}">🤚 {p['bat_hand']} Hand Bat</span>
                            <span class="badge" style="background:{bwc}22;color:{bwc}">🎯 {p['bowl_style'] if pd.notna(p.get('bowl_style','')) else 'N/A'}</span>
                        </div>""",unsafe_allow_html=True)
                    if len(b_s)>0:
                        r=b_s.iloc[0]
                        st.markdown(f"""<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px">
                            <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:16px;font-weight:900;color:#1a6fd4">{int(r['runs']):,}</div><div style="font-size:9px;color:#607080">Runs</div></div>
                            <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:16px;font-weight:900;color:#f4820a">{r['sr']}</div><div style="font-size:9px;color:#607080">Strike Rate</div></div>
                            <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:16px;font-weight:900;color:#1c8b6e">{int(r['sixes'])}</div><div style="font-size:9px;color:#607080">Sixes</div></div>
                            <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:16px;font-weight:900;color:#a855f7">{int(r['fours'])}</div><div style="font-size:9px;color:#607080">Fours</div></div>
                        </div>""",unsafe_allow_html=True)
                    if len(bw_s)>0 and bw_s.iloc[0]['wickets']>0:
                        r=bw_s.iloc[0]
                        st.markdown(f"""<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">
                            <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:16px;font-weight:900;color:#e63946">{int(r['wickets'])}</div><div style="font-size:9px;color:#607080">Wickets</div></div>
                            <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:16px;font-weight:900;color:#f4820a">{r['economy']}</div><div style="font-size:9px;color:#607080">Economy</div></div>
                            <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:16px;font-weight:900;color:#fff">{r['avg'] if pd.notna(r['avg']) else 'N/A'}</div><div style="font-size:9px;color:#607080">Bowl Avg</div></div>
                        </div>""",unsafe_allow_html=True)
                    st.markdown('</div>',unsafe_allow_html=True)
            else: st.info("No player found. Try a different name.")
        else: st.markdown('<div style="color:#607080;font-size:13px;text-align:center;padding:2rem">Search any player to see their batting style, bowling style, and career stats.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: DEEP ANALYTICS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=='deep':
    st.markdown('<div class="page-title">🔬 <span>Deep Analytics</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Phase · Dismissals · Bowl Styles + Wicket Rate · Allrounders · Cities · Death Bowlers · PP Batters · Caught Leaders</div>',unsafe_allow_html=True)
    dtab=st.radio("",["⚡ Phase","🎯 Dismissals","🏏 Bowl Styles","🌟 Allrounders","🏙️ Cities","💀 Death Bowlers","🔓 PP Batters","🤝 Caught Leaders"],horizontal=True,key="dtab",label_visibility="collapsed")

    if dtab=="⚡ Phase":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">POWERPLAY vs MIDDLE vs DEATH OVERS ANALYSIS</div>',unsafe_allow_html=True)
        phase_colors={'Powerplay (0-5)':'#1a6fd4','Middle (6-14)':'#f4820a','Death (15-19)':'#e63946'}
        for _,row in phase_stats.iterrows():
            color=phase_colors.get(str(row['phase']),'#fff')
            st.markdown(f"""<div style="background:#07111e;border-radius:12px;padding:1.2rem;margin-bottom:10px;border-left:4px solid {color}">
                <div style="font-size:14px;font-weight:800;color:{color}">{row['phase']}</div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px">
                    <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:18px;font-weight:900;color:#fff">{int(row['runs']):,}</div><div style="font-size:9px;color:#607080">Total Runs</div></div>
                    <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:18px;font-weight:900;color:{color}">{row['run_rate']}</div><div style="font-size:9px;color:#607080">Run Rate</div></div>
                    <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:18px;font-weight:900;color:#fff">{int(row['wickets']):,}</div><div style="font-size:9px;color:#607080">Wickets</div></div>
                    <div style="text-align:center;background:#0c1828;border-radius:8px;padding:8px"><div style="font-size:18px;font-weight:900;color:#e63946">{row['wicket_rate']}%</div><div style="font-size:9px;color:#607080">Wicket Rate</div></div>
                </div></div>""",unsafe_allow_html=True)
        st.markdown("""<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:.5rem">
            <div style="background:#07111e;border-radius:10px;padding:1rem;border-left:3px solid #1a6fd4"><div style="color:#1a6fd4;font-weight:700;font-size:11px">⚡ POWERPLAY (0–5)</div><div style="color:#ccd;font-size:12px;margin-top:6px;line-height:1.8">Run rate: ~7.3<br>Lowest wicket risk<br>Only 2 fielders outside ring<br>Set the foundation here</div></div>
            <div style="background:#07111e;border-radius:10px;padding:1rem;border-left:3px solid #f4820a"><div style="color:#f4820a;font-weight:700;font-size:11px">🏏 MIDDLE OVERS (6–14)</div><div style="color:#ccd;font-size:12px;margin-top:6px;line-height:1.8">Run rate: ~7.3<br>Most total wickets taken<br>Spinners dominate here<br>Build partnerships</div></div>
            <div style="background:#07111e;border-radius:10px;padding:1rem;border-left:3px solid #e63946"><div style="color:#e63946;font-weight:700;font-size:11px">💀 DEATH OVERS (15–19)</div><div style="color:#ccd;font-size:12px;margin-top:6px;line-height:1.8">Run rate: ~9.0<br>Highest wicket rate ~8%<br>Most explosive phase<br>Yorkers & slower balls key</div></div>
        </div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif dtab=="🎯 Dismissals":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">HOW BATTERS GET OUT — ALL IPL HISTORY</div>',unsafe_allow_html=True)
        total_wk=dismissal_stats['count'].sum()
        colors=['#1a6fd4','#e63946','#f4820a','#1c8b6e','#a855f7','#00b4d8','#ff6b6b']
        for i,(_,row) in enumerate(dismissal_stats.iterrows()):
            pct=round(row['count']/total_wk*100,1); color=colors[i%len(colors)]
            st.markdown(f"""<div class="player-row"><div class="player-rank">#{i+1}</div>
                <div style="flex:1"><div class="player-name">{str(row['dismissal_kind']).title()}</div>
                <div style="background:#1a2a40;height:6px;border-radius:3px;margin-top:5px"><div style="width:{pct}%;height:100%;border-radius:3px;background:{color}"></div></div></div>
                <div style="text-align:right"><div class="player-val" style="color:{color}">{pct}%</div>
                <div style="font-size:9px;color:#607080">{int(row['count']):,} dismissals</div></div></div>""",unsafe_allow_html=True)
        st.markdown(f'<div style="background:#07111e;border-radius:10px;padding:1rem;margin-top:1rem;border:.5px solid #1a2a40;text-align:center"><div style="color:#607080;font-size:11px">TOTAL WICKETS ANALYZED</div><div style="font-size:24px;font-weight:900;color:#f4820a">{total_wk:,}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif dtab=="🏏 Bowl Styles":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">BOWLING STYLE — WICKETS & WICKET RATE PER BALL</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #f4820a">
            <div class="insight-title" style="color:#f4820a">📊 Wicket Count vs Wicket Rate</div>
            <div class="insight-body">
            <b style="color:#fff">Wicket Count</b> = total wickets taken (volume-dependent — bowlers who bowl more overs take more).<br>
            <b style="color:#fff">Wicket Rate %</b> = wickets per ball bowled (true effectiveness — how dangerous is each delivery?)<br><br>
            Right arm Medium leads in raw wickets because they bowl the most overs overall. But <b style="color:#fff">Right arm Fast has a higher wicket rate</b> — each delivery is more dangerous. Legbreak Googly (wrist spin) has a high wicket rate too, despite fewer balls bowled.
            </div></div>""",unsafe_allow_html=True)
        style_colors={'Right arm Fast':'#e63946','Right arm Medium':'#f4820a','Right arm Fast medium':'#ff6b6b','Right arm Offbreak':'#1c8b6e','Slow Left arm Orthodox':'#1a6fd4','Legbreak Googly':'#a855f7','Right arm Medium fast':'#f97316','Left arm Fast medium':'#00b4d8','Left arm Medium fast':'#06b6d4','Legbreak':'#8b5cf6'}
        mxw=bowl_style_stats['wickets'].max(); mxr=bowl_style_stats['wicket_rate'].max()
        th='<table class="stat-table"><thead><tr><th>Bowl Style</th><th>Wickets</th><th>Wicket Rate %</th><th>Balls</th></tr></thead><tbody>'
        for _,row in bowl_style_stats.iterrows():
            color=style_colors.get(str(row['bowl_style']),'#607080')
            wp=row['wickets']/mxw*100; rp=row['wicket_rate']/mxr*100 if mxr>0 else 0
            th+=f"""<tr>
                <td><span class="badge" style="background:{color}22;color:{color}">{row['bowl_style']}</span></td>
                <td><div style="display:flex;align-items:center;gap:6px"><div style="background:#1a2a40;height:5px;border-radius:3px;width:55px"><div style="width:{wp:.0f}%;height:100%;border-radius:3px;background:{color}"></div></div><span style="color:{color};font-weight:700">{int(row['wickets']):,}</span></div></td>
                <td><div style="display:flex;align-items:center;gap:6px"><div style="background:#1a2a40;height:5px;border-radius:3px;width:55px"><div style="width:{rp:.0f}%;height:100%;border-radius:3px;background:#e63946"></div></div><span style="color:#e63946;font-weight:700">{row['wicket_rate']}%</span></div></td>
                <td style="color:#607080">{int(row['balls']):,}</td></tr>"""
        st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif dtab=="🌟 Allrounders":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">TOP IPL ALLROUNDERS — 500+ RUNS & 30+ WICKETS</div>',unsafe_allow_html=True)
        th='<table class="stat-table"><thead><tr><th>#</th><th>Player</th><th>Runs</th><th>Wickets</th><th>All-round Score</th></tr></thead><tbody>'
        for i,(_,row) in enumerate(allrounders_df.iterrows(),1):
            score=round((row['runs']/100)+(row['wickets']*2))
            th+=f'<tr><td style="color:#607080">{i}</td><td style="color:#fff;font-weight:700">{row["player"]}</td><td style="color:#1a6fd4;font-weight:700">{int(row["runs"]):,}</td><td style="color:#e63946;font-weight:700">{int(row["wickets"])}</td><td><span class="badge" style="background:#f4820a22;color:#f4820a">{score}</span></td></tr>'
        st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif dtab=="🏙️ Cities":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">CITY-WISE IPL MATCH DISTRIBUTION</div>',unsafe_allow_html=True)
        mxm=city_stats['matches_played'].max()
        ccols=['#1a6fd4','#f4820a','#e63946','#1c8b6e','#a855f7','#00b4d8','#f97316','#8b5cf6','#06b6d4','#ff6b6b','#D4AF37','#C8A951']
        for i,(_,row) in enumerate(city_stats.iterrows()):
            pct=row['matches_played']/mxm*100; color=ccols[i%len(ccols)]
            st.markdown(f"""<div class="player-row"><div class="player-rank">#{i+1}</div>
                <div style="flex:1"><div class="player-name">🏙️ {row['city']}</div>
                <div style="background:#1a2a40;height:6px;border-radius:3px;margin-top:5px"><div style="width:{pct:.0f}%;height:100%;border-radius:3px;background:{color}"></div></div></div>
                <div class="player-val" style="color:{color}">{int(row['matches_played'])} matches</div></div>""",unsafe_allow_html=True)
        st.markdown('<div class="divider" style="margin-top:1rem">CITY-WISE TEAM WIN RATES</div>',unsafe_allow_html=True)
        sel_team=st.selectbox("Select a team",ACTIVE_TEAMS,key="city_team_sel")
        if sel_team in city_team_wins:
            ctw=city_team_wins[sel_team]; ti=TEAM_INFO[sel_team]
            th=f'<table class="stat-table"><thead><tr><th>City</th><th>Played</th><th>Wins</th><th>Win Rate</th></tr></thead><tbody>'
            for _,row in ctw.iterrows():
                th+=f'<tr><td style="color:#fff">🏙️ {row["city"]}</td><td style="color:#607080">{int(row["played"])}</td><td style="color:{ti["color"]};font-weight:700">{int(row["wins"])}</td><td><span class="badge" style="background:{ti["color"]}22;color:{ti["color"]}">{row["win_rate"]}%</span></td></tr>'
            st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif dtab=="💀 Death Bowlers":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">BEST DEATH OVERS BOWLERS (OVERS 15–19)</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #e63946">
            <div class="insight-title" style="color:#e63946">💀 What is death bowling?</div>
            <div class="insight-body">Overs 15–19 are the most high-pressure phase. Batters swing hard; bowlers must execute yorkers, slower balls and bouncers under extreme pressure. This table shows the <b style="color:#fff">top wicket-takers specifically in the death overs</b> (min 120 balls bowled in this phase). Both wicket count and wicket rate are shown.</div></div>""",unsafe_allow_html=True)
        mxw=death_bowlers['wickets'].max()
        th='<table class="stat-table"><thead><tr><th>#</th><th>Bowler</th><th>Wickets</th><th>Wicket Rate</th><th>Economy</th><th>Balls</th></tr></thead><tbody>'
        for i,(_,row) in enumerate(death_bowlers.iterrows(),1):
            bar=row['wickets']/mxw*100
            th+=f'<tr><td style="color:#607080">{i}</td><td style="color:#fff;font-weight:700">{row["bowler"]}</td><td><div style="display:flex;align-items:center;gap:6px"><div style="background:#1a2a40;height:5px;border-radius:3px;width:50px"><div style="width:{bar:.0f}%;height:100%;border-radius:3px;background:#e63946"></div></div><span style="color:#e63946;font-weight:700">{int(row["wickets"])}</span></div></td><td style="color:#a855f7;font-weight:700">{row["wicket_rate"]}%</td><td style="color:#f4820a">{row["economy"]}</td><td style="color:#607080">{int(row["balls"])}</td></tr>'
        st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif dtab=="🔓 PP Batters":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">BEST POWERPLAY BATTERS (OVERS 0–5)</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #1a6fd4">
            <div class="insight-title" style="color:#1a6fd4">⚡ Powerplay specialists</div>
            <div class="insight-body">The first 6 overs (0–5) restrict fielders to inside the 30-yard circle. Elite openers and top-order batters use this window to score freely. This table ranks batters by <b style="color:#fff">total runs scored in powerplay overs only</b> (min 100 balls faced in this phase). Strike rate color-coded: <span style="color:#1c8b6e">green ≥140</span>, <span style="color:#f4820a">orange ≥120</span>.</div></div>""",unsafe_allow_html=True)
        mxr=pp_bat['runs'].max()
        th='<table class="stat-table"><thead><tr><th>#</th><th>Batter</th><th>PP Runs</th><th>Balls</th><th>Strike Rate</th></tr></thead><tbody>'
        for i,(_,row) in enumerate(pp_bat.iterrows(),1):
            bar=row['runs']/mxr*100; src='#1c8b6e' if row['sr']>=140 else ('#f4820a' if row['sr']>=120 else '#607080')
            th+=f'<tr><td style="color:#607080">{i}</td><td style="color:#fff;font-weight:700">{row["batter"]}</td><td><div style="display:flex;align-items:center;gap:6px"><div style="background:#1a2a40;height:5px;border-radius:3px;width:50px"><div style="width:{bar:.0f}%;height:100%;border-radius:3px;background:#1a6fd4"></div></div><span style="color:#1a6fd4;font-weight:700">{int(row["runs"])}</span></div></td><td style="color:#607080">{int(row["balls"])}</td><td><span class="badge" style="background:{src}22;color:{src}">{row["sr"]}</span></td></tr>'
        st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    elif dtab=="🤝 Caught Leaders":
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="divider">BOWLERS WITH MOST CAUGHT DISMISSALS</div>',unsafe_allow_html=True)
        st.markdown("""<div class="insight-box" style="border-left:3px solid #00b4d8">
            <div class="insight-title" style="color:#00b4d8">🤝 What are caught dismissals?</div>
            <div class="insight-body">A caught dismissal happens when the batter edges, top-edges, or mistimes a shot and a fielder takes the catch. It's the most common wicket type in T20. Bowlers who generate <b style="color:#fff">edges (pace), top-edges (bouncers), and miscued drives (slower balls/spin)</b> lead this list. Columns also show what % of their total wickets were caught outs.</div></div>""",unsafe_allow_html=True)
        mxc=caught_bowl['caught_wickets'].max()
        th='<table class="stat-table"><thead><tr><th>#</th><th>Bowler</th><th>Caught Wickets</th><th>% of Total Wkts</th></tr></thead><tbody>'
        for i,(_,row) in enumerate(caught_bowl.iterrows(),1):
            bar=row['caught_wickets']/mxc*100
            twv=bowl[bowl['bowler']==row['bowler']]['wickets'].values
            twi=int(twv[0]) if len(twv)>0 else 1
            cpct=round(row['caught_wickets']/max(twi,1)*100)
            th+=f'<tr><td style="color:#607080">{i}</td><td style="color:#fff;font-weight:700">{row["bowler"]}</td><td><div style="display:flex;align-items:center;gap:6px"><div style="background:#1a2a40;height:5px;border-radius:3px;width:60px"><div style="width:{bar:.0f}%;height:100%;border-radius:3px;background:#00b4d8"></div></div><span style="color:#00b4d8;font-weight:700">{int(row["caught_wickets"])}</span></div></td><td><span class="badge" style="background:#00b4d822;color:#00b4d8">{cpct}%</span></td></tr>'
        st.markdown(th+'</tbody></table>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: CHARTS
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=='charts':
    st.markdown('<div class="page-title">📊 <span>Analytics Charts</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Visual insights from 18 seasons of IPL data (2008–2026)</div>',unsafe_allow_html=True)
    ctab=st.radio("",["Runs trend","Toss decisions","Team wins","Top batters","Top bowlers"],horizontal=True,key="ctab",label_visibility="collapsed")
    plt.rcParams.update({'font.family':'DejaVu Sans'})

    chart_info={
        "Runs trend":{"color":"#1a6fd4","icon":"📈","title":"Average Runs Per Match — Season-by-Season Trend",
            "body":"This line chart shows the <b style='color:#fff'>average total runs scored per match</b> each IPL season. It directly reflects how batting-friendly T20 has become over 18 years. Early seasons (2008–2012) averaged ~310 runs per match. By 2020–2026, that number has risen to ~340+. The upward trend is driven by better bats, smaller boundaries at some venues, aggressive batting techniques, and the impact player rule introduced in 2023. Average scores have risen <b style='color:#f4820a'>27%+ from 2009 to 2026</b>."},
        "Toss decisions":{"color":"#1c8b6e","icon":"🪙","title":"Bat First vs Field First — Toss Decision Trend",
            "body":"This grouped bar chart tracks how many captains chose to <b style='color:#fff'>bat first vs field (chase) after winning the toss</b>, season by season. In early seasons (2008–2012), batting first was preferred — teams wanted to post a big target. From ~2013 onward, <b style='color:#f4820a'>chasing became dominant</b> due to: (1) clearer target psychology, (2) dew advantage for batting teams in night matches, and (3) data showing chasers win more often. By 2020–2026, over 70% of toss-winners opt to field first."},
        "Team wins":{"color":"#f4820a","icon":"🏆","title":"All-Time Wins by Active IPL Team (2008–2026)",
            "body":"This horizontal bar chart ranks all 10 active IPL franchises by <b style='color:#fff'>total wins across all IPL seasons</b>. <b style='color:#D4AF37'>CSK</b> and <b style='color:#1E4E91'>MI</b> dominate due to consistency over 15+ seasons of play. <b style='color:#36495E'>Gujarat Titans</b> and <b style='color:#00C8FF'>Lucknow Super Giants</b> joined in 2022 and have fewer matches played. Win count alone doesn't measure dominance — win rate (wins/matches) is a better indicator which is shown in the Teams page. This chart shows franchise legacy and longevity."},
        "Top batters":{"color":"#1a6fd4","icon":"🏏","title":"Top 15 Run Scorers — All-Time IPL (2008–2026)",
            "body":"This chart lists the <b style='color:#fff'>15 highest run-scorers in IPL history</b> (min 500 balls faced). Players here are almost exclusively openers or #3 batters who bat across all phases. The chart reveals the massive gap between the elite few and the rest — consistent high scorers bat through entire innings. A player appearing here has played many seasons and batted at the top of the order consistently — it reflects both individual brilliance and team role."},
        "Top bowlers":{"color":"#e63946","icon":"🎯","title":"Top 15 Wicket Takers — All-Time IPL (2008–2026)",
            "body":"This chart shows the <b style='color:#fff'>15 highest wicket-takers in IPL history</b> (min 300 balls bowled). Spinners like <b style='color:#fff'>YS Chahal, SP Narine, PP Chawla</b> dominate the top because they bowl through the middle overs and can be used in powerplay too. Pace bowlers like <b style='color:#fff'>Bhuvneshwar Kumar, JJ Bumrah</b> appear despite bowling fewer overs per game because their wicket rate is high. Being in this chart requires sustained excellence across many seasons — no flash-in-the-pan players here."},
    }

    if ctab in chart_info:
        ci=chart_info[ctab]
        st.markdown(f"""<div class="insight-box" style="border-left:3px solid {ci['color']};margin-bottom:1rem">
            <div class="insight-title" style="color:{ci['color']}">{ci['icon']} {ci['title']}</div>
            <div class="insight-body">{ci['body']}</div></div>""",unsafe_allow_html=True)

    if ctab=="Runs trend":
        fig,ax=plt.subplots(figsize=(10,4),facecolor=BG); ax.set_facecolor(CARD)
        seasons=list(season_avg.index); vals=list(season_avg.values)
        ax.fill_between(seasons,vals,alpha=0.15,color='#1a6fd4')
        ax.plot(seasons,vals,color='#1a6fd4',linewidth=2.5,marker='o',markersize=5,markerfacecolor='#1a6fd4')
        for s,v in zip(seasons,vals): ax.annotate(f'{v:.0f}',(s,v),textcoords='offset points',xytext=(0,8),ha='center',fontsize=8,color='#8899aa')
        ax.set_title('Average total runs per match by season (2008–2026)',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=9)
        for sp in ax.spines.values(): sp.set_color(GRID); sp.set_linewidth(.5)
        ax.grid(axis='y',color=GRID,linewidth=.5,alpha=.5); fig.tight_layout()
        st.image(fig_to_b64(fig),use_container_width=True); plt.close()
    elif ctab=="Toss decisions":
        fig,ax=plt.subplots(figsize=(10,4),facecolor=BG); ax.set_facecolor(CARD)
        td2=td.reset_index(); seasons=td2['season'].tolist()
        bat_vals=(td2['bat'].tolist() if 'bat' in td2.columns else [0]*len(td2))
        field_vals=(td2['field'].tolist() if 'field' in td2.columns else [0]*len(td2))
        x=np.arange(len(seasons)); w=0.38
        ax.bar(x-w/2,bat_vals,w,label='Bat first',color='#1a6fd4',alpha=.85)
        ax.bar(x+w/2,field_vals,w,label='Field first',color='#1c8b6e',alpha=.85)
        ax.set_xticks(x); ax.set_xticklabels(seasons,rotation=45,fontsize=8,color='#607080')
        ax.tick_params(colors='#607080',labelsize=9)
        ax.set_title('Toss decision trend: bat vs field (2008–2026)',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.legend(facecolor=CARD,edgecolor=GRID,labelcolor=TEXT,fontsize=9)
        for sp in ax.spines.values(): sp.set_color(GRID); sp.set_linewidth(.5)
        ax.grid(axis='y',color=GRID,linewidth=.5,alpha=.5); fig.tight_layout()
        st.image(fig_to_b64(fig),use_container_width=True); plt.close()
    elif ctab=="Team wins":
        active_tw=[(n,tw.get(n,0),TEAM_INFO[n]['color']) for n in ACTIVE_TEAMS]; active_tw.sort(key=lambda x:-x[1])
        fig,ax=plt.subplots(figsize=(10,5),facecolor=BG); ax.set_facecolor(CARD)
        names=[TEAM_INFO[x[0]]['abbr'] for x in active_tw]; wins=[x[1] for x in active_tw]; colors=[x[2] for x in active_tw]
        bars=ax.barh(names,wins,color=colors,alpha=.85,height=.6)
        for bar,val in zip(bars,wins): ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,str(val),va='center',ha='left',color=TEXT,fontsize=10,fontweight='bold')
        ax.set_title('All-time wins by active team (2008–2026)',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=10)
        for sp in ax.spines.values(): sp.set_color(GRID); sp.set_linewidth(.5)
        ax.invert_yaxis(); ax.grid(axis='x',color=GRID,linewidth=.5,alpha=.5); ax.set_xlim(0,max(wins)+20); fig.tight_layout()
        st.image(fig_to_b64(fig),use_container_width=True); plt.close()
    elif ctab=="Top batters":
        top15=bat[bat['balls']>=500].sort_values('runs',ascending=False).head(15)
        fig,ax=plt.subplots(figsize=(10,6),facecolor=BG); ax.set_facecolor(CARD)
        bars=ax.barh(top15['batter'][::-1],top15['runs'][::-1],color='#1a6fd4',alpha=.85,height=.65)
        for bar,val in zip(bars,top15['runs'][::-1]): ax.text(bar.get_width()+30,bar.get_y()+bar.get_height()/2,f'{int(val):,}',va='center',ha='left',color=TEXT,fontsize=9,fontweight='bold')
        ax.set_title('Top 15 run scorers — IPL 2008–2026',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=9)
        for sp in ax.spines.values(): sp.set_color(GRID); sp.set_linewidth(.5)
        ax.grid(axis='x',color=GRID,linewidth=.5,alpha=.5); ax.set_xlim(0,max(top15['runs'])+500); fig.tight_layout()
        st.image(fig_to_b64(fig),use_container_width=True); plt.close()
    elif ctab=="Top bowlers":
        top15b=bowl[bowl['balls']>=300].sort_values('wickets',ascending=False).head(15)
        fig,ax=plt.subplots(figsize=(10,6),facecolor=BG); ax.set_facecolor(CARD)
        bars=ax.barh(top15b['bowler'][::-1],top15b['wickets'][::-1],color='#e63946',alpha=.85,height=.65)
        for bar,val in zip(bars,top15b['wickets'][::-1]): ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,str(int(val)),va='center',ha='left',color=TEXT,fontsize=9,fontweight='bold')
        ax.set_title('Top 15 wicket takers — IPL 2008–2026',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=9)
        for sp in ax.spines.values(): sp.set_color(GRID); sp.set_linewidth(.5)
        ax.grid(axis='x',color=GRID,linewidth=.5,alpha=.5); fig.tight_layout()
        st.image(fig_to_b64(fig),use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE: H2H
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=='h2h':
    st.markdown('<div class="page-title">⚔️ <span>Head to Head</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Compare any two teams — full rivalry breakdown (2008–2026)</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: ht1=st.selectbox("Team 1",ACTIVE_TEAMS,index=0,key="h1")
    with c2: ht2=st.selectbox("Team 2",[t for t in ACTIVE_TEAMS if t!=ht1],index=1,key="h2")
    t1i_h,t2i_h=TEAM_INFO[ht1],TEAM_INFO[ht2]
    h2h=matches[((matches['team1']==ht1)&(matches['team2']==ht2))|((matches['team1']==ht2)&(matches['team2']==ht1))]
    w1=int((h2h['winner']==ht1).sum()); w2=int((h2h['winner']==ht2).sum()); tot=len(h2h)
    if tot>0:
        p1=round(w1/tot*100); p2=100-p1
        bat_first=h2h[h2h['toss_decision']=='bat']; field_first=h2h[h2h['toss_decision']=='field']
        st.markdown(f"""<div class="card">
            <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:1rem;align-items:center;text-align:center;margin-bottom:1rem">
                <div><img src="{t1i_h['logo']}" width="64" height="64" style="border-radius:50%;box-shadow:0 0 16px {t1i_h['color']}55">
                    <div style="font-size:38px;font-weight:900;color:{t1i_h['color']};margin-top:6px">{w1}</div>
                    <div style="font-size:11px;font-weight:600;color:#aab;margin-top:2px">{ht1}</div></div>
                <div><div style="font-size:11px;font-weight:700;color:#607080">{tot}</div><div style="font-size:9px;color:#304050;margin-top:2px">MATCHES</div></div>
                <div><img src="{t2i_h['logo']}" width="64" height="64" style="border-radius:50%;box-shadow:0 0 16px {t2i_h['color']}55">
                    <div style="font-size:38px;font-weight:900;color:{t2i_h['color']};margin-top:6px">{w2}</div>
                    <div style="font-size:11px;font-weight:600;color:#aab;margin-top:2px">{ht2}</div></div>
            </div>
            <div class="prob-bar-outer" style="height:10px;margin-bottom:6px">
                <div style="width:{p1}%;background:{t1i_h['color']};height:100%;border-radius:5px 0 0 5px"></div>
                <div style="width:{p2}%;background:{t2i_h['color']};height:100%;border-radius:0 5px 5px 0"></div>
            </div>
            <div class="prob-labels">
                <span style="color:{t1i_h['color']};font-weight:700">{p1}%</span>
                <span>HEAD-TO-HEAD WIN %</span>
                <span style="color:{t2i_h['color']};font-weight:700">{p2}%</span>
            </div></div>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="divider">RECENT 8 MEETINGS</div>',unsafe_allow_html=True)
            recent=h2h[['date','venue','winner']].tail(8).sort_values('date',ascending=False)
            rh='<table class="stat-table"><thead><tr><th>Date</th><th>Venue</th><th>Winner</th></tr></thead><tbody>'
            for _,row in recent.iterrows():
                wi=TEAM_INFO.get(row['winner'],{"color":"#fff","abbr":str(row['winner'])[:3].upper()})
                rh+=f'<tr><td style="color:#607080;font-size:11px">{row["date"]}</td><td style="color:#8899aa;font-size:11px">{str(row["venue"]).split(",")[0][:20]}</td><td><span class="badge" style="background:{wi["color"]}22;color:{wi["color"]}">{wi["abbr"]}</span></td></tr>'
            st.markdown(rh+'</tbody></table>',unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card"><div class="divider">RIVALRY INSIGHTS</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
                    <div class="metric-card"><div class="metric-val" style="color:{t1i_h['color']};font-size:20px">{w1}</div><div class="metric-lbl">{t1i_h['abbr']} wins</div></div>
                    <div class="metric-card"><div class="metric-val" style="color:{t2i_h['color']};font-size:20px">{w2}</div><div class="metric-lbl">{t2i_h['abbr']} wins</div></div>
                </div>
                <div style="font-size:12px;color:#8899aa;line-height:2">
                    <div>🏏 Total matches: <b style="color:#fff">{tot}</b></div>
                    <div>🏟️ Bat-first matches: <b style="color:#fff">{len(bat_first)}</b></div>
                    <div>🌾 Field-first matches: <b style="color:#fff">{len(field_first)}</b></div>
                    <div>{'🟡' if w1>=w2 else '🔴'} Series leader: <b style="color:{t1i_h['color'] if w1>=w2 else t2i_h['color']}">{t1i_h['abbr'] if w1>=w2 else t2i_h['abbr']}</b></div>
                </div></div>""",unsafe_allow_html=True)
    else:
        st.info("No matches found between these two teams.")

# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif st.session_state.page=='about':
    st.markdown('<div class="page-title">ℹ️ <span>About</span></div>',unsafe_allow_html=True)

    total_m=len(matches)
    total_d=len(deliveries)
    total_p=len(players)
    acc_pct=round(acc*100,1)

    left_n=int((players['bat_hand']=='Left').sum())
    right_n=int((players['bat_hand']=='Right').sum())
    left_pct=round(left_n/total_p*100)
    right_pct=100-left_pct
    pace_n=int((players['bowl_type']=='Pace').sum())
    spin_n=int((players['bowl_type']=='Spin').sum())
    pace_pct=round(pace_n/total_p*100)
    spin_pct=round(spin_n/total_p*100)
    other_n=total_p-left_n-right_n
    other_bowl_n=total_p-pace_n-spin_n

    st.markdown(f"""
    <div class="card"><div class="divider">📋 PROJECT</div>
        <div style="color:#ccd;font-size:13px;line-height:2.1">
        <b style="color:#fff">IPL Analytics & Match Predictor</b> is an end-to-end data analytics and ML project
        built on IPL data <b style="color:#f4820a">2008–2026</b> covering
        <b style="color:#f4820a">{total_m:,} matches</b>, <b style="color:#f4820a">{total_d:,} deliveries</b>,
        and <b style="color:#f4820a">{total_p} players</b>.<br><br>
        Uses a <b style="color:#f4820a">Random Forest Classifier</b> with flip-team data augmentation.<br><br>
        <b style="color:#8899aa">Model accuracy on held-out test set:</b><br>
        🌲 <b style="color:#f4820a">Random Forest:</b> <b style="color:#e63946">~52.67%</b> &nbsp;|&nbsp;
        📈 <b style="color:#a855f7">Gradient Boosting:</b> <b style="color:#1c8b6e">~54.73%</b> &nbsp;|&nbsp;
        🤝 <b style="color:#1a6fd4">Ensemble (RF+GB):</b> <b style="color:#1c8b6e">~54.73%</b><br><br>
        T20 cricket is inherently unpredictable — player fitness, pitch conditions, dew, in-match momentum,
        and match-day form cannot be captured by pre-match statistics alone.
        <b style="color:#fff">~54–55% is a strong benchmark</b> for this class of problem and consistent with academic literature on T20 prediction.
        </div></div>

    <div class="card"><div class="divider">🧑‍🤝‍🧑 PLAYER DATASET BREAKDOWN (ACTUAL DATA)</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1rem">
            <div class="metric-card"><div class="metric-val" style="color:#1a6fd4">{left_pct}%</div><div class="metric-lbl">Left-hand batters</div><div class="metric-sub">{left_n} of {total_p} players</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#f4820a">{right_pct}%</div><div class="metric-lbl">Right-hand batters</div><div class="metric-sub">{right_n} of {total_p} players</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#e63946">{pace_pct}%</div><div class="metric-lbl">Pace bowlers</div><div class="metric-sub">{pace_n} of {total_p} players</div></div>
            <div class="metric-card"><div class="metric-val" style="color:#1c8b6e">{spin_pct}%</div><div class="metric-lbl">Spin bowlers</div><div class="metric-sub">{spin_n} of {total_p} players</div></div>
        </div>
        <div style="color:#607080;font-size:11px;text-align:center">Note: {other_n} players have unclassified batting hand; {other_bowl_n} have unclassified bowling type — all percentages are relative to {total_p} total players in the dataset.</div>
    </div>

    <div class="card"><div class="divider">🔍 KEY FINDINGS</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;color:#94a3b8;font-size:12px;line-height:2">
            <div>✅ Win rate differential is the <b style="color:#fff">strongest predictor</b></div>
            <div>✅ Toss outcome has <b style="color:#fff">near-zero impact (~50.8%)</b></div>
            <div>✅ Field-first preference <b style="color:#fff">grew 70%+ post-2013</b></div>
            <div>✅ Average match scores <b style="color:#fff">rose 27%+ from 2009 to 2026</b></div>
            <div>✅ Left-handers score <b style="color:#fff">slightly higher per ball vs pace</b></div>
            <div>✅ Pace takes <b style="color:#fff">~2.2× more wickets</b> than spin overall</div>
            <div>✅ Left arm pace vs right-hand bat = <b style="color:#fff">highest wicket rate (5.41%)</b></div>
            <div>✅ Death over run rate (~9.0) <b style="color:#fff">far exceeds powerplay (~7.3)</b></div>
            <div>✅ RF accuracy ~52–53%; Ensemble (RF+GB) reaches <b style="color:#fff">~54–55%</b></div>
            <div>✅ T20 unpredictability means <b style="color:#fff">54–55% is a strong ML benchmark</b></div>
        </div></div>

    <div class="card"><div class="divider">🛠️ TECH STACK</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px">
            {''.join([f'<span class="badge" style="background:{c[1]}22;color:{c[1]};padding:6px 14px;font-size:11px">{c[0]}</span>' for c in [("Python 3.11","#f4820a"),("Scikit-learn","#1a6fd4"),("Pandas","#e63946"),("Streamlit","#1c8b6e"),("Matplotlib","#a855f7"),("NumPy","#00b4d8"),("Random Forest","#f4820a"),("HuggingFace Datasets","#ffcc00")]])}
        </div></div>

    <div class="card"><div class="divider">👤 BUILT BY</div>
        <div style="color:#ccd;font-size:13px;line-height:2.2">
            <b style="color:#fff;font-size:15px">Shibang Maity</b><br>
            Computer Science, KIIT University · Final Year · 2026–2027<br>
            <span style="color:#607080">📎 github.com/shibangmaity</span><br>
            <span style="color:#607080">💼 linkedin.com/in/shibang-maity-865ba4304</span>
        </div></div>""",unsafe_allow_html=True)

st.markdown('<div class="footer">🏏 Built by Shibang Maity · IPL Data 2008–2026 · Random Forest + Player Style Analytics · KIIT University 2026</div>',unsafe_allow_html=True)

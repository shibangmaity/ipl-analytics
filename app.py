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
    'Dubai International Cricket Stadium'])

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

BG_SVG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjEwODAiIHZpZXdCb3g9IjAgMCAxOTIwIDEwODAiPgogIDxkZWZzPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwIiB5MT0iMCIgeDI9IjAiIHkyPSIxIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMDEwNDA4Ii8+PHN0b3Agb2Zmc2V0PSI0MCUiIHN0b3AtY29sb3I9IiMwMzA5MWEiLz48c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwNTBkMjAiLz48L2xpbmVhckdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJvdXRmaWVsZCIgY3g9IjUwJSIgY3k9Ijk1JSIgcj0iNjUlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMGE2MDI1IiBzdG9wLW9wYWNpdHk9IjAuNjAiLz48c3RvcCBvZmZzZXQ9IjQ1JSIgc3RvcC1jb2xvcj0iIzA1MmYxMCIgc3RvcC1vcGFjaXR5PSIwLjI4Ii8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDAwIiBzdG9wLW9wYWNpdHk9IjAiLz48L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJwaXRjaF9nbG93IiBjeD0iNTAlIiBjeT0iNTUlIiByPSI1MCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNjODkxM2EiIHN0b3Atb3BhY2l0eT0iMC4zNSIvPjxzdG9wIG9mZnNldD0iNzAlIiBzdG9wLWNvbG9yPSIjMDAwIiBzdG9wLW9wYWNpdHk9IjAiLz48L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJzdHVtcF9nbG93IiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNmZmUwOTAiIHN0b3Atb3BhY2l0eT0iMC41NSIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzAwMCIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD4KICAgIDxyYWRpYWxHcmFkaWVudCBpZD0iZmxfdGwiIGN4PSIwJSIgY3k9IjAlIiByPSIxMDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZmZlMDkwIiBzdG9wLW9wYWNpdHk9IjAuOTUiLz48c3RvcCBvZmZzZXQ9IjYlIiBzdG9wLWNvbG9yPSIjZmZiODMwIiBzdG9wLW9wYWNpdHk9IjAuNTUiLz48c3RvcCBvZmZzZXQ9IjI1JSIgc3RvcC1jb2xvcj0iI2ZmOGMwMCIgc3RvcC1vcGFjaXR5PSIwLjIwIi8+PHN0b3Agb2Zmc2V0PSI2MCUiIHN0b3AtY29sb3I9IiNmZjY2MDAiIHN0b3Atb3BhY2l0eT0iMC4wNiIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzAwMCIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD4KICAgIDxyYWRpYWxHcmFkaWVudCBpZD0iZmxfdHIiIGN4PSIxMDAlIiBjeT0iMCUiIHI9IjEwMCUiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNmZmUwOTAiIHN0b3Atb3BhY2l0eT0iMC45NSIvPjxzdG9wIG9mZnNldD0iNiUiIHN0b3AtY29sb3I9IiNmZmI4MzAiIHN0b3Atb3BhY2l0eT0iMC41NSIvPjxzdG9wIG9mZnNldD0iMjUlIiBzdG9wLWNvbG9yPSIjZmY4YzAwIiBzdG9wLW9wYWNpdHk9IjAuMjAiLz48c3RvcCBvZmZzZXQ9IjYwJSIgc3RvcC1jb2xvcj0iI2ZmNjYwMCIgc3RvcC1vcGFjaXR5PSIwLjA2Ii8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDAwIiBzdG9wLW9wYWNpdHk9IjAiLz48L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJiYWxsX2wiIGN4PSIwJSIgY3k9IjUyJSIgcj0iNDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjY2MxYTBhIiBzdG9wLW9wYWNpdHk9IjAuMzAiLz48c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwMDAiIHN0b3Atb3BhY2l0eT0iMCIvPjwvcmFkaWFsR3JhZGllbnQ+CiAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImJhbGxfciIgY3g9IjEwMCUiIGN5PSI1MiUiIHI9IjQwJSI+PHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iI2NjMWEwYSIgc3RvcC1vcGFjaXR5PSIwLjMwIi8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDAwIiBzdG9wLW9wYWNpdHk9IjAiLz48L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJjd2FzaCIgY3g9IjUwJSIgY3k9IjM4JSIgcj0iNTUlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjZmZhMDIwIiBzdG9wLW9wYWNpdHk9IjAuMTEiLz48c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwMDAiIHN0b3Atb3BhY2l0eT0iMCIvPjwvcmFkaWFsR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImJlYW1fdGwiIHgxPSIwIiB5MT0iMCIgeDI9IjAuNiIgeTI9IjEiPjxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiNmZmQwNjAiIHN0b3Atb3BhY2l0eT0iMC4yMiIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI2ZmZDA2MCIgc3RvcC1vcGFjaXR5PSIwIi8+PC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iYmVhbV90ciIgeDE9IjEiIHkxPSIwIiB4Mj0iMC40IiB5Mj0iMSI+PHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iI2ZmZDA2MCIgc3RvcC1vcGFjaXR5PSIwLjIyIi8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjZmZkMDYwIiBzdG9wLW9wYWNpdHk9IjAiLz48L2xpbmVhckdyYWRpZW50PgogICAgPGZpbHRlciBpZD0iZ2xvdyI+PGZlR2F1c3NpYW5CbHVyIHN0ZERldmlhdGlvbj0iMyIgcmVzdWx0PSJibHVyIi8+PGZlTWVyZ2U+PGZlTWVyZ2VOb2RlIGluPSJibHVyIi8+PGZlTWVyZ2VOb2RlIGluPSJTb3VyY2VHcmFwaGljIi8+PC9mZU1lcmdlPjwvZmlsdGVyPgogICAgPGZpbHRlciBpZD0ic2ciPjxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjIuNSIgcmVzdWx0PSJibHVyIi8+PGZlTWVyZ2U+PGZlTWVyZ2VOb2RlIGluPSJibHVyIi8+PGZlTWVyZ2VOb2RlIGluPSJTb3VyY2VHcmFwaGljIi8+PC9mZU1lcmdlPjwvZmlsdGVyPgogIDwvZGVmcz4KICA8cmVjdCB3aWR0aD0iMTkyMCIgaGVpZ2h0PSIxMDgwIiBmaWxsPSJ1cmwoI3NreSkiLz4KICA8ZyBmaWx0ZXI9InVybCgjZ2xvdykiPgogICAgPGNpcmNsZSBjeD0iMTgwIiBjeT0iNTUiIHI9IjEuMyIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NSIvPjxjaXJjbGUgY3g9IjM0MCIgY3k9IjMwIiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNjUiLz4KICAgIDxjaXJjbGUgY3g9IjUyMCIgY3k9Ijc1IiByPSIxLjEiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzUiLz48Y2lyY2xlIGN4PSI3MDAiIGN5PSI0MiIgcj0iMS40IiBmaWxsPSIjZmZlOGEwIiBvcGFjaXR5PSIwLjkwIi8+CiAgICA8Y2lyY2xlIGN4PSI5MDAiIGN5PSIyMiIgcj0iMC45IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjU1Ii8+PGNpcmNsZSBjeD0iMTEwMCIgY3k9IjY4IiByPSIxLjIiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzUiLz4KICAgIDxjaXJjbGUgY3g9IjEzMDAiIGN5PSIzOCIgcj0iMS4wIiBmaWxsPSIjZmZlOGEwIiBvcGFjaXR5PSIwLjg1Ii8+PGNpcmNsZSBjeD0iMTUwMCIgY3k9IjUyIiByPSIxLjMiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNjUiLz4KICAgIDxjaXJjbGUgY3g9IjE3MDAiIGN5PSIyOCIgcj0iMS4xIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjc1Ii8+PGNpcmNsZSBjeD0iMTg1NSIgY3k9IjYyIiByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNTUiLz4KICAgIDxjaXJjbGUgY3g9IjQ1MCIgY3k9IjE4IiByPSIwLjgiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNjAiLz48Y2lyY2xlIGN4PSIxMjAwIiBjeT0iMTUiIHI9IjEuMCIgZmlsbD0iI2ZmZThhMCIgb3BhY2l0eT0iMC43MCIvPgogIDwvZz4KICA8cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iMTkyMCIgaGVpZ2h0PSIxNzUiIGZpbGw9IiMwMTA4MTAiIG9wYWNpdHk9IjAuODgiLz4KICA8ZWxsaXBzZSBjeD0iMTYwIiBjeT0iMTcyIiByeD0iMjAwIiByeT0iNTUiIGZpbGw9IiMwMTA4MTAiIG9wYWNpdHk9IjAuOTUiLz4KICA8ZWxsaXBzZSBjeD0iNTIwIiBjeT0iMTY4IiByeD0iMjYwIiByeT0iNTIiIGZpbGw9IiMwMTA4MTAiIG9wYWNpdHk9IjAuOTIiLz4KICA8ZWxsaXBzZSBjeD0iOTYwIiBjeT0iMTYzIiByeD0iMjgwIiByeT0iNDgiIGZpbGw9IiMwMTA4MTAiIG9wYWNpdHk9IjAuOTUiLz4KICA8ZWxsaXBzZSBjeD0iMTQwMCIgY3k9IjE2OCIgcng9IjI2MCIgcnk9IjUyIiBmaWxsPSIjMDEwODEwIiBvcGFjaXR5PSIwLjkyIi8+CiAgPGVsbGlwc2UgY3g9IjE3NjAiIGN5PSIxNzIiIHJ4PSIyMDAiIHJ5PSI1NSIgZmlsbD0iIzAxMDgxMCIgb3BhY2l0eT0iMC45NSIvPgogIDxwb2x5Z29uIHBvaW50cz0iMCwwIDUwMCwxMDgwIDAsMTA4MCIgZmlsbD0idXJsKCNiZWFtX3RsKSIgb3BhY2l0eT0iMC4yOCIvPgogIDxwb2x5Z29uIHBvaW50cz0iMCwwIDMyMCwxMDgwIDAsOTAwIiBmaWxsPSJ1cmwoI2JlYW1fdGwpIiBvcGFjaXR5PSIwLjE0Ii8+CiAgPHBvbHlnb24gcG9pbnRzPSIxOTIwLDAgMTQyMCwxMDgwIDE5MjAsMTA4MCIgZmlsbD0idXJsKCNiZWFtX3RyKSIgb3BhY2l0eT0iMC4yOCIvPgogIDxwb2x5Z29uIHBvaW50cz0iMTkyMCwwIDE2MDAsMTA4MCAxOTIwLDkwMCIgZmlsbD0idXJsKCNiZWFtX3RyKSIgb3BhY2l0eT0iMC4xNCIvPgogIDxjaXJjbGUgY3g9IjM4IiBjeT0iMzgiIHI9IjcwIiBmaWxsPSJ1cmwoI2ZsX3RsKSIvPgogIDxjaXJjbGUgY3g9IjE4ODIiIGN5PSIzOCIgcj0iNzAiIGZpbGw9InVybCgjZmxfdHIpIi8+CiAgPGNpcmNsZSBjeD0iMzgiIGN5PSIzOCIgcj0iMjAiIGZpbGw9IiNmZmZiZTAiIG9wYWNpdHk9IjAuNjUiLz4KICA8Y2lyY2xlIGN4PSIxODgyIiBjeT0iMzgiIHI9IjIwIiBmaWxsPSIjZmZmYmUwIiBvcGFjaXR5PSIwLjY1Ii8+CiAgPGNpcmNsZSBjeD0iMzgiIGN5PSIzOCIgcj0iOCIgZmlsbD0iI2ZmZmZmZiIgb3BhY2l0eT0iMC45NSIvPgogIDxjaXJjbGUgY3g9IjE4ODIiIGN5PSIzOCIgcj0iOCIgZmlsbD0iI2ZmZmZmZiIgb3BhY2l0eT0iMC45NSIvPgogIDxyZWN0IHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjEwODAiIGZpbGw9InVybCgjb3V0ZmllbGQpIi8+CiAgPHJlY3Qgd2lkdGg9IjE5MjAiIGhlaWdodD0iMTA4MCIgZmlsbD0idXJsKCNiYWxsX2wpIi8+CiAgPHJlY3Qgd2lkdGg9IjE5MjAiIGhlaWdodD0iMTA4MCIgZmlsbD0idXJsKCNiYWxsX3IpIi8+CiAgPHJlY3Qgd2lkdGg9IjE5MjAiIGhlaWdodD0iMTA4MCIgZmlsbD0idXJsKCNjd2FzaCkiLz4KICA8cmVjdCB4PSI4NDgiIHk9IjMyMCIgd2lkdGg9IjIyNCIgaGVpZ2h0PSI0NjAiIHJ4PSIxMCIgZmlsbD0idXJsKCNwaXRjaF9nbG93KSIvPgogIDwhLS0gQ3JlYXNlIGdsb3cgaGFsb3MgLS0+CiAgPGVsbGlwc2UgY3g9Ijk2MCIgY3k9IjQ2OCIgcng9IjQ1IiByeT0iOSIgZmlsbD0idXJsKCNzdHVtcF9nbG93KSIgb3BhY2l0eT0iMC43NSIvPgogIDxlbGxpcHNlIGN4PSI5NjAiIGN5PSI1OTYiIHJ4PSI0NSIgcnk9IjkiIGZpbGw9InVybCgjc3R1bXBfZ2xvdykiIG9wYWNpdHk9IjAuNzUiLz4KICA8IS0tIEJhdHRpbmcgY3JlYXNlIGJyaWdodCBsaW5lIC0tPgogIDxsaW5lIHgxPSI4MzIiIHkxPSI0NjgiIHgyPSIxMDg4IiB5Mj0iNDY4IiBzdHJva2U9IiNmZmU4YTAiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLW9wYWNpdHk9IjAuNjAiLz4KICA8bGluZSB4MT0iODMyIiB5MT0iNTk2IiB4Mj0iMTA4OCIgeTI9IjU5NiIgc3Ryb2tlPSIjZmZlOGEwIiBzdHJva2Utd2lkdGg9IjMiIHN0cm9rZS1vcGFjaXR5PSIwLjYwIi8+CiAgPCEtLSBSZXR1cm4gY3JlYXNlIHRpY2tzIC0tPgogIDxsaW5lIHgxPSI4MzIiIHkxPSI0NjAiIHgyPSI4MzIiIHkyPSI0NzYiIHN0cm9rZT0iI2ZmZThhMCIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1vcGFjaXR5PSIwLjQwIi8+CiAgPGxpbmUgeDE9IjEwODgiIHkxPSI0NjAiIHgyPSIxMDg4IiB5Mj0iNDc2IiBzdHJva2U9IiNmZmU4YTAiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2Utb3BhY2l0eT0iMC40MCIvPgogIDxsaW5lIHgxPSI4MzIiIHkxPSI1ODgiIHgyPSI4MzIiIHkyPSI2MDQiIHN0cm9rZT0iI2ZmZThhMCIgc3Ryb2tlLXdpZHRoPSIyLjUiIHN0cm9rZS1vcGFjaXR5PSIwLjQwIi8+CiAgPGxpbmUgeDE9IjEwODgiIHkxPSI1ODgiIHgyPSIxMDg4IiB5Mj0iNjA0IiBzdHJva2U9IiNmZmU4YTAiIHN0cm9rZS13aWR0aD0iMi41IiBzdHJva2Utb3BhY2l0eT0iMC40MCIvPgogIDwhLS0gVE9QIHN0dW1wcyAtIGdsb3dpbmcgcm9kcyAtLT4KICA8ZyBmaWx0ZXI9InVybCgjc2cpIj4KICAgIDxsaW5lIHgxPSI5NDgiIHkxPSI0MjgiIHgyPSI5NDgiIHkyPSI0NjgiIHN0cm9rZT0iI2ZmZTA5MCIgc3Ryb2tlLXdpZHRoPSI1IiBzdHJva2Utb3BhY2l0eT0iMC45MiIvPgogICAgPGxpbmUgeDE9Ijk2MCIgeTE9IjQyNSIgeDI9Ijk2MCIgeTI9IjQ2OCIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjUuNSIgc3Ryb2tlLW9wYWNpdHk9IjAuOTgiLz4KICAgIDxsaW5lIHgxPSI5NzIiIHkxPSI0MjgiIHgyPSI5NzIiIHkyPSI0NjgiIHN0cm9rZT0iI2ZmZTA5MCIgc3Ryb2tlLXdpZHRoPSI1IiBzdHJva2Utb3BhY2l0eT0iMC45MiIvPgogICAgPGxpbmUgeDE9Ijk0NCIgeTE9IjQyOCIgeDI9Ijk3NiIgeTI9IjQyOCIgc3Ryb2tlPSIjZmZlMDkwIiBzdHJva2Utd2lkdGg9IjMuNSIgc3Ryb2tlLW9wYWNpdHk9IjAuODUiLz4KICA8L2c+CiAgPCEtLSBCT1RUT00gc3R1bXBzIC0tPgogIDxnIGZpbHRlcj0idXJsKCNzZykiPgogICAgPGxpbmUgeDE9Ijk0OCIgeTE9IjU5NiIgeDI9Ijk0OCIgeTI9IjYzOCIgc3Ryb2tlPSIjZmZlMDkwIiBzdHJva2Utd2lkdGg9IjUiIHN0cm9rZS1vcGFjaXR5PSIwLjkyIi8+CiAgICA8bGluZSB4MT0iOTYwIiB5MT0iNTk2IiB4Mj0iOTYwIiB5Mj0iNjQwIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iNS41IiBzdHJva2Utb3BhY2l0eT0iMC45OCIvPgogICAgPGxpbmUgeDE9Ijk3MiIgeTE9IjU5NiIgeDI9Ijk3MiIgeTI9IjYzOCIgc3Ryb2tlPSIjZmZlMDkwIiBzdHJva2Utd2lkdGg9IjUiIHN0cm9rZS1vcGFjaXR5PSIwLjkyIi8+CiAgICA8bGluZSB4MT0iOTQ0IiB5MT0iNjM4IiB4Mj0iOTc2IiB5Mj0iNjM4IiBzdHJva2U9IiNmZmUwOTAiIHN0cm9rZS13aWR0aD0iMy41IiBzdHJva2Utb3BhY2l0eT0iMC44NSIvPgogIDwvZz4KICA8IS0tIFBpdGNoIGNlbnRyZSBsaW5lIC0tPgogIDxsaW5lIHgxPSI5NjAiIHkxPSI0NjgiIHgyPSI5NjAiIHkyPSI1OTYiIHN0cm9rZT0iI2M4OTEzYSIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1vcGFjaXR5PSIwLjIwIi8+CiAgPCEtLSBCb3VuZGFyeSByb3BlIC0tPgogIDxlbGxpcHNlIGN4PSI5NjAiIGN5PSI3MTAiIHJ4PSI4NDAiIHJ5PSIzNDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIyLjgiIHN0cm9rZS1vcGFjaXR5PSIwLjE0IiBzdHJva2UtZGFzaGFycmF5PSIxNCw5Ii8+CiAgPGVsbGlwc2UgY3g9Ijk2MCIgY3k9IjY0NSIgcng9IjM5MCIgcnk9IjE1OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjEuOCIgc3Ryb2tlLW9wYWNpdHk9IjAuMDgiIHN0cm9rZS1kYXNoYXJyYXk9IjksMTIiLz4KICA8IS0tIE92ZXJsYXlzIC0tPgogIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjI4MCIgZmlsbD0iIzAwMCIgb3BhY2l0eT0iMC40MiIvPgogIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSIxMTAiIGhlaWdodD0iMTA4MCIgZmlsbD0iIzAwMCIgb3BhY2l0eT0iMC4zMiIvPgogIDxyZWN0IHg9IjE4MTAiIHk9IjAiIHdpZHRoPSIxMTAiIGhlaWdodD0iMTA4MCIgZmlsbD0iIzAwMCIgb3BhY2l0eT0iMC4zMiIvPgogIDxyZWN0IHg9IjAiIHk9Ijg4MCIgd2lkdGg9IjE5MjAiIGhlaWdodD0iMjAwIiBmaWxsPSIjMDAwIiBvcGFjaXR5PSIwLjQyIi8+Cjwvc3ZnPgo="

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}

.stApp {{
    background-color: #02040a !important;
    background-image: url("{BG_SVG}") !important;
    background-size: cover !important;
    background-position: center top !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    min-height: 100vh;
}}

section[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="collapsedControl"]  {{ display: none !important; }}
#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding-top:0 !important;padding-bottom:1rem;max-width:1200px;position:relative;z-index:1;}}

/* ── TOP HEADER ── */
.top-header {{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 20px 10px 16px;
    background:linear-gradient(90deg,#07111e 0%,#0a1628 50%,#07111e 100%);
    border-bottom:1.5px solid #1a2a40;
    margin-bottom:0; position:relative; overflow:hidden;
}}
.top-header::after {{
    content:''; position:absolute; bottom:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,#f4820a88,transparent);
}}
.brand-wrap {{ display:flex; align-items:center; gap:12px; }}
.brand-text-ipl {{ font-size:20px; font-weight:900; color:#f4820a; line-height:1; letter-spacing:1px; }}
.brand-text-sub {{ font-size:10px; font-weight:700; color:#607080; letter-spacing:2px; margin-top:3px; text-transform:uppercase; }}
.trophy-badge {{
    display:flex; align-items:center; gap:10px;
    background:linear-gradient(135deg,#0c1828,#111e30);
    border:1px solid #f4820a55; border-radius:14px;
    padding:8px 16px 8px 12px;
    box-shadow:0 0 24px rgba(244,130,10,0.18);
    position:relative; overflow:hidden;
}}
.trophy-badge::before {{
    content:''; position:absolute; inset:0;
    background:linear-gradient(135deg,rgba(244,130,10,0.08),transparent);
    pointer-events:none;
}}
.trophy-icon {{ font-size:28px; animation:trophy-glow 2.5s ease-in-out infinite; }}
@keyframes trophy-glow {{
    0%,100% {{ filter:drop-shadow(0 0 5px rgba(244,130,10,0.5)); }}
    50%      {{ filter:drop-shadow(0 0 14px rgba(244,130,10,1.0)); }}
}}
.trophy-text-year {{ font-size:15px; font-weight:900; color:#f4820a; line-height:1; letter-spacing:0.5px; }}
.trophy-text-tagline {{ font-size:9px; color:#7090a0; font-weight:600; letter-spacing:1px; margin-top:3px; line-height:1.3; }}

/* ── NAV BUTTONS — IPL Theme (dark navy + orange border + white text) ── */
.stButton>button {{
    background: linear-gradient(135deg, #0a1628 0%, #0d1e35 100%) !important;
    color: #e0eaf5 !important;
    border: 1px solid #2a4060 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 10px 8px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
}}
.stButton>button:hover {{
    background: linear-gradient(135deg, #f4820a 0%, #d4620a 100%) !important;
    color: #ffffff !important;
    border: 1px solid #f4820a !important;
    box-shadow: 0 4px 18px rgba(244,130,10,0.45) !important;
    transform: translateY(-1px) !important;
}}

/* Predict button special */
div[data-testid="stVerticalBlock"] > div:last-child .stButton>button {{
    background: linear-gradient(135deg, #1a6fd4 0%, #1050a0 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    padding: 14px !important;
    box-shadow: 0 4px 20px rgba(26,111,212,0.4) !important;
    letter-spacing: 0.5px !important;
}}
div[data-testid="stVerticalBlock"] > div:last-child .stButton>button:hover {{
    background: linear-gradient(135deg, #f4820a 0%, #d4620a 100%) !important;
    box-shadow: 0 6px 24px rgba(244,130,10,0.5) !important;
}}

/* ── NAVIGATE label ── */
.nav-label {{
    font-size:11px; color:#8899aa; font-weight:700;
    letter-spacing:2px; padding: 0 4px;
    text-transform:uppercase;
}}

/* ── PAGE ELEMENTS ── */
.page-title{{font-size:26px;font-weight:900;color:#fff;letter-spacing:-.5px;}}
.page-title span{{color:#f4820a;}}
.page-sub{{font-size:13px;color:#9aaabb;margin-top:3px;margin-bottom:1.25rem;text-shadow:0 1px 4px rgba(0,0,0,0.9);}}
.divider{{display:flex;align-items:center;gap:10px;margin:0.75rem 0 1rem;color:#6080a0;font-size:10px;font-weight:700;letter-spacing:2px;}}
.divider::before,.divider::after{{content:'';flex:1;height:.5px;background:#1a2a40;}}
.card{{background:rgba(12,24,40,0.92);border-radius:14px;padding:1.25rem;margin-bottom:1rem;border:.5px solid #1a2a40;backdrop-filter:blur(4px);}}
.metric-card{{background:#07111e;border-radius:12px;padding:1rem 1.25rem;border:.5px solid #1a2a40;text-align:center;}}
.metric-val{{font-size:28px;font-weight:900;margin-bottom:2px;}}
.metric-lbl{{font-size:10px;color:#8899aa;font-weight:600;letter-spacing:.5px;}}
.metric-sub{{font-size:10px;color:#5a7080;margin-top:3px;}}
.team-card{{border-radius:14px;padding:1.25rem .75rem;text-align:center;border:2px solid transparent;}}
.team-card-name{{font-size:12px;font-weight:800;color:#fff;line-height:1.3;margin-top:8px;}}
.team-card-abbr{{font-size:10px;color:#607080;margin-top:3px;}}
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
.player-rank{{font-size:12px;font-weight:700;color:#5a7080;min-width:20px;}}
.player-name{{font-size:13px;font-weight:700;color:#fff;flex:1;}}
.player-val{{font-size:13px;font-weight:800;}}
.footer{{text-align:center;color:#4a6070;font-size:10px;margin-top:1rem;padding-bottom:1rem;}}

/* ── ALL RADIO LABELS BRIGHT WHITE ── */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] p {{
    color:#ffffff !important; font-weight:600 !important;
    font-size:13px !important;
    text-shadow:0 1px 4px rgba(0,0,0,0.9) !important;
}}
div[data-testid="stRadio"] label:hover {{ color:#f4820a !important; }}
div[data-testid="stHorizontalBlock"] label,
div[data-testid="stHorizontalBlock"] span,
div[data-testid="stHorizontalBlock"] p {{
    color:#ffffff !important; font-weight:600 !important;
    text-shadow:0 1px 4px rgba(0,0,0,0.8) !important;
}}

.stSelectbox>div>div{{background:#0c1828 !important;color:#fff !important;border-color:#1a2a40 !important;}}
label[data-baseweb="label"]{{color:#8899aa !important;}}
</style>
""", unsafe_allow_html=True)

# ── TOP HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
    <div class="brand-wrap">
        <img src="{IPL_LOGO}" width="56" height="56"
             style="border-radius:50%;box-shadow:0 0 24px rgba(255,215,0,0.7),0 0 8px rgba(255,215,0,0.4);
                    flex-shrink:0;border:2.5px solid rgba(255,215,0,0.5);">
        <div>
            <div class="brand-text-ipl">IPL ANALYTICS</div>
            <div class="brand-text-sub">Cricket Intelligence</div>
        </div>
    </div>
    <div class="trophy-badge">
        <div class="trophy-icon">🏆</div>
        <div>
            <div class="trophy-text-year">IPL 2024</div>
            <div class="trophy-text-tagline">THE ULTIMATE<br>CRICKET SHOWDOWN</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load & Train ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    MATCHES_URL = "https://huggingface.co/datasets/shibangmaity/ipl-analytics-data/resolve/main/matches.csv"
    DELIVERIES_URL = "https://huggingface.co/datasets/shibangmaity/ipl-analytics-data/resolve/main/deliveries.csv"
    matches = pd.read_csv(MATCHES_URL)
    deliveries = pd.read_csv(DELIVERIES_URL)
    sm={'2007/08':'2008','2009/10':'2010','2020/21':'2021'}
    tm={'Rising Pune Supergiant':'Rising Pune Supergiants','Royal Challengers Bangalore':'Royal Challengers Bengaluru','Delhi Daredevils':'Delhi Capitals','Kings XI Punjab':'Punjab Kings'}
    matches['season']=matches['season'].replace(sm)
    for col in ['team1','team2','toss_winner','winner']: matches[col]=matches[col].replace(tm)
    deliveries['batting_team']=deliveries['batting_team'].replace(tm)
    deliveries['bowling_team']=deliveries['bowling_team'].replace(tm)
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
    matches_flip['toss_win_bat']=matches['toss_win_bat']
    matches_flip['h2h_winrate']=1-matches['h2h_winrate']
    matches_flip['form_diff']=-matches['form_diff']
    matches_flip['form5_diff']=-matches['form5_diff']
    matches_flip['venue_wr_diff']=-matches['venue_wr_diff']
    matches_flip['wr_diff']=-matches['wr_diff']
    matches_flip['team1_form']=matches['team2_form']
    matches_flip['team2_form']=matches['team1_form']
    matches_flip['team1_form5']=matches['team2_form5']
    matches_flip['team2_form5']=matches['team1_form5']
    matches_flip['team1_overall_wr']=matches['team2_overall_wr']
    matches_flip['team2_overall_wr']=matches['team1_overall_wr']
    matches_flip['team1_won']=1-matches['team1_won']
    import pandas as pd_inner
    matches_aug=pd_inner.concat([matches,matches_flip],ignore_index=True)
    matches_aug=matches_aug.sample(frac=1,random_state=42).reset_index(drop=True)
    X,y=matches_aug[feats],matches_aug['team1_won']
    split=int(len(matches)*0.75)
    X_train,y_train=X[:split*2],y[:split*2]
    X_test,y_test=matches[feats][split:],matches['team1_won'][split:]
    rf=RandomForestClassifier(n_estimators=1000,max_depth=6,min_samples_leaf=8,max_features='sqrt',random_state=42)
    rf.fit(X_train,y_train); acc=rf.score(X_test,y_test)
    bat=deliveries.groupby('batter').agg(runs=('batsman_runs','sum'),balls=('ball','count'),innings=('match_id','nunique')).reset_index()
    bat['sr']=(bat['runs']/bat['balls']*100).round(1)
    bat['avg']=(bat['runs']/bat['innings']).round(1)
    wk_df=deliveries[deliveries['is_wicket']==1]
    wk_df=wk_df[~wk_df['dismissal_kind'].isin(['run out','retired hurt','obstructing the field'])]
    bowl=deliveries.groupby('bowler').agg(runs=('total_runs','sum'),balls=('ball','count'),matches=('match_id','nunique')).reset_index()
    wk_grp=wk_df.groupby('bowler')['is_wicket'].count().reset_index(); wk_grp.columns=['bowler','wickets']
    bowl=bowl.merge(wk_grp,on='bowler',how='left').fillna(0)
    bowl['wickets']=bowl['wickets'].astype(int)
    bowl['economy']=(bowl['runs']/(bowl['balls']/6)).round(2)
    bowl['avg']=(bowl['runs']/bowl['wickets'].replace(0,np.nan)).round(1)
    sixes=deliveries[deliveries['batsman_runs']==6].groupby('batter').size().reset_index(); sixes.columns=['batter','sixes']
    fours=deliveries[deliveries['batsman_runs']==4].groupby('batter').size().reset_index(); fours.columns=['batter','fours']
    bat=bat.merge(sixes,on='batter',how='left').merge(fours,on='batter',how='left').fillna(0)
    bat['sixes']=bat['sixes'].astype(int); bat['fours']=bat['fours'].astype(int)
    bat2=deliveries.merge(matches[['id','season']],left_on='match_id',right_on='id')
    oc=bat2.groupby(['season','batter'])['batsman_runs'].sum().reset_index()
    oc=oc.loc[oc.groupby('season')['batsman_runs'].idxmax()].sort_values('season')
    wk2=wk_df.merge(matches[['id','season']],left_on='match_id',right_on='id')
    pc=wk2.groupby(['season','bowler'])['is_wicket'].count().reset_index()
    pc=pc.loc[pc.groupby('season')['is_wicket'].idxmax()].sort_values('season')
    ss=deliveries.merge(matches[['id','season']],left_on='match_id',right_on='id')
    season_avg=ss.groupby(['season','match_id'])['total_runs'].sum().reset_index().groupby('season')['total_runs'].mean().round(1)
    td=matches.groupby(['season','toss_decision']).size().unstack(fill_value=0)
    td['field_pct']=(td.get('field',0)/(td.get('bat',0)+td.get('field',0))*100).round(1)
    t_fours=deliveries[deliveries['batsman_runs']==4].groupby('batting_team').size().reset_index(); t_fours.columns=['team','fours']
    t_sixes=deliveries[deliveries['batsman_runs']==6].groupby('batting_team').size().reset_index(); t_sixes.columns=['team','sixes']
    team_bnd=t_fours.merge(t_sixes,on='team')
    return rf,le,ve,matches,tw,tm2,acc,bat,bowl,oc,pc,season_avg,td,team_bnd,deliveries

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
    X=[[int(t1_toss),toss_win_bat,h2hwr,f1-f2,f1_5-f2_5,vwr1-vwr2,wr1-wr2,2024,1.0,f1,f2,f1_5,f2_5,wr1,wr2]]
    prob=rf.predict_proba(X)[0]; winner=t1 if prob[1]>prob[0] else t2
    return winner,round(prob[1]*100,1),round(prob[0]*100,1),h1,h2,round(f1*100),round(f2*100)

def fig_to_base64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor=fig.get_facecolor()); buf.seek(0)
    return "data:image/png;base64,"+base64.b64encode(buf.read()).decode()

BG='#080f1e'; CARD='#0c1828'; GRID='#1a2a40'; TEXT='#ccd'

with st.spinner("🏏 Loading model & stats..."):
    rf,le,ve,matches,tw,tm2,acc,bat,bowl,oc,pc,season_avg,td,team_bnd,deliveries=load_and_train()

# ── TOP NAV BAR ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#05101e,#07152a,#05101e);
            border-bottom:1px solid #1a2a40;padding:10px 16px 8px;
            margin-bottom:1rem;display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
    <span class="nav-label">⚡ Navigate</span>
""", unsafe_allow_html=True)

nav_cols=st.columns(6)
nav_items=[("predict","⚡ Predict"),("teams","👥 Teams"),("players","🏏 Players"),
           ("charts","📊 Charts"),("h2h","⚔️ H2H"),("about","ℹ️ About")]
for col,(key,label) in zip(nav_cols,nav_items):
    with col:
        if st.button(label,key=f"top_{key}",use_container_width=True):
            st.session_state.page=key; st.rerun()
st.markdown("</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════
if st.session_state.page=='predict':
    st.markdown('<div class="page-title">IPL MATCH <span>PREDICTOR</span> ⚡</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Predict. Analyze. Win.</div>',unsafe_allow_html=True)
    with st.container():
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
            st.markdown(f'<div class="team-card" style="border-color:{t1i["color"]};background:{t1i["bg"]}"><img src="{t1i["logo"]}" width="72" height="72" style="border-radius:50%"><div class="team-card-name">{team1.upper()}</div><div class="team-card-abbr">{t1i["abbr"]}</div></div>',unsafe_allow_html=True)
        with cv:
            st.markdown("<br><br>",unsafe_allow_html=True)
            st.markdown('<div class="vs-badge">VS</div>',unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="team-card" style="border-color:{t2i["color"]};background:{t2i["bg"]}"><img src="{t2i["logo"]}" width="72" height="72" style="border-radius:50%"><div class="team-card-name">{team2.upper()}</div><div class="team-card-abbr">{t2i["abbr"]}</div></div>',unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        d1,d2,d3,d4=st.columns(4)
        with d1: venue=st.selectbox("🏟️ Venue",VENUES,key="venue")
        with d2: st.date_input("📅 Date",key="mdate")
        with d3: toss_winner=st.radio("🪙 Toss winner",[team1,team2],key="tw")
        with d4: toss_decision=st.radio("🏏 Decision",["bat","field"],key="td_val")
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("⚡  Predict match outcome",use_container_width=True):
        t1_toss=1 if toss_winner==team1 else 0
        winner,p1,p2,h1,h2,f1p,f2p=predict_match(rf,le,ve,matches,tw,tm2,team1,team2,venue,t1_toss,toss_decision)
        if winner:
            wc=t1i['color'] if winner==team1 else t2i['color']
            h_lbl=f"{t1i['abbr']} leads {h1}–{h2}" if h1>=h2 else f"{t2i['abbr']} leads {h2}–{h1}"
            vs=venue.split(',')[0]
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
                    <span style="font-size:11px;font-weight:700;color:#607080;letter-spacing:1px">📈 PREDICTION RESULT</span>
                    <span class="ai-badge">AI POWERED</span>
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
                <div class="prob-labels"><span>{t1i['abbr']}</span><span style="color:#304050">WIN PROBABILITY</span><span>{t2i['abbr']}</span></div>
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

elif st.session_state.page=='teams':
    st.markdown('<div class="page-title">👥 <span>Teams</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">All IPL 2024 franchises — performance overview</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    for col,lbl,val,sub,color in [(m1,"Total matches",len(matches),"2008–2024","#1a6fd4"),(m2,"Active teams",10,"IPL 2024","#f4820a"),(m3,"Model accuracy",f"{round(acc*100,1)}%","test set","#1c8b6e"),(m4,"Toss effect","50.8%","wins match","#a855f7")]:
        with col: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{val}</div><div class="metric-lbl">{lbl}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    cols=st.columns(5)
    for i,(name,info) in enumerate(TEAM_INFO.items()):
        w=tw.get(name,0); t=tm2.get(name,1); wr=round(w/t*100)
        bnd=team_bnd[team_bnd['team']==name]
        sx=int(bnd['sixes'].values[0]) if len(bnd)>0 else 0
        with cols[i%5]:
            st.markdown(f"""<div class="card" style="border:1.5px solid {info['color']}33;text-align:center;padding:1rem .5rem">
                <img src="{info['logo_sm']}" width="52" height="52" style="border-radius:50%">
                <div style="font-size:11px;font-weight:800;color:#fff;line-height:1.3;margin-top:8px">{name}</div>
                <div style="font-size:10px;color:#607080;margin-top:2px">{info['abbr']}</div>
                <div style="margin-top:10px"><div style="font-size:18px;font-weight:900;color:{info['color']}">{wr}%</div><div style="font-size:9px;color:#607080">win rate</div></div>
                <div style="display:flex;justify-content:space-around;margin-top:10px;padding-top:8px;border-top:.5px solid #1a2a40">
                    <div style="text-align:center"><div style="font-size:12px;font-weight:700;color:#fff">{w}</div><div style="font-size:9px;color:#607080">wins</div></div>
                    <div style="text-align:center"><div style="font-size:12px;font-weight:700;color:#f4820a">{sx}</div><div style="font-size:9px;color:#607080">sixes</div></div>
                    <div style="text-align:center"><div style="font-size:12px;font-weight:700;color:#fff">{t}</div><div style="font-size:9px;color:#607080">played</div></div>
                </div></div>""",unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-top:.5rem">',unsafe_allow_html=True)
    st.markdown('<div class="divider">ALL-TIME WIN RANKINGS</div>',unsafe_allow_html=True)
    rows=sorted([{"_n":n,"abbr":TEAM_INFO[n]["abbr"],"color":TEAM_INFO[n]["color"],"w":tw.get(n,0),"t":tm2.get(n,1)} for n in ACTIVE_TEAMS],key=lambda x:-x["w"])
    th='<table class="stat-table"><thead><tr><th>#</th><th>Team</th><th>Wins</th><th>Matches</th><th>Win rate</th><th>Loss</th></tr></thead><tbody>'
    for i,r in enumerate(rows,1):
        wr2=round(r['w']/r['t']*100); loss=r['t']-r['w']
        th+=f'<tr><td style="color:#607080">{i}</td><td><span style="color:{r["color"]};font-weight:700">{r["abbr"]}</span>  {r["_n"]}</td><td style="color:{r["color"]};font-weight:700">{r["w"]}</td><td style="color:#607080">{r["t"]}</td><td><span class="badge" style="background:{r["color"]}22;color:{r["color"]}">{wr2}%</span></td><td style="color:#607080">{loss}</td></tr>'
    th+='</tbody></table>'; st.markdown(th,unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.page=='players':
    st.markdown('<div class="page-title">🏏 <span>Player Stats</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">IPL 2008–2024 — batting, bowling, and milestone records</div>',unsafe_allow_html=True)
    tab=st.radio("View",["🏏 Batting","🎯 Bowling","🏆 Season Awards","🌟 Milestones"],horizontal=True,key="ptab",label_visibility="collapsed")
    if tab=="🏏 Batting":
        st.markdown('<div class="divider" style="margin-top:.5rem">SORT BY</div>',unsafe_allow_html=True)
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
            val=row[col_key]
            disp=f"{val:.2f}" if col_key in ('economy','avg') else str(int(val))
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
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown('<div class="divider">🟠 ORANGE CAP — MOST RUNS</div>',unsafe_allow_html=True)
            th='<table class="stat-table"><thead><tr><th>Season</th><th>Batter</th><th>Runs</th></tr></thead><tbody>'
            for _,r in oc.sort_values('season',ascending=False).iterrows():
                th+=f'<tr><td style="color:#f4820a;font-weight:700">{r["season"]}</td><td style="color:#fff">{r["batter"]}</td><td style="color:#f4820a;font-weight:700">{int(r["batsman_runs"])}</td></tr>'
            th+='</tbody></table>'; st.markdown(th,unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown('<div class="divider">💜 PURPLE CAP — MOST WICKETS</div>',unsafe_allow_html=True)
            th2='<table class="stat-table"><thead><tr><th>Season</th><th>Bowler</th><th>Wickets</th></tr></thead><tbody>'
            for _,r in pc.sort_values('season',ascending=False).iterrows():
                th2+=f'<tr><td style="color:#a855f7;font-weight:700">{r["season"]}</td><td style="color:#fff">{r["bowler"]}</td><td style="color:#a855f7;font-weight:700">{int(r["is_wicket"])}</td></tr>'
            th2+='</tbody></table>'; st.markdown(th2,unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
    elif tab=="🌟 Milestones":
        c1,c2=st.columns(2)
        with c1:
            top6=bat.sort_values('sixes',ascending=False).head(10)
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown('<div class="divider">💥 MOST SIXES</div>',unsafe_allow_html=True)
            for i,r in top6.reset_index(drop=True).iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["batter"]}</div><div class="player-val" style="color:#f4820a">{int(r["sixes"])} 6s</div></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
        with c2:
            top4=bat.sort_values('fours',ascending=False).head(10)
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown('<div class="divider">🔵 MOST FOURS</div>',unsafe_allow_html=True)
            for i,r in top4.reset_index(drop=True).iterrows():
                st.markdown(f'<div class="player-row"><div class="player-rank">#{i+1}</div><div class="player-name">{r["batter"]}</div><div class="player-val" style="color:#1a6fd4">{int(r["fours"])} 4s</div></div>',unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.page=='charts':
    st.markdown('<div class="page-title">📊 <span>Analytics Charts</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Visual insights from 16 seasons of IPL data</div>',unsafe_allow_html=True)
    chart_tab=st.radio("",["Runs trend","Toss decisions","Team wins","Top batters","Top bowlers"],horizontal=True,key="ctab",label_visibility="collapsed")
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    if chart_tab=="Runs trend":
        fig,ax=plt.subplots(figsize=(10,4),facecolor=BG); ax.set_facecolor(CARD)
        seasons=list(season_avg.index); vals=list(season_avg.values)
        ax.fill_between(seasons,vals,alpha=0.15,color='#1a6fd4')
        ax.plot(seasons,vals,color='#1a6fd4',linewidth=2.5,marker='o',markersize=5,markerfacecolor='#1a6fd4')
        for s,v in zip(seasons,vals): ax.annotate(f'{v:.0f}',(s,v),textcoords='offset points',xytext=(0,8),ha='center',fontsize=8,color='#8899aa')
        ax.set_title('Average total runs per match by season',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=9); ax.spines[:].set_color(GRID)
        for spine in ax.spines.values(): spine.set_linewidth(.5)
        ax.set_ylim(260,390); ax.grid(axis='y',color=GRID,linewidth=.5,alpha=.5); fig.tight_layout()
        st.image(fig_to_base64(fig),use_container_width=True); plt.close()
        st.markdown('<div class="card" style="margin-top:.5rem"><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;text-align:center"><div><div style="font-size:20px;font-weight:900;color:#f4820a">365.8</div><div style="font-size:10px;color:#607080">Avg runs in 2024 (highest)</div></div><div><div style="font-size:20px;font-weight:900;color:#1a6fd4">286.9</div><div style="font-size:10px;color:#607080">Avg runs in 2009 (lowest)</div></div><div><div style="font-size:20px;font-weight:900;color:#1c8b6e">+27%</div><div style="font-size:10px;color:#607080">Increase 2009 → 2024</div></div></div></div>',unsafe_allow_html=True)
    elif chart_tab=="Toss decisions":
        fig,ax=plt.subplots(figsize=(10,4),facecolor=BG); ax.set_facecolor(CARD)
        td2=td.reset_index(); seasons=td2['season'].tolist()
        bat_vals=td2.get('bat',pd.Series([0]*len(td2))).tolist()
        field_vals=td2.get('field',pd.Series([0]*len(td2))).tolist()
        x=np.arange(len(seasons)); w=0.38
        ax.bar(x-w/2,bat_vals,w,label='Bat first',color='#1a6fd4',alpha=.85)
        ax.bar(x+w/2,field_vals,w,label='Field first',color='#1c8b6e',alpha=.85)
        ax.set_xticks(x); ax.set_xticklabels(seasons,rotation=45,fontsize=8,color='#607080')
        ax.tick_params(colors='#607080',labelsize=9)
        ax.set_title('Toss decision trend: bat vs field by season',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.legend(facecolor=CARD,edgecolor=GRID,labelcolor=TEXT,fontsize=9); ax.spines[:].set_color(GRID)
        for spine in ax.spines.values(): spine.set_linewidth(.5)
        ax.grid(axis='y',color=GRID,linewidth=.5,alpha=.5); fig.tight_layout()
        st.image(fig_to_base64(fig),use_container_width=True); plt.close()
        st.markdown('<div class="card" style="margin-top:.5rem"><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;text-align:center"><div><div style="font-size:20px;font-weight:900;color:#1c8b6e">83%</div><div style="font-size:10px;color:#607080">Teams chose field in 2018</div></div><div><div style="font-size:20px;font-weight:900;color:#1a6fd4">39</div><div style="font-size:10px;color:#607080">Bat-first choices in 2010</div></div><div><div style="font-size:20px;font-weight:900;color:#f4820a">73%</div><div style="font-size:10px;color:#607080">Field-first in 2024</div></div></div></div>',unsafe_allow_html=True)
    elif chart_tab=="Team wins":
        active_tw=[(n,tw.get(n,0),TEAM_INFO[n]['color']) for n in ACTIVE_TEAMS]; active_tw.sort(key=lambda x:-x[1])
        fig,ax=plt.subplots(figsize=(10,5),facecolor=BG); ax.set_facecolor(CARD)
        names=[x[0].replace('Royal Challengers Bengaluru','RCB').replace('Chennai Super Kings','CSK').replace('Mumbai Indians','MI').replace('Kolkata Knight Riders','KKR').replace('Delhi Capitals','DC').replace('Rajasthan Royals','RR').replace('Punjab Kings','PBKS').replace('Sunrisers Hyderabad','SRH').replace('Gujarat Titans','GT').replace('Lucknow Super Giants','LSG') for x in active_tw]
        wins=[x[1] for x in active_tw]; colors=[x[2] for x in active_tw]
        bars=ax.barh(names,wins,color=colors,alpha=.85,height=.6)
        for bar,val in zip(bars,wins): ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,str(val),va='center',ha='left',color=TEXT,fontsize=10,fontweight='bold')
        ax.set_title('All-time wins by team (active franchises)',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=10); ax.spines[:].set_color(GRID)
        for spine in ax.spines.values(): spine.set_linewidth(.5)
        ax.invert_yaxis(); ax.grid(axis='x',color=GRID,linewidth=.5,alpha=.5); ax.set_xlim(0,max(wins)+20); fig.tight_layout()
        st.image(fig_to_base64(fig),use_container_width=True); plt.close()
    elif chart_tab=="Top batters":
        top15=bat[bat['balls']>=500].sort_values('runs',ascending=False).head(15)
        fig,ax=plt.subplots(figsize=(10,6),facecolor=BG); ax.set_facecolor(CARD)
        bars=ax.barh(top15['batter'][::-1],top15['runs'][::-1],color='#1a6fd4',alpha=.85,height=.65)
        for bar,val in zip(bars,top15['runs'][::-1]): ax.text(bar.get_width()+30,bar.get_y()+bar.get_height()/2,f'{int(val):,}',va='center',ha='left',color=TEXT,fontsize=9,fontweight='bold')
        ax.set_title('Top 15 run scorers — IPL all time',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=9); ax.spines[:].set_color(GRID)
        for spine in ax.spines.values(): spine.set_linewidth(.5)
        ax.grid(axis='x',color=GRID,linewidth=.5,alpha=.5); ax.set_xlim(0,9500); fig.tight_layout()
        st.image(fig_to_base64(fig),use_container_width=True); plt.close()
    elif chart_tab=="Top bowlers":
        top15b=bowl[bowl['balls']>=300].sort_values('wickets',ascending=False).head(15)
        fig,ax=plt.subplots(figsize=(10,6),facecolor=BG); ax.set_facecolor(CARD)
        bars=ax.barh(top15b['bowler'][::-1],top15b['wickets'][::-1],color='#e63946',alpha=.85,height=.65)
        for bar,val in zip(bars,top15b['wickets'][::-1]): ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,str(int(val)),va='center',ha='left',color=TEXT,fontsize=9,fontweight='bold')
        ax.set_title('Top 15 wicket takers — IPL all time',color=TEXT,fontsize=13,fontweight='bold',pad=12)
        ax.tick_params(colors='#607080',labelsize=9); ax.spines[:].set_color(GRID)
        for spine in ax.spines.values(): spine.set_linewidth(.5)
        ax.grid(axis='x',color=GRID,linewidth=.5,alpha=.5); fig.tight_layout()
        st.image(fig_to_base64(fig),use_container_width=True); plt.close()

elif st.session_state.page=='h2h':
    st.markdown('<div class="page-title">⚔️ <span>Head to Head</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Compare any two teams — full rivalry breakdown</div>',unsafe_allow_html=True)
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
                <div><img src="{t1i_h['logo']}" width="64" height="64" style="border-radius:50%">
                    <div style="font-size:38px;font-weight:900;color:{t1i_h['color']};margin-top:6px">{w1}</div>
                    <div style="font-size:11px;font-weight:600;color:#aab;margin-top:2px">{ht1}</div></div>
                <div><div style="font-size:11px;font-weight:700;color:#607080">{tot}</div><div style="font-size:9px;color:#304050;margin-top:2px">MATCHES</div></div>
                <div><img src="{t2i_h['logo']}" width="64" height="64" style="border-radius:50%">
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
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown('<div class="divider">RECENT 8 MEETINGS</div>',unsafe_allow_html=True)
            recent=h2h[['date','venue','winner','toss_decision']].tail(8).sort_values('date',ascending=False)
            rh='<table class="stat-table"><thead><tr><th>Date</th><th>Venue</th><th>Winner</th></tr></thead><tbody>'
            for _,row in recent.iterrows():
                wi=TEAM_INFO.get(row['winner'],{"color":"#fff","abbr":str(row['winner'])[:3].upper()})
                rh+=f'<tr><td style="color:#607080;font-size:11px">{row["date"]}</td><td style="color:#8899aa;font-size:11px">{str(row["venue"]).split(",")[0][:22]}</td><td><span class="badge" style="background:{wi["color"]}22;color:{wi["color"]}">{wi["abbr"]}</span></td></tr>'
            rh+='</tbody></table>'; st.markdown(rh,unsafe_allow_html=True); st.markdown('</div>',unsafe_allow_html=True)
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
                    <div>{'🟡' if w1>w2 else '🔴'} Series leader: <b style="color:{t1i_h['color'] if w1>=w2 else t2i_h['color']}">{t1i_h['abbr'] if w1>=w2 else t2i_h['abbr']}</b></div>
                </div></div>""",unsafe_allow_html=True)
    else:
        st.info("No matches found between these two teams.")

elif st.session_state.page=='about':
    st.markdown('<div class="page-title">ℹ️ <span>About</span></div>',unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="margin-top:1rem"><div class="divider">PROJECT</div>
        <div style="color:#ccd;font-size:13px;line-height:2">
            <b style="color:#fff">IPL Analytics & Match Predictor</b> is an end-to-end data analytics and ML project
            built on IPL data 2008–2024 covering <b style="color:#f4820a">1,090 matches</b> and <b style="color:#f4820a">260,920 deliveries</b>.<br><br>
            Uses a <b style="color:#f4820a">Random Forest Classifier</b> with data augmentation achieving
            <b style="color:#1c8b6e">{round(acc*100,1)}% accuracy</b> on held-out test set.
        </div></div>
    <div class="card"><div class="divider">KEY FINDINGS</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;color:#aab;font-size:12px;line-height:2">
            <div>✅ Win rate differential is the strongest predictor</div>
            <div>✅ Toss outcome has near-zero impact (~50.8%)</div>
            <div>✅ Field-first preference grew 55% (2008) → 83% (2018)</div>
            <div>✅ Average match scores rose 27% from 2009 to 2024</div>
            <div>✅ V. Kohli leads all-time runs with 8,014</div>
            <div>✅ YS Chahal leads all-time wickets with 205</div>
        </div></div>
    <div class="card"><div class="divider">TECH STACK</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px">
            {''.join([f'<span class="badge" style="background:{c[1]}22;color:{c[1]};padding:6px 14px">{c[0]}</span>' for c in [("Python","#f4820a"),("Scikit-learn","#1a6fd4"),("Pandas","#e63946"),("Streamlit","#1c8b6e"),("Matplotlib","#a855f7"),("NumPy","#00b4d8")]])}
        </div></div>
    <div class="card"><div class="divider">BUILT BY</div>
        <div style="color:#ccd;font-size:13px;line-height:2">
            <b style="color:#fff">Shibang Maity</b> · Computer Science, KIIT University<br>
            <span style="color:#607080">shibangmaity@gmail.com · github.com/shibangmaity</span>
        </div></div>""",unsafe_allow_html=True)

st.markdown('<div class="footer">Built by Shibang Maity · IPL Data 2008–2024 · Random Forest Classifier</div>',unsafe_allow_html=True)
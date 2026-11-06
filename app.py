"""
직육면체·정육면체 전개도 실험실 (5학년 수학 5단원 직육면체)
실행: streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from geometry import (CUBE_NETS, FACE_LABELS, rows_to_cells, build_net, fold,
                      check_cube_net, identify_net, is_connected, rects_overlap_2d)

st.set_page_config(page_title="전개도 실험실", page_icon="🧊", layout="wide")

# 면 색깔 (면마다 다른 색)
PALETTE = ["#FF6B6B", "#FFB347", "#FFE066", "#6BCB77", "#4D96FF", "#B983FF"]
# 마주 보는 면끼리 같은 색 (축 0,1,2)
PAIR_COLORS = ["#FF6B6B", "#4D96FF", "#6BCB77"]


# ------------------------------------------------------------
# 그림 그리는 함수
# ------------------------------------------------------------
def face_colors(faces, mode):
    if mode == "마주 보는 면끼리 같은 색":
        return [PAIR_COLORS[f["axis_n"]] for f in faces]
    if mode == "모두 같은 색":
        return ["#9AD0F5"] * len(faces)
    return [PALETTE[i % 6] for i in range(len(faces))]


def net_2d_figure(faces, colors, show_size=False):
    fig = go.Figure()
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = f["rect"]
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], fill="toself",
            fillcolor=colors[i], line=dict(color="#333", width=3),
            mode="lines", hoverinfo="skip", showlegend=False))
        text = FACE_LABELS[i]
        if show_size:
            text += f"<br><span style='font-size:13px'>{x1 - x0:g}×{y1 - y0:g}</span>"
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text,
                           showarrow=False, font=dict(size=22, color="#222"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def _frame_traces(faces, colors, angle):
    pts = fold(faces, angle)
    traces = []
    for i, p in enumerate(pts):
        traces.append(go.Mesh3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
            color=colors[i], opacity=0.93, flatshading=True,
            hoverinfo="skip", showscale=False))
    ex, ey, ez = [], [], []
    for p in pts:
        loop = np.vstack([p, p[:1]])
        ex += list(loop[:, 0]) + [None]
        ey += list(loop[:, 1]) + [None]
        ez += list(loop[:, 2]) + [None]
    traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color="#222", width=5),
                               hoverinfo="skip", showlegend=False))
    c = np.array([p.mean(axis=0) for p in pts])
    traces.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="text",
                               text=FACE_LABELS[:len(pts)],
                               textfont=dict(size=20, color="#111"),
                               hoverinfo="skip", showlegend=False))
    return traces


def fold_3d_figure(faces, colors, start_angle=0):
    angles = list(range(0, 91, 5))
    # 화면 범위를 고정해서 애니메이션 중에 흔들리지 않게 함
    allp = np.vstack([np.vstack(fold(faces, a)) for a in angles])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    pad = 0.3
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    ratio = span / span.max()

    frames = [go.Frame(data=_frame_traces(faces, colors, a), name=str(a)) for a in angles]
    fig = go.Figure(data=_frame_traces(faces, colors, start_angle), frames=frames)

    play_args = {"frame": {"duration": 90, "redraw": True}, "fromcurrent": True,
                 "transition": {"duration": 0}, "mode": "immediate"}
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(range=[lo[0], hi[0]], visible=False),
            yaxis=dict(range=[lo[1], hi[1]], visible=False),
            zaxis=dict(range=[lo[2], hi[2]], visible=False),
            aspectmode="manual",
            aspectratio=dict(x=ratio[0], y=ratio[1], z=ratio[2]),
            camera=dict(eye=dict(x=1.3, y=-1.6, z=1.2)),
        ),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="▶ 접기", method="animate", args=[None, play_args]),
                dict(label="⏸ 멈춤", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ])],
        sliders=[dict(
            active=angles.index(start_angle) if start_angle in angles else 0,
            x=0.25, len=0.75, y=0.0, yanchor="bottom",
            currentvalue=dict(prefix="접은 각도: ", suffix="°", font=dict(size=14)),
            steps=[dict(label=str(a), method="animate",
                        args=[[str(a)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}]) for a in angles],
        )],
    )
    return fig


def show_net(faces, mode, show_size=False, key=""):
    colors = face_colors(faces, mode)
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("**펼친 전개도**")
        st.plotly_chart(net_2d_figure(faces, colors, show_size), key=key + "2d")
    with right:
        st.markdown("**접어 보기** — ▶ 버튼 또는 슬라이더, 마우스로 돌려 보기")
        st.plotly_chart(fold_3d_figure(faces, colors), key=key + "3d")


def opposite_quiz(faces, key):
    with st.expander("🤔 접기 전에 예상해 보기: 마주 보는 면 찾기"):
        labels = FACE_LABELS[:len(faces)]
        q = st.selectbox("기준이 되는 면", labels, key=key + "q")
        qi = labels.index(q)
        others = [l for l in labels if l != q]
        ans = st.radio(f"{q}번 면과 마주 보는 면은?", others, horizontal=True,
                       index=None, key=key + "a")
        if st.button("정답 확인", key=key + "b"):
            correct = [labels[j] for j, f in enumerate(faces)
                       if j != qi and f["axis_n"] == faces[qi]["axis_n"]]
            if ans is None:
                st.info("먼저 답을 골라 주세요.")
            elif correct and ans == correct[0]:
                st.success(f"정답! {q}번 면과 {correct[0]}번 면은 마주 보는 면(평행한 면)이에요. "
                           f"나머지 4개 면은 {q}번 면과 수직이에요.")
                st.balloons()
            else:
                st.warning("다시 생각해 보세요. ‘마주 보는 면끼리 같은 색’으로 바꾸고 접어 보면 확인할 수 있어요.")


# ------------------------------------------------------------
# 파이어베이스 (선택 사항: 설정이 없으면 저장 기능만 꺼짐)
# ------------------------------------------------------------
@st.cache_resource
def get_db():
    try:
        if "firebase" not in st.secrets:
            return None
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception:
        return None


def save_submission(db, name, cells, valid, net_type):
    from firebase_admin import firestore
    db.collection("nets").add({
        "name": name,
        "cells": [f"{r},{c}" for r, c in cells],
        "valid": bool(valid),
        "type": net_type or "",
        "created": firestore.SERVER_TIMESTAMP,
    })


def load_submissions(db):
    from firebase_admin import firestore
    docs = (db.collection("nets")
            .order_by("created", direction=firestore.Query.DESCENDING)
            .limit(500).stream())
    return [d.to_dict() for d in docs]


db = get_db()

# ------------------------------------------------------------
# 화면
# ------------------------------------------------------------
st.title("🧊 직육면체·정육면체 전개도 실험실")
st.caption("5학년 수학 5단원 ‘직육면체’ — 전개도를 접어 보며 마주 보는 면, 만나는 면, 길이가 같은 모서리를 탐구해요.")

tab_names = ["🧊 정육면체 전개도", "📦 직육면체 전개도", "✏️ 내가 만든 전개도"]
if db is not None:
    tab_names.append("👩‍🏫 우리 반 결과")
tabs = st.tabs(tab_names)

COLOR_MODES = ["면마다 다른 색", "마주 보는 면끼리 같은 색", "모두 같은 색"]

# ---------------- 탭 1: 정육면체 ----------------
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        net_name = st.selectbox("정육면체 전개도 11가지 중 고르기", list(CUBE_NETS.keys()), key="cube_sel")
    with c2:
        mode1 = st.radio("색칠 방법", COLOR_MODES, horizontal=True, key="cube_color")
    faces1 = build_net(rows_to_cells(CUBE_NETS[net_name]))
    show_net(faces1, mode1, key="cube")
    opposite_quiz(faces1, "cubeq")

# ---------------- 탭 2: 직육면체 ----------------
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a = st.slider("가로 (cm)", 1, 6, 4, key="a")
    with c2:
        b = st.slider("세로 (cm)", 1, 6, 2, key="b")
    with c3:
        c = st.slider("높이 (cm)", 1, 6, 3, key="c")
    with c4:
        shape = st.selectbox("전개도 모양", list(CUBE_NETS.keys()), index=4, key="box_sel")
    mode2 = st.radio("색칠 방법", COLOR_MODES, index=1, horizontal=True, key="box_color")
    faces2 = build_net(rows_to_cells(CUBE_NETS[shape]), a, b, c)
    if rects_overlap_2d(faces2):
        st.warning("이 크기에서는 펼친 전개도의 면끼리 겹쳐요. 다른 크기나 모양을 골라 보세요.")
    if a == b == c:
        st.info("가로, 세로, 높이가 모두 같아요 → 정육면체예요! 정육면체도 직육면체라고 할 수 있어요.")
    show_net(faces2, mode2, show_size=True, key="box")
    st.caption("각 면에 적힌 숫자는 ‘가로×세로’ 길이예요. 같은 색끼리(마주 보는 면) 크기가 같은지 확인해 보세요.")
    opposite_quiz(faces2, "boxq")

# ---------------- 탭 3: 내가 그린 전개도 ----------------
N = 5
if "grid" not in st.session_state:
    st.session_state.grid = set()
if "found" not in st.session_state:
    st.session_state.found = set()


def toggle(rc):
    g = st.session_state.grid
    g.symmetric_difference_update({rc})


with tabs[2]:
    st.markdown("칸을 눌러 **정사각형 6개**로 전개도를 그려 보세요. 정육면체가 될지 먼저 예상한 뒤 확인해요!")
    gl, gr = st.columns([1, 1.6])
    with gl:
        for r in range(N):
            cols = st.columns(N)
            for col_i in range(N):
                on = (r, col_i) in st.session_state.grid
                cols[col_i].button("🟦" if on else "⬜", key=f"g{r}_{col_i}",
                                   on_click=toggle, args=((r, col_i),))
        if st.button("🧹 모두 지우기", key="clear"):
            st.session_state.grid = set()
            st.rerun()

    cells = sorted(st.session_state.grid)
    with gr:
        st.markdown(f"**칠한 칸: {len(cells)}개**")
        ok, reason = check_cube_net(cells) if cells else (False, "칸을 눌러 그려 보세요.")
        net_type = identify_net(cells) if ok else None
        if ok:
            st.success(f"✅ {reason}  (11가지 중 **{net_type}** 와 같은 모양)")
            st.session_state.found.add(net_type)
        elif cells:
            st.error(f"❌ {reason}")

        if len(cells) == 6 and is_connected(cells):
            st.plotly_chart(fold_3d_figure(build_net(cells), PALETTE), key="mine3d")
            if not ok:
                st.caption("끝까지 접어서 어느 면이 겹치는지 찾아보세요.")

        found = st.session_state.found
        st.markdown(f"**내가 찾은 정육면체 전개도: {len(found)} / 11**")
        st.progress(len(found) / 11)
        if found:
            st.write(", ".join(sorted(found)))

        if db is not None and len(cells) > 0:
            name = st.text_input("모둠 이름 또는 번호 (실명 대신 번호나 별명)", key="who")
            if st.button("📤 선생님께 제출", key="send"):
                if not name.strip():
                    st.info("이름(번호)을 먼저 적어 주세요.")
                else:
                    try:
                        save_submission(db, name.strip(), cells, ok, net_type)
                        st.success("제출했어요!")
                    except Exception as e:
                        st.error(f"제출하지 못했어요: {e}")

# ---------------- 탭 4: 우리 반 결과 (파이어베이스 설정 시) ----------------
if db is not None:
    with tabs[3]:
        st.markdown("학생들이 제출한 전개도를 모아 봐요. 우리 반이 11가지를 모두 찾았을까요?")
        if st.button("🔄 새로고침", key="reload"):
            st.rerun()
        try:
            subs = load_submissions(db)
        except Exception as e:
            subs = []
            st.error(f"불러오지 못했어요: {e}")
        found_class = {s.get("type") for s in subs if s.get("valid") and s.get("type")}
        m1, m2, m3 = st.columns(3)
        m1.metric("제출 수", len(subs))
        m2.metric("정육면체가 되는 전개도", sum(1 for s in subs if s.get("valid")))
        m3.metric("찾은 종류", f"{len(found_class)} / 11")

        cols = st.columns(4)
        for i, (nm, rows) in enumerate(CUBE_NETS.items()):
            with cols[i % 4]:
                mark = "✅" if nm in found_class else "⬜"
                st.markdown(f"{mark} **{nm}**")
                st.code("\n".join(rows).replace(".", "□"), language=None)

        if subs:
            st.markdown("**제출 목록**")
            st.dataframe([{"이름": s.get("name"), "판정": "⭕" if s.get("valid") else "❌",
                           "모양": s.get("type") or "-"} for s in subs])

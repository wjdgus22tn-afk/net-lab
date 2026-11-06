"""
전개도 접기 계산 모듈 (화면과 상관없는 수학 부분)

- 정육면체 전개도 11가지
- 격자 전개도 → 직육면체(가로 a, 세로 b, 높이 c) 전개도로 바꾸기
- 접는 각도(0~90도)에 따라 각 면의 3D 꼭짓점 계산
- 내가 그린 전개도가 정육면체가 되는지 판별
"""
from collections import deque
import numpy as np

# ------------------------------------------------------------
# 1. 정육면체 전개도 11가지 (■ = 면)
# ------------------------------------------------------------
CUBE_NETS = {
    "1-4-1형 ①": ["■...", "■■■■", "■..."],
    "1-4-1형 ②": ["■...", "■■■■", ".■.."],
    "1-4-1형 ③": ["■...", "■■■■", "..■."],
    "1-4-1형 ④": ["■...", "■■■■", "...■"],
    "1-4-1형 ⑤": [".■..", "■■■■", ".■.."],
    "1-4-1형 ⑥": [".■..", "■■■■", "..■."],
    "2-3-1형 ①": ["■■..", ".■■■", ".■.."],
    "2-3-1형 ②": ["■■..", ".■■■", "..■."],
    "2-3-1형 ③": ["■■..", ".■■■", "...■"],
    "2-2-2형":   ["■■..", ".■■.", "..■■"],
    "3-3형":     ["■■■..", "..■■■"],
}

FACE_LABELS = ["①", "②", "③", "④", "⑤", "⑥"]


def rows_to_cells(rows):
    """문자열 그림 → (행, 열) 칸 목록"""
    return [(r, c) for r, line in enumerate(rows) for c, ch in enumerate(line) if ch == "■"]


# ------------------------------------------------------------
# 2. 모양 비교 (돌리거나 뒤집으면 같은 전개도인지)
# ------------------------------------------------------------
def _normalize(cells):
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def canonical(cells):
    """회전 4가지 × 뒤집기 2가지 중 가장 작은 모양을 대표로 사용"""
    forms = []
    pts = list(cells)
    for flip in (False, True):
        cur = [(r, -c) for r, c in pts] if flip else pts
        for _ in range(4):
            cur = [(c, -r) for r, c in cur]  # 90도 회전
            forms.append(_normalize(cur))
    return min(forms)


CANONICAL_TO_NAME = {canonical(rows_to_cells(v)): k for k, v in CUBE_NETS.items()}


def identify_net(cells):
    """내가 그린 모양이 11가지 중 어느 것과 같은지 (없으면 None)"""
    return CANONICAL_TO_NAME.get(canonical(cells))


# ------------------------------------------------------------
# 3. 격자 전개도 → 면(직사각형)과 접는 선(경첩) 만들기
# ------------------------------------------------------------
DIRS = {(0, 1): "+x", (0, -1): "-x", (-1, 0): "+y", (1, 0): "-y"}  # 행이 위로(-1) 가면 2D에서 +y


def _choose_root(cells):
    s = set(cells)
    cr = sum(r for r, _ in cells) / len(cells)
    cc = sum(c for _, c in cells) / len(cells)

    def score(cell):
        r, c = cell
        deg = sum((r + dr, c + dc) in s for dr, dc in DIRS)
        return (-deg, (r - cr) ** 2 + (c - cc) ** 2)

    return min(cells, key=score)


def is_connected(cells):
    if not cells:
        return False
    s = set(cells)
    seen = {cells[0]}
    q = deque([cells[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if n in s and n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen) == len(s)


def build_net(cells, a=1.0, b=1.0, c=1.0):
    """
    cells : 격자 칸 목록 (6개, 이어져 있어야 함)
    a, b, c : 직육면체의 가로, 세로, 높이 (정육면체는 모두 1)
    반환 : 면 정보 목록 (BFS 순서, 0번이 밑면)
      각 면: rect(x0,y0,x1,y1), parent, hinge(점1, 점2, 바깥방향), axis_n(이 면의 법선 축)
    """
    dims = [float(a), float(b), float(c)]
    s = set(cells)
    root = _choose_root(cells)
    faces = []
    index = {}

    # 밑면: 2D x축 ↔ 가로(0), 2D y축 ↔ 세로(1), 법선 ↔ 높이(2)
    faces.append(dict(cell=root, parent=None, ax_x=0, ax_y=1, axis_n=2,
                      rect=(0.0, 0.0, dims[0], dims[1]), hinge=None))
    index[root] = 0
    q = deque([root])
    while q:
        cur = q.popleft()
        p = faces[index[cur]]
        x0, y0, x1, y1 = p["rect"]
        for (dr, dc), d in DIRS.items():
            n = (cur[0] + dr, cur[1] + dc)
            if n not in s or n in index:
                continue
            if d in ("+x", "-x"):
                ax_x, ax_y, axis_n = p["axis_n"], p["ax_y"], p["ax_x"]
                w = dims[ax_x]
                if d == "+x":
                    rect = (x1, y0, x1 + w, y1)
                    hinge = ((x1, y0), (x1, y1), (1.0, 0.0))
                else:
                    rect = (x0 - w, y0, x0, y1)
                    hinge = ((x0, y0), (x0, y1), (-1.0, 0.0))
            else:
                ax_x, ax_y, axis_n = p["ax_x"], p["axis_n"], p["ax_y"]
                h = dims[ax_y]
                if d == "+y":
                    rect = (x0, y1, x1, y1 + h)
                    hinge = ((x0, y1), (x1, y1), (0.0, 1.0))
                else:
                    rect = (x0, y0 - h, x1, y0)
                    hinge = ((x0, y0), (x1, y0), (0.0, -1.0))
            index[n] = len(faces)
            faces.append(dict(cell=n, parent=index[cur], ax_x=ax_x, ax_y=ax_y,
                              axis_n=axis_n, rect=rect, hinge=hinge))
            q.append(n)
    return faces


# ------------------------------------------------------------
# 4. 접기 계산
# ------------------------------------------------------------
def _rotation_about_line(p, axis, theta):
    """점 p를 지나고 방향이 axis인 직선을 축으로 theta(라디안) 회전하는 4x4 행렬"""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    M = np.eye(4)
    M[:3, :3] = R
    p = np.asarray(p, float)
    M[:3, 3] = p - R @ p
    return M


def fold(faces, angle_deg):
    """접는 각도(0=펼침, 90=완성)에서 각 면의 3D 꼭짓점 4개(반시계)를 반환"""
    theta = np.radians(angle_deg)
    mats = []
    out = []
    for f in faces:
        if f["parent"] is None:
            M = np.eye(4)
        else:
            (px, py), _, (nx, ny) = f["hinge"]
            axis = (ny, -nx, 0.0)  # n × z : 면이 위(+z)로 들리도록
            M = mats[f["parent"]] @ _rotation_about_line((px, py, 0.0), axis, theta)
        mats.append(M)
        x0, y0, x1, y1 = f["rect"]
        corners = np.array([[x0, y0, 0, 1], [x1, y0, 0, 1], [x1, y1, 0, 1], [x0, y1, 0, 1]], float)
        out.append((M @ corners.T).T[:, :3])
    return out


def check_folds_into_box(faces):
    """다 접었을 때 6개 면이 서로 다른 자리에 가는지 확인"""
    if len(faces) != 6:
        return False
    pts = fold(faces, 90)
    centers = {tuple(np.round(p.mean(axis=0), 4)) for p in pts}
    return len(centers) == 6


def rects_overlap_2d(faces, eps=1e-9):
    """펼친 전개도에서 직사각형끼리 겹치는지 (직육면체 크기에 따라 생길 수 있음)"""
    for i in range(len(faces)):
        ax0, ay0, ax1, ay1 = faces[i]["rect"]
        for j in range(i + 1, len(faces)):
            bx0, by0, bx1, by1 = faces[j]["rect"]
            if ax0 < bx1 - eps and bx0 < ax1 - eps and ay0 < by1 - eps and by0 < ay1 - eps:
                return True
    return False


def check_cube_net(cells):
    """내가 그린 전개도 판별 → (정육면체가 되는가, 이유)"""
    n = len(cells)
    if n != 6:
        return False, f"면이 {n}개예요. 정육면체는 면이 6개여야 해요."
    if not is_connected(cells):
        return False, "떨어져 있는 면이 있어요. 모든 면이 변으로 이어져야 해요."
    s = set(cells)
    for r, c in cells:
        if {(r + 1, c), (r, c + 1), (r + 1, c + 1)} <= s:
            return False, "2×2로 뭉친 부분이 있어요. 접으면 면이 겹쳐요."
    if not check_folds_into_box(build_net(cells)):
        return False, "접어 보면 두 면이 같은 자리에 겹쳐서 빈 곳이 생겨요."
    return True, "정육면체의 전개도가 맞아요!"

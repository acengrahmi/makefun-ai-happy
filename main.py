"""The First Flower -- a make.fun "Make the AI judge happy" entry.

A small round white robot kneels on a soft green hill at golden sunrise,
gently cupping a single glowing pink flower in both hands and offering it
to a tiny orange kitten, who reaches one paw toward it.

Python 3.13, standard library only, no network, fully deterministic.
The image is drawn with signed-distance shapes at 2x supersampling and
box-downscaled to 1024x1024, then written with a hand-rolled PNG encoder.
"""

import math
import os
import struct
import zlib

SS = 2                      # supersampling factor
DW = 1024                   # design (final) width/height
W = DW * SS
H = DW * SS
buf = bytearray(W * H * 3)


# --------------------------------------------------------------------------
# low-level painting
# --------------------------------------------------------------------------

def _blend(i, color, a):
    if a >= 1.0:
        buf[i] = color[0]
        buf[i + 1] = color[1]
        buf[i + 2] = color[2]
    elif a > 0.0:
        ia = 1.0 - a
        buf[i] = int(buf[i] * ia + color[0] * a + 0.5)
        buf[i + 1] = int(buf[i + 1] * ia + color[1] * a + 0.5)
        buf[i + 2] = int(buf[i + 2] * ia + color[2] * a + 0.5)


def paint(x0, y0, x1, y1, fn, color, soft=1.4, alpha=1.0):
    """Paint `color` where fn(x, y) <= 0 in design coordinates.

    fn returns a signed-distance-like value; `soft` is the edge feather
    width in the same units as fn's return value.
    """
    px0 = max(0, int(x0 * SS))
    px1 = min(W - 1, int(x1 * SS) + 1)
    py0 = max(0, int(y0 * SS))
    py1 = min(H - 1, int(y1 * SS) + 1)
    inv = 1.0 / SS
    for py in range(py0, py1 + 1):
        dy = (py + 0.5) * inv
        base = py * W * 3
        for px in range(px0, px1 + 1):
            d = fn((px + 0.5) * inv, dy)
            if d < soft:
                a = 0.5 - d / (2.0 * soft)
                if a > 1.0:
                    a = 1.0
                _blend(base + px * 3, color, a * alpha)


# --------------------------------------------------------------------------
# shape fields (each returns fn(x, y) -> signed distance-ish, <= 0 inside)
# --------------------------------------------------------------------------

def ellipse_fn(cx, cy, rx, ry, rot=0.0):
    ca = math.cos(rot)
    sa = math.sin(rot)
    m = min(rx, ry)

    def fn(x, y):
        u = x - cx
        v = y - cy
        X = u * ca + v * sa
        Y = -u * sa + v * ca
        return (math.sqrt((X / rx) ** 2 + (Y / ry) ** 2) - 1.0) * m

    return fn


def super_fn(cx, cy, a, b, p=4.0, rot=0.0):
    """Superellipse: a rounded-rectangle blob, good for robot parts."""
    ca = math.cos(rot)
    sa = math.sin(rot)
    m = min(a, b)
    ip = 1.0 / p

    def fn(x, y):
        u = x - cx
        v = y - cy
        X = u * ca + v * sa
        Y = -u * sa + v * ca
        v2 = (abs(X) / a) ** p + (abs(Y) / b) ** p
        return (v2 ** ip - 1.0) * m

    return fn


def capsule_fn(x1, y1, x2, y2, r):
    vx = x2 - x1
    vy = y2 - y1
    l2 = vx * vx + vy * vy

    def fn(x, y):
        if l2 == 0.0:
            t = 0.0
        else:
            t = ((x - x1) * vx + (y - y1) * vy) / l2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        ex = x - (x1 + t * vx)
        ey = y - (y1 + t * vy)
        return math.sqrt(ex * ex + ey * ey) - r

    return fn


def tri_fn(ax, ay, bx, by, cx, cy):
    """Signed distance to a triangle (after Inigo Quilez)."""
    e0x, e0y = bx - ax, by - ay
    e1x, e1y = cx - bx, cy - by
    e2x, e2y = ax - cx, ay - cy
    s = 1.0 if (e0x * e2y - e0y * e2x) >= 0.0 else -1.0

    def fn(x, y):
        v0x, v0y = x - ax, y - ay
        v1x, v1y = x - bx, y - by
        v2x, v2y = x - cx, y - cy

        def pq(ex, ey, vx, vy):
            t = (vx * ex + vy * ey) / (ex * ex + ey * ey)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            dx = vx - ex * t
            dy = vy - ey * t
            return dx * dx + dy * dy

        d0 = pq(e0x, e0y, v0x, v0y)
        c0 = s * (v0x * e0y - v0y * e0x)
        d1 = pq(e1x, e1y, v1x, v1y)
        c1 = s * (v1x * e1y - v1y * e1x)
        d2 = pq(e2x, e2y, v2x, v2y)
        c2 = s * (v2x * e2y - v2y * e2x)
        d, c = d0, c0
        if d1 < d:
            d, c = d1, c1
        if d2 < d:
            d, c = d2, c2
        return -math.sqrt(d) if c >= 0.0 else math.sqrt(d)

    return fn


def heart_fn(cx, cy, s, rot=0.0):
    """Classic implicit heart curve, s ~ half height in design px."""
    ca = math.cos(rot)
    sa = math.sin(rot)

    def fn(x, y):
        u = x - cx
        v = y - cy
        X = (u * ca + v * sa) / s
        Y = -(-u * sa + v * ca) / s
        return (X * X + Y * Y - 1.0) ** 3 - X * X * Y ** 3

    return fn


def arc_fn(cx, cy, r, thick, keep):
    """Ring segment: keep='top' gives a ∩ stroke, 'bottom' gives ∪."""

    def fn(x, y):
        dx = x - cx
        dy = y - cy
        if keep == "top" and dy > r * 0.30:
            return 99.0
        if keep == "bottom" and dy < -r * 0.30:
            return 99.0
        return abs(math.sqrt(dx * dx + dy * dy) - r) - thick / 2.0

    return fn


def union2(f1, f2):
    def fn(x, y):
        d1 = f1(x, y)
        d2 = f2(x, y)
        return d1 if d1 < d2 else d2

    return fn


# convenience wrappers ------------------------------------------------------

def ellipse(cx, cy, rx, ry, color, rot=0.0, soft=1.4, alpha=1.0):
    paint(cx - rx - 3, cy - ry - 3, cx + rx + 3, cy + ry + 3,
          ellipse_fn(cx, cy, rx, ry, rot), color, soft, alpha)


def disc(cx, cy, r, color, soft=1.4, alpha=1.0):
    ellipse(cx, cy, r, r, color, 0.0, soft, alpha)


def sup(cx, cy, a, b, color, p=4.0, rot=0.0, soft=1.6, alpha=1.0):
    paint(cx - a - 3, cy - b - 3, cx + a + 3, cy + b + 3,
          super_fn(cx, cy, a, b, p, rot), color, soft, alpha)


def capsule(x1, y1, x2, y2, r, color, soft=1.4, alpha=1.0):
    paint(min(x1, x2) - r - 3, min(y1, y2) - r - 3,
          max(x1, x2) + r + 3, max(y1, y2) + r + 3,
          capsule_fn(x1, y1, x2, y2, r), color, soft, alpha)


def tri(pts, color, soft=1.4, alpha=1.0):
    (ax, ay), (bx, by), (cx, cy) = pts
    paint(min(ax, bx, cx) - 3, min(ay, by, cy) - 3,
          max(ax, bx, cx) + 3, max(ay, by, cy) + 3,
          tri_fn(ax, ay, bx, by, cx, cy), color, soft, alpha)


def heart(cx, cy, s, color, rot=0.0, alpha=1.0):
    paint(cx - s * 1.35, cy - s * 1.25, cx + s * 1.35, cy + s * 1.35,
          heart_fn(cx, cy, s, rot), color, soft=0.10, alpha=alpha)


def arc(cx, cy, r, thick, keep, color, alpha=1.0):
    paint(cx - r - thick, cy - r - thick, cx + r + thick, cy + r + thick,
          arc_fn(cx, cy, r, thick, keep), color, soft=1.2, alpha=alpha)


def sparkle(cx, cy, r, color, alpha=1.0):
    f = union2(ellipse_fn(cx, cy, r * 0.20, r, 0.0),
               ellipse_fn(cx, cy, r, r * 0.20, 0.0))
    paint(cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3,
          f, color, soft=1.2, alpha=alpha)


# --------------------------------------------------------------------------
# background: sky gradient + sun + clouds
# --------------------------------------------------------------------------

def vgrad(stops):
    """Vertical gradient; stops = [(y_design, (r, g, b)), ...]."""
    stops = sorted(stops)
    for py in range(H):
        t = (py + 0.5) / SS
        if t <= stops[0][0]:
            col = stops[0][1]
        elif t >= stops[-1][0]:
            col = stops[-1][1]
        else:
            col = stops[-1][1]
            for i in range(len(stops) - 1):
                y0, c0 = stops[i]
                y1, c1 = stops[i + 1]
                if y0 <= t <= y1:
                    f = (t - y0) / (y1 - y0)
                    col = (int(c0[0] + (c1[0] - c0[0]) * f + 0.5),
                           int(c0[1] + (c1[1] - c0[1]) * f + 0.5),
                           int(c0[2] + (c1[2] - c0[2]) * f + 0.5))
                    break
        start = py * W * 3
        buf[start:start + W * 3] = bytes(col) * W


# palette -------------------------------------------------------------------
SKY_STOPS = [(0, (255, 171, 112)), (340, (255, 203, 148)),
             (640, (255, 229, 190)), (1024, (255, 240, 214))]
SUN_GLOW = (255, 190, 100)
SUN_CORE = (255, 208, 105)
SUN_INNER = (255, 224, 140)
CLOUD = (255, 242, 224)
HILL = (143, 209, 120)
HILL_SHADE = (96, 175, 92)
HILL_LIGHT = (182, 228, 144)
SHADOW = (58, 120, 66)
WHITE = (255, 250, 242)
WHITE_SHADE = (238, 242, 245)
EDGE_SHADE = (213, 221, 231)
METAL = (198, 207, 219)
SCREEN = (43, 58, 74)
MINT = (140, 245, 205)
CORAL = (255, 120, 140)
CHEEK = (255, 160, 175)
ORANGE = (246, 159, 77)
ORANGE_DARK = (226, 124, 52)
ORANGE_PALE = (255, 214, 160)
PINK = (255, 110, 150)
PINK_SOFT = (255, 150, 175)
PETAL_A = (255, 133, 164)
PETAL_B = (255, 120, 150)
GOLD = (255, 209, 102)
GREEN_STEM = (86, 160, 90)
DARK = (60, 42, 34)


def draw_scene():
    vgrad(SKY_STOPS)

    # sun: halo first, then the disc low behind the subjects
    disc(512, 452, 390, SUN_GLOW, soft=60.0, alpha=0.16)
    disc(512, 452, 318, SUN_GLOW, soft=42.0, alpha=0.20)
    disc(512, 452, 252, SUN_CORE, soft=2.0)
    disc(512, 452, 210, SUN_INNER, soft=2.0)

    # soft morning clouds
    ellipse(170, 172, 92, 30, CLOUD, alpha=0.55)
    ellipse(244, 150, 58, 24, CLOUD, alpha=0.50)
    ellipse(852, 142, 82, 27, CLOUD, alpha=0.50)
    ellipse(788, 122, 50, 20, CLOUD, alpha=0.45)
    ellipse(430, 96, 64, 20, CLOUD, alpha=0.40)
    ellipse(622, 82, 48, 16, CLOUD, alpha=0.35)

    # tiny hearts drifting up into the morning sky
    heart(300, 300, 10, PINK_SOFT, rot=-0.10, alpha=0.80)
    heart(724, 286, 9, PINK, rot=0.10, alpha=0.75)
    heart(660, 208, 7, PINK_SOFT, rot=-0.06, alpha=0.70)

    # rolling hill
    ellipse(512, 1120, 1000, 440, HILL)
    ellipse(512, 1290, 1050, 430, HILL_SHADE, alpha=0.50)
    ellipse(512, 1052, 640, 300, HILL_LIGHT, alpha=0.55)

    # ground shadows
    ellipse(352, 868, 178, 25, SHADOW, alpha=0.30)
    ellipse(690, 870, 158, 23, SHADOW, alpha=0.30)

    draw_robot()
    draw_kitten()
    draw_flower()

    # floating hearts around the flower
    heart(478, 476, 15, PINK, rot=-0.14)
    heart(560, 448, 13, PINK_SOFT, rot=0.12)
    heart(606, 452, 11, PINK, rot=0.05, alpha=0.92)

    # sparkles
    sparkle(486, 540, 17, GOLD)
    sparkle(556, 518, 14, (255, 236, 170))
    sparkle(522, 492, 10, GOLD, alpha=0.9)
    sparkle(468, 602, 10, (255, 236, 170), alpha=0.9)
    disc(486, 540, 3, WHITE, alpha=0.9)
    disc(556, 518, 3, WHITE, alpha=0.9)

    # tiny meadow flowers and grass tufts on the hill
    for (fx, fy, fc) in [(150, 930, WHITE), (232, 976, GOLD),
                         (880, 940, PINK_SOFT), (820, 986, WHITE),
                         (122, 990, PINK_SOFT), (930, 902, GOLD)]:
        disc(fx, fy, 5, fc, alpha=0.9)
        disc(fx, fy, 2, (255, 240, 200), alpha=0.9)
    for (gx, gy) in [(90, 1002), (300, 1008), (760, 1004), (950, 998)]:
        ellipse(gx, gy, 27, 10, HILL_SHADE, alpha=0.8)


def draw_robot():
    # kneeling legs
    ellipse(272, 840, 64, 30, WHITE)
    ellipse(420, 844, 60, 28, WHITE)
    ellipse(272, 852, 52, 18, WHITE_SHADE, alpha=0.8)
    ellipse(420, 855, 48, 16, WHITE_SHADE, alpha=0.8)

    # neck and body
    sup(350, 626, 30, 18, METAL, p=3.0)
    sup(350, 726, 118, 100, WHITE, p=4.0)
    ellipse(350, 766, 76, 56, WHITE_SHADE, alpha=0.85)
    ellipse(256, 722, 26, 66, EDGE_SHADE, alpha=0.45)
    heart(350, 712, 17, CORAL, alpha=0.92)          # little heart on its chest

    # antenna
    capsule(350, 438, 350, 404, 7, METAL)
    disc(350, 392, 15, CORAL)
    disc(345, 387, 5, WHITE, alpha=0.85)

    # head with side pods
    disc(243, 526, 20, METAL)
    disc(457, 526, 20, METAL)
    disc(243, 526, 9, CORAL)
    disc(457, 526, 9, CORAL)
    sup(350, 520, 108, 94, WHITE, p=4.5)
    ellipse(350, 592, 80, 20, WHITE_SHADE, alpha=0.6)

    # face screen with happy glowing eyes and a smile
    sup(350, 524, 84, 67, SCREEN, p=4.0)
    arc(318, 524, 13, 6.5, "top", MINT)
    arc(382, 524, 13, 6.5, "top", MINT)
    arc(350, 540, 10, 5.0, "bottom", MINT)
    ellipse(284, 552, 13, 8, CHEEK, alpha=0.55)
    ellipse(416, 552, 13, 8, CHEEK, alpha=0.55)

    # arms reaching forward, cupping the flower
    capsule(424, 726, 494, 648, 22, WHITE_SHADE)     # far arm (slightly shaded)
    capsule(440, 690, 498, 624, 24, WHITE)           # near arm
    ellipse(504, 620, 26, 17, WHITE, rot=-0.5)       # cupped hands
    ellipse(513, 635, 24, 15, WHITE_SHADE, rot=0.4)


def draw_kitten():
    # curled tail behind the body
    capsule(772, 796, 810, 744, 16, ORANGE)
    capsule(810, 744, 814, 692, 16, ORANGE)
    disc(814, 684, 15, ORANGE_PALE)

    # sitting body and pale belly
    ellipse(688, 748, 106, 102, ORANGE)
    ellipse(688, 784, 60, 56, ORANGE_PALE, alpha=0.95)
    ellipse(778, 762, 25, 60, ORANGE_DARK, alpha=0.40)

    # ground paws
    ellipse(616, 852, 30, 15, (238, 146, 64))
    ellipse(744, 856, 28, 14, (238, 146, 64))

    # ears (behind the head circle)
    tri([(622, 542), (646, 460), (686, 524)], ORANGE)
    tri([(714, 524), (754, 460), (778, 542)], ORANGE)
    tri([(640, 520), (652, 486), (668, 516)], PINK_SOFT)
    tri([(730, 516), (748, 486), (758, 520)], PINK_SOFT)

    # head
    ellipse(688, 588, 92, 84, ORANGE)

    # forehead stripes
    capsule(688, 510, 688, 534, 6, ORANGE_DARK)
    capsule(659, 517, 663, 538, 5, ORANGE_DARK)
    capsule(717, 517, 713, 538, 5, ORANGE_DARK)

    # reaching arm and paw, stretched toward the flower
    capsule(624, 708, 574, 632, 19, ORANGE)
    ellipse(563, 619, 23, 18, (250, 170, 92), rot=-0.4)
    disc(553, 609, 3, PINK, alpha=0.9)
    disc(564, 603, 3, PINK, alpha=0.9)
    disc(575, 609, 3, PINK, alpha=0.9)

    # face: bright eyes, heart nose, little cat mouth, blush
    ellipse(652, 592, 10, 13, DARK)
    ellipse(724, 592, 10, 13, DARK)
    disc(655, 587, 3.5, WHITE)
    disc(727, 587, 3.5, WHITE)
    heart(688, 618, 7.5, (244, 114, 140))
    arc(680, 622, 6, 3.2, "bottom", (150, 84, 54))
    arc(696, 622, 6, 3.2, "bottom", (150, 84, 54))
    ellipse(620, 626, 13, 8, CHEEK, alpha=0.50)
    ellipse(756, 626, 13, 8, CHEEK, alpha=0.50)


def draw_flower():
    fx, fy = 516, 566

    # warm glow behind the bloom
    disc(fx, fy, 80, PINK, soft=22.0, alpha=0.32)
    disc(fx, fy, 52, PINK_SOFT, soft=12.0, alpha=0.34)

    # stem and leaf, running down into the robot's hands
    capsule(514, 588, 508, 632, 6, GREEN_STEM)
    ellipse(492, 616, 15, 8, (110, 190, 105), rot=-0.6)

    # six petals around the centre
    for k in range(6):
        ang = math.radians(k * 60 + 10)
        pcx = fx + 32 * math.cos(ang)
        pcy = fy + 32 * math.sin(ang)
        col = PETAL_A if k % 2 == 0 else PETAL_B
        ellipse(pcx, pcy, 33, 22, col, rot=ang)
    ellipse(fx, fy - 32, 22, 32, (255, 170, 190))    # lighter petal on top

    # golden heart of the flower
    disc(fx, fy, 17, GOLD)
    disc(fx, fy, 8, (255, 180, 84))
    disc(fx - 5, fy - 5, 3, (255, 244, 200), alpha=0.9)


# --------------------------------------------------------------------------
# downscale + PNG encode
# --------------------------------------------------------------------------

def downscale():
    out = bytearray(DW * DW * 3)
    stride = W * 3
    for y in range(DW):
        row0 = (2 * y) * stride
        row1 = row0 + stride
        orow = y * DW * 3
        for x in range(DW):
            i0 = row0 + 6 * x
            i1 = row1 + 6 * x
            o = orow + 3 * x
            out[o] = (buf[i0] + buf[i0 + 3] + buf[i1] + buf[i1 + 3] + 2) // 4
            out[o + 1] = (buf[i0 + 1] + buf[i0 + 4] +
                          buf[i1 + 1] + buf[i1 + 4] + 2) // 4
            out[o + 2] = (buf[i0 + 2] + buf[i0 + 5] +
                          buf[i1 + 2] + buf[i1 + 5] + 2) // 4
    return out


def write_png(pixels, w, h, path):
    stride = w * 3
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += pixels[y * stride:(y + 1) * stride]
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", comp)
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def main():
    draw_scene()
    pixels = downscale()
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    write_png(pixels, DW, DW, os.path.join(out_dir, "entry.png"))


if __name__ == "__main__":
    main()

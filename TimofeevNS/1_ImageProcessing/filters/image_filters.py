"""
Функции обработки изображений для практической работы №1.
Изображения: np.ndarray (H, W, 3), BGR, uint8.
Точки: (y, x).
"""

import random
import math
from collections import deque
from typing import List, Tuple, Set, Dict, Any, Optional

import numpy as np

Point = Tuple[int, int]

def _to_grayscale(image: np.ndarray) -> np.ndarray: #p obvious
    b = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    r = image[..., 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


def _convolve2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray: #p convolution
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray, ((ph, ph), (pw, pw)), mode='edge')
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def _dilate(mask: np.ndarray, thickness: int) -> np.ndarray: #p dilatation for edges
    if thickness <= 1:
        return mask.copy()
    r = (thickness - 1) // 2
    h, w = mask.shape
    padded = np.pad(mask, ((r, r), (r, r)), mode='constant')
    out = np.zeros((h, w), dtype=mask.dtype)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out = np.maximum(out, padded[r + dy:r + dy + h, r + dx:r + dx + w])
    return out


def _gaussian_kernel_1d(sigma: float) -> np.ndarray: #p math
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray: #p math
    k = _gaussian_kernel_1d(sigma)
    r = len(k) // 2
    h, w = gray.shape

    padded = np.pad(gray, ((r, r), (0, 0)), mode='edge')
    tmp = np.zeros_like(gray, dtype=np.float32)
    for i, kv in enumerate(k):
        tmp += kv * padded[i:i + h, :]

    padded = np.pad(tmp, ((0, 0), (r, r)), mode='edge')
    out = np.zeros_like(gray, dtype=np.float32)
    for i, kv in enumerate(k):
        out += kv * padded[:, i:i + w]
    return out


def _ellipse_mask(h: int, w: int) -> np.ndarray: #p obvious but math
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    a, b = w / 2.0, h / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    inside = ((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2 <= 1.0
    return inside.astype(np.uint8) * 255

def generate_random_points(image_shape: Tuple[int, ...], count: int) -> List[Point]: #p obvious
    h, w = image_shape[:2]
    return [(random.randint(0, h - 1), random.randint(0, w - 1))
            for _ in range(count)]

def add_white_noise(image: np.ndarray, count: int) -> np.ndarray: #p obvious
    result = image.copy()
    for (y, x) in generate_random_points(result.shape, count):
        result[y, x] = (255, 255, 255)
    return result


def add_black_noise(image: np.ndarray, count: int) -> np.ndarray: #p obvious
    result = image.copy()
    for (y, x) in generate_random_points(result.shape, count):
        result[y, x] = (0, 0, 0)
    return result


def add_salt_pepper_noise(image: np.ndarray, count: int) -> np.ndarray: #p obvious
    salt_count = count // 2
    pepper_count = count - salt_count
    result = add_white_noise(image, salt_count)
    result = add_black_noise(result, pepper_count)
    return result

def apply_ellipse_mask(image: np.ndarray, #p obvious
                       color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    h, w = image.shape[:2]
    mask = _ellipse_mask(h, w)
    result = image.copy()
    result[mask == 0] = color
    return result

FILM_DEFECT_TABLE: Dict[str, Dict[str, Any]] = {
    'infrared':    {'defect': 'color_shift', 'params': {'shift': (40, -20, -10)}},
    'faded':       {'defect': 'desaturate',  'params': {'percent': 0.7}},
}

def _fill_rhombus(mask: np.ndarray, #p math to fill 1 scratch preset
                  p1: Point, p2: Point,
                  half_diag_fraction: float = 0.02) -> None:
    """
    tbd
    """
    h, w = mask.shape[:2]
    y1, x1 = int(p1[0]), int(p1[1])
    y2, x2 = int(p2[0]), int(p2[1])

    dy = y2 - y1
    dx = x2 - x1
    length = math.hypot(dy, dx)

    if length == 0:
        if 0 <= y1 < h and 0 <= x1 < w:
            mask[y1, x1] = 255
        return

    sin_a = dy / length
    cos_a = dx / length

    my = (y1 + y2) / 2.0
    mx = (x1 + x2) / 2.0

    offset = length * half_diag_fraction
    x_off = offset * sin_a
    y_off = offset * cos_a

    py3 = my + y_off
    px3 = mx + x_off
    py4 = my - y_off
    px4 = mx - x_off

    cross_3 = (y2 - y1) * (px3 - x1) - (x2 - x1) * (py3 - y1)
    cross_4 = (y2 - y1) * (px4 - x1) - (x2 - x1) * (py4 - y1)
    if abs(cross_3) < 1e-6 and abs(cross_4) < 1e-6:
        steps = max(abs(dy), abs(dx), 1)
        for i in range(steps + 1):
            t = i / steps
            yy = int(round(y1 + t * dy))
            xx = int(round(x1 + t * dx))
            if 0 <= yy < h and 0 <= xx < w:
                mask[yy, xx] = 255
        return

    ys_all = (y1, y2, py3, py4)
    xs_all = (x1, x2, px3, px4)
    y_min = max(0, int(math.floor(min(ys_all))) - 1)
    y_max = min(h - 1, int(math.ceil(max(ys_all))) + 1)
    x_min = max(0, int(math.floor(min(xs_all))) - 1)
    x_max = min(w - 1, int(math.ceil(max(xs_all))) + 1)

    cy, cx = my, mx

    ref_1 = (px3 - x1) * (cy - y1) - (py3 - y1) * (cx - x1)
    ref_2 = (x2 - px3) * (cy - py3) - (y2 - py3) * (cx - px3)
    ref_3 = (px4 - x2) * (cy - y2) - (py4 - y2) * (cx - x2)
    ref_4 = (x1 - px4) * (cy - py4) - (y1 - py4) * (cx - px4)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            s1 = (px3 - x1) * (y - y1) - (py3 - y1) * (x - x1)
            s2 = (x2 - px3) * (y - py3) - (y2 - py3) * (x - px3)
            s3 = (px4 - x2) * (y - y2) - (py4 - y2) * (x - x2)
            s4 = (x1 - px4) * (y - py4) - (y1 - py4) * (x - px4)
            if (s1 * ref_1 >= 0 and s2 * ref_2 >= 0
                    and s3 * ref_3 >= 0 and s4 * ref_4 >= 0):
                mask[y, x] = 255


def generate_scratches(image_shape: Tuple[int, ...], #p generate scratch presets in a controlled manner
                       scratch_coefficient: float,
                       max_segments: int = None,
                       max_thickness: int = 3,
                       max_length: float = None
                       ) -> List[Tuple[Point, Point, int]]:
    """
    tbd
    """
    h, w = image_shape[:2]
    # Количество: не более area*0.025, если не задано явно
    if max_segments is None or max_segments <= 0:
        max_segments = max(1, (h * w) * 0.025)
        #max_segments = max(1, (h * w) // 10000)

    n = max(1, int(max_segments * scratch_coefficient))

    # Длина: не более 2.5% от наибольшей стороны, если не задано явно
    if max_length is None or max_length <= 0:
        max_length = 0.025 * max(h, w)

    if max_length <= 0:
        max_length = max_segments + 5

    scratches: List[Tuple[Point, Point, int]] = []
    attempts = 0
    max_attempts = n * 100

    while len(scratches) < n and attempts < max_attempts:
        remaining = n - len(scratches)
        batch: List[Tuple[Point, Point, int]] = []

        for _ in range(remaining):
            if attempts >= max_attempts:
                break
            p1 = (random.randint(0, h - 1), random.randint(0, w - 1))
            p2 = (random.randint(0, h - 1), random.randint(0, w - 1))
            attempts += 1

            length = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
            if length <= max_length:
                thickness = random.randint(1, max_thickness)
                batch.append((p1, p2, thickness))

        need = n - len(scratches)
        scratches.extend(batch[:need])

    return scratches


def scratches_to_mask(image_shape: Tuple[int, ...], #p go through each scratch preset
                      scratches: List[Tuple[Point, Point, int]],
                      half_diag_fraction: float = 0.02) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for entry in scratches:
        p1, p2 = entry[0], entry[1]
        _fill_rhombus(mask, p1, p2, half_diag_fraction)
    return mask


def apply_scratches(image: np.ndarray, #p gen_scr + scratches->mask + img+[color where scr mask > 0]
                    scratch_coefficient: float,
                    scratch_color: Tuple[int, int, int] = (255, 255, 255),
                    intensity: float = 0.5,
                    max_segments: int = None,
                    max_thickness: int = 3,
                    half_diag_fraction: float = 0.02) -> np.ndarray:
    scratches = generate_scratches(image.shape, scratch_coefficient,
                                   max_segments, max_thickness)
    mask = scratches_to_mask(image.shape, scratches, half_diag_fraction)
    mask_bool = mask > 0

    result = image.copy()
    color_layer = np.full_like(image, scratch_color)
    result[mask_bool] = (
        result[mask_bool].astype(np.float32) * (1 - intensity)
        + color_layer[mask_bool].astype(np.float32) * intensity
    ).astype(np.uint8)
    return result

_SOBEL_X = np.array([[-1, 0, 1], #p
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

_SOBEL_Y = np.array([[-1, -2, -1], #p
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)

def detect_edges(image: np.ndarray, #p sobel
                 thickness: int = 1,
                 low_threshold: int = 100,
                 high_threshold: int = 200) -> np.ndarray:
    gray = _to_grayscale(image)
    gx = _convolve2d(gray, _SOBEL_X)
    gy = _convolve2d(gray, _SOBEL_Y)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    strong = (mag >= high_threshold)
    weak = (mag >= low_threshold) & ~strong

    edges = np.zeros_like(mag, dtype=np.uint8)
    edges[strong] = 255

    if weak.any() and strong.any():
        strong_dil = _dilate(edges, 3)
        edges[weak & (strong_dil > 0)] = 255

    if thickness > 1:
        edges = _dilate(edges, thickness)
    return edges

def neonify(image: np.ndarray, #p edges + blur + img postprocessing + summarising
            thickness: int = 3,
            glow_color: Tuple[int, int, int] = (255, 0, 255),
            glow_intensity: float = 0.8,
            blur_sigma: float = 7.0) -> np.ndarray:
    edges = detect_edges(image, thickness=thickness)
    glow = _gaussian_blur(edges.astype(np.float32), sigma=blur_sigma)
    glow = np.clip(glow, 0, 255)

    glow_colored = np.zeros_like(image)
    glow_colored[:] = glow_color

    alpha = np.clip((glow / 255.0) * glow_intensity, 0.0, 1.0)[..., None]

    darkened = image.astype(np.float32) * 0.3
    result = darkened * (1 - alpha) + glow_colored.astype(np.float32) * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)

    result[edges > 0] = glow_color
    return result

# 16-17. edxdy — регионы между вложенными эллипсами

def _ellipse_boundary(cx: float, cy: float, #p kinda obvious
                      a: float, b: float,
                      W: int, H: int) -> Set[Point]:
    if a <= 0 or b <= 0:
        return set()

    perimeter = math.pi * (
        3 * (a + b) - math.sqrt(max(0.0, (3 * a + b) * (a + 3 * b)))
    )
    n_samples = max(8, int(perimeter * 4))

    pts: Set[Point] = set()
    for k in range(n_samples):
        theta = 2.0 * math.pi * k / n_samples
        x = int(round(cx + a * math.cos(theta)))
        y = int(round(cy + b * math.sin(theta)))
        if 0 <= y < H and 0 <= x < W:
            pts.add((y, x))
    return pts


def edxdy(image: np.ndarray, #p return elliptic rings
          center: Point,
          h_semi: float, v_semi: float,
          dx: float, dy: float,
          max_rings: int = 500) -> List[Set[Point]]:
    H, W = image.shape[:2]
    cy, cx = center

    if dx <= 0 or dy <= 0:
        raise ValueError('dx and dy must be positive')
    if h_semi <= 0 or v_semi <= 0:
        raise ValueError('semi-axes must be positive')

    ellipses: List[Tuple[float, float]] = []
    i = 0
    while i < max_rings:
        a = h_semi + i * dx
        b = v_semi + i * dy
        ellipses.append((a, b))

        max_h = max(abs(cx - 0.0), abs(cx - (W - 1)))
        max_v = max(abs(cy - 0.0), abs(cy - (H - 1)))
        if (max_h / a) ** 2 + (max_v / b) ** 2 <= 1.0:
            break
        i += 1

    N = len(ellipses)
    if N == 0:
        return []

    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]

    result: List[Set[Point]] = []
    prev_inside = None

    for i, (a, b) in enumerate(ellipses):
        e = ((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2
        cur_inside = e <= 1.0

        if i == 0:
            region_mask = cur_inside
        else:
            region_mask = cur_inside & ~prev_inside

        ys_r, xs_r = np.where(region_mask)
        result.append(set(zip(ys_r.tolist(), xs_r.tolist())))
        prev_inside = cur_inside

    return result

def _apply_vignette_random_sets(image: np.ndarray, #p applies procedure for each ring
                                rings: List[Set[Point]],
                                brighten: bool,
                                interval_len: float,
                                strength: float) -> np.ndarray:
    N = len(rings)
    if N <= 1:
        return image.copy()

    L = float(interval_len)
    if not 0.0 < L <= 1.0:
        raise ValueError('interval_len must be in (0, 1]')

    result = image.astype(np.float32)
    for i, ring in enumerate(rings):
        t = i / (N - 1)
        lo = t * (1.0 - L)
        hi = lo + L
        for (y, x) in ring:
            r = random.uniform(lo, hi) * strength
            r = min(max(r, 0.0), 1.0)
            px = result[y, x]
            if brighten:
                result[y, x] = px + (255.0 - px) * r
            else:
                result[y, x] = px * (1.0 - r)
    return np.clip(result, 0, 255).astype(np.uint8)


def vignette_random_rings(image: np.ndarray, #p edxdy + for each ring
                          center: Point,
                          h_semi: float, v_semi: float,
                          dx: float, dy: float,
                          brighten: bool,
                          interval_len: float = 0.2,
                          strength: float = 1.0) -> np.ndarray:
    rings = edxdy(image, center, h_semi, v_semi, dx, dy)
    return _apply_vignette_random_sets(image, rings, brighten,
                                       interval_len, strength)

_SEPIA_MATRIX = np.array([ #p
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131],
], dtype=np.float32)


def apply_sepia(image: np.ndarray, #p
                strength: float = 1.0) -> np.ndarray:
    r = image[..., 2].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    b = image[..., 0].astype(np.float32)

    new_r = _SEPIA_MATRIX[0, 0] * r + _SEPIA_MATRIX[0, 1] * g + _SEPIA_MATRIX[0, 2] * b
    new_g = _SEPIA_MATRIX[1, 0] * r + _SEPIA_MATRIX[1, 1] * g + _SEPIA_MATRIX[1, 2] * b
    new_b = _SEPIA_MATRIX[2, 0] * r + _SEPIA_MATRIX[2, 1] * g + _SEPIA_MATRIX[2, 2] * b

    sepia = np.stack([new_b, new_g, new_r], axis=-1)
    sepia = np.clip(sepia, 0, 255)

    if strength >= 1.0:
        return sepia.astype(np.uint8)
    if strength <= 0.0:
        return image.copy()

    blended = image.astype(np.float32) * (1.0 - strength) + sepia * strength
    return np.clip(blended, 0, 255).astype(np.uint8)

def apply_aging(image: np.ndarray, #p sepia + scratches + vingette
                center: Point,
                sepia_strength: float = 1.0,
                scratch_coefficient: float = 0.8,
                scratch_color: Tuple[int, int, int] = (255, 255, 255),
                scratch_intensity: float = 0.6,
                max_segments: int = None,
                max_thickness: int = 3,
                half_diag_fraction: float = 0.02,
                vignette_h_semi: Optional[float] = None,
                vignette_v_semi: Optional[float] = None,
                vignette_dx: float = 10.0,
                vignette_dy: float = 10.0,
                vignette_brighten: bool = False,
                vignette_interval_len: float = 0.2,
                vignette_strength: float = 1.0) -> np.ndarray:
    H, W = image.shape[:2]
    if vignette_h_semi is None:
        vignette_h_semi = W / 4.0
    if vignette_v_semi is None:
        vignette_v_semi = H / 4.0

    result = apply_sepia(image, strength=sepia_strength)
    result = apply_scratches(result,
                             scratch_coefficient=scratch_coefficient,
                             scratch_color=scratch_color,
                             intensity=scratch_intensity,
                             max_segments=max_segments,
                             max_thickness=max_thickness,
                             half_diag_fraction=half_diag_fraction)
    result = vignette_random_rings(result,
                                   center=center,
                                   h_semi=vignette_h_semi,
                                   v_semi=vignette_v_semi,
                                   dx=vignette_dx,
                                   dy=vignette_dy,
                                   brighten=vignette_brighten,
                                   interval_len=vignette_interval_len,
                                   strength=vignette_strength)
    return result

def resize_image(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray: #p nearest neighbour
    h, w = image.shape[:2]
    if new_width == w and new_height == h:
        return image.copy()

    out = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    x_scale = w / new_width
    y_scale = h / new_height

    for y in range(new_height):
        src_y = int(y * y_scale)
        if src_y >= h:
            src_y = h - 1
        for x in range(new_width):
            src_x = int(x * x_scale)
            if src_x >= w:
                src_x = w - 1
            out[y, x] = image[src_y, src_x]

    return out


def convert_to_grayscale(image: np.ndarray) -> np.ndarray: #p all 3 channels are the same
    gray = _to_grayscale(image)  # float32
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


def desaturate_image(image: np.ndarray, percent: float = 1.0) -> np.ndarray: #p mix gray and normal img
    f = image.astype(np.float32)
    gray = _to_grayscale(image)  # 2D float32
    gray_img = np.stack([gray, gray, gray], axis=-1)
    out = f + (gray_img - f) * percent
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_film_whole(image: np.ndarray, film_type: str) -> np.ndarray: #p action depending on film type
    if film_type not in FILM_DEFECT_TABLE:
        return image.copy()

    spec = FILM_DEFECT_TABLE[film_type]
    kind = spec['defect']
    params = spec['params']
    f = image.astype(np.float32)

    if kind == 'color_shift':
        out = f + np.array(params['shift'], dtype=np.float32)
    elif kind == 'desaturate':
        p = params['percent']
        # crutch for avoiding format problems, could've
        # call grayscale with orig img, then image.astype(np.float32)
        gray = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
        gray_img = np.stack([gray, gray, gray], axis=-1)
        out = f + (gray_img - f) * p
    else:
        return image.copy()

    return np.clip(out, 0, 255).astype(np.uint8)

FILM_TYPES = tuple(FILM_DEFECT_TABLE.keys())

def apply_antique(image: np.ndarray, #p
                  center: Point,
                  desaturate_percent: float = 0.7,
                  noise_count: int = 500,
                  vignette_h_semi: Optional[float] = None,
                  vignette_v_semi: Optional[float] = None,
                  vignette_dx: float = 10.0,
                  vignette_dy: float = 10.0,
                  vignette_brighten: bool = False,
                  vignette_interval_len: float = 0.2,
                  vignette_strength: float = 1.0,
                  sepia_strength: float = 1.0) -> np.ndarray:
    """
        1) выцветание (desaturate),
        2) наложение шума (соль и перец),
        3) случайная виньетка по кольцам,
        4) сепия.
    """
    H, W = image.shape[:2]
    if vignette_h_semi is None:
        vignette_h_semi = W / 4.0
    if vignette_v_semi is None:
        vignette_v_semi = H / 4.0

    result = desaturate_image(image, percent=desaturate_percent)
    result = add_salt_pepper_noise(result, count=noise_count)
    result = vignette_random_rings(result,
                                   center=center,
                                   h_semi=vignette_h_semi,
                                   v_semi=vignette_v_semi,
                                   dx=vignette_dx,
                                   dy=vignette_dy,
                                   brighten=vignette_brighten,
                                   interval_len=vignette_interval_len,
                                   strength=vignette_strength)
    result = apply_sepia(result, strength=sepia_strength)
    return result

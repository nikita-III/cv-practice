import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np

from image_filters import (
    apply_ellipse_mask,
    neonify,
    apply_aging,
    convert_to_grayscale,
    resize_image,
    desaturate_image,
    apply_film_whole,
    apply_antique,
)


def parse_color(s: str) -> Tuple[int, int, int]:
    parts = s.split(',')
    return tuple(int(p.strip()) for p in parts)


def get_filter_map():
    """Словарь: имя фильтра -> функция(image, params)."""
    m = {
        'grayscale': lambda img, p: convert_to_grayscale(img), #p grayscale
        'resize': lambda img, p: resize_image(img, p['resize_width'], p['resize_height']), #p resize
        'ellipse_mask': lambda img, p: apply_ellipse_mask(img, parse_color(p['mask_color'])), #p matte
        'neon': lambda img, p: neonify( #p neon
            img,
            thickness=p['neon_thickness'],
            glow_color=parse_color(p['neon_glow_color']),
            glow_intensity=p['neon_glow_intensity'],
            blur_sigma=p['neon_blur_sigma'],
        ),
        'aging': lambda img, p: apply_aging( #p antique
            img,
            center=(img.shape[0] // 2, img.shape[1] // 2),
            sepia_strength=p['aging_sepia_strength'],
            scratch_coefficient=p['aging_scratch_coefficient'],
            scratch_color=parse_color(p['aging_scratch_color']),
            scratch_intensity=p['aging_scratch_intensity'],
            max_segments=p['aging_max_segments'],
            max_thickness=p['aging_max_thickness'],
            half_diag_fraction=p['aging_half_diag_fraction'],
            vignette_h_semi=p['aging_vignette_h_semi'],
            vignette_v_semi=p['aging_vignette_v_semi'],
            vignette_dx=p['aging_vignette_dx'],
            vignette_dy=p['aging_vignette_dy'],
            vignette_brighten=p['aging_vignette_brighten'],
            vignette_interval_len=p['aging_vignette_interval_len'],
            vignette_strength=p['aging_vignette_strength'],
        ),
        'desaturate': lambda img, p: desaturate_image(img, percent=p['desaturate_percent']), #p fade
        'antique': lambda img, p: apply_antique(  # p antique
            img,
            center=(img.shape[0] // 2, img.shape[1] // 2),
            desaturate_percent=p['antique_desaturate_percent'],
            noise_count=p['antique_noise_count'],
            vignette_h_semi=p['antique_vignette_h_semi'],
            vignette_v_semi=p['antique_vignette_v_semi'],
            vignette_dx=p['antique_vignette_dx'],
            vignette_dy=p['antique_vignette_dy'],
            vignette_brighten=p['antique_vignette_brighten'],
            vignette_interval_len=p['antique_vignette_interval_len'],
            vignette_strength=p['antique_vignette_strength'],
            sepia_strength=p['antique_sepia_strength'],
        ),
        'infrared': lambda img, p: apply_film_whole(img, 'infrared'),  # p infrared film
    }
    return m


def apply_filters(image: np.ndarray, filter_names: List[str], params: dict) -> np.ndarray:
    """Последовательно применяет фильтры к изображению."""
    fmap = get_filter_map()
    result = image.copy()
    for name in filter_names:
        if name not in fmap:
            raise ValueError(f'Неизвестный фильтр: {name}')
        result = fmap[name](result, params)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Применение фильтров к изображению.')
    parser.add_argument('input', nargs='?', help='Путь к входному изображению.')
    parser.add_argument('-f', '--filter', dest='filters', action='append', default=[],
                        help='Имя фильтра (можно несколько раз).')
    parser.add_argument('-o', '--output', default='output', help='Каталог для результатов.')
    parser.add_argument('-l', '--list', action='store_true', help='Показать доступные фильтры.')
    parser.add_argument('--resize-width', type=int, default=100)
    parser.add_argument('--resize-height', type=int, default=100)
    parser.add_argument('--noise-count', type=int, default=None)
    parser.add_argument('--mask-color', default='255,255,255')
    parser.add_argument('--scratch-coefficient', type=float, default=1.0)
    parser.add_argument('--scratch-color', default='255,255,255')
    parser.add_argument('--scratch-intensity', type=float, default=0.5)
    parser.add_argument('--max-segments', type=int, default=None)
    parser.add_argument('--max-thickness', type=int, default=3)
    parser.add_argument('--half-diag-fraction', type=float, default=0.02)
    parser.add_argument('--edges-thickness', type=int, default=1)
    parser.add_argument('--edges-low-threshold', type=int, default=100)
    parser.add_argument('--edges-high-threshold', type=int, default=200)
    parser.add_argument('--neon-thickness', type=int, default=3)
    parser.add_argument('--neon-glow-color', default='255,0,255')
    parser.add_argument('--neon-glow-intensity', type=float, default=0.8)
    parser.add_argument('--neon-blur-sigma', type=float, default=7.0)
    parser.add_argument('--sepia-strength', type=float, default=1.0)
    parser.add_argument('--aging-sepia-strength', type=float, default=1.0)
    parser.add_argument('--aging-scratch-coefficient', type=float, default=0.8)
    parser.add_argument('--aging-scratch-color', default='255,255,255')
    parser.add_argument('--aging-scratch-intensity', type=float, default=0.6)
    parser.add_argument('--aging-max-segments', type=int, default=None)
    parser.add_argument('--aging-max-thickness', type=int, default=3)
    parser.add_argument('--aging-half-diag-fraction', type=float, default=0.02)
    parser.add_argument('--aging-vignette-h-semi', type=float, default=None)
    parser.add_argument('--aging-vignette-v-semi', type=float, default=None)
    parser.add_argument('--aging-vignette-dx', type=float, default=10.0)
    parser.add_argument('--aging-vignette-dy', type=float, default=10.0)
    parser.add_argument('--aging-vignette-brighten', action='store_true', default=False)
    parser.add_argument('--aging-vignette-interval-len', type=float, default=0.2)
    parser.add_argument('--aging-vignette-strength', type=float, default=1.0)
    parser.add_argument('--desaturate-percent', type=float, default=0.7)
    parser.add_argument('--vignette-h-semi', type=float, default=None)
    parser.add_argument('--vignette-v-semi', type=float, default=None)
    parser.add_argument('--vignette-dx', type=float, default=10.0)
    parser.add_argument('--vignette-dy', type=float, default=10.0)
    parser.add_argument('--vignette-brighten', action='store_true', default=False)
    parser.add_argument('--vignette-interval-len', type=float, default=0.2)
    parser.add_argument('--vignette-strength', type=float, default=1.0)
    parser.add_argument('--antique-desaturate-percent', type=float, default=0.7)
    parser.add_argument('--antique-noise-count', type=int, default=500)
    parser.add_argument('--antique-vignette-h-semi', type=float, default=None)
    parser.add_argument('--antique-vignette-v-semi', type=float, default=None)
    parser.add_argument('--antique-vignette-dx', type=float, default=10.0)
    parser.add_argument('--antique-vignette-dy', type=float, default=10.0)
    parser.add_argument('--antique-vignette-brighten', action='store_true', default=False)
    parser.add_argument('--antique-vignette-interval-len', type=float, default=0.2)
    parser.add_argument('--antique-vignette-strength', type=float, default=1.0)
    parser.add_argument('--antique-sepia-strength', type=float, default=1.0)
    parser.add_argument('--all', action='store_true',
                        help='Применить все фильтры по отдельности с дефолтными параметрами.')

    return parser

def apply_all_filters(image: np.ndarray, params: dict,
                      output_dir: str, base: str) -> List[str]:
    """
    Применяет каждый фильтр по отдельности к исходному изображению
    с параметрами из params (по умолчанию — дефолтные) и сохраняет
    результат каждого в отдельный файл в output_dir.
    """
    fmap = get_filter_map()
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    for name in sorted(fmap.keys()):
        try:
            result = fmap[name](image.copy(), params)
            out_path = os.path.join(output_dir, f'{base}_{name}.png')
            cv2.imwrite(out_path, result)
            saved.append(out_path)
            print(f'  -> {out_path}')
        except Exception as e:
            print(f'  !! {name}: {e}')

    return saved

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print('Доступные фильтры:')
        for name in sorted(get_filter_map().keys()):
            print(f'  {name}')
        return

    if args.input is None:
        parser.error('Не указан входной файл (или используйте --list).')

    image = cv2.imread(args.input)
    if image is None:
        parser.error(f'Не удалось прочитать: {args.input}')

    params = vars(args)
    base = os.path.splitext(os.path.basename(args.input))[0]

    if args.all:
        saved = apply_all_filters(image, params, args.output, base)
        print(f'\nГотово. Сохранено файлов: {len(saved)}')
        return

    try:
        result = apply_filters(image, args.filters, params)
    except ValueError as e:
        parser.error(str(e))
        return

    os.makedirs(args.output, exist_ok=True)
    suffix = '_'.join(args.filters) if args.filters else 'all'
    out_path = os.path.join(args.output, f'{base}_{suffix}.png')
    cv2.imwrite(out_path, result)
    print(f'Сохранено: {out_path}')


if __name__ == '__main__':
    main()

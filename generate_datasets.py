"""Генериране на синтетични видео-набори за тестване на приложението.

Всеки „субект“ има уникални морфологични и кинематични параметри:
ръст, телосложение, дължина на крачката, амплитуда на ръцете, каданс.
За всеки субект се записват N видеа, в които субектът пресича кадъра
(с малки случайни вариации в начална позиция и скорост).

Резултатът се записва в `data/datasets/<dataset_name>/<subject>/<video>.mp4`
– структурата, която приложението очаква от ZIP-архив.

    python generate_datasets.py                  # малък набор: 6 × 5
    python generate_datasets.py --large          # + голям набор: 12 × 8
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data" / "datasets"

# Размери на кадъра
FRAME_H = 240
FRAME_W = 480
FPS = 25

# Размери на "сцената" в пиксели, в които се рисува субектът
PERSON_H_RANGE = (120, 180)  # вертикален габарит на тялото


@dataclass
class SubjectProfile:
    """Уникални параметри, които дефинират „походката“ на един субект."""

    subject_id: str
    height: int            # общ ръст в пиксели
    head_r: int            # радиус на главата
    body_w: int            # ширина на тялото
    body_h: int            # височина на тялото
    arm_len: int
    leg_len: int
    stride_freq: float     # цикъла/кадър (колко бързо „крачи“)
    arm_swing: float       # амплитуда на ръцете в радиани
    leg_swing: float       # амплитуда на краката в радиани
    torso_lean: float      # наклон напред в радиани
    walk_speed: float      # пиксели/кадър хоризонтално
    bg_color: tuple[int, int, int]  # фонов оттенък (леко уникален на субект, за по-добър MOG2)


def sample_subject(rng: np.random.Generator, sid: int) -> SubjectProfile:
    """Генерира случайни, но различими един от друг профили."""
    # Дискретни „типове“ – гарантират видима разлика между субектите
    kinds = [
        dict(height=140, body_w=22, body_h=50, leg_len=45, arm_len=38),   # дребен
        dict(height=160, body_w=28, body_h=56, leg_len=55, arm_len=44),   # среден
        dict(height=180, body_w=34, body_h=64, leg_len=62, arm_len=50),   # висок
        dict(height=170, body_w=40, body_h=58, leg_len=55, arm_len=46),   # широк
    ]
    base = kinds[sid % len(kinds)]

    return SubjectProfile(
        subject_id=f"subject_{sid:02d}",
        height=base["height"] + int(rng.integers(-6, 7)),
        head_r=int(rng.integers(10, 14)),
        body_w=base["body_w"] + int(rng.integers(-3, 4)),
        body_h=base["body_h"] + int(rng.integers(-3, 4)),
        arm_len=base["arm_len"] + int(rng.integers(-3, 4)),
        leg_len=base["leg_len"] + int(rng.integers(-3, 4)),
        stride_freq=float(rng.uniform(0.08, 0.16)),
        arm_swing=float(rng.uniform(0.35, 0.80)),
        leg_swing=float(rng.uniform(0.35, 0.70)),
        torso_lean=float(rng.uniform(-0.10, 0.10)),
        walk_speed=float(rng.uniform(2.8, 4.2)),
        bg_color=(
            int(rng.integers(140, 200)),
            int(rng.integers(140, 200)),
            int(rng.integers(140, 200)),
        ),
    )


def draw_person(
    frame: np.ndarray, profile: SubjectProfile, x_center: int, phase: float
) -> None:
    """Рисува „човек“ като система от прости геометрични форми."""
    h, w = frame.shape[:2]

    # Крака – долната точка на тялото
    ground_y = int(h * 0.92)
    body_bottom_y = ground_y - profile.leg_len

    body_top_y = body_bottom_y - profile.body_h
    head_cy = body_top_y - profile.head_r - 4

    color = (50, 50, 50)  # тъмен силует – контрастен спрямо фона
    thickness = -1

    lean_dx = int(profile.torso_lean * profile.body_h)

    # Глава (с малка биомеханична нестационарност – ±1px вертикално)
    head_offset_y = int(2 * np.sin(phase * 2))
    cv2.circle(
        frame,
        (x_center + lean_dx, head_cy + head_offset_y),
        profile.head_r,
        color,
        thickness,
    )

    # Тяло – трапец
    bw_top = profile.body_w - 4
    bw_bot = profile.body_w
    pts = np.array(
        [
            [x_center - bw_top // 2 + lean_dx, body_top_y],
            [x_center + bw_top // 2 + lean_dx, body_top_y],
            [x_center + bw_bot // 2, body_bottom_y],
            [x_center - bw_bot // 2, body_bottom_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], color)

    # Ръце – люлеене в противофаза
    arm_angle_l = np.sin(phase) * profile.arm_swing
    arm_angle_r = -arm_angle_l
    shoulder_l = (x_center - bw_top // 2 + lean_dx, body_top_y + 6)
    shoulder_r = (x_center + bw_top // 2 + lean_dx, body_top_y + 6)
    _draw_limb(frame, shoulder_l, profile.arm_len, arm_angle_l, thickness=5, color=color)
    _draw_limb(frame, shoulder_r, profile.arm_len, arm_angle_r, thickness=5, color=color)

    # Крака – в противофаза на ръцете, но с двойна стъпка
    leg_angle_l = np.sin(phase + np.pi) * profile.leg_swing
    leg_angle_r = -leg_angle_l
    hip_l = (x_center - bw_bot // 4, body_bottom_y)
    hip_r = (x_center + bw_bot // 4, body_bottom_y)
    _draw_limb(frame, hip_l, profile.leg_len, leg_angle_l, thickness=7, color=color)
    _draw_limb(frame, hip_r, profile.leg_len, leg_angle_r, thickness=7, color=color)


def _draw_limb(
    frame: np.ndarray,
    start: tuple[int, int],
    length: int,
    angle: float,
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    """Рисува крайник с „коляно/лакът“ в средата за по-естествена форма."""
    # Горна част – накланя се според angle
    mid = (
        int(start[0] + np.sin(angle) * length * 0.5),
        int(start[1] + np.cos(angle) * length * 0.5),
    )
    # Долна част – продължава по същата посока, но с допълнителна флексия
    flex = 0.2 * np.cos(angle * 2)
    end = (
        int(mid[0] + np.sin(angle + flex) * length * 0.5),
        int(mid[1] + np.cos(angle + flex) * length * 0.5),
    )
    cv2.line(frame, start, mid, color, thickness)
    cv2.line(frame, mid, end, color, thickness)


def make_background(profile: SubjectProfile, seed: int) -> np.ndarray:
    """Неподвижен фон с лека шумова структура – MOG2 работи по-чисто."""
    rng = np.random.default_rng(seed)
    bg = np.full((FRAME_H, FRAME_W, 3), profile.bg_color, dtype=np.uint8)
    # Добавяме лек градиент и шум, за да не е монотонен фонът
    gradient = np.linspace(-20, 20, FRAME_W, dtype=np.float32)
    bg = bg.astype(np.float32) + gradient[None, :, None]
    bg += rng.normal(0, 2.0, bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def draw_bag(frame: np.ndarray, profile: SubjectProfile, x_center: int, phase: float) -> None:
    """Добавя чанта/раница на субекта – изменя силуета (covariate „bag“)."""
    h = frame.shape[0]
    ground_y = int(h * 0.92)
    body_bottom_y = ground_y - profile.leg_len
    body_top_y = body_bottom_y - profile.body_h

    # Раница на гърба – фиксирана
    bag_w = int(profile.body_w * 0.55)
    bag_h = int(profile.body_h * 0.5)
    # малко люшкане в ритъм с походката
    sway = int(2 * np.sin(phase))
    bag_x = x_center - profile.body_w // 2 - bag_w + 2 + sway
    bag_y = body_top_y + 6
    cv2.rectangle(frame, (bag_x, bag_y), (bag_x + bag_w, bag_y + bag_h), (40, 40, 40), -1)


def render_video(
    profile: SubjectProfile,
    path: Path,
    rng: np.random.Generator,
    warmup_frames: int = 25,
    walk_frames: int = 125,
    carrying: bool = False,
    distractor_noise: float = 0.0,
) -> None:
    """Записва едно видео с даден субект.

    carrying: ако е True, субектът „носи“ раница – изменя силуета.
    distractor_noise: амплитуда (0..1) на неподвижни шумови точки във фона –
        проверява устойчивостта на MOG2.
    """
    start_x = int(rng.integers(-30, 0))
    speed = profile.walk_speed * float(rng.uniform(0.92, 1.08))
    phase_offset = float(rng.uniform(0, 2 * np.pi))

    bg = make_background(profile, seed=int(rng.integers(0, 10_000)))

    if distractor_noise > 0:
        noise_rng = np.random.default_rng(int(rng.integers(0, 10_000)))
        n_specks = int(40 * distractor_noise)
        for _ in range(n_specks):
            cx = int(noise_rng.integers(0, FRAME_W))
            cy = int(noise_rng.integers(0, FRAME_H))
            r = int(noise_rng.integers(2, 5))
            shade = int(noise_rng.integers(60, 200))
            cv2.circle(bg, (cx, cy), r, (shade, shade, shade), -1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise IOError(f"Не може да се създаде видеото: {path}")

    try:
        for _ in range(warmup_frames):
            writer.write(bg)

        for t in range(walk_frames):
            frame = bg.copy()
            x = start_x + int(speed * t)
            if x > FRAME_W + 40:
                break
            phase = phase_offset + t * profile.stride_freq * 2 * np.pi
            draw_person(frame, profile, x_center=x, phase=phase)
            if carrying:
                draw_bag(frame, profile, x_center=x, phase=phase)
            writer.write(frame)
    finally:
        writer.release()


def build_dataset(
    name: str,
    n_subjects: int,
    n_videos_per_subject: int,
    seed: int,
    force: bool = False,
    carrying_ratio: float = 0.0,
    distractor_noise: float = 0.0,
) -> Path:
    """Създава цяла директория с набор и записва MP4 файлове.

    carrying_ratio: дял видеа, в които субектът носи раница (0..1).
    distractor_noise: амплитуда на шумови точки във фона (0..1).
    """
    target = DATA_DIR / name
    if target.exists():
        if not force:
            print(f"  наборът вече съществува: {target}  (използвайте --force за презаписване)")
            return target
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(seed)
    extras = []
    if carrying_ratio > 0:
        extras.append(f"bag={int(carrying_ratio * 100)}%")
    if distractor_noise > 0:
        extras.append(f"noise={distractor_noise:.2f}")
    extra_str = f"  [{', '.join(extras)}]" if extras else ""
    print(f"Набор '{name}': {n_subjects} субекта × {n_videos_per_subject} видеа{extra_str}")

    for sid in range(n_subjects):
        profile = sample_subject(master_rng, sid)
        subject_dir = target / profile.subject_id
        subject_dir.mkdir(exist_ok=True)

        for vid in range(n_videos_per_subject):
            video_path = subject_dir / f"walk_{vid:02d}.mp4"
            video_rng = np.random.default_rng(master_rng.integers(0, 10**9))
            carrying = (vid / max(1, n_videos_per_subject)) < carrying_ratio
            render_video(
                profile, video_path, video_rng,
                carrying=carrying,
                distractor_noise=distractor_noise,
            )
            tag = " 🎒" if carrying else ""
            print(f"  ✓ {profile.subject_id}/walk_{vid:02d}.mp4{tag}")

    return target


PRESETS = {
    "small":       dict(n_subjects=6,  n_videos_per_subject=5,  seed_offset=0),
    "medium":      dict(n_subjects=10, n_videos_per_subject=6,  seed_offset=1),
    "large":       dict(n_subjects=12, n_videos_per_subject=8,  seed_offset=2),
    "xlarge":      dict(n_subjects=20, n_videos_per_subject=6,  seed_offset=3),
    "bag":         dict(n_subjects=8,  n_videos_per_subject=8,  seed_offset=4, carrying_ratio=0.5),
    "noisy":       dict(n_subjects=8,  n_videos_per_subject=6,  seed_offset=5, distractor_noise=0.6),
    "hard":        dict(n_subjects=15, n_videos_per_subject=8,  seed_offset=6, carrying_ratio=0.35, distractor_noise=0.3),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--preset",
        nargs="+",
        choices=list(PRESETS.keys()) + ["all"],
        default=["small"],
        help="кой/кои предварителни набори да се генерират (може няколко)",
    )
    ap.add_argument("--force", action="store_true", help="презапиши съществуващи набори")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    presets = list(PRESETS.keys()) if "all" in args.preset else args.preset
    for preset in presets:
        cfg = PRESETS[preset].copy()
        build_dataset(
            name=f"synthetic_{preset}",
            n_subjects=cfg["n_subjects"],
            n_videos_per_subject=cfg["n_videos_per_subject"],
            seed=args.seed + cfg["seed_offset"],
            force=args.force,
            carrying_ratio=cfg.get("carrying_ratio", 0.0),
            distractor_noise=cfg.get("distractor_noise", 0.0),
        )

    print("\nГотово. Наборите са достъпни в раздел „Обучение“.")
    print(f"Пътят към наборите: {DATA_DIR}")


if __name__ == "__main__":
    main()

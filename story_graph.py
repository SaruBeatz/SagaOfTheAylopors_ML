"""Load chapter JSON assets and simulate playthrough paths."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import (
    ASSETS_DIR,
    FEATURES,
    apply_delta,
    clamp_vector,
    effects_json_to_delta,
    l2_distance,
    neutral_vector,
)

MAX_STEPS_PER_CHAPTER = 800


@dataclass
class ChoiceOption:
    choice_id: str
    index: int
    effects: dict
    next_dialogue: Optional[str]


@dataclass
class DialogueNode:
    dialogue_id: str
    scene_id: str
    chapter_id: int
    has_choices: bool
    next_dialogue: Optional[str]
    only_for_companion: Optional[str]
    choices: List[ChoiceOption] = field(default_factory=list)


@dataclass
class SceneInfo:
    scene_id: str
    order: int
    dialogue_ids: List[str]


@dataclass
class ChapterGraph:
    chapter_id: int
    scenes: List[SceneInfo]
    dialogues: Dict[str, DialogueNode]
    start_dialogue_id: str


def load_chapter(chapter_path: Path) -> ChapterGraph:
    data = json.loads(chapter_path.read_text(encoding="utf-8"))
    chapter_id = int(data["chapter_id"])
    dialogues: Dict[str, DialogueNode] = {}
    scenes: List[SceneInfo] = []

    for scene_json in sorted(data["scenes"], key=lambda s: s["order"]):
        scene_id = scene_json["id"]
        dialogue_ids: List[str] = []
        for dlg in sorted(scene_json["dialogues"], key=lambda d: d["order"]):
            dlg_id = dlg["id"]
            dialogue_ids.append(dlg_id)
            choices: List[ChoiceOption] = []
            if dlg.get("has_choices") and "choices" in dlg:
                for idx, ch in enumerate(dlg["choices"]):
                    choices.append(
                        ChoiceOption(
                            choice_id=ch.get("id", f"{dlg_id}_c{idx+1}"),
                            index=idx,
                            effects=ch.get("effects"),
                            next_dialogue=ch.get("next_dialogue"),
                        )
                    )
            dialogues[dlg_id] = DialogueNode(
                dialogue_id=dlg_id,
                scene_id=scene_id,
                chapter_id=chapter_id,
                has_choices=bool(dlg.get("has_choices")),
                next_dialogue=dlg.get("next_dialogue"),
                only_for_companion=dlg.get("only_for_companion"),
                choices=choices,
            )
        scenes.append(SceneInfo(scene_id=scene_id, order=scene_json["order"], dialogue_ids=dialogue_ids))

    start_dialogue_id = scenes[0].dialogue_ids[0]
    return ChapterGraph(
        chapter_id=chapter_id,
        scenes=scenes,
        dialogues=dialogues,
        start_dialogue_id=start_dialogue_id,
    )


def load_all_chapters(assets_dir: Path | None = None) -> Dict[int, ChapterGraph]:
    assets_dir = assets_dir or ASSETS_DIR
    chapters: Dict[int, ChapterGraph] = {}
    for path in sorted(assets_dir.glob("chapter_*.json")):
        ch = load_chapter(path)
        chapters[ch.chapter_id] = ch
    return chapters


def _scene_index(graph: ChapterGraph, scene_id: str) -> int:
    for i, s in enumerate(graph.scenes):
        if s.scene_id == scene_id:
            return i
    return -1


def _first_dialogue_of_scene(graph: ChapterGraph, scene_id: str) -> Optional[str]:
    for s in graph.scenes:
        if s.scene_id == scene_id and s.dialogue_ids:
            return s.dialogue_ids[0]
    return None


def _next_scene_first_dialogue(graph: ChapterGraph, scene_id: str) -> Optional[str]:
    idx = _scene_index(graph, scene_id)
    if idx < 0 or idx >= len(graph.scenes) - 1:
        return None
    return graph.scenes[idx + 1].dialogue_ids[0]


def _skip_companion_filtered(graph: ChapterGraph, dialogue_id: str) -> Optional[str]:
    """Skip only_for_companion lines (no companion selected in simulation)."""
    current = dialogue_id
    safety = 60
    while safety > 0:
        safety -= 1
        node = graph.dialogues.get(current)
        if node is None:
            return None
        if not node.only_for_companion:
            return current
        if node.next_dialogue and node.next_dialogue in graph.dialogues:
            current = node.next_dialogue
        else:
            return None
    return None


def _advance_linear(graph: ChapterGraph, dialogue_id: str) -> Optional[str]:
    node = graph.dialogues.get(dialogue_id)
    if node is None:
        return None
    if node.next_dialogue and node.next_dialogue in graph.dialogues:
        return _skip_companion_filtered(graph, node.next_dialogue)
    nxt = _next_scene_first_dialogue(graph, node.scene_id)
    if nxt:
        return _skip_companion_filtered(graph, nxt)
    return None


def pick_choice_index(
    state: np.ndarray,
    options: List[ChoiceOption],
    target: np.ndarray,
    rng: random.Random,
    epsilon: float,
    delta_noise: float,
) -> int:
    scored: List[Tuple[float, int]] = []
    for opt in options:
        delta = effects_json_to_delta(opt.effects)
        if delta_noise > 0:
            jitter = np.array([rng.uniform(-delta_noise, delta_noise) for _ in FEATURES])
            delta = delta + jitter
        predicted = apply_delta(state, delta)
        scored.append((l2_distance(predicted, target), opt.index))
    scored.sort(key=lambda x: x[0])
    if len(scored) == 1:
        return scored[0][1]
    if rng.random() < epsilon:
        return scored[1][1] if len(scored) > 1 else scored[0][1]
    return scored[0][1]


@dataclass
class SimulationResult:
    final_state: np.ndarray
    chapter_snapshots: Dict[int, np.ndarray]
    choice_count: int
    final_distance: float


def simulate_playthrough(
    chapters: Dict[int, ChapterGraph],
    target: np.ndarray,
    rng: random.Random,
    epsilon: float = 0.12,
    delta_noise: float = 0.02,
    final_noise: float = 0.03,
) -> SimulationResult:
    state = neutral_vector()
    chapter_snapshots: Dict[int, np.ndarray] = {}
    current_chapter = 1
    dialogue_id = chapters[1].start_dialogue_id
    choice_count = 0
    steps = 0

    while current_chapter <= 7:
        graph = chapters[current_chapter]
        dialogue_id = _skip_companion_filtered(graph, dialogue_id)
        if dialogue_id is None:
            chapter_snapshots[current_chapter] = state.copy()
            current_chapter += 1
            if current_chapter > 7:
                break
            dialogue_id = chapters[current_chapter].start_dialogue_id
            continue

        node = graph.dialogues[dialogue_id]
        steps += 1
        if steps > MAX_STEPS_PER_CHAPTER * 7:
            break

        if node.has_choices and node.choices:
            choice_count += 1
            idx = pick_choice_index(state, node.choices, target, rng, epsilon, delta_noise)
            opt = node.choices[idx]
            delta = effects_json_to_delta(opt.effects)
            if delta_noise > 0:
                delta = delta + np.array(
                    [rng.uniform(-delta_noise, delta_noise) for _ in FEATURES]
                )
            state = apply_delta(state, delta)
            if opt.next_dialogue and opt.next_dialogue in graph.dialogues:
                dialogue_id = _skip_companion_filtered(graph, opt.next_dialogue)
            else:
                dialogue_id = _advance_linear(graph, dialogue_id)
            continue

        dialogue_id = _advance_linear(graph, dialogue_id)
        if dialogue_id is None:
            chapter_snapshots[current_chapter] = state.copy()
            current_chapter += 1
            if current_chapter > 7:
                break
            dialogue_id = chapters[current_chapter].start_dialogue_id

    if final_noise > 0:
        noise = np.array([rng.uniform(-final_noise, final_noise) for _ in FEATURES])
        state = clamp_vector(state + noise)

    return SimulationResult(
        final_state=state,
        chapter_snapshots=chapter_snapshots,
        choice_count=choice_count,
        final_distance=l2_distance(state, target),
    )

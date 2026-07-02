"""Gymnasium environment for PPO training with SCT-style drone events.

The environment exposes a discrete action space where each action selects an
event. If ``sct.py`` provides events, they are discovered automatically. If not,
the environment falls back to a small built-in event set so training code can
run immediately.

Supported SCT integration shapes:
    - ``EVENTS = ["hover", "forward", ...]``
    - ``def get_events(): return [...]``
    - event objects with a ``name`` attribute and optional ``enabled(state)``
      and ``apply(state)`` methods
    - your ``SCT`` controller passed as ``sct_model`` or constructed from
      ``sct_filename``
    - generic controllers with optional methods: ``reset()``,
      ``enabled_events(state)``, ``is_enabled(event, state)``, ``fire(event,
      state)``, ``apply_event(event, state)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - compatibility for older stacks
    import gym
    from gym import spaces


StateDict = Dict[str, np.ndarray | float | int | bool]


@dataclass(frozen=True)
class DroneEvent:
    """A simple event adapter used when SCT events are names/functions."""

    name: str
    delta_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def enabled(self, state: StateDict) -> bool:
        del state
        return True

    def apply(self, state: StateDict) -> StateDict:
        state = copy_state(state)
        state["velocity"] = np.asarray(state["velocity"], dtype=np.float32) + np.asarray(
            self.delta_velocity, dtype=np.float32
        )
        return state


DEFAULT_EVENTS: Tuple[DroneEvent, ...] = (
    DroneEvent("hover", (0.0, 0.0, 0.0)),
    DroneEvent("forward", (1.0, 0.0, 0.0)),
    DroneEvent("backward", (-1.0, 0.0, 0.0)),
    DroneEvent("left", (0.0, 1.0, 0.0)),
    DroneEvent("right", (0.0, -1.0, 0.0)),
    DroneEvent("up", (0.0, 0.0, 1.0)),
    DroneEvent("down", (0.0, 0.0, -1.0)),
)


@dataclass
class LearnedSCTBuilder:
    """Build a supervisor YAML graph from PPO simulator transitions."""

    event_names: Sequence[str]
    world_size: float
    position_bins: int = 6
    velocity_bins: int = 3
    transitions: Dict[int, Set[Tuple[str, int]]] = field(default_factory=dict)
    state_ids: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    initial_state_id: int = 0

    def state_id(self, state: StateDict) -> int:
        key = self.state_key(state)
        if key not in self.state_ids:
            self.state_ids[key] = len(self.state_ids)
        return self.state_ids[key]

    def record(self, source: StateDict, event_name_value: str, target: StateDict) -> None:
        source_id = self.state_id(source)
        target_id = self.state_id(target)
        self.transitions.setdefault(source_id, set()).add((event_name_value, target_id))
        self.transitions.setdefault(target_id, set())

    def set_initial_state(self, state: StateDict) -> None:
        self.initial_state_id = self.state_id(state)

    def state_key(self, state: StateDict) -> Tuple[int, ...]:
        position = np.asarray(state["position"], dtype=np.float32)
        velocity = np.asarray(state["velocity"], dtype=np.float32)
        target = np.asarray(state["target"], dtype=np.float32)
        battery = float(state["battery"])

        pos_key = discretize(position, -self.world_size, self.world_size, self.position_bins)
        target_key = discretize(target, -self.world_size, self.world_size, self.position_bins)
        vel_key = discretize(velocity, -3.0, 3.0, self.velocity_bins)
        battery_key = int(np.clip(np.floor(battery * 4.0), 0, 4))
        return tuple(pos_key + vel_key + target_key + [battery_key])

    def to_dict(self) -> Dict[str, Any]:
        num_states = max(self.state_ids.values(), default=0) + 1
        sup_data: List[Any] = []
        for state_id in range(num_states):
            outgoing = sorted(self.transitions.get(state_id, set()))
            sup_data.append(len(outgoing))
            for event_name_value, target_id in outgoing:
                high, low = divmod(target_id, 256)
                sup_data.extend([event_name_value, high, low])

        return {
            "num_events": len(self.event_names),
            "num_supervisors": 1,
            "events": list(self.event_names),
            "ev_controllable": [1] * len(self.event_names),
            "sup_events": [[1] * len(self.event_names)],
            "sup_init_state": [self.initial_state_id],
            "sup_current_state": [self.initial_state_id],
            "sup_data_pos": [0],
            "sup_data": sup_data,
        }

    def export(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)


class DroneSCTEnv(gym.Env):
    """Drone navigation environment driven by SCT events.

    Observation:
        A 13-dimensional float vector:
        ``position(3), velocity(3), target(3), target_delta(3), battery(1)``.

    Action:
        Discrete index into the discovered event list.

    Reward:
        Progress toward target, small step cost, invalid-event penalty, success
        bonus, crash penalty, and battery depletion penalty.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        sct_model: Optional[Any] = None,
        sct_filename: Optional[str] = None,
        learned_sct_filename: Optional[str] = None,
        events: Optional[Sequence[Any]] = None,
        *,
        world_size: float = 10.0,
        max_steps: int = 300,
        dt: float = 0.2,
        acceleration: float = 1.0,
        drag: float = 0.08,
        max_speed: float = 3.0,
        target_radius: float = 0.5,
        battery_capacity: float = 1.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if sct_model is None and sct_filename is not None:
            from sct import SCT  # type: ignore

            sct_model = SCT(sct_filename)

        self.sct_model = sct_model
        raw_events = events if events is not None else discover_sct_events(sct_model)
        self.events = normalize_events(raw_events, sct_model=sct_model)
        self.event_names = [event_name(event, sct_model=sct_model) for event in self.events]
        self.learned_sct_filename = learned_sct_filename

        self.world_size = float(world_size)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.acceleration = float(acceleration)
        self.drag = float(drag)
        self.max_speed = float(max_speed)
        self.target_radius = float(target_radius)
        self.battery_capacity = float(battery_capacity)
        self.render_mode = render_mode
        self.sct_builder = (
            LearnedSCTBuilder(self.event_names, self.world_size) if sct_model is None else None
        )

        self.action_space = spaces.Discrete(len(self.events))
        high = np.array(
            [
                self.world_size,
                self.world_size,
                self.world_size,
                self.max_speed,
                self.max_speed,
                self.max_speed,
                self.world_size,
                self.world_size,
                self.world_size,
                2.0 * self.world_size,
                2.0 * self.world_size,
                2.0 * self.world_size,
                self.battery_capacity,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space.low[-1] = 0.0

        self.np_random: np.random.Generator
        self.state: StateDict = {}
        self.steps = 0
        self.last_distance = 0.0
        self.last_event = "reset"
        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})

        if hasattr(self.sct_model, "reset"):
            self.sct_model.reset()
        elif self.sct_model is not None and hasattr(self.sct_model, "sup_init_state"):
            self.sct_model.sup_current_state = list(self.sct_model.sup_init_state)
            self.sct_model.last_events = [0] * int(self.sct_model.num_events)
            self.sct_model.input_buffer = []

        position = np.asarray(
            options.get(
                "position",
                self.np_random.uniform(-0.2 * self.world_size, 0.2 * self.world_size, size=3),
            ),
            dtype=np.float32,
        )
        target = np.asarray(
            options.get(
                "target",
                self.np_random.uniform(-0.8 * self.world_size, 0.8 * self.world_size, size=3),
            ),
            dtype=np.float32,
        )
        target[2] = max(0.5, float(target[2]))
        position[2] = max(0.5, float(position[2]))

        self.state = {
            "position": np.clip(position, -self.world_size, self.world_size),
            "velocity": np.zeros(3, dtype=np.float32),
            "target": np.clip(target, -self.world_size, self.world_size),
            "battery": float(options.get("battery", self.battery_capacity)),
            "crashed": False,
            "reached_target": False,
        }
        self.steps = 0
        self.last_distance = self._distance_to_target()
        self.last_event = "reset"
        if self.sct_builder is not None:
            self.sct_builder.set_initial_state(self.state)

        return self._observation(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        self.steps += 1

        invalid = action < 0 or action >= len(self.events)
        event = self.events[0] if invalid else self.events[action]
        event_label = event_name(event, sct_model=self.sct_model)
        enabled = False if invalid else self._is_event_enabled(event)
        previous_state = copy_state(self.state)

        if enabled:
            self.state = self._apply_event(event)
            self.last_event = event_label
        else:
            self.last_event = f"invalid:{event_label}"

        self._integrate_physics()
        if self.sct_builder is not None and enabled:
            self.sct_builder.record(previous_state, event_label, self.state)
        reward, terminated = self._reward(enabled)
        truncated = self.steps >= self.max_steps

        return self._observation(), float(reward), bool(terminated), bool(truncated), self._info()

    def render(self) -> Optional[str]:
        text = (
            f"step={self.steps} event={self.last_event} "
            f"pos={np.round(self.state['position'], 2)} "
            f"vel={np.round(self.state['velocity'], 2)} "
            f"target={np.round(self.state['target'], 2)} "
            f"battery={self.state['battery']:.2f}"
        )
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        if self.learned_sct_filename is not None:
            self.export_learned_sct(self.learned_sct_filename)
        return None

    def export_learned_sct(self, filename: str) -> None:
        if self.sct_builder is None:
            raise RuntimeError("No learned SCT builder exists when an SCT model is already active.")
        self.sct_builder.export(filename)

    def _observation(self) -> np.ndarray:
        position = np.asarray(self.state["position"], dtype=np.float32)
        velocity = np.asarray(self.state["velocity"], dtype=np.float32)
        target = np.asarray(self.state["target"], dtype=np.float32)
        battery = np.array([float(self.state["battery"])], dtype=np.float32)
        return np.concatenate([position, velocity, target, target - position, battery]).astype(
            np.float32
        )

    def _info(self) -> Dict[str, Any]:
        return {
            "distance": self._distance_to_target(),
            "event": self.last_event,
            "event_names": tuple(self.event_names),
            "position": np.asarray(self.state["position"], dtype=np.float32).copy(),
            "target": np.asarray(self.state["target"], dtype=np.float32).copy(),
            "battery": float(self.state["battery"]),
        }

    def _distance_to_target(self) -> float:
        return float(
            np.linalg.norm(
                np.asarray(self.state["target"], dtype=np.float32)
                - np.asarray(self.state["position"], dtype=np.float32)
            )
        )

    def _is_event_enabled(self, event: Any) -> bool:
        if self.sct_model is not None:
            if hasattr(self.sct_model, "enabled_events"):
                enabled_events = self.sct_model.enabled_events(copy_state(self.state))
                return event_label_in(event, enabled_events, sct_model=self.sct_model)
            if hasattr(self.sct_model, "is_enabled"):
                return bool(self.sct_model.is_enabled(event, copy_state(self.state)))
            if hasattr(self.sct_model, "get_active_controllable_events"):
                active_events = self.sct_model.get_active_controllable_events()
                event_id = event_index(event, self.sct_model)
                return event_id is not None and bool(active_events[event_id])

        if hasattr(event, "enabled"):
            return bool(event.enabled(copy_state(self.state)))
        if callable(event) and hasattr(event, "is_enabled"):
            return bool(event.is_enabled(copy_state(self.state)))
        return True

    def _apply_event(self, event: Any) -> StateDict:
        state_copy = copy_state(self.state)

        if self.sct_model is not None:
            if hasattr(self.sct_model, "fire"):
                result = self.sct_model.fire(event, state_copy)
                return merge_state(self.state, result)
            if hasattr(self.sct_model, "apply_event"):
                result = self.sct_model.apply_event(event, state_copy)
                return merge_state(self.state, result)
            if hasattr(self.sct_model, "make_transition"):
                event_id = event_index(event, self.sct_model)
                if event_id is None:
                    return state_copy
                self.sct_model.make_transition(event_id)
                if hasattr(self.sct_model, "exec_callback"):
                    self.sct_model.exec_callback(event_id)
                return self._apply_named_event(event_name(event_id, sct_model=self.sct_model))

        if hasattr(event, "apply"):
            return merge_state(self.state, event.apply(state_copy))
        if callable(event):
            result = event(state_copy)
            return merge_state(self.state, result)

        return self._apply_named_event(event_name(event, sct_model=self.sct_model))

    def _apply_named_event(self, name: str) -> StateDict:
        state = copy_state(self.state)
        aliases = {
            "hover": (0.0, 0.0, 0.0),
            "stay": (0.0, 0.0, 0.0),
            "forward": (1.0, 0.0, 0.0),
            "backward": (-1.0, 0.0, 0.0),
            "left": (0.0, 1.0, 0.0),
            "right": (0.0, -1.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "takeoff": (0.0, 0.0, 1.0),
            "down": (0.0, 0.0, -1.0),
            "land": (0.0, 0.0, -1.0),
        }
        delta = np.asarray(aliases.get(name.lower(), (0.0, 0.0, 0.0)), dtype=np.float32)
        state["velocity"] = np.asarray(state["velocity"], dtype=np.float32) + (
            self.acceleration * delta
        )
        return state

    def _integrate_physics(self) -> None:
        velocity = np.asarray(self.state["velocity"], dtype=np.float32)
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed:
            velocity = velocity / speed * self.max_speed

        position = np.asarray(self.state["position"], dtype=np.float32) + velocity * self.dt
        crashed = bool(position[2] <= 0.0 or np.any(np.abs(position) > self.world_size))
        position = np.clip(position, -self.world_size, self.world_size)
        position[2] = max(0.0, float(position[2]))

        self.state["position"] = position.astype(np.float32)
        self.state["velocity"] = (velocity * (1.0 - self.drag)).astype(np.float32)
        self.state["battery"] = max(0.0, float(self.state["battery"]) - self._battery_cost())
        self.state["crashed"] = crashed
        self.state["reached_target"] = self._distance_to_target() <= self.target_radius

    def _battery_cost(self) -> float:
        velocity = np.asarray(self.state["velocity"], dtype=np.float32)
        motion_cost = 0.0015 * float(np.linalg.norm(velocity))
        return 0.001 + motion_cost

    def _reward(self, enabled: bool) -> Tuple[float, bool]:
        distance = self._distance_to_target()
        progress = self.last_distance - distance
        self.last_distance = distance

        reward = 2.0 * progress - 0.01
        if not enabled:
            reward -= 0.25
        if bool(self.state["reached_target"]):
            reward += 10.0
            return reward, True
        if bool(self.state["crashed"]):
            reward -= 10.0
            return reward, True
        if float(self.state["battery"]) <= 0.0:
            reward -= 5.0
            return reward, True
        return reward, False


def discover_sct_events(sct_model: Optional[Any] = None) -> Sequence[Any]:
    """Load events from ``sct.py`` if it defines them, else use defaults."""

    if sct_model is not None:
        if hasattr(sct_model, "get_events"):
            events, controllable = sct_model.get_events()
            return controllable_event_ids(events, controllable)
        if hasattr(sct_model, "EV"):
            controllable = getattr(sct_model, "ev_controllable", None)
            return controllable_event_ids(sct_model.EV, controllable)

    try:
        import sct  # type: ignore
    except ImportError:
        return DEFAULT_EVENTS

    if hasattr(sct, "get_events"):
        events = sct.get_events()
        if events:
            return events
    if hasattr(sct, "EVENTS"):
        events = getattr(sct, "EVENTS")
        if events:
            return events
    if hasattr(sct, "events"):
        events = getattr(sct, "events")
        if events:
            return events
    return DEFAULT_EVENTS


def normalize_events(events: Sequence[Any], sct_model: Optional[Any] = None) -> List[Any]:
    if not events:
        return list(DEFAULT_EVENTS)
    if is_sct_get_events_result(events):
        ev_map, controllable = events
        return controllable_event_ids(ev_map, controllable)
    if isinstance(events, Mapping):
        if all(isinstance(value, Integral) for value in events.values()):
            if sct_model is not None and hasattr(sct_model, "ev_controllable"):
                return controllable_event_ids(events, sct_model.ev_controllable)
            return [value for _, value in sorted(events.items(), key=lambda item: item[1])]
        return [DroneEvent(str(name), tuple(value)) for name, value in events.items()]
    return list(events)


def event_name(event: Any, sct_model: Optional[Any] = None) -> str:
    if sct_model is not None and isinstance(event, Integral) and hasattr(sct_model, "EV"):
        for name, idx in sct_model.EV.items():
            if idx == int(event):
                return str(name)
    if isinstance(event, str):
        return event
    if hasattr(event, "name"):
        return str(event.name)
    if hasattr(event, "__name__"):
        return str(event.__name__)
    return str(event)


def event_index(event: Any, sct_model: Optional[Any]) -> Optional[int]:
    if isinstance(event, Integral):
        return int(event)
    if isinstance(event, str) and sct_model is not None and hasattr(sct_model, "EV"):
        return int(sct_model.EV[event])
    if hasattr(event, "index"):
        return int(event.index)
    return None


def event_label_in(
    event: Any,
    enabled_events: Iterable[Any],
    sct_model: Optional[Any] = None,
) -> bool:
    expected = event_name(event, sct_model=sct_model)
    return any(
        candidate == event or event_name(candidate, sct_model=sct_model) == expected
        for candidate in enabled_events
    )


def is_sct_get_events_result(events: Any) -> bool:
    return (
        isinstance(events, tuple)
        and len(events) == 2
        and isinstance(events[0], Mapping)
        and isinstance(events[1], Sequence)
    )


def controllable_event_ids(
    events: Mapping[str, int],
    controllable: Optional[Sequence[int | bool]],
) -> List[int]:
    ordered_ids = [idx for _, idx in sorted(events.items(), key=lambda item: item[1])]
    if controllable is None:
        return ordered_ids
    return [idx for idx in ordered_ids if bool(controllable[idx])]


def discretize(values: np.ndarray, low: float, high: float, bins: int) -> List[int]:
    scaled = (np.asarray(values, dtype=np.float32) - low) / (high - low)
    indices = np.floor(np.clip(scaled, 0.0, 0.999999) * bins).astype(np.int32)
    return indices.tolist()


def copy_state(state: StateDict) -> StateDict:
    copied: StateDict = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def merge_state(current: StateDict, result: Any) -> StateDict:
    """Merge SCT return values while preserving required state keys."""

    if result is None:
        return copy_state(current)
    if isinstance(result, Mapping):
        merged = copy_state(current)
        for key, value in result.items():
            merged[key] = np.asarray(value, dtype=np.float32) if key in {"position", "velocity", "target"} else value
        return merged
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], Mapping):
        return merge_state(current, result[0])
    raise TypeError(
        "SCT event handlers must return None, a state mapping, or (state, info). "
        f"Got {type(result).__name__}."
    )


def make_env(**kwargs: Any) -> Callable[[], DroneSCTEnv]:
    """Factory compatible with Stable-Baselines3 vectorized env helpers."""

    def _init() -> DroneSCTEnv:
        return DroneSCTEnv(**kwargs)

    return _init


__all__ = ["DroneEvent", "DroneSCTEnv", "make_env"]

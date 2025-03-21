"""
Code inspired by
- https://github.com/resgroup/yaw-control-simulation/blob/674a925fcc3b7635c26f6d26b5f572992273e1c0/yaw_control_simulation/controller/hilloftowie_controller.py#L18
- https://github.com/m-lundberg/simple-pid/blob/6e6f4b000cdf7830635674cb89a6e823f5025784/simple_pid/pid.py#L12
"""

import datetime as dt


class PID:
    def __init__(
        self,
        p_gain: float,
        d_gain: float,
        i_gain: float,
        min_command: float | None = None,
        max_command: float | None = None,
    ):
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.i_gain = i_gain
        self.min_command = min_command
        self.max_command = max_command

        self.timestamp: dt.datetime | None = None
        self.value: float | None = None
        self.setpoint: float | None = None
        self.accumulated_i_term: float = 0

    def _store_last(self, timestamp: dt.datetime, value: float, setpoint: float):
        self.timestamp = timestamp
        self.value = value
        self.setpoint = setpoint

    def _clip(self, value: float) -> float:
        if self.min_command and value < self.min_command:
            return self.min_command
        if self.max_command and value > self.max_command:
            return self.max_command
        return value

    def __call__(self, timestamp: dt.datetime, value: float, setpoint: float) -> float | None:
        if self.timestamp is None:
            self._store_last(timestamp, value, setpoint)
            return None

        err = value - setpoint
        p_term = self.p_gain * err

        setpoint_diff = 0  # setpoint - self.setpoint
        value_diff = value - self.value
        time_diff = (timestamp - self.timestamp).total_seconds()
        d_term = self.d_gain * (value_diff - setpoint_diff) / time_diff

        self.accumulated_i_term = self.accumulated_i_term + self.i_gain * err * time_diff
        self._store_last(timestamp, value, setpoint)
        return self._clip(p_term + self.accumulated_i_term + d_term)

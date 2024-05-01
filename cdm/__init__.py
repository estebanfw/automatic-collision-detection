"""cdm package"""

""" Pandas Package"""
import pandas as pd
from pandas import DataFrame


class Event:
    """Event Class
    Conjunction Event Class
    """

    def __init__(self, event_id):
        self.event_id = event_id


class Cdm(Event):
    """cdm class"""

    def __init__(
        self,
        event_id,
        time_to_tca,
        miss_distance,
        pc,
        relative_speed,
        relative_position_r,
        relative_position_t,
        relative_position_n,
        relative_velocity_r,
        relative_velocity_t,
        relative_velocity_n,
        mahalanobis_distance,
    ):
        super().__init__(event_id)

        self.time_to_tca = time_to_tca
        self.miss_distance = miss_distance
        self.pc = pc
        self.relative_position_r = relative_position_r
        self.relative_position_t = relative_position_t
        self.relative_position_n = relative_position_n
        self.relative_velocity_r = relative_velocity_r
        self.relative_velocity_t = relative_velocity_t
        self.relative_velocity_n = relative_velocity_n
        self.relative_speed = relative_speed
        self.mahalanobis_distance = mahalanobis_distance

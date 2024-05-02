from typing import List, Tuple
from app.core import events


def load_events(events_module: List[Tuple, str]):
    """
    This function loads events from the events module

    Args:
        events_module (List[Tuple, string]): List of functions from events module in tuple or
    """
    for event in events_module:
        if isinstance(event, tuple):
            func = getattr(events, event[0], None)
            if func is not None:
                func(*event[1:])
        else:
            func = getattr(events, event, None)
            if func is not None:
                event()

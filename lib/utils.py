import numpy as np
from fourier_ica import FourierICA


def filter_triggers(triggers, sfreq, start, end, rad):
    """
    check if this trigger is ok with following conditions:
      * it is at least ``rad`` samples away from other triggers
      * it is at least ``rad`` samples away from start or end
    """

    valid_triggers = []
    for trigger in triggers:

        if trigger < start + rad:
            continue

        if trigger > end - rad:
            continue

        valid = True

        for other in triggers:
            # don't compare to itself
            if other == trigger:
                continue

            if trigger < other + rad and trigger > other - rad:
                valid = False

        if not valid:
            continue

        valid_triggers.append(trigger)

    return np.array(valid_triggers, dtype=triggers.dtype)

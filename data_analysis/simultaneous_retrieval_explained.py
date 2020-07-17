# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def detect_simultaneous_retrieval(two_retrieved, two_max_overlap, tS, onset):
    """
    Finds simultaneous retrieval

    Parameters
    ----------
    two_retrieved -- 2D array of ints
        Storing the identity of the two patterns with the
        highest overlap at each recording time (row). The first column is for
        the one with maximal overlap.

    two_max_overlap -- 2D array of floats
        Storing the overlap with the two patterns with the
        highest overlap at each recording time.

    tS -- 1D array of floats
        Times at which data were recorded

    onset -- float
        Time when the network was cued

    Returns
    -------
    Simult_ret -- List of tuples
        Each element corresponds to a simultaneous retrieval detection, stored
        as (retrieved, outsider, detection time).
    validation_times -- List of floats
        Times at which transitions were validated
    """
    simult_ret = []
    validation_times = []
    previously_retrieved = -1
    waiting_validation = False  # Is a transition waiting to be
                                # validated?
    was_blocked = False         # Was the retrieval blocked because of
                                # undistinguishable patterns?
    last_blocked = -1
    last_blocker = -1

    for iT in range(len(tS)):
        if tS[iT] >= onset:
            retrieved = two_retrieved[iT, 0]
            outsider = two_retrieved[iT, 1]
            max_overlap = two_max_overlap[iT, 0]
            max2_overlap = two_max_overlap[iT, 1]

            # A transition is detected because the retrieved pattern
            # changed. It still has to be validated.
            if retrieved != previously_retrieved \
               and not waiting_validation:
                print('Detected, %d %d' % (previously_retrieved, retrieved))
                waiting_validation = True
                was_blocked = False

            # If the difference between the two maximal overlaps
            # sufficient to validate the transtition, it is validated
            if waiting_validation and max_overlap > 0.5 \
               and max_overlap - max2_overlap > 0.2:
                print('Validated, %d %d' % (previously_retrieved, retrieved))
                waiting_validation = False
                was_blocked = False
                previously_retrieved = retrieved
                validation_times.append(tS[iT])

            is_blocked = waiting_validation and max_overlap > 0.5 \
                and max_overlap - max2_overlap <= 0.2

            # A blocking is recorded is the retrieval was blocked at
            # the previous step, but the blocking conditions changed,
            # either because the retrieved pattern changed or the
            # outsider changed
            if was_blocked:
                blocked = retrieved
                blocker = outsider
                if blocker != last_blocker or blocked != last_blocked:
                    print("Blocked")
                    print(blocked, last_blocked, blocker, last_blocker)
                    simult_ret.append((last_blocked, last_blocker, tS[iT]))

            # One has to recall the blocked and blocker in order to
            # check if they change at the next step
            if is_blocked:
                last_blocked = retrieved
                last_blocker = outsider

            was_blocked = is_blocked

    plt.ion()
    plt.close('all')
    plt.figure('visual_check_blocking_')
    plt.plot(tS, two_max_overlap)
    for event in simult_ret:
        plt.axvline(event[2], color='tab:green', label='Blocking')
    for tt in validation_times:
        plt.axvline(tt, color='tab:red', label='Validation')

    for iT in range(0, len(tS), 100):
        plt.text(tS[iT], two_max_overlap[iT, 0]+0.05, str(int(two_retrieved[iT, 0])))
    for iT in range(4, len(tS), 100):
        plt.text(tS[iT], two_max_overlap[iT, 1]-0.05, str(int(two_retrieved[iT, 1])))

    return simult_ret, validation_times


""" Testing """
import file_handling

key = '47402f476fb3076f46f9604929e997ed'

two_retrieved = file_handling.load_two_first(0, 0, key)
tS = np.array(file_handling.load_time(0, key)[0])

max_m_mu = file_handling.load_max_m_mu(0, key)[0]
max2_m_mu = file_handling.load_max2_m_mu(0, key)[0]
two_max_overlap = np.zeros((len(max_m_mu), 2))
two_max_overlap[:, 0] = max_m_mu
two_max_overlap[:, 1] = max2_m_mu

recorded = tS > 0.
tS = tS[recorded]
two_retrieved = two_retrieved[recorded, :]
two_max_overlap = two_max_overlap[recorded, :]

trans_times = file_handling.load_transition_time(0,  key)
onset = trans_times[0][1]

simult_ret, validation_times = detect_simultaneous_retrieval(two_retrieved, two_max_overlap, tS, onset)

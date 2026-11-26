"""HTCondor GPU requirements for the feedback and scale experiments."""

HOPPER = '(Capability == 9.0 && regexp("H100|H200",DeviceName))'
_A100 = '(Capability == 8.0 && regexp("A100",DeviceName))'
_BLACKWELL = '(Capability == 12.0 && regexp("^NVIDIA RTX PRO 6000 Blackwell Server Edition$",DeviceName))'
FEEDBACK_FALLBACK = f'({HOPPER} || {_A100})'
SCALE_FALLBACK = f'({HOPPER} || {_A100} || {_BLACKWELL})'
FEEDBACK_COMMON = ('((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver))'
                   ' && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)')
SCALE_COMMON = FEEDBACK_COMMON + ' && (DeviceUuid =!= "7067f38b-b4c2-5568-ec82-f698473ddd07")'
RANK = (f'10.0 * ifThenElse(countMatches({HOPPER}, TARGET.AvailableGPUs)>0,1.0,0.0)'
        ' + ifThenElse(regexp("^backfill", TARGET.Name),0.0,1.0)')


def feedback_gpus(fallback=False):
    return (FEEDBACK_FALLBACK if fallback else HOPPER) + ' && ' + FEEDBACK_COMMON


def scale_gpus(fallback):
    return (SCALE_FALLBACK if fallback else HOPPER) + ' && ' + SCALE_COMMON


def desired_fallback(ad, now):
    status = ad['JobStatus']
    if status == 1:
        since = ad.get('EnteredCurrentStatus')
        return isinstance(since, (int, float)) and now - since >= 3600
    if status in (2, 6, 7):
        # Restore Hopper preference for the next queue wait after an eviction.
        return False
    return None

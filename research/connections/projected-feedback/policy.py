"""Hopper first; validated FA2 A100 backend after one uninterrupted idle hour."""
HOPPER='(Capability == 9.0 && regexp("H100|H200",DeviceName))'
FALLBACK='('+HOPPER+' || (Capability == 8.0 && regexp("A100",DeviceName)))'
COMMON='((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)'
RANK='10.0 * ifThenElse(countMatches('+HOPPER+', TARGET.AvailableGPUs)>0,1.0,0.0) + ifThenElse(regexp("^backfill", TARGET.Name),0.0,1.0)'


def require_gpus(fallback=False):
    return (FALLBACK if fallback else HOPPER)+' && '+COMMON


def desired_fallback(ad,now):
    if ad['JobStatus']==1:
        since=ad.get('EnteredCurrentStatus')
        return isinstance(since,(int,float)) and now-since>=3600
    if ad['JobStatus'] in (2,6,7):
        # A requirements edit does not migrate the active allocation.
        return False
    return None

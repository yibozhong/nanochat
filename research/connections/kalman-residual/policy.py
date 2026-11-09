"""GPU eligibility for this new run; never mutates an already-running allocation."""
HOPPER = '(Capability == 9.0 && regexp("H100|H200",DeviceName))'
FALLBACK = '('+HOPPER+' || (Capability == 8.0 && regexp("A100",DeviceName)) || (Capability == 12.0 && regexp("^NVIDIA RTX PRO 6000 Blackwell Server Edition$",DeviceName)))'
COMMON = '((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName) && (DeviceUuid =!= "7067f38b-b4c2-5568-ec82-f698473ddd07")'
RANK = '10.0 * ifThenElse(countMatches('+HOPPER+', TARGET.AvailableGPUs)>0,1.0,0.0) + ifThenElse(regexp("^backfill", TARGET.Name),0.0,1.0)'


def require_gpus(fallback):
    return (FALLBACK if fallback else HOPPER)+' && '+COMMON


def desired_fallback(ad, now):
    status=ad['JobStatus']
    if status==1:
        since=ad.get('EnteredCurrentStatus')
        return isinstance(since,(int,float)) and now-since>=3600
    if status in (2,6,7):
        # Restore first-hour preference for the next eviction/requeue. Updating
        # requirements does not evict the current A100/PRO6000 allocation.
        return False
    return None

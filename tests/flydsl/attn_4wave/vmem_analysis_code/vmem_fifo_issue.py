# SPDX-License-Identifier: MIT
"""Optional issue-retirement rule; default is the original finite FIFO.

This is a user-requested counterfactual, not measured retirement. Natural
completion happens first at a tie; then input order resolves simultaneous
issues. Cut windows keep the original S/L rate; excluded bytes are reported.
"""
import math

import numpy as np

from vmem_inflight_fifo import MAX_BINS, plain, reference_diagnostic


ABC = ("a_load", "b_dma", "c_store")
POLICY = (
    "新issue退休规则为独立配置项，默认关闭。关闭时保留原FIFO容量限制，"
    "满则等待自然完成释放槽位，不截断任何请求。开启时，"
    "每条新ATT成功issue先结束最早发出且仍贡献带宽的请求，并立即释放其槽位；"
    "不是仅在FIFO满时触发，也不限于曾排队的请求。自然完成先于同刻issue，"
    "同刻多个issue按原始稳定顺序逐条处理。窗口截断后仍按原S/L计带宽，"
    "剩余payload不补摊；这是模型排除量，不是丢失的真实数据。"
    "仅在开启时，从空池起始最多保留一个正长度服务窗口，因此C>=1不改变接纳。"
    "A/B/C缓存命中各自覆盖为20ns，仅作用于该实际请求类别，不改变metadata/atomic。"
)


def latency_values(data, ns, hits=None):
    hits = hits or {}
    if any(k not in ABC or not isinstance(v, bool) for k, v in hits.items()):
        raise ValueError("cache flags must be A/B/C booleans")
    keys = [c["key"] for c in data["classes"]]
    for key in keys:
        if isinstance(ns[key], bool) or not math.isfinite(ns[key]) or not 1 <= ns[key] <= 10000:
            raise ValueError("each latency must be finite in [1,10000] ns")
    if not math.isfinite(data["nsPerCycle"]) or data["nsPerCycle"] <= 0:
        raise ValueError("invalid ATT clock")
    effective = {k: 20. if hits.get(k, False) else ns[k] for k in keys}
    event_ns = []
    for e in data["events"]:
        key = keys[e[4]]
        event_ns.append(20. if hits.get(key, False) else ns[keys[e[8] if len(e) > 8 else e[4]]])
    return effective, np.asarray(event_ns, dtype=float)


def schedule(data, ns, capacity, hits=None, issue_retirement=False):
    if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 8192):
        raise ValueError("capacity must be integer in [1,8192] or None")
    if not isinstance(issue_retirement, bool):
        raise ValueError("issue retirement must be a boolean")
    retire = issue_retirement and capacity is not None
    effective, event_ns = latency_values(data, ns, hits)
    lengths = event_ns/data["nsPerCycle"]
    n = len(data["events"])
    starts, expires, nominal = np.zeros(n), np.zeros(n), np.zeros(n)
    cut_by = np.full(n, -1, dtype=np.int64)
    free = np.full(capacity or 1, -math.inf); slots = np.full(n, -1, dtype=np.int64)
    live = []; previous_arrival = -math.inf; previous_start = -math.inf
    for i, event in enumerate(data["events"]):
        t = event[0]
        if not math.isfinite(t) or t < previous_arrival:
            raise ValueError("ATT successful issues must be finite and sorted")
        if not math.isfinite(event[5]) or event[5] <= 0:
            raise ValueError("payload must be positive")
        # Direct list/vector reference, independent of the browser stream rule.
        if retire:
            live = [j for j in live if starts[j] <= t < expires[j]]
        if retire and live:
            j = min(live, key=lambda j: (data["events"][j][0], j))
            expires[j] = t; cut_by[j] = i; free[slots[j]] = t; live.remove(j)
        if capacity is None:
            start = t
        else:
            slot = int(np.argmin(free)); start = max(t, previous_start, float(free[slot])); slots[i] = slot
        starts[i] = start; nominal[i] = expires[i] = start+lengths[i]
        if capacity is not None: free[slots[i]] = expires[i]
        if retire: live.append(i)
        previous_arrival, previous_start = t, start
    edges = sorted([(float(starts[i]), 1) for i in range(n) if expires[i] > starts[i]]+
                   [(float(expires[i]), -1) for i in range(n) if expires[i] > starts[i]])
    active = peak = 0
    for _, delta in edges:
        active += delta; assert active >= 0; peak = max(peak, active)
    assert active == 0 and (capacity is None or peak <= (1 if retire else capacity))
    delays = starts-np.asarray([e[0] for e in data["events"]])
    payload = math.fsum(e[5] for e in data["events"])
    counted = math.fsum(e[5] if cut_by[i] < 0 else e[5]*(expires[i]-starts[i])/lengths[i] for i, e in enumerate(data["events"]))
    return {"starts": starts, "expires": expires, "naturalExpires": nominal, "cutBy": cut_by,
            "delays": delays, "eventLatencies": lengths,
            "latencies": np.asarray([effective[c["key"]]/data["nsPerCycle"] for c in data["classes"]]),
            "latencyNs": effective, "baseLatencyNs": dict(ns), "cacheHits": {k: bool((hits or {}).get(k)) for k in ABC},
            "capacity": capacity, "peakActive": peak, "delayed": int(np.count_nonzero(delays)),
            "totalDelay": float(delays.sum()), "maxDelay": float(delays.max()) if n else 0.,
            "lastCompletion": max(data["begin"], float(expires.max())) if n else data["begin"],
            "truncated": int(np.count_nonzero(cut_by >= 0)), "issuedPayloadBytes": payload,
            "contributingPayloadBytes": counted, "excludedPayloadBytes": max(0., payload-counted),
            "issueRetiresOldest": retire, "issueRetirementSelected": issue_retirement}


def reference_model(data, ns, capacity, hits=None, issue_retirement=False):
    p = schedule(data, ns, capacity, hits, issue_retirement)
    begin, step = data["begin"], data["step"]
    bins = math.ceil((max(data["end"], p["lastCompletion"])-begin)/step)
    if bins <= 0 or bins > MAX_BINS: raise ValueError("invalid/excessive bins; no truncation")
    kinds = len(data["classes"])
    counts, gbs = np.zeros((kinds,bins)), np.zeros((kinds,bins))
    for i,e in enumerate(data["events"]):
        a,b = p["starts"][i],p["expires"][i]
        if b <= a: continue
        l,r = max(0,math.floor((a-begin)/step)),min(bins,math.ceil((b-begin)/step))
        times = begin+np.arange(l,r)*step
        fraction = np.maximum(0.,np.minimum(times+step,b)-np.maximum(times,a))/step
        counts[e[4],l:r] += fraction
        gbs[e[4],l:r] += fraction*e[5]/(p["eventLatencies"][i]*data["nsPerCycle"])
    omitted = {"starts","expires","naturalExpires","cutBy","delays","eventLatencies","latencies","latencyNs","baseLatencyNs","cacheHits"}
    fifo = {k:v for k,v in p.items() if k not in omitted}
    end = begin+bins*step
    # Integrate the original arrival -> admission interval separately. Waiting
    # contributes no bandwidth and reappears when the retirement option is off.
    waiting_edges, flux = np.zeros(bins), np.zeros(bins+1)
    for sign, times in ((1, [e[0] for e in data["events"]]), (-1, p["starts"])):
        for pos in (np.asarray(times)-begin)/step:
            if pos < 0: flux[0] += sign
            elif pos < bins:
                index = math.floor(pos)
                waiting_edges[index] += sign*(index+1-pos)
                flux[index+1] += sign
    waiting = waiting_edges+np.cumsum(flux[:-1])
    waiting[np.abs(waiting) < 1e-9] = 0
    assert (waiting >= 0).all()
    fifo.update(peakQueued=float(waiting.max()),captureEnd=data["end"],tailCycles=end-data["end"],policy=POLICY)
    result = {k:p[k] for k in omitted}
    result.update(bandwidth=gbs.sum(axis=0)*256/1000,count=counts.sum(axis=0),classGBs=gbs,classCount=counts,
                  waitingCount=waiting,end=end,fifo=fifo)
    if p["issueRetiresOldest"]:
        assert not p["delays"].any(), "enabled empty-pool rule must not generate a waiting queue"
    if data["events"] and data["events"][0][0] >= begin:
        area = math.fsum(result["bandwidth"])*step*data["nsPerCycle"]*1000/256
        assert math.isclose(area,p["contributingPayloadBytes"],rel_tol=1e-10,abs_tol=1e-5)
        assert math.isclose(area+p["excludedPayloadBytes"],p["issuedPayloadBytes"],rel_tol=1e-10,abs_tol=1e-5)
        assert math.isclose(math.fsum(waiting)*step,p["totalDelay"],rel_tol=1e-10,abs_tol=1e-5)
    return result


def make_reference(data, ns, capacity, hits=None, issue_retirement=False):
    m = reference_model(data,ns,capacity,hits,issue_retirement)
    ds = {d:{k:v for k,v in reference_diagnostic(data,m,d).items() if k!="gaps"} for d in ("interior","capture","full")}
    return {"ns":dict(ns),"capacity":capacity,"cacheHits":hits or {},"issueRetirement":issue_retirement,"model":plain(m),"diagnostics":ds}
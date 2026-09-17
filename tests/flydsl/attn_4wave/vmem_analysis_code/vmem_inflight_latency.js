// SPDX-License-Identifier: MIT
"use strict";

// Numeric-only routines shared by the page and browser regression tests.
// Changing L changes BOTH lifetime windows and the bytes/L contribution.
// The captured issue stream, payloads and observation interval stay fixed.
(() => {
  const lowerBound = (array, value) => {
    let left = 0, right = array.length;
    while (left < right) { const middle = (left + right) >>> 1; if (array[middle] < value) left = middle + 1; else right = middle; }
    return left;
  };

  function model(data, latencyNs) {
    const keys = data.classes.map((kind) => kind.key), types = keys.length;
    const ns = keys.map((key) => latencyNs[key]);
    if (ns.some((value) => !Number.isFinite(value) || value < 1 || value > 10000)) throw new Error("每个延迟必须是1～10000ns的有限数值");
    const lengths = ns.map((value) => value / data.nsPerCycle);
    const bins = (data.end - data.begin) / data.step;
    const countEdges = Array.from({length: types}, () => new Float64Array(bins));
    const byteEdges = Array.from({length: types}, () => new Float64Array(bins));
    const countDelta = Array.from({length: types}, () => new Float64Array(bins + 1));
    const byteDelta = Array.from({length: types}, () => new Float64Array(bins + 1));
    for (const event of data.events) {
      const [issue, , , , kind, payload] = event, latencyKind = event[8] ?? kind;
      const a = Math.max(0, (issue - data.begin) / data.step);
      const b = Math.min(bins, (issue + lengths[latencyKind] - data.begin) / data.step);
      if (b <= a) continue;
      const l = Math.floor(a), r = Math.ceil(b) - 1, gbs = payload / ns[latencyKind];
      if (l === r) { countEdges[kind][l] += b - a; byteEdges[kind][l] += (b - a) * gbs; }
      else {
        countEdges[kind][l] += l + 1 - a; byteEdges[kind][l] += (l + 1 - a) * gbs;
        countEdges[kind][r] += b - r; byteEdges[kind][r] += (b - r) * gbs;
        countDelta[kind][l + 1] += 1; countDelta[kind][r] -= 1;
        byteDelta[kind][l + 1] += gbs; byteDelta[kind][r] -= gbs;
      }
    }
    const count = new Float64Array(bins), bandwidth = new Float64Array(bins);
    for (let kind = 0; kind < types; kind++) {
      let nc = 0, nb = 0;
      for (let i = 0; i < bins; i++) {
        nc += countDelta[kind][i]; nb += byteDelta[kind][i];
        const c = countEdges[kind][i] + nc, b = byteEdges[kind][i] + nb;
        countEdges[kind][i] = Math.abs(c) < 1e-10 ? 0 : c;
        byteEdges[kind][i] = Math.abs(b) < 1e-9 ? 0 : b;
        if (countEdges[kind][i] < 0 || byteEdges[kind][i] < 0) throw new Error("在途积分出现负值");
        count[i] += countEdges[kind][i]; bandwidth[i] += byteEdges[kind][i] * 256 / 1000;
      }
    }
    return {latencies: lengths, bandwidth, count, classGBs: byteEdges, classCount: countEdges};
  }

  function supply(data, values = null) {
    const groups = [], times = [];
    data.events.forEach((event, id) => {
      const time = values?.starts?.[id] ?? event[0];
      if (times.at(-1) !== time) { times.push(time); groups.push([]); }
      groups[groups.length - 1].push(id);
    });
    return {times, groups};
  }

  function diagnostic(data, values, domain, cachedSupply) {
    const [begin, end] = domain === "interior" ? data.interior : [data.begin, domain === "capture" ? data.end : values.end ?? data.end];
    const first = (begin - data.begin) / data.step, stop = (end - data.begin) / data.step;
    const size = stop - first, {times, groups} = values.starts ? supply(data, values) : cachedSupply || supply(data);
    const eventTimes = values.starts || data.events.map((event) => event[0]);
    const deficit = new Float64Array(size), prefix = new Float64Array(size + 1), lowPrefix = new Float64Array(size + 1);
    let lowCycles = 0, sumBw = 0, bwCorrection = 0, areaCorrection = 0;
    for (let i = 0; i < size; i++) {
      const bw = values.bandwidth[first + i], d = Math.max(0, data.target - bw);
      // Long FIFO drain tails can repeat the same fractional value for
      // hundreds of thousands of bins. Compensated sums retain area accuracy
      // without relaxing the independently verified numeric tolerances.
      deficit[i] = d;
      const term = d * data.step - areaCorrection, nextArea = prefix[i] + term;
      areaCorrection = (nextArea - prefix[i]) - term; prefix[i + 1] = nextArea;
      if (d > 0) lowCycles += data.step;
      lowPrefix[i + 1] = lowCycles;
      const bwTerm = bw - bwCorrection, nextBw = sumBw + bwTerm;
      bwCorrection = (nextBw - sumBw) - bwTerm; sumBw = nextBw;
    }
    const areaAt = (time, low = false) => {
      const offset = Math.max(0, Math.min(size, (time - begin) / data.step));
      if (offset === size) return (low ? lowPrefix : prefix)[size];
      const i = Math.floor(offset);
      return (low ? lowPrefix : prefix)[i] + (offset - i) * data.step * (low ? (deficit[i] > 0 ? 1 : 0) : deficit[i]);
    };
    const cuts = [begin];
    for (let i = lowerBound(times, begin + Number.EPSILON); i < times.length && times[i] < end; i++) if (times[i] > begin) cuts.push(times[i]);
    cuts.push(end);
    const gaps = [], hot = new Map();
    const total = prefix[size];
    for (let i = 0; i + 1 < cuts.length; i++) {
      const a = cuts[i], b = cuts[i + 1], next = lowerBound(times, b);
      const ids = next < groups.length ? groups[next] : [];
      const pcs = [...new Set(ids.map((id) => data.events[id][1]))].sort((a, b) => a - b), key = pcs.join(",");
      const area = Math.max(0, areaAt(b) - areaAt(a)), low = Math.max(0, areaAt(b, true) - areaAt(a, true));
      const gap = {begin: a, end: b, nextIds: ids, pcs, deficit: area, lowCycles: low};
      gaps.push(gap);
      if (!hot.has(key)) hot.set(key, {pcs, deficit: 0, lowCycles: 0, gapCount: 0, lowGapCount: 0, waveTasks: new Set(), example: gap});
      const item = hot.get(key); item.deficit += area; item.lowCycles += low; item.gapCount++;
      if (low > 1e-7) item.lowGapCount++;
      ids.forEach((id) => item.waveTasks.add(`${data.events[id][2]}:${data.events[id][3]}`));
      if (area > item.example.deficit) item.example = gap;
    }
    const hotspots = [...hot.values()].map((item) => ({...item, waveTasks: item.waveTasks.size, share: total ? item.deficit / total : 0}));
    const compare = (a, b) => b.deficit - a.deficit || String(a.pcs).localeCompare(String(b.pcs), "en", {numeric: true});
    hotspots.sort(compare).forEach((item, rank) => { item.rank = rank + 1; });
    const regions = [];
    let i = 0, gapIndex = 0;
    while (i < size) {
      if (deficit[i] <= 0) { i++; continue; }
      const l = i;
      let minimum = values.bandwidth[first + i], minIndex = first + i, bwSum = 0;
      while (i < size && deficit[i] > 0) {
        const bw = values.bandwidth[first + i]; bwSum += bw;
        if (bw < minimum) { minimum = bw; minIndex = first + i; } i++;
      }
      const a = begin + l * data.step, b = begin + i * data.step, area = prefix[i] - prefix[l], byPc = new Map();
      while (gapIndex < gaps.length && gaps[gapIndex].end <= a) gapIndex++;
      for (let g = gapIndex; g < gaps.length && gaps[g].begin < b; g++) {
        const gap = gaps[g], key = gap.pcs.join(","), part = areaAt(Math.min(b, gap.end)) - areaAt(Math.max(a, gap.begin));
        if (!byPc.has(key)) byPc.set(key, {pcs: gap.pcs, deficit: 0}); byPc.get(key).deficit += part;
      }
      const top = [...byPc.values()].sort(compare)[0];
      regions.push({begin: a, end: b, cycles: b - a, deficit: area, share: total ? area / total : 0,
        meanBandwidth: bwSum / (i - l), minBandwidth: minimum, index: minIndex,
        issued: lowerBound(eventTimes, b) - lowerBound(eventTimes, a), topPcs: top?.pcs || [], topShare: area ? (top?.deficit || 0) / area : 0});
    }
    regions.sort((a, b) => b.deficit - a.deficit || a.begin - b.begin).forEach((r, rank) => { r.rank = rank + 1; });
    return {begin, end, cycles: end - begin, deficit: total, lowCycles, lowFraction: lowCycles / (end - begin),
      deficitFraction: total / (data.target * (end - begin)), meanBandwidth: sumBw / size, regions, hotspots};
  }

  window.VmemLatency = {model, diagnostic, supply};
})();
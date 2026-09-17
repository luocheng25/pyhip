// SPDX-License-Identifier: MIT
"use strict";

// Counterfactual CU-wide FCFS admission, not hardware FIFO reconstruction.
// ATT successful-issue timestamps are immutable arrivals to this model.
(() => {
  const MAX_BINS = 1000000;
  function push(heap, value) {
    let i = heap.length; heap.push(value);
    while (i > 0) { const p = (i - 1) >>> 1; if (heap[p] <= value) break; heap[i] = heap[p]; i = p; }
    heap[i] = value;
  }
  function pop(heap) {
    const first = heap[0], last = heap.pop();
    if (heap.length) {
      let i = 0;
      while (2 * i + 1 < heap.length) {
        let child = 2 * i + 1;
        if (child + 1 < heap.length && heap[child + 1] < heap[child]) child++;
        if (heap[child] >= last) break;
        heap[i] = heap[child]; i = child;
      }
      heap[i] = last;
    }
    return first;
  }

  function schedule(data, latencyNs, capacity) {
    if (capacity !== null && (!Number.isInteger(capacity) || capacity < 1 || capacity > 8192)) throw new Error("FIFO容量必须是1～8192的整数（条/CU）");
    if (!Number.isFinite(data.nsPerCycle) || data.nsPerCycle <= 0) throw new Error("无效ATT时钟");
    const lengths = data.classes.map((c) => {
      const ns = latencyNs[c.key];
      if (!Number.isFinite(ns) || ns < 1 || ns > 10000) throw new Error("每个延迟必须是1～10000ns的有限数值");
      return ns / data.nsPerCycle;
    });
    const n = data.events.length, starts = new Float64Array(n), expires = new Float64Array(n), delays = new Float64Array(n);
    const heap = [];
    let previous = -Infinity, previousArrival = -Infinity, peakActive = 0, delayed = 0, totalDelay = 0, maxDelay = 0, lastCompletion = data.begin;
    for (let i = 0; i < n; i++) {
      const e = data.events[i], arrival = e[0], kind = e[8] ?? e[4];
      if (!Number.isFinite(arrival) || arrival < previousArrival || !Number.isFinite(lengths[kind])) throw new Error("访问顺序或延迟类别无效");
      previousArrival = arrival;
      let start = Math.max(arrival, previous);
      while (heap.length && heap[0] <= start) pop(heap);
      if (capacity !== null && heap.length >= capacity) {
        start = heap[0];
        while (heap.length && heap[0] <= start) pop(heap);
      }
      const finish = start + lengths[kind], wait = start - arrival;
      starts[i] = start; expires[i] = finish; delays[i] = wait;
      push(heap, finish); previous = start; peakActive = Math.max(peakActive, heap.length);
      if (wait > 0) delayed++;
      totalDelay += wait; maxDelay = Math.max(maxDelay, wait); lastCompletion = Math.max(lastCompletion, finish);
    }
    return {starts, expires, delays, latencies: lengths, capacity, peakActive, delayed, totalDelay, maxDelay, lastCompletion};
  }

  function model(data, latencyNs, capacity) {
    const planned = schedule(data, latencyNs, capacity);
    const end = data.begin + Math.ceil((Math.max(data.end, planned.lastCompletion) - data.begin) / data.step) * data.step;
    const bins = (end - data.begin) / data.step;
    if (!Number.isSafeInteger(bins) || bins > MAX_BINS) throw new Error(`模型排空尾部需要${bins.toLocaleString()}个bin，超过页面上限；请增加FIFO容量或缩短L（未丢弃请求）`);
    const shifted = data.events.map((e, i) => [planned.starts[i], ...e.slice(1)]);
    const values = VmemLatency.model({...data, end, events: shifted}, latencyNs);
    const edges = new Float64Array(bins), delta = new Float64Array(bins + 1);
    for (let i = 0; i < data.events.length; i++) {
      const a = Math.max(0, (data.events[i][0] - data.begin) / data.step), b = Math.min(bins, (planned.starts[i] - data.begin) / data.step);
      if (b <= a) continue;
      const l = Math.floor(a), r = Math.ceil(b) - 1;
      if (l === r) edges[l] += b - a;
      else { edges[l] += l + 1 - a; edges[r] += b - r; delta[l + 1]++; delta[r]--; }
    }
    let count = 0, peakQueued = 0;
    for (let i = 0; i < bins; i++) { count += delta[i]; edges[i] += count; peakQueued = Math.max(peakQueued, edges[i]); }
    return {...values, starts: planned.starts, expires: planned.expires, delays: planned.delays,
      end, waitingCount: edges, fifo: {...planned, starts: undefined, expires: undefined, delays: undefined,
        peakQueued, captureEnd: data.end, tailCycles: Math.max(0, end - data.end),
        policy: "CU-wide FCFS, one slot per VMEM; stable ATT order; completions release slots at start+L"}};
  }

  globalThis.VmemFifo = {schedule, model, MAX_BINS};
})();
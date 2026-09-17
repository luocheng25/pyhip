// SPDX-License-Identifier: MIT
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const number = (value, digits = 3) => Number(value).toFixed(digits);
  const state = {model: "mean", unit: "cycles", domain: "interior", view: "bandwidth", index: 0, pc: null, event: null, ready: false, computing: false};
  let data, times, maxLatency, waveEvents, instructionTimes, drawing = false, hoverFrame = 0, pointerStart = null, nativeClicks = 0;
  let overviewBound = false, selectedRegion = null;
  let customBase = "mean", cachedSupply = null;
  const api = {ready: false};
  window.VmemTimeline = api;
  const currentModel = () => data.models[state.model];
  const serviceTimes = (m = currentModel()) => m.starts || times;
  const eventStart = (id, m = currentModel()) => m.starts?.[id] ?? data.events[id][0];
  const modelEnd = () => currentModel().end ?? data.end;
  const fifoEnabled = () => Boolean(data.fifoConfig && $("fifo-enabled").checked);
  // Capability is separate from the optional, default-off retirement setting.
  const issueRule = () => Boolean(data?.fifoConfig?.supportsIssueRetirement || data?.fifoConfig?.issueRetiresOldest);
  const retirementSelected = () => issueRule() && $("issue-retire-enabled").checked;
  const cacheFields = [["a_load", "a"], ["b_dma", "b"], ["c_store", "c"]];
  const cacheHitFlags = () => issueRule() ? Object.fromEntries(cacheFields.map(([key, field]) => [key, $("cache-hit-" + field).checked])) : {};
  const eventLength = (id, m = currentModel()) => m.eventLatencies?.[id] ?? m.latencies[data.events[id][8] ?? data.events[id][4]];
  const eventExpiry = (id, m = currentModel()) => m.expires?.[id] ?? eventStart(id, m) + eventLength(id, m);
  const maximumLength = (m) => peakOf(m.eventLatencies || m.latencies);

  function lowerBound(array, value) {
    let lo = 0, hi = array.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (array[mid] < value) lo = mid + 1; else hi = mid; }
    return lo;
  }

  function inspectBin(index, model = state.model) {
    index = Math.max(0, Math.min(data.models[model].bandwidth.length - 1, Math.floor(index)));
    const a = data.begin + index * data.step, b = a + data.step, m = data.models[model];
    const starts = serviceTimes(m);
    const active = [], groups = new Map();
    const left = lowerBound(starts, a - maxLatency[model]);
    const right = lowerBound(starts, b);
    for (let id = left; id < right; id++) {
      const [issue, pc, wave, task, kind, payload] = data.events[id];
      const start = starts[id], length = eventLength(id, m), expiry = eventExpiry(id, m);
      const overlap = Math.max(0, Math.min(b, expiry) - Math.max(a, start));
      if (overlap <= 0) continue;
      const fraction = overlap / data.step;
      const bandwidth = fraction * payload / (length * data.nsPerCycle) * 256 / 1000;
      const event = {id, issue, start, wait: start - issue, pc, wave, task, kind, payload, expiry, fraction, bandwidth, isNew: start >= a};
      active.push(event);
      if (!groups.has(pc)) groups.set(pc, {pc, bandwidth: 0, count: 0, ids: [], waves: new Set()});
      const group = groups.get(pc);
      group.bandwidth += bandwidth; group.count += fraction; group.ids.push(id); group.waves.add(wave);
    }
    const issuedIds = Array.from({length: right - lowerBound(starts, a)}, (_, i) => lowerBound(starts, a) + i);
    const arrivalIds = Array.from({length: lowerBound(times, b) - lowerBound(times, a)}, (_, i) => lowerBound(times, a) + i);
    const waiting = [];
    if (m.starts) for (let id = lowerBound(starts, a); id < lowerBound(times, b); id++) {
      const overlap = Math.min(b, starts[id]) - Math.max(a, times[id]);
      if (overlap > 0) waiting.push({id, fraction: overlap / data.step});
    }
    return {index, a, b, model, active, issuedIds, arrivalIds, waiting, waitingCount: m.waitingCount?.[index] ?? 0,
      groups: [...groups.values()].map((g) => ({...g, waves: [...g.waves].sort((a, b) => a - b)})).sort((a, b) => b.bandwidth - a.bandwidth || a.pc - b.pc),
      bandwidth: m.bandwidth[index], count: m.count[index],
      sumBandwidth: active.reduce((s, e) => s + e.bandwidth, 0),
      sumCount: active.reduce((s, e) => s + e.fraction, 0)};
  }

  const coordinate = (cycle) => state.unit === "cycles" ? cycle : (cycle - data.begin) * data.nsPerCycle / 1000;
  const cycleAt = (x) => state.unit === "cycles" ? x : x * 1000 / data.nsPerCycle + data.begin;
  const domain = () => state.domain === "interior" ? data.interior : [data.begin, state.domain === "capture" ? data.end : modelEnd()];
  const indexAt = (cycle) => Math.max(0, Math.min(currentModel().bandwidth.length - 1, Math.floor((cycle - data.begin) / data.step)));
  const diagnosis = () => data.diagnostics?.[state.model]?.[state.domain];
  const deficitView = () => Boolean(data.diagnostics) && state.view === "deficit";
  const inflightView = () => state.view === "inflight";
  const peakOf = (array) => array.reduce((peak, value) => Math.max(peak, value), 0);
  const deficitValues = () => data.models[state.model].bandwidth.map((value) => Math.max(0, data.target - value));
  const percent = (value) => `${number(value * 100, 2)}%`;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function kindDot(kind) {
    const node = element("span", "", "kind");
    node.style.background = data.classes[kind]?.color || "#9baec6";
    return node;
  }

  function codeButton(pc, label, eventId = null) {
    const button = element("button", label);
    button.dataset.pc = String(pc);
    if (eventId !== null) button.dataset.event = String(eventId);
    button.addEventListener("click", () => { state.pc = pc; state.event = eventId; showSource(pc, eventId); markRows(); highlight(); });
    return button;
  }

  function snippet(lines, target, radius = 5) {
    const pre = element("pre");
    for (let line = Math.max(1, target - radius); line <= Math.min(lines.length, target + radius); line++) {
      const row = element("span", undefined, "code-line" + (line === target ? " hot" : ""));
      row.append(element("span", String(line), "line-number"), document.createTextNode(lines[line - 1]));
      pre.append(row);
    }
    return pre;
  }

  function showSource(pc, eventId = null) {
    const site = data.sites[pc], content = $("source-content");
    content.replaceChildren();
    if (!site) {
      content.textContent = "所选 bin 没有模型在途访问；这不等于硬件空闲。可单独点击前后请求查看其代码。";
      $("source-badge").textContent = "无在途访问来源";
      if (data.diagnostics) {
        const next = lowerBound(serviceTimes(), data.begin + (state.index + .5) * data.step);
        if (next < times.length) { const row = element("p", "下一补充请求（不是当前来源）：", "source-jump");
          row.append(codeButton(data.events[next][1], eventLabel(next), next)); content.append(row); }
      }
      return;
    }
    $("source-badge").textContent = `PC ${pc} · ${site.phase}`;
    if (data.diagnostics) {
      const active = inspectBin(state.index);
      const contributes = eventId === null ? active.groups.some((g) => g.pc === pc) : active.active.some((e) => e.id === eventId);
      content.append(element("p", site.kind === null ? "代码上下文（非访存，不分摊带宽缺口）" : contributes
        ? "当前在途访问来源（不是带宽不足的原因证明）" : "补充请求 / 代码上下文（不是当前流量来源，也不是根因证明）", "wave-low"));
    }
    const title = element("p", site.meaning);
    title.prepend(kindDot(site.kind)); content.append(title);
    content.append(element("p", `${site.kind === null ? "非 VMEM：仅用于代码上下文，不计入访问来源" : `${site.payload} B / wave（${site.payloadSemantics}）`} · 原代码地址 ${site.address}`, "source-meta muted"));
    content.append(element("p", site.op, "source-meta"));
    if (eventId !== null) {
      const e = data.events[eventId], length = eventLength(eventId);
      const overlap = inspectBin(state.index).active.find((item) => item.id === eventId);
      const packet = e.length > 6 ? ` · ${e[6] >= 0 ? `执行 q${e[6]}` : data.adjustableLatency && ["pyhip", "legacy"].includes(data.family) ? "未标记执行阶段" : "任务启动 / metadata"}${e[7] >= 0 ? ` / 供给 q${e[7]}` : ""}` : "";
      const start = eventStart(eventId);
      content.append(element("p", `Wave ${e[2]} / task ${e[3]}${packet} · ATT issue/模型到达 ${e[0]} → 接纳 ${number(start, 3)} → 贡献结束/释放 ${number(eventExpiry(eventId), 3)} cycles · 等待 ${number((start - e[0]) * data.nsPerCycle, 3)}ns · ${overlap ? `本 bin 贡献 ${number(overlap.bandwidth, 6)} TB/s` : "不覆盖当前服务窗口；仅查看该访问的代码"}`, "source-meta"));
      if (issueRule() && currentModel().naturalExpires) {
        const m = currentModel(), trigger = m.cutBy[eventId];
        content.append(element("p", `配置L=${number(length * data.nsPerCycle, 3)}ns；自然完成 ${number(m.naturalExpires[eventId], 3)} cycles；${trigger >= 0 ? `被新ATT issue #${trigger} / PC ${data.events[trigger][1]} 于 ${data.events[trigger][0]} cycles截断并释放槽位；此前仍按S/L贡献，不补摊剩余字节` : "未被新issue截断"}。`, "source-meta wave-low"));
      }
    }
    const appendCode = (label, lines, line, radius, open = true) => {
      const block = element(data.diagnostics ? "details" : "div", undefined, "source-block");
      if (data.diagnostics) block.open = open;
      block.append(element(data.diagnostics ? "summary" : "div", label, "source-title"), snippet(lines, line, radius));
      content.append(block);
    };
    if (site.file && site.line) appendCode(`${site.sourceMapping || "DWARF 冻结来源"}：${site.file}:${site.line}`, data.sources[site.file].lines, site.line, 4, site.anchors.length === 0);
    else content.append(element("p", "该PC没有可靠源码行映射；只显示准确ISA及单独标注的helper，不伪造DWARF。", "source-meta muted"));
    for (const [index, anchor] of site.anchors.entries()) {
      appendCode(`语义定位（非 DWARF）：${anchor.label} · ${anchor.file}:${anchor.line}`, data.sources[anchor.file].lines, anchor.line, 4, index === 0);
    }
    appendCode(`准确 ISA：${data.isaName || "22_final_isa.s"}:${site.isaLine} · PC ${pc}`, data.isa, site.isaLine, 6);
  }

  function markRows() {
    for (const row of document.querySelectorAll("#origin-table tbody tr, #access-table tbody tr")) {
      row.classList.toggle("selected", Number(row.dataset.pc) === state.pc && (state.event === null || !row.dataset.event || Number(row.dataset.event) === state.event));
    }
  }

  function renderDetails(preferredEvent = null) {
    const bin = inspectBin(state.index);
    state.index = bin.index;
    if (state.ready) $("load-status").textContent = `已加载 ${data.candidate || "M128 persistent"} 全部访问。当前：${state.model} / ${$("domain").selectedOptions[0].textContent}；缺口只由服务中inflight带宽计算，不使用issue-stall。`;
    $("selected-time").textContent = String(bin.a);
    $("selected-us").textContent = `[${bin.a}, ${bin.b}) · ${number((bin.a - data.begin) * data.nsPerCycle / 1000)} µs`;
    $("selected-bw").textContent = number(bin.bandwidth);
    $("selected-count").textContent = number(bin.count);
    $("selected-deficit").textContent = `${number(Math.max(0, data.target - bin.bandwidth))} TB/s`;
    if (data.diagnostics) {
      const deficit = Math.max(0, data.target - bin.bandwidth), lengths = data.models[state.model].latencies;
      const equivalent = (kind) => deficit * 1000 / 256 * lengths[kind] * data.nsPerCycle / 1024;
      $("missing-requests").textContent = `缺 ${number(equivalent(0), 2)} 条 B 等效 / ${number(equivalent(2), 2)} 条 C 等效（两种表达，不相加）`;
      const classes = $("class-counts"); classes.replaceChildren();
      data.classes.forEach((kind, k) => {
        const item = element("span", `${kind.label}：${number(data.models[state.model].classCount[k][bin.index], 2)} 条在途 / ${number(data.models[state.model].classGBs[k][bin.index] * 256 / 1000, 3)} TB/s`);
        item.prepend(kindDot(k)); classes.append(item);
      });
      renderWaveContext(bin); renderRegionAt(bin);
    }
    $("selected-domain").textContent = `${state.model} L · ${bin.a >= data.end ? "模型排空尾部（非ATT采集）" : bin.a >= data.interior[0] && bin.b <= data.interior[1] ? "原内部观察区" : "捕获首尾；不纳入原内部统计"}`;
    $("issued-count").textContent = `到达 ${bin.arrivalIds.length} / 接纳 ${bin.issuedIds.length} 条；服务窗口 ${bin.active.length} 个`;
    if (data.fifoConfig) renderQueue(bin);
    $("source-check").textContent = `Σ来源 ${number(bin.sumBandwidth, 6)} · 数值误差 ${Math.abs(bin.sumBandwidth - bin.bandwidth).toExponential(1)}`;
    $("cycle-input").value = String(bin.a); $("time-slider").value = String(state.index);
    $("origin-count").textContent = `${bin.groups.length} 个 PC · 所有来源均列出`;
    $("access-summary").textContent = `动态访问明细（${bin.active.length} 条，非整数在途均值 ${number(bin.sumCount)}）`;
    $("origin-empty").hidden = bin.active.length !== 0;
    $("origin-empty").textContent = "该 bin 在所选 L 模型下没有在途访问。下方最近补充请求不是这段零带宽的来源。";
    const origins = $("origin-table").tBodies[0], access = $("access-table").tBodies[0];
    origins.replaceChildren(); access.replaceChildren();
    for (const g of bin.groups) {
      const site = data.sites[g.pc], row = element("tr"); row.dataset.pc = String(g.pc);
      const name = element("td"); name.append(kindDot(site.kind), codeButton(g.pc, `${site.meaning} / PC ${g.pc}`));
      row.append(name, element("td", g.waves.map((w) => `W${w}`).join(", ")),
        element("td", number(g.count, 4), "num"), element("td", number(g.bandwidth, 6), "num"),
        element("td", bin.sumBandwidth > 0 ? `${number(100 * g.bandwidth / bin.sumBandwidth, 1)}%` : "—", "num"));
      origins.append(row);
    }
    for (const e of bin.active) {
      const row = element("tr"); row.dataset.pc = String(e.pc); row.dataset.event = String(e.id);
      const name = element("td"); name.append(kindDot(e.kind), codeButton(e.pc, `PC ${e.pc} · ${data.sites[e.pc].phase}`, e.id));
      if (e.isNew) name.append(element("span", "新接纳", "pill"));
      row.append(name, element("td", `W${e.wave} / ${e.task}`), element("td", `${e.issue} → ${number(e.start, 2)} → ${number(e.expiry, 2)}；等待 ${number(e.wait * data.nsPerCycle, 2)}ns`),
        element("td", `${number(e.fraction * 100, 2)}%`, "num"), element("td", number(e.bandwidth, 6), "num"));
      access.append(row);
    }
    const issued = $("issued-list"); issued.replaceChildren();
    if (!bin.issuedIds.length) issued.textContent = "该16-cycle bin没有新的服务接纳；带宽仍可能由更早的服务中访问贡献。";
    for (const id of bin.issuedIds) {
      const e = data.events[id], p = element("p"); p.append(codeButton(e[1], `接纳 ${number(eventStart(id), 3)} / ATT到达 ${e[0]} cycles · W${e[2]} · ${data.sites[e[1]].meaning} · PC ${e[1]}`, id)); issued.append(p);
    }
    const neighbors = $("neighbors"); neighbors.replaceChildren();
    const starts = serviceTimes(), after = lowerBound(starts, (bin.a + bin.b) / 2);
    for (const [label, id] of [["中心之前", after - 1], ["中心及之后", after]]) {
      const block = element("p", `${label}：`);
      if (id < 0 || id >= times.length) block.append(document.createTextNode("捕获中没有记录"));
      else {
        const t = starts[id], left = lowerBound(starts, t);
        let right = left; while (right < starts.length && starts[right] === t) right++;
        for (let j = left; j < right; j++) {
          const e = data.events[j];
          block.append(codeButton(e[1], `${t} · W${e[2]} · ${data.sites[e[1]].meaning} / PC ${e[1]}`, j), document.createTextNode("  "));
        }
      }
      neighbors.append(block);
    }
    state.event = preferredEvent;
    state.pc = preferredEvent !== null ? data.events[preferredEvent][1] : bin.groups[0]?.pc ?? null;
    showSource(state.pc, state.event); markRows();
    const bounds = domain(); $("previous").disabled = bin.a <= bounds[0]; $("next").disabled = bin.b >= bounds[1];
    return bin;
  }

  function renderQueue(bin) {
    const m = currentModel(), f = m.fifo;
    $("queue-summary").textContent = f ? `${f.capacity === null ? "无背压对照" : `共享FIFO ${f.capacity}条/CU`} · 排队过 ${f.delayed}/${data.events.length} 条 · 平均等待 ${number(f.totalDelay / data.events.length * data.nsPerCycle, 3)}ns · 最长等待 ${number(f.maxDelay * data.nsPerCycle, 3)}ns · 服务峰值 ${f.peakActive} 条/CU · 尾部 ${number(f.tailCycles * data.nsPerCycle / 1000, 3)}µs` : "原始无背压保存模型：数据未更改；其历史窗口边界保持原样。";
    if (issueRule()) {
      const retirement = Boolean(f?.issueRetiresOldest);
      $("fifo-controls").querySelector("h2").textContent = retirement ? "FIFO背压：新ATT issue退休最老在途请求" : "FIFO背压：先等槽位，再开始L窗口";
      $("fifo-controls").querySelector(".card-title small").textContent = retirement ? "停止带宽贡献，同时释放槽位" : "每CU共享 · 自然完成释放槽位";
      $("issue-rule-notice").textContent = retirement
        ? "新issue退休已开启：每条新ATT成功issue结束最老且仍贡献带宽的请求并立即释放槽位。该规则从空池起最多保留1条正长度服务窗口，因此C≥1不再改变接纳；同刻按原始稳定顺序处理，贡献率仍为S/L，不补摊剩余字节。"
        : "新issue退休默认关闭：当前不根据新issue截断旧请求，恢复原FIFO容量限制、等待接纳与自然完成释放槽位。A/B/C缓存命中开关仍可独立使用。";
      if (f?.issueRetirementSelected && f.capacity === null) $("issue-rule-notice").textContent = "新issue退休已勾选，但有限FIFO已关闭；当前为完整窗口无背压对照，该规则暂不生效。";
      $("issue-cutoff-summary").hidden = !f;
      $("issue-cutoff-summary").textContent = f ? `新issue提前结束 ${f.truncated ?? 0}/${data.events.length} 条；原始payload ${number((f.issuedPayloadBytes ?? 0) / 1048576, 3)} MiB，贡献积分等效 ${number((f.contributingPayloadBytes ?? 0) / 1048576, 3)} MiB，截断排除 ${number((f.excludedPayloadBytes ?? 0) / 1048576, 3)} MiB。排除量不是实测数据丢失，也没有重新分配给其他请求。` : "";
    }
    const effectiveCapacity = f?.issueRetiresOldest ? 1 : f?.capacity;
    const bound = f?.capacity == null ? Infinity : effectiveCapacity * data.events.reduce((v, e, id) => Math.max(v, e[5] / (eventLength(id, m) * data.nsPerCycle)), 0) * 256 / 1000;
    $("fifo-ceiling").hidden = !(bound < data.target);
    $("fifo-ceiling").textContent = `${f?.issueRetiresOldest ? "每次新issue退休一条，使服务峰值最多1条/CU；C≥1不起限流作用。" : ""}当前规则与L使模型带宽上界仅${number(bound, 3)} TB/s < ${data.target}目标，因此“100%低于目标 / 单一连续低谷”是参数限定结果，不能解释成每个时刻硬件都缺供给。上界不是实测HBM；请结合取消背压对照。`;
    $("queue-bin").textContent = `当前bin：服务中 ${number(bin.count, 4)} 条/CU；等待 ${number(bin.waitingCount, 4)} 条/CU（不贡献带宽）；ATT到达 ${bin.arrivalIds.length}，模型接纳 ${bin.issuedIds.length}。`;
    const body = $("queue-table").tBodies[0]; body.replaceChildren();
    for (const {id, fraction} of bin.waiting.slice(0, 100)) {
      const e = data.events[id], row = element("tr"), cell = element("td");
      cell.append(codeButton(e[1], `#${id} · W${e[2]}/task ${e[3]} · PC ${e[1]}`, id));
      row.append(cell, element("td", `${e[0]} → ${number(eventStart(id), 3)}`),
        element("td", number((eventStart(id) - e[0]) * data.nsPerCycle, 3), "num"), element("td", percent(fraction), "num"));
      body.append(row);
    }
    $("queue-access-count").textContent = `等待明细：${bin.waiting.length} 条覆盖本bin（最多显示前100条；统计包含全部），不是在途带宽来源`;
    $("fifo-tail").disabled = !f || f.tailCycles <= 0;
  }

  function eventLabel(id) {
    const e = data.events[id], site = data.sites[e[1]];
    const packet = e.length > 6 && e[6] >= 0 ? ` q${e[6]}` : "";
    return `PC ${e[1]} · ${site.meaning}${packet} · task ${e[3]}`;
  }

  function renderWaveContext(bin) {
    const tbody = $("wave-table").tBodies[0]; tbody.replaceChildren();
    const center = (bin.a + bin.b) / 2;
    let shown = 0;
    data.waves.forEach((wave) => {
        if (data.adjustableLatency && !$("all-wave-context").checked && !(wave.begin <= center && center < wave.end)
          && !bin.active.some((e) => e.wave === wave.id) && !bin.waiting.some((e) => data.events[e.id][2] === wave.id)) return;
      shown++;
      const row = element("tr"), count = bin.active.filter((e) => e.wave === wave.id).reduce((sum, e) => sum + e.fraction, 0);
      row.append(element("td", `W${wave.id} · S${wave.simd}/${wave.slot}`), element("td", number(count, 3), count < 1 ? "wave-low num" : "num"));
      const local = waveEvents[wave.id], at = lowerBound(local.ids.map((id) => eventStart(id)), center);
      for (const [index, next] of [[at - 1, false], [at, true]]) {
        const cell = element("td");
        if (index < 0 || index >= local.ids.length) cell.textContent = "捕获范围内无记录";
        else {
          const id = local.ids[index], event = data.events[id];
          cell.append(codeButton(event[1], eventLabel(id), id), element("div", `接纳 ${number(eventStart(id), 3)} / ATT ${event[0]} cycles · ${next ? "+" : "−"}${number(Math.abs(eventStart(id) - center), 0)} cycles`, "muted"));
        }
        row.append(cell);
      }
      const cell = element("td"), records = data.waveInstructions[wave.id], i = lowerBound(instructionTimes[wave.id], center);
      if (i < records.length) {
        const [time, pc, task, phase] = records[i];
        cell.append(codeButton(pc, `PC ${pc} · ${phase}`), element("div", `原始ATT（未重排）：${data.sites[pc].op} · task ${task} · +${number(time - center, 0)} cycles`, "muted"));
      } else cell.textContent = "无下一条指令；保留该bin，不缩小观察区";
      row.append(cell); tbody.append(row);
    });
    $("wave-context-title").textContent = `所选时间的wave供给上下文 · 显示${shown}/${data.waves.length}条（统计包含全部）`;
  }

  function pcText(pcs) {
    return pcs.length ? pcs.map((pc) => `PC ${pc} · ${data.sites[pc].meaning}`).join(" + ") : "捕获内无下一次供给";
  }

  function renderRegionAt(bin) {
    const d = diagnosis(); if (!d) return;
    selectedRegion = d.regions.find((row) => row.begin <= bin.a && bin.a < row.end) || null;
    const r = selectedRegion;
    $("region-title").textContent = r ? `连续低谷 #${r.rank} · [${r.begin}, ${r.end})` : "当前 bin 不在低带宽区间内";
    $("region-metrics").textContent = r ? `${r.cycles} cycles / ${number(r.cycles * data.nsPerCycle / 1000, 3)} µs · 区间平均 ${number(r.meanBandwidth)} TB/s · 累计缺口 ${number(r.deficit, 1)} TB/s·cycles · 占观察区全部缺口 ${percent(r.share)}` : "可点击任意红色缺口或下方排名，定位缺少在途请求的区间。";
    const sources = $("region-source"); sources.replaceChildren();
    if (r) {
      sources.append(element("span", `区间主要下一补充位置（占本区间缺口 ${percent(r.topShare)}，非因果归因）： `));
      r.topPcs.forEach((pc) => sources.append(codeButton(pc, `PC ${pc} · ${data.sites[pc].meaning}`), document.createTextNode("  ")));
    }
    for (const row of $("region-table").tBodies[0].rows) row.classList.toggle("rank-active", Number(row.dataset.rank) === r?.rank);
  }

  async function zoomCycles(left, right) {
    const [a, b] = domain();
    await Plotly.relayout($("chart"), {"xaxis.range": [coordinate(Math.max(a, left)), coordinate(Math.min(b, right))]});
    await fitY(); updateOverviewFocus();
  }

  async function focusRegion(rank) {
    const r = diagnosis()?.regions.find((row) => row.rank === rank); if (!r) return;
    selectCycle(data.begin + r.index * data.step);
    // Surface an explicitly labeled next supplier without inventing an
    // origin in a zero-inflight bin. The source table remains unchanged.
    if (!inspectBin(state.index).active.length) {
      const center = data.begin + (state.index + .5) * data.step;
      const id = data.events.findIndex((event, id) => eventStart(id) >= center && r.topPcs.includes(event[1]));
      if (id >= 0) { state.pc = data.events[id][1]; state.event = id; showSource(state.pc, id); highlight(); }
    }
    const margin = Math.max(512, Math.min(2048, r.cycles * .4));
    await zoomCycles(r.begin - margin, r.end + margin);
  }

  async function focusHotspot(row) {
    const {begin, end, nextIds} = row.example;
    let index = indexAt(begin), best = -1;
    for (let i = indexAt(begin); i <= indexAt(end - 1); i++) {
      const deficit = Math.max(0, data.target - data.models[state.model].bandwidth[i]);
      if (deficit > best) { best = deficit; index = i; }
    }
    selectCycle(data.begin + index * data.step);
    const id = nextIds[0];
    if (id !== undefined) { state.pc = data.events[id][1]; state.event = id; showSource(state.pc, id); markRows(); }
    const margin = Math.max(512, Math.min(2048, (end - begin) * .4));
    await zoomCycles(begin - margin, end + margin); highlight();
  }

  function rankShare(value) {
    const cell = element("td", percent(value), "num"), bar = element("div", undefined, "rank-bar");
    bar.style.width = `${Math.max(.5, value * 100)}%`; cell.append(bar); return cell;
  }

  function renderHotspots() {
    const body = $("hotspot-table").tBodies[0], filter = $("hotspot-search").value.toLowerCase(); body.replaceChildren();
    for (const row of diagnosis().hotspots) {
      const title = pcText(row.pcs); if (filter && !title.toLowerCase().includes(filter)) continue;
      const tr = element("tr"), cell = element("td"), button = element("button", `#${row.rank} ${title}`, "link-button");
      tr.dataset.rank = String(row.rank); button.addEventListener("click", () => focusHotspot(row)); cell.append(button);
      tr.append(cell, element("td", `${row.lowGapCount} / ${row.gapCount}`, "num"), rankShare(row.share));
      body.append(tr);
    }
  }

  function renderRankings() {
    const d = diagnosis(); if (!d) return;
    $("summary-low").textContent = percent(d.lowFraction);
    $("summary-low-cycles").textContent = `${number(d.lowCycles, 0)} / ${number(d.cycles, 0)} cycles`;
    $("summary-deficit").textContent = percent(d.deficitFraction);
    $("summary-regions").textContent = String(d.regions.length);
    $("summary-outside").textContent = d.deficit > 0 ? percent(1 - d.regions.slice(0, 3).reduce((s, r) => s + r.share, 0)) : "—";
    $("region-count").textContent = `${d.regions.length} 个 · 全部列出`;
    const body = $("region-table").tBodies[0]; body.replaceChildren();
    for (const row of d.regions) {
      const tr = element("tr"), cell = element("td"), button = element("button", `#${row.rank} · ${row.begin}`, "link-button");
      tr.dataset.rank = String(row.rank); button.addEventListener("click", () => focusRegion(row.rank)); cell.append(button);
      tr.append(cell, element("td", String(row.cycles), "num"), element("td", number(row.meanBandwidth), "num"), rankShare(row.share));
      body.append(tr);
    }
    renderHotspots(); renderParameters();
  }

  function renderParameters() {
    const parent = $("parameter-content"); parent.replaceChildren();
    parent.append(element("p", `${data.candidate} · M${data.config.block_m} / sort${data.config.sort_block_m} · ${data.config.num_waves} waves/CTA · ${data.config.persistent ? `${data.config.persistent_workgroups} persistent CTA` : "独立CTA"} · SE0/CU0；共 ${data.waves.reduce((s, w) => s + w.tasks, 0)} wave-task。`));
    parent.append(element("p", `目标 ${data.target} TB/s；16-cycle bins。D = Σ 16 × max(0,目标−B)，不做正负抵消。ATT校准 ${number(data.nsPerCycle, 9)} ns/cycle；延迟/容量的来源在上方列明。`));
    if (data.fifoConfig) parent.append(element("p", data.fifoConfig.assumptions, "wave-low"));
    const table = element("table"), head = element("thead"), row = element("tr");
    ["类型", `${state.model} L / ns`, "新ATT cycles", "payload单位"].forEach((name) => row.append(element("th", name)));
    head.append(row); table.append(head); const body = element("tbody");
    data.classes.forEach((kind, k) => { const tr = element("tr"); tr.append(element("td", kind.label),
      element("td", number(data.latencyNs[state.model][kind.key], 6), "num"), element("td", number(data.models[state.model].latencies[k], 6), "num"),
      element("td", data.payloadsByClass ? `${data.payloadsByClass[kind.key].join(" / ") || "无请求"} B / wave` : k < 3 ? "1024B / wave" : k === 4 ? "4B operand" : "4 / 96 / 256B")); body.append(tr); });
    table.append(body); parent.append(table);
    parent.append(element("p", data.adjustableLatency
      ? `边界：A/C payload为容量上界，ATT没有实际行掩码。实际B策略：${data.cachePolicies.b_dma.join(" / ")}，C策略：${data.cachePolicies.c_store.join(" / ")}。延迟是探针代理；满载L含路径排队，作为服务L再次排队可能重复计入压力。自定义L是假设，不是新测量；改变L必须重建接纳时间及窗口，不是仅除以新延迟。`
      : "边界：A/C为64-lane容量上界，ATT无实际执行掩码；metadata的A-load L是代理。原C延迟来自 nt sc1 探针，本M256是 nt（无sc1），不是策略匹配的延迟测量。新4/8GiB store复测未替换旧L。", "wave-low"));
    if (data.domainDescription) parent.append(element("p", data.domainDescription));
    parent.append(element("p", "P90/P95/P99只是替代固定L，不是置信区间。×256为均匀CU外推，不能当实测HBM或预测加速；等效B/C缺包是两种单位，不能相加。FIFO已满时，缺口等效条数不能直接再塞入当前容量，不是新增发射建议。"));
  }

  function latencyInputs() {
    const ns = {...(issueRule() ? currentModel().baseLatencyNs || data.latencyNs[state.model] : data.latencyNs[state.model])};
    for (const [key, field] of cacheFields) {
      const input = $("latency-" + field);
      ns[key] = Number(issueRule() && $("cache-hit-" + field).checked ? input.dataset.missValue : input.value);
    }
    ns.metadata = ns.a_load;
    return ns;
  }

  function fillLatencyInputs(ns) {
    for (const [key, field] of cacheFields) {
      const input = $("latency-" + field), hit = issueRule() && $("cache-hit-" + field).checked;
      input.dataset.missValue = String(ns[key]); input.value = String(hit ? 20 : ns[key]);
      if (issueRule()) input.disabled = hit;
    }
  }

  function restoreCacheFlags(model = currentModel()) {
    if (!issueRule()) return;
    for (const [key, field] of cacheFields) $("cache-hit-" + field).checked = Boolean(model.cacheHits?.[key]);
    $("issue-retire-enabled").checked = Boolean(model.fifo?.issueRetirementSelected ?? model.fifo?.issueRetiresOldest);
    $("issue-retire-enabled").disabled = model.fifo?.capacity == null;
    fillLatencyInputs(model.baseLatencyNs || data.latencyNs[state.model]);
  }

  function clearCacheFlags() {
    if (issueRule()) {
      for (const [, field] of cacheFields) $("cache-hit-" + field).checked = false;
      $("issue-retire-enabled").checked = false;
      $("issue-retire-enabled").disabled = !fifoEnabled();
    }
  }

  async function applyLatencies(ns) {
    if (!data.adjustableLatency || state.computing || drawing) return;
    const restoreSwitch = () => { if (data.fifoConfig) $("fifo-enabled").checked = currentModel().fifo?.capacity != null; restoreCacheFlags(); };
    if (Object.values(ns).some((value) => !Number.isFinite(value) || value < 1 || value > 10000)) {
      $("latency-status").textContent = "延迟必须是1～10000ns的有限数值；未更改当前模型。";
      restoreSwitch();
      return;
    }
    const capacity = fifoEnabled() ? Number($("fifo-capacity").value) : null;
    if (capacity !== null && (!Number.isInteger(capacity) || capacity < 1 || capacity > 8192)) {
      $("latency-status").textContent = "FIFO容量必须是1～8192的整数（条/CU）；未更改当前模型。";
      restoreSwitch();
      return;
    }
    state.computing = true; $("latency-apply").disabled = true;
    const controls = ["model", "domain", "unit", "view", "latency-reset", "latency-preset", ...(data.fifoConfig ? ["fifo-enabled", "fifo-capacity", "fifo-reset", "fifo-tail", "fifo-preset", "fifo-apply"] : []), ...(issueRule() ? ["issue-retire-enabled", "cache-hit-a", "cache-hit-b", "cache-hit-c", "latency-a", "latency-b", "latency-c"] : [])];
    controls.forEach((id) => { $(id).disabled = true; });
    $("latency-status").textContent = "正在重算接纳/等待/完成时间、服务inflight、带宽及全部缺口排名…";
    await new Promise((resolve) => requestAnimationFrame(() => resolve()));
    try {
      api.customError = null;
      const started = performance.now();
      const computed = data.fifoConfig ? VmemFifo.model(data, ns, capacity, cacheHitFlags(), retirementSelected()) : VmemLatency.model(data, ns);
      const diagnostics = Object.fromEntries((data.fifoConfig ? ["interior", "capture", "full"] : ["interior", "full"]).map((domain) => [domain, VmemLatency.diagnostic(data, computed, domain, cachedSupply)]));
      const name = data.fifoConfig && capacity !== null ? "fifo" : "custom";
      data.models[name] = computed; data.latencyNs[name] = {...(computed.latencyNs || ns)}; data.diagnostics[name] = diagnostics;
      maxLatency[name] = maximumLength(computed);
      state.model = name; $("model").value = name; state.index = indexAt(data.begin + state.index * data.step);
      fillLatencyInputs(ns); renderRankings(); renderDetails();
      await draw(); selectCycle(data.begin + state.index * data.step);
      const shown = computed.latencyNs || ns;
      $("latency-status").textContent = `已应用${capacity === null ? "无背压" : `FIFO ${capacity}条/CU`}：B/DMA ${number(shown.b_dma, 3)}ns，A ${number(shown.a_load, 3)}ns，C ${number(shown.c_store, 3)}ns；${issueRule() ? "A/B/C缓存开关只改变自身实际类别，metadata/atomic不变；" : ""}接纳/等待/尾部/来源/排名已重算，原始Mean保持不变（${number(performance.now() - started, 0)}ms）。`;
      api.customRevision = (api.customRevision || 0) + 1;
    } catch (error) {
      $("latency-status").textContent = `重算失败：${error.message}`;
      api.customError = String(error.stack || error);
    } finally {
      state.computing = false; $("latency-apply").disabled = false;
      controls.forEach((id) => { $(id).disabled = false; });
      if (data.fifoConfig) {
        $("fifo-enabled").checked = currentModel().fifo?.capacity != null;
        $("fifo-tail").disabled = !(currentModel().fifo?.tailCycles > 0);
      }
      if (issueRule()) restoreCacheFlags();
    }
  }

  async function resetLatencies() {
    if (state.computing || drawing) return;
    state.model = customBase = "mean"; $("model").value = "mean";
    if (data.fifoConfig) $("fifo-enabled").checked = false;
    clearCacheFlags();
    $("latency-preset").value = "mean"; fillLatencyInputs(data.latencyNs.mean);
    state.index = indexAt(data.begin + state.index * data.step);
    renderRankings(); renderDetails(); await draw(false); selectCycle(data.begin + state.index * data.step);
    $("latency-status").textContent = "已恢复原始Mean：使用保存的原始bin，未拟合或改写默认数据。";
  }

  async function restoreFifo(preset = data.fifoConfig.defaultPreset, resetCapacity = false) {
    if (state.computing || drawing) return;
    const values = data.fifoConfig.presets[preset];
    $("fifo-preset").value = preset; $("fifo-enabled").checked = true;
    if (resetCapacity) { $("fifo-capacity").value = String(data.fifoConfig.defaultCapacity); clearCacheFlags(); }
    fillLatencyInputs(values.latencyNs); customBase = "mean";
    $("latency-preset").value = "full-load";
    await applyLatencies(values.latencyNs);
  }

  async function drawOverview() {
    const d = diagnosis(); if (!d) return;
    const first = indexAt(d.begin), last = indexAt(d.end - data.step) + 1;
    const stride = Math.max(1, Math.ceil((last - first) / 720));
    const x = [coordinate(d.begin)], values = [], custom = [], text = [];
    for (let l = first; l < last; l += stride) {
      const r = Math.min(last, l + stride); let sum = 0, worst = l, peak = -1;
      for (let i = l; i < r; i++) {
        const deficit = Math.max(0, data.target - data.models[state.model].bandwidth[i]); sum += deficit;
        if (deficit > peak) { peak = deficit; worst = i; }
      }
      const a = data.begin + l * data.step, b = data.begin + r * data.step;
      x.push(coordinate(b)); values.push(sum / (r - l) / data.target * 100); custom.push(worst);
      text.push(`[${a}, ${b}) cycles<br>正缺口均值 ${number(sum / (r - l), 3)} TB/s<br>点击定位格内最不足的16-cycle bin`);
    }
    await Plotly.react($("overview"), [{type: "heatmap", x, y: ["正缺口"], z: [values], zmin: 0, zmax: 100,
      colorscale: [[0, "#1b524a"], [.35, "#be6d49"], [1, "#ff4568"]], showscale: false, zsmooth: false,
      customdata: [custom], text: [text], hovertemplate: "%{text}<extra></extra>"}],
      {paper_bgcolor: "#192332", plot_bgcolor: "#192332", font: {family: "TimelineCJK, sans-serif", color: "#adc0d9", size: 11},
       margin: {l: 76, r: 24, t: 12, b: 33}, xaxis: {range: [coordinate(d.begin), coordinate(d.end)], fixedrange: true},
       yaxis: {fixedrange: true}, dragmode: false}, {responsive: true, displayModeBar: false});
    $("overview-caption").textContent = `范围 [${d.begin}, ${d.end}) · 概览最多720格，每格${stride}个bin；缺口先逐16-cycle计算再聚合，保证总面积不变。下方主图保留16-cycle分辨率。`;
    if (!overviewBound) {
      $("overview").on("plotly_click", async (event) => {
        const index = event.points[0]?.customdata; if (!Number.isFinite(index)) return;
        selectCycle(data.begin + index * data.step);
        if (selectedRegion) await focusRegion(selectedRegion.rank);
        else await zoomCycles(data.begin + index * data.step - 2048, data.begin + index * data.step + 2048);
      });
      overviewBound = true;
    }
    updateOverviewFocus();
  }

  function updateOverviewFocus() {
    if (!data.diagnostics || !$("overview")._fullLayout || !$("chart")._fullLayout) return;
    const range = $("chart")._fullLayout.xaxis.range;
    return Plotly.relayout($("overview"), {shapes: [{type: "rect", xref: "x", x0: range[0], x1: range[1],
      yref: "paper", y0: 0, y1: 1, fillcolor: "rgba(255,255,255,.1)", line: {color: "#e5f0ff", width: 2}}]});
  }

  function selectionShapes() {
    const a = data.begin + state.index * data.step;
    const shapes = [
      {type: "rect", xref: "x", x0: coordinate(a), x1: coordinate(a + data.step), yref: "paper", y0: 0, y1: 1, fillcolor: "rgba(130,186,255,.22)", line: {width: 0}},
      {type: "line", xref: "x", x0: coordinate(a + data.step / 2), x1: coordinate(a + data.step / 2), yref: "paper", y0: 0, y1: 1, line: {color: "#c9e2ff", width: 1}}
    ];
    if (!deficitView() && !inflightView()) shapes.unshift({type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: data.target, y1: data.target, line: {color: "#ff7b81", width: 1.4, dash: "dash"}});
    if (data.diagnostics && selectedRegion) shapes.unshift({type: "rect", xref: "x", x0: coordinate(selectedRegion.begin), x1: coordinate(selectedRegion.end),
      yref: "paper", y0: 0, y1: 1, fillcolor: "rgba(245,94,113,.09)", line: {color: "#a85563", width: 1}, layer: "below"});
    if (state.event !== null) {
      const e = data.events[state.event];
      const track = data.waves[e[2]].track ?? e[2];
      const start = eventStart(state.event);
      if (start > e[0]) shapes.push({type: "rect", xref: "x", x0: coordinate(e[0]), x1: coordinate(start),
        yref: "y2", y0: track - .35, y1: track + .35, fillcolor: "rgba(182,153,255,.15)", line: {color: "#b699ff", width: 1, dash: "dot"}});
      shapes.push({type: "rect", xref: "x", x0: coordinate(start), x1: coordinate(eventExpiry(state.event)),
        yref: "y2", y0: track - .35, y1: track + .35, fillcolor: "rgba(245,185,92,.26)", line: {color: "#f5b95c", width: 1}});
    }
    if (modelEnd() > data.end) shapes.push({type: "line", xref: "x", x0: coordinate(data.end), x1: coordinate(data.end),
      yref: "paper", y0: 0, y1: 1, line: {color: "#b699ff", width: 2, dash: "dot"}});
    return shapes;
  }

  function highlight() { if (state.ready && !drawing) return Plotly.relayout($("chart"), {shapes: selectionShapes()}); }

  function fitY() {
    if (drawing || !state.ready || !$("auto-y").checked || deficitView()) return;
    const range = $("chart")._fullLayout.xaxis.range;
    const left = indexAt(cycleAt(range[0])), right = indexAt(cycleAt(range[1]));
    let peak = inflightView() ? 1 : data.target;
    const values = inflightView() ? data.models[state.model].count : data.models[state.model].bandwidth;
    for (let i = left; i <= right; i++) peak = Math.max(peak, values[i]);
    const upper = Math.max(inflightView() ? 2 : 6, peak * 1.08);
    if (Math.abs($("chart")._fullLayout.yaxis.range[1] - upper) > 1e-6) return Plotly.relayout($("chart"), {"yaxis.range": [0, upper]});
  }

  function selectCycle(cycle, eventId = null, ensureVisible = false) {
    if (!Number.isFinite(cycle)) return;
    const [a, b] = domain();
    state.index = indexAt(Math.max(a, Math.min(b - data.step, cycle)));
    renderDetails(eventId); highlight();
    if (ensureVisible && state.ready) {
      const x = coordinate(data.begin + state.index * data.step), axis = $("chart")._fullLayout.xaxis;
      if (x < axis.range[0] || x > axis.range[1]) {
        const width = Math.min(axis.range[1] - axis.range[0], coordinate(b) - coordinate(a));
        const lo = Math.max(coordinate(a), Math.min(coordinate(b) - width, x - width / 2));
        Plotly.relayout($("chart"), {"xaxis.range": [lo, lo + width]});
      }
    }
    api.selection = {index: state.index, cycle: data.begin + state.index * data.step, pc: state.pc, event: state.event, model: state.model};
    updateOverviewFocus();
  }

  async function draw(resetRange = true) {
    drawing = true;
    const m = data.models[state.model], n = m.bandwidth.length;
    const x = Array.from({length: n}, (_, i) => coordinate(data.begin + (i + .5) * data.step));
    const deficit = deficitView(), counts = inflightView();
    const traces = [{x, y: deficit ? deficitValues() : counts ? m.count : m.bandwidth, name: deficit ? "缺口：max(0,目标−B)" : counts ? "全部在途（不能用统一门槛判断）" : "总在途带宽", mode: "lines", type: "scatter",
      line: {color: deficit ? "#ff7b81" : "#ecf4ff", width: 1.4, shape: "hvh", simplify: false}, fill: deficit ? "tozeroy" : "none", fillcolor: "rgba(255,90,117,.3)",
      customdata: Array.from({length: n}, (_, i) => i), hovertemplate: `${deficit ? "带宽缺口" : "总量"} %{y:.5f} ${counts ? "条/CU" : "TB/s"}<extra></extra>`, meta: deficit ? "deficit" : counts ? "inflight" : "bandwidth"}];
    if (!deficit) data.classes.forEach((kind, k) => {
      traces.push({x, y: counts ? m.classCount[k] : m.classGBs[k].map((v) => v * 256 / 1000), name: kind.label, type: "scatter", mode: "lines", meta: counts ? "inflight" : "bandwidth",
        visible: k < 3 ? true : "legendonly", legendgroup: kind.key,
        line: {color: kind.color, width: 1, shape: "hvh", simplify: false}, customdata: traces[0].customdata,
        hovertemplate: `${kind.label} %{y:.5f} ${counts ? "条/CU" : "TB/s"}<extra></extra>`});
    });
    if (counts && m.waitingCount) traces.push({x, y: m.waitingCount, name: "等待队列（不贡献带宽）", type: "scatter", mode: "lines",
      meta: "inflight", visible: "legendonly", line: {color: "#b699ff", width: 1, shape: "hvh", simplify: false},
      customdata: traces[0].customdata, hovertemplate: "等待 %{y:.5f} 条/CU<extra></extra>"});
    data.classes.forEach((kind, k) => {
      const ids = []; for (let i = 0; i < data.events.length; i++) if (data.events[i][4] === k) ids.push(i);
      traces.push({x: ids.map((id) => coordinate(eventStart(id))), y: ids.map((id) => data.waves[data.events[id][2]].track ?? data.events[id][2]),
        customdata: ids, text: ids.map((id) => { const e = data.events[id]; return `${eventLabel(id)}<br>W${e[2]} · ATT到达 ${e[0]} → 模型接纳 ${number(eventStart(id), 3)} cycles`; }),
        yaxis: "y2", name: kind.label + " admission", legendgroup: kind.key + "-issue", showlegend: false,
        mode: "markers", type: "scatter", marker: {color: kind.color, size: 5, opacity: .8, symbol: "line-ns-open"},
        meta: "event", hovertemplate: "%{text}<extra></extra>"});
    });
    if (data.fifoConfig) traces.push({x: times.map(coordinate), y: data.events.map((e) => data.waves[e[2]].track ?? e[2]),
      customdata: data.events.map((_, id) => id), text: data.events.map((e, id) => `${eventLabel(id)}<br>原ATT成功issue / 模型到达 ${e[0]} cycles`),
      yaxis: "y2", name: "原始ATT到达（只作对照）", visible: "legendonly", mode: "markers", type: "scatter",
      marker: {color: "#90a5bf", size: 4, symbol: "circle-open"}, meta: "arrival", hovertemplate: "%{text}<extra></extra>"});
    const [a, b] = domain(), chart = $("chart");
    const oldRange = chart._fullLayout?.xaxis.range;
    const layout = {paper_bgcolor: "#192332", plot_bgcolor: "#131d2b", font: {family: "TimelineCJK, system-ui, sans-serif", color: "#adc0d9", size: 12},
      margin: {t: 56, b: 58, l: 83, r: 26}, dragmode: "zoom", hovermode: "closest", hoverdistance: 24,
      legend: {orientation: "h", x: 0, y: 1.13, font: {size: 12}},
      xaxis: {title: {text: state.unit === "cycles" ? "ATT cycle（绝对坐标）" : `µs（从原CSV起点${data.begin} cycles）`},
        range: resetRange || !oldRange || oldRange[0] >= coordinate(b) || oldRange[1] <= coordinate(a) ? [coordinate(a), coordinate(b)] : [Math.max(coordinate(a), oldRange[0]), Math.min(coordinate(b), oldRange[1])], gridcolor: "#29394e", zeroline: false, showspikes: true, spikemode: "across", spikecolor: "#9fc6ee", spikethickness: 1},
      yaxis: {domain: [.39, 1], title: {text: deficit ? "缺口 / TB/s ↑ 更不足" : counts ? "平均在途条数 / CU" : "估算带宽 / TB/s"}, range: [0, deficit ? data.target * 1.05 : Math.max(counts ? 2 : 6, peakOf(counts ? m.count : m.bandwidth) * 1.08)], gridcolor: "#29394e", zeroline: false, fixedrange: true},
      yaxis2: {domain: [0, .25], tickvals: data.tracks ? data.tracks.map((t) => t.id) : data.waves.map((w) => w.id), ticktext: data.tracks ? data.tracks.map((t) => t.label) : data.waves.map((w) => `W${w.id} · S${w.simd}/${w.slot}`),
        range: [(data.tracks?.length ?? data.waves.length) - .4, -.6], gridcolor: "#29394e", zeroline: false, fixedrange: true},
      annotations: [{xref: "paper", yref: "paper", x: 0, y: .30, text: m.starts ? "模型服务接纳（非新ATT）；紫色虚线=原捕获末端" : "原ATT成功 VMEM 发射（点击查看代码）", showarrow: false, xanchor: "left", font: {size: 12, color: "#9baec6"}}],
      shapes: selectionShapes()};
    await Plotly.react(chart, traces, layout, {responsive: true, scrollZoom: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"]});
    drawing = false;
    if (data.diagnostics) {
      $("view-help").textContent = deficit ? `红色 = 距 ${data.target} TB/s 目标的正缺口；越高越不足，0 = 达到目标。此图不显示高带宽峰值。`
        : counts ? "B load、A load和C store的在途条数分开看；不同L/S不能共用请求数门槛。下方事件按SIMD/slot轨道显示，实际wave/task在提示中。"
        : `当前L下模型带宽${currentModel().fifo?.issueRetiresOldest ? "（请求窗口按新issue退休规则截断，非峰值裁平）" : "（未裁平峰值）"}；低于虚线才是不足。可切回缺口视图查看严重程度。`;
      await drawOverview();
    }
    await fitY();
    $("time-slider").min = String(indexAt(a)); $("time-slider").max = String(indexAt(b - data.step));
    $("cycle-input").min = String(a); $("cycle-input").max = String(b - data.step);
  }

  function cycleFromPointer(event) {
    const chart = $("chart"), box = chart.getBoundingClientRect(), axis = chart._fullLayout?.xaxis;
    if (!axis) return null;
    const px = event.clientX - box.left - axis._offset;
    const y = event.clientY - box.top;
    if (px < 0 || px > axis._length || y < chart._fullLayout.margin.t || y > box.height - chart._fullLayout.margin.b) return null;
    return cycleAt(axis.range[0] + px / axis._length * (axis.range[1] - axis.range[0]));
  }

  function bind() {
    const chart = $("chart");
    chart.on("plotly_relayout", (event) => { if (Object.keys(event).some((key) => key.startsWith("xaxis.range") || key === "xaxis.autorange")) { fitY(); updateOverviewFocus(); } });
    new ResizeObserver(() => { if (state.ready && !drawing) Plotly.Plots.resize(chart); }).observe(chart);
    if (data.diagnostics) new ResizeObserver(() => { if (state.ready && !drawing) Plotly.Plots.resize($("overview")); }).observe($("overview"));
    chart.on("plotly_click", (event) => {
      const p = event.points[0]; if (!p) return;
      nativeClicks++;
      if (p.data.meta === "event" || p.data.meta === "arrival") { const id = p.customdata; selectCycle(p.data.meta === "arrival" ? data.events[id][0] : eventStart(id), id); }
      else selectCycle(data.begin + p.customdata * data.step);
    });
    chart.addEventListener("pointerdown", (event) => { pointerStart = {x: event.clientX, y: event.clientY, clicks: nativeClicks}; });
    chart.addEventListener("pointerup", (event) => {
      if (!pointerStart || event.button !== 0 || Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4) return;
      const cycle = cycleFromPointer(event), clicks = pointerStart.clicks; pointerStart = null;
      // A whitespace click also selects time; let Plotly's precise point click
      // take precedence when an access marker was hit.
      setTimeout(() => { if (cycle !== null && clicks === nativeClicks) selectCycle(cycle); }, 0);
    });
    chart.addEventListener("pointermove", (event) => {
      const cycle = cycleFromPointer(event); if (cycle === null) return;
      const index = indexAt(cycle), value = data.models[state.model].bandwidth[index];
      $("hover-readout").textContent = `悬停 [${data.begin + index * data.step}, ${data.begin + (index + 1) * data.step}) cycles · 带宽 ${number(value, 5)} / 缺口 ${number(Math.max(0, data.target - value), 5)} TB/s · 单击固定并查看来源`;
      if ($("follow").checked && !pointerStart) { cancelAnimationFrame(hoverFrame); hoverFrame = requestAnimationFrame(() => selectCycle(cycle)); }
    });
    document.addEventListener("pointerup", () => { pointerStart = null; });
    chart.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); selectCycle(data.begin + (state.index + (event.key === "ArrowLeft" ? -1 : 1)) * data.step, null, true); }
    });
    $("model").addEventListener("change", async () => {
      if ($("model").value === "custom" && !data.models.custom) { $("model").value = state.model; return; }
      state.model = $("model").value;
      if (data.adjustableLatency) {
        if (!["custom", "fifo"].includes(state.model)) customBase = state.model;
        if (data.fifoConfig) {
          $("fifo-enabled").checked = currentModel().fifo?.capacity != null;
          if (currentModel().fifo?.capacity != null) $("fifo-capacity").value = String(currentModel().fifo.capacity);
        }
        if (issueRule()) restoreCacheFlags(); else fillLatencyInputs(data.latencyNs[state.model]);
        $("latency-status").textContent = state.model === "custom" ? "当前为自定义L；可恢复原始Mean。" : `当前采用保存的 ${state.model} L；输入新数值后点击应用。`;
      }
      state.index = indexAt(data.begin + state.index * data.step);
      renderRankings(); renderDetails(); await draw(false); selectCycle(data.begin + state.index * data.step);
    });
    $("domain").addEventListener("change", async () => { state.domain = $("domain").value; renderRankings(); selectCycle(data.begin + state.index * data.step); await draw(); });
    $("unit").addEventListener("change", async () => { state.unit = $("unit").value; await draw(); });
    $("jump").addEventListener("click", () => selectCycle(Number($("cycle-input").value), null, true));
    $("cycle-input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("jump").click(); });
    $("previous").addEventListener("click", () => selectCycle(data.begin + (state.index - 1) * data.step, null, true));
    $("next").addEventListener("click", () => selectCycle(data.begin + (state.index + 1) * data.step, null, true));
    $("reset").addEventListener("click", () => draw());
    $("auto-y").addEventListener("change", () => { if (deficitView()) return; if ($("auto-y").checked) fitY(); else Plotly.relayout(chart, {"yaxis.range": [0, Math.max(inflightView() ? 2 : 6, peakOf(inflightView() ? data.models[state.model].count : data.models[state.model].bandwidth) * 1.08)]}); });
    $("time-slider").addEventListener("input", () => selectCycle(data.begin + Number($("time-slider").value) * data.step, null, true));
    $("bookmark").addEventListener("change", async () => {
      if (!$("bookmark").value) return;
      const cycle = Number($("bookmark").value); selectCycle(cycle);
      await Plotly.relayout(chart, {"xaxis.range": [coordinate(cycle - 1600), coordinate(cycle + 1600)]});
    });
    if (data.diagnostics) {
      $("view").addEventListener("change", async () => { state.view = $("view").value; await draw(false); });
      $("worst").addEventListener("click", () => focusRegion(1));
      $("hotspot-search").addEventListener("input", renderHotspots);
      $("all-wave-context").addEventListener("change", () => renderWaveContext(inspectBin(state.index)));
    }
    if (data.adjustableLatency) {
      $("latency-apply").addEventListener("click", () => applyLatencies(latencyInputs()));
      $("latency-reset").addEventListener("click", resetLatencies);
      $("latency-preset").addEventListener("change", () => {
        const preset = $("latency-preset").value;
        if (!preset) return;
        if (preset === "mean") { resetLatencies(); return; }
        if (preset === "full-load") { restoreFifo(); return; }
        const mean = data.latencyNs.mean; let b = mean.b_dma, a = mean.a_load, c = mean.c_store;
        if (preset === "short" || preset === "long") { const factor = preset === "short" ? .5 : 2; b *= factor; a *= factor; c *= factor; }
        if (preset === "load-short") [b, a, c] = [300, 300, 600];
        if (preset === "store-short") [b, a, c] = [900, 700, 150];
        if (preset === "balanced") [b, a, c] = [600, 500, 320];
        customBase = "mean";
        applyLatencies({...mean, b_dma: b, a_load: a, c_store: c, metadata: a});
      });
      for (const id of ["latency-b", "latency-a", "latency-c"]) $(id).addEventListener("keydown", (event) => {
        if (event.key === "Enter") applyLatencies(latencyInputs());
      });
      if (issueRule()) for (const [, field] of cacheFields) $("cache-hit-" + field).addEventListener("change", () => {
        if (state.computing || drawing) { restoreCacheFlags(); return; }
        const input = $("latency-" + field), checked = $("cache-hit-" + field).checked;
        if (checked) input.dataset.missValue = input.value;
        input.value = checked ? "20" : input.dataset.missValue; input.disabled = checked;
        applyLatencies(latencyInputs());
      });
      if (issueRule()) $("issue-retire-enabled").addEventListener("change", () => {
        if (state.computing || drawing) { restoreCacheFlags(); return; }
        applyLatencies(latencyInputs());
      });
      if (data.fifoConfig) {
        $("fifo-enabled").addEventListener("change", () => applyLatencies(latencyInputs()));
        $("fifo-apply").addEventListener("click", () => applyLatencies(latencyInputs()));
        $("fifo-capacity").addEventListener("keydown", (event) => { if (event.key === "Enter") applyLatencies(latencyInputs()); });
        $("fifo-preset").addEventListener("change", () => restoreFifo($("fifo-preset").value));
        $("fifo-reset").addEventListener("click", () => restoreFifo(undefined, true));
        $("fifo-tail").addEventListener("click", async () => {
          state.domain = "full"; $("domain").value = "full"; renderRankings(); await draw();
          selectCycle(data.end); await zoomCycles(Math.max(data.begin, data.end - 512), modelEnd());
        });
      }
    }
  }

  async function start() {
    if (typeof DecompressionStream === "undefined") throw new Error("此自包含页面需要支持 DecompressionStream 的现代浏览器（Chrome/Edge/Firefox）。");
    const bytes = Uint8Array.from(atob($("timeline-payload").textContent.trim()), (c) => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    data = JSON.parse(await new Response(stream).text());
    await document.fonts.load("14px TimelineCJK");
    await document.fonts.ready;
    times = data.events.map((e) => e[0]);
    maxLatency = Object.fromEntries(Object.entries(data.models).map(([name, m]) => [name, maximumLength(m)]));
    api.inspectBin = inspectBin; api.selectCycle = selectCycle; api.data = data; api.getState = () => ({...state, drawing});
    api.focusRegion = focusRegion; api.diagnosis = diagnosis; api.applyLatencies = applyLatencies;
    if (data.adjustableLatency) {
      $("latency-controls").hidden = false;
      if (data.navigation) {
        $("suite-navigation").hidden = false;
        for (const entry of data.navigation) { const option = element("option", entry.candidate); option.value = entry.href; option.selected = entry.candidate === data.candidate; $("suite-case").append(option); }
        $("suite-case").addEventListener("change", () => { location.href = $("suite-case").value; });
      }
      const option = element("option", "自定义load/store L"); option.value = "custom"; $("model").append(option);
      fillLatencyInputs(data.latencyNs.mean);
      $("latency-status").textContent = "原始Mean已载入。可分别调整B/DMA、A/load、C/store，或选择快捷档位。";
      cachedSupply = VmemLatency.supply(data);
      const tracks = [...new Set(data.waves.map((w) => `${w.simd}/${w.slot}`))].sort((a, b) => a.localeCompare(b, "en", {numeric: true}));
      data.tracks = tracks.map((label, id) => ({id, label: `S${label}`}));
      data.waves.forEach((wave) => { wave.track = tracks.indexOf(`${wave.simd}/${wave.slot}`); });
      if (data.fifoConfig) {
        $("fifo-controls").hidden = false; $("queue-panel").hidden = false;
        if (issueRule()) {
          $("cache-hit-controls").hidden = false; $("issue-rule-notice").hidden = false;
          $("issue-retire-control").hidden = false;
          $("issue-retire-enabled").checked = Boolean(data.fifoConfig.defaultIssueRetirement ?? data.fifoConfig.issueRetiresOldest);
        }
        $("fifo-assumptions").textContent = data.fifoConfig.assumptions;
        $("fifo-provenance").textContent = data.fifoConfig.provenance;
        for (const key of Object.keys(data.fifoConfig.presets)) {
          const item = element("option", `${key.toUpperCase()} 满载L（容量不变）`); item.value = key; $("fifo-preset").append(item);
        }
        $("fifo-preset").value = data.fifoConfig.defaultPreset;
        const values = data.fifoConfig.presets[data.fifoConfig.defaultPreset];
        $("fifo-enabled").checked = true; $("fifo-capacity").value = String(data.fifoConfig.defaultCapacity);
        $("fifo-default").textContent = `默认${data.fifoConfig.defaultCapacity}条/CU · 用户可调的模型参数，不是实测硬件容量`;
        const item = element("option", "FIFO + 当前L（服务中/排队分离）"); item.value = "fifo"; $("model").append(item);
        const full = element("option", "本配置满载L（容量不变）"); full.value = "full-load"; $("latency-preset").append(full);
        $("latency-preset").value = "full-load";
        const capture = element("option", "原捕获范围（不含模型尾部）"); capture.value = "capture"; $("domain").append(capture);
        $("domain").querySelector('[value="full"]').textContent = "完整模型（含排空尾部）";
        for (const [name, diagnostics] of Object.entries(data.diagnostics)) diagnostics.capture = diagnostics.full;
        data.models.fifo = VmemFifo.model(data, values.latencyNs, data.fifoConfig.defaultCapacity, {}, retirementSelected());
        data.latencyNs.fifo = {...(data.models.fifo.latencyNs || values.latencyNs)}; maxLatency.fifo = maximumLength(data.models.fifo);
        data.diagnostics.fifo = Object.fromEntries(["interior", "capture", "full"].map((d) => [d, VmemLatency.diagnostic(data, data.models.fifo, d)]));
        state.model = "fifo"; $("model").value = "fifo"; fillLatencyInputs(values.latencyNs);
        $("latency-status").textContent = `已启用FIFO默认${data.fifoConfig.defaultCapacity}条/CU；容量和延迟均可独立调整。取消背压可用相同L对照，恢复原始Mean保留历史数据。`;
        $("window-note").innerHTML = "带宽为<b>服务中payload等效估算</b>，不是HBM实测。FIFO窗口为<b>[接纳, 接纳+L)</b>；等待队列不贡献带宽。下一供给按接纳时间分组，不是根因证明。";
        if (issueRule()) $("window-note").innerHTML = "旧FIFO时间轴：<b>新issue退休为可选规则，默认关闭</b>。关闭时按容量排队并在接纳+L自然完成；勾选后才按新issue提前结束最老在途请求并释放槽位，贡献率仍为S/L。A/B/C缓存命中20ns独立可选，不是HBM实测或真实退休。";
      }
    }
    if (data.diagnostics) {
      state.view = "deficit";
      document.body.classList.add("deficit-mode");
      $("analysis-workspace").classList.add("rich");
      $("analysis-workspace").append($("source-inspector"));
      $("analysis-workspace").before($("region-detail"));
      for (const id of ["diagnosis-overview", "view-label", "worst", "view-help", "rankings", "region-detail", "class-counts", "wave-context", "model-parameters"]) $(id).hidden = false;
      $("page-title").textContent = `${data.candidate} · 在途带宽缺口`;
      $("page-subtitle").textContent = data.fifoConfig ? `有限FIFO背压 · 容量可调（默认${data.fifoConfig.defaultCapacity}） · 等待/服务分离 · 历史无背压对照` : data.adjustableLatency ? "分开load/store在途数 · 可调L · 全部16-cycle数据与代码定位" : "先看不足，再看代码 · 新M256 ATT / 原16-cycle与固定L口径";
      if (issueRule()) $("page-subtitle").textContent = "旧FIFO界面 · 新issue退休可选（默认关闭） · A/B/C缓存命中20ns · 原始无背压对照保留";
      document.title = `${data.candidate} · 在途带宽缺口`;
      $("capture-badge").textContent = `${data.config.num_waves} waves/CTA · ${data.config.persistent ? `${data.config.persistent_workgroups} persistent CTA` : "独立CTA"} · SE0/CU0 ×256`;
      $("packet-note").textContent = data.packetNote || "M256：q0..23为实际N64处理顺序；Memory(q)写C(q−2)、补B(q+3)。中间循环PC会重复，动态q由每wave执行序列重建。没有M128的worker N旋转；DWARF、语义helper与ISA分别标明。";
      waveEvents = data.waves.map((wave) => { const ids = []; data.events.forEach((e, id) => { if (e[2] === wave.id) ids.push(id); }); return {ids, times: ids.map((id) => times[id])}; });
      instructionTimes = data.waveInstructions.map((rows) => rows.map((row) => row[0]));
      renderRankings();
    }
    for (const bookmark of data.bookmarks) { const option = element("option", bookmark.label); option.value = String(bookmark.cycle); $("bookmark").append(option); }
    $("provenance").textContent = `原CSV SHA256 ${data.csvSha256} · 冻结kernel SHA256 ${data.sourceSha256} · ${data.events.length}条VMEM / ${data.models.mean.bandwidth.length}个bin。离线页面，不访问网络。`;
    state.index = indexAt(data.interior[0]);
    renderDetails(); await draw(); bind(); state.ready = true;
    selectCycle(data.interior[0]);
    if (data.diagnostics) await focusRegion(1);
    api.ready = true;
    $("load-status").textContent = `已加载 ${data.candidate || "M128 persistent"} 全部原始bin与访问来源。当前：${state.model} / 内部观察区；无issue-stall指标。`;
  }

  start().catch((error) => {
    api.error = String(error.stack || error);
    $("load-status").textContent = "加载失败：" + error.message;
    $("load-status").classList.add("error");
    console.error(error);
  });
})();
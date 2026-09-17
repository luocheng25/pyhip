# SPDX-License-Identifier: MIT
"""Generate new restored FIFO pages, retaining both earlier HTML releases."""
import argparse
import base64
from copy import deepcopy
import gzip
from html import escape
import json
from pathlib import Path
import re

from build_vmem_service_html import SLUGS, checked_outputs, sha, write_json
from vmem_fifo_issue import POLICY, make_reference
from vmem_inflight_fifo import packed
from vmem_inflight_timeline import embedded_font


HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[3]


def build(old,output,font):
    output.mkdir(parents=True,exist_ok=False)
    protected=checked_outputs(old)
    for slug in SLUGS:protected.update(checked_outputs(old/slug))
    template=(HERE/"vmem_inflight_timeline.template.html").read_text()
    names=("vmem_inflight_latency.js","vmem_inflight_fifo.js","vmem_fifo_issue.js","vmem_inflight_timeline.js")
    app="\n".join((HERE/name).read_text() for name in names)
    license_path=WORKSPACE/"tests/flydsl/moe_8w_down/figures/FONT-LICENSE.txt"
    license_text=license_path.read_text()
    files=[Path(__file__),HERE/"vmem_fifo_issue.py",HERE/"vmem_inflight_fifo.py",HERE/"vmem_inflight_timeline.py",
           HERE/"build_vmem_service_html.py",HERE/"vmem_inflight_timeline.template.html",font,license_path,*(HERE/name for name in names)]
    inputs={str(p):sha(p) for p in files}
    from plotly.offline import get_plotlyjs
    plotly=get_plotlyjs()
    summaries=[]
    for slug in SLUGS:
        source=old/slug/"timeline_data.json.gz"
        previous=json.loads(gzip.decompress(source.read_bytes()));data=deepcopy(previous)
        assert data["schema"]==4 and data["fifoConfig"]["defaultCapacity"]==12
        data["schema"]=6;cfg=data["fifoConfig"]
        cfg.update(supportsIssueRetirement=True,issueRetiresOldest=False,defaultIssueRetirement=False,
               cacheHitNs=20,cacheHitClasses=["a_load","b_dma","c_store"],assumptions=POLICY,
                   retiredWindowPolicy="earliest successful issue still contributing; free slot immediately; natural completion first")
        cfg["provenance"] += " 本版恢复旧模型与旧默认L，不采用服务率模型。新issue退休为独立可选配置，缺省不勾选；关闭时使用原有限FIFO自然完成逻辑。旧L和缓存20ns均为历史/用户模型假设，不是新测量。"
        cfg["knownDegeneracy"]="仅开启新issue退休时：从空池开始最多1条正长度服务窗口，C>=1不再影响接纳。默认关闭，无此退化。"
        cfg["recoveredFrom"]={"path":str(source),"sha256":sha(source)}
        ns=cfg["presets"][cfg["defaultPreset"]]["latencyNs"]
        references={"default":make_reference(data,ns,12),"unbounded":make_reference(data,ns,None),
                "enabled":make_reference(data,ns,12,issue_retirement=True),
                    "hits":{key:make_reference(data,ns,12,{key:True}) for key in ("a_load","b_dma","c_store")},
                "allHits":make_reference(data,ns,12,{key:True for key in ("a_load","b_dma","c_store")}),
                "enabledAllHits":make_reference(data,ns,12,{key:True for key in ("a_load","b_dma","c_store")},issue_retirement=True)}
        for key in ("events","sources","isa","waves","waveInstructions","models","latencyNs","diagnostics"):
            assert data[key]==previous[key],(slug,key)
        dest=output/slug;dest.mkdir()
        payload=packed(data);encoded_font=embedded_font(font,template+app+json.dumps(data,ensure_ascii=False))
        mapping={"FONT":encoded_font,"FONT_LICENSE":escape(license_text),"APP":app,"PLOTLY":plotly,"PAYLOAD":base64.b64encode(payload).decode()}
        html=re.sub(r"@@(FONT|FONT_LICENSE|APP|PLOTLY|PAYLOAD)@@",lambda m:mapping[m[1]],template)
        with (dest/"inflight_timeline.html").open("x") as f:f.write(html)
        # Direct index alias allows opening a case directory without a redirect.
        with (dest/"index.html").open("x") as f:f.write(html)
        with (dest/"timeline_data.json.gz").open("xb") as f:f.write(payload)
        with (dest/"fifo_reference.json.gz").open("xb") as f:f.write(packed(references))
        reference=references["default"]
        summary={"slug":slug,"candidate":data["candidate"],"events":len(data["events"]),"capacity":12,"ns":ns,
                 "fifo":reference["model"]["fifo"],"diagnostics":reference["diagnostics"],
                 "allHitsInterior":references["allHits"]["diagnostics"]["interior"],"originalPayloadSHA256":sha(source),
                 "defaultIssueRetirement":False,"enabledFIFO":references["enabled"]["model"]["fifo"],
                 "note":"Default is original FIFO; optional cutoff exclusions are not physical lost traffic or measured retirement."}
        write_json(dest/"summary.json",summary)
        write_json(dest/"verified.json",{"status":"PASS","input_sha256":{**inputs,str(source):sha(source)},
                   "output_sha256":{p.name:sha(p) for p in dest.iterdir() if p.is_file()},"scope":"Python reference and preserved raw inputs; browser verification separate."})
        summaries.append(summary)
        print("FIFO_ISSUE_PAGE",slug,len(data["events"]),"cut",summary["fifo"]["truncated"],"peak",summary["fifo"]["peakActive"],flush=True)
    rows="".join(f'<tr><td><a href="{s["slug"]}/inflight_timeline.html">{escape(s["candidate"])}</a></td><td>12（原FIFO，可调整）</td><td>{s["ns"]["b_dma"]:.3f} / {s["ns"]["a_load"]:.3f} / {s["ns"]["c_store"]:.3f}</td><td>关闭 / {s["fifo"]["truncated"]}条截断</td><td>{s["diagnostics"]["interior"]["deficitFraction"]:.3%}</td></tr>' for s in summaries)
    index=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>恢复FIFO模型 · 新issue退休 / A-B-C缓存</title><style>body{{margin:28px;background:#101620;color:#e1e9f6;font:15px/1.8 TimelineCJK,system-ui,sans-serif}}main{{max-width:1400px;margin:auto}}a{{color:#92c6ff}}td,th{{padding:12px;border-bottom:1px solid #30415a;text-align:left}}table{{border-collapse:collapse;width:100%}}.scroll{{overflow:auto}}.notice{{background:#192332;padding:16px;border-left:3px solid #f5b95c;margin:18px 0}}small{{color:#b5c5db}}h1{{font-size:26px}}</style><main>
<h1>旧FIFO时间轴 · 新issue退休可选</h1><p>新issue规则已做成独立配置项，<b>缺省不选中</b>。默认C=12、原默认L与A/B/C缓存开关保持不变。</p>
<div class="notice"><b>默认关闭：按原FIFO容量限制排队，等自然完成释放槽位，不截断请求。</b> 勾选“新issue退休最老请求”后，才按新ATT成功issue结束最早且仍贡献带宽的请求并释放槽位。仅开启时从空池开始最多1条正长度窗口，C≥1不再影响接纳。取消勾选立即恢复原调度。重载、切换页面和恢复默认均不勾选。</div>
<p><b>A、B、C各自增加“缓存命中：20ns”开关。</b> 选中立即设置20ns并重算，取消恢复此前L。只改自身实际类别，不改变metadata或atomic；全部默认未选中。命中流量含cache侧payload，不是HBM带宽。</p>
<div class="scroll"><table><thead><tr><th>配置</th><th>容量 / CU</th><th>B / A / C 默认L ns</th><th>默认新issue规则</th><th>原内部区正缺口</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><a href="../../inflight_fifo_20260915/default12/html_final/index.html">保留的上一版FIFO模型</a> · <a href="../../inflight_service_20260916/html_final/index.html">保留的服务率实验模型</a> · <a href="../README.md">修改说明与验收</a></p>
<small>16-cycle逐bin正缺口；全部原始事件、冻结源码、ISA、历史Mean/P90/P95/P99不改。不使用issue-stall排名，不预测kernel加速。本次无GPU测量。缺口是模型结果。</small><details><summary>内嵌字体许可证</summary><pre>{escape(license_text)}</pre></details></main></html>'''
    font_data=embedded_font(font,index)
    index=index.replace("<style>",f"<style>@font-face{{font-family:TimelineCJK;src:url(data:font/woff;base64,{font_data}) format('woff')}}",1)
    with (output/"index.html").open("x") as f:f.write(index)
    write_json(output/"suite_summary.json",{"status":"PASS","rules":POLICY,"cases":summaries,"events":sum(s["events"] for s in summaries),"GPU_runs":0})
    for p,h in protected.items():assert sha(p)==h,p
    write_json(output/"verified.json",{"status":"PASS","pages":7,"input_sha256":inputs,"protected_sha256":protected,
               "output_sha256":{str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file()},"GPU_runs":0})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("old-root","output","font"):p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args();build(a.old_root.resolve(),a.output.resolve(),a.font.resolve())
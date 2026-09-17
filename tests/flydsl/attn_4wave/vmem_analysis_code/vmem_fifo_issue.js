// SPDX-License-Identifier: MIT
"use strict";
// Restore the previous FIFO UI, replacing its model only for explicitly
// versioned issue-retirement data. Old generated pages remain unchanged.
(() => {
  const legacy = globalThis.VmemFifo;
  const ABC = ["a_load","b_dma","c_store"];
  function push(heap,value) {
    let i=heap.length;heap.push(value);
    while(i){const p=(i-1)>>>1;if(heap[p]<=value)break;heap[i]=heap[p];i=p;}heap[i]=value;
  }
  function pop(heap) {
    const first=heap[0],last=heap.pop();
    if(heap.length){let i=0;while(2*i+1<heap.length){let c=2*i+1;if(c+1<heap.length&&heap[c+1]<heap[c])c++;
      if(heap[c]>=last)break;heap[i]=heap[c];i=c;}heap[i]=last;}return first;
  }
  function values(data,ns,hits={}) {
    for(const [k,v] of Object.entries(hits))if(!ABC.includes(k)||typeof v!=="boolean")throw Error("A/B/C缓存开关必须为布尔值");
    for(const c of data.classes)if(!Number.isFinite(ns[c.key])||ns[c.key]<1||ns[c.key]>10000)throw Error("延迟须为1～10000ns有限数值");
    if(!Number.isFinite(data.nsPerCycle)||data.nsPerCycle<=0)throw Error("无效ATT时钟");
    const effective=Object.fromEntries(data.classes.map(c=>[c.key,hits[c.key]?20:ns[c.key]]));
    const lengths=Float64Array.from(data.events,e=>{
      const actual=data.classes[e[4]]?.key,k=data.classes[e[8]??e[4]]?.key;
      if(!actual||!k)throw Error("无效事件类别");
      return (hits[actual]?20:ns[k])/data.nsPerCycle;
    });
    return {effective,lengths};
  }
  function schedule(data,ns,capacity,hits={},issueRetirement=false) {
    if(capacity!==null&&(!Number.isInteger(capacity)||capacity<1||capacity>8192))throw Error("FIFO容量必须为1～8192整数");
    if(typeof issueRetirement!=="boolean")throw Error("新issue退休开关必须为布尔值");
    const retire=issueRetirement&&capacity!==null;
    const {effective,lengths}=values(data,ns,hits),n=data.events.length;
    const starts=new Float64Array(n),expires=new Float64Array(n),naturalExpires=new Float64Array(n),cutBy=new Int32Array(n).fill(-1),delays=new Float64Array(n);
    let previous=-Infinity,previousStart=-Infinity;
    const releases=[];
    for(let i=0;i<n;i++){
      const t=data.events[i][0];if(!Number.isFinite(t)||t<previous)throw Error("ATT成功issue未排序");
      if(!Number.isFinite(data.events[i][5])||data.events[i][5]<=0)throw Error("无效payload");
      let start=t;
      if(capacity!==null&&!retire){
        start=Math.max(t,previousStart);
        while(releases.length&&releases[0]<=start)pop(releases);
        if(releases.length>=capacity){start=releases[0];while(releases.length&&releases[0]<=start)pop(releases);}
      }
      starts[i]=start;expires[i]=naturalExpires[i]=start+lengths[i];delays[i]=start-t;
      if(capacity!==null&&!retire)push(releases,expires[i]);
      // Empty-pool induction: each new issue retires the sole live predecessor
      // (if it has not completed naturally), frees its slot, and admits now.
      // Stable order handles multiple issues at exactly the same timestamp.
      if(retire&&i>0&&expires[i-1]>t){expires[i-1]=t;cutBy[i-1]=i;}
      previous=t;previousStart=start;
    }
    const edges=[];let payload=0,counted=0,pc=0,cc=0,truncated=0,last=data.begin,delayed=0,totalDelay=0,maxDelay=0;
    for(let i=0;i<n;i++){
      if(expires[i]>starts[i])edges.push([starts[i],1],[expires[i],-1]);
      if(cutBy[i]>=0)truncated++;
      const s=data.events[i][5],part=cutBy[i]<0?s:s*(expires[i]-starts[i])/lengths[i];
      let y=s-pc,t=payload+y;pc=(t-payload)-y;payload=t;
      y=part-cc;t=counted+y;cc=(t-counted)-y;counted=t;
      last=Math.max(last,expires[i]);
      if(delays[i]>0)delayed++;totalDelay+=delays[i];maxDelay=Math.max(maxDelay,delays[i]);
    }
    edges.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);let active=0,peak=0;
    for(const [,d] of edges){active+=d;peak=Math.max(peak,active);if(active<0)throw Error("负占用");}
    if(active!==0||capacity!==null&&peak>(retire?1:capacity))throw Error("FIFO/退休规则不一致");
    return {starts,expires,naturalExpires,cutBy,delays,eventLatencies:lengths,
      latencies:data.classes.map(c=>effective[c.key]/data.nsPerCycle),latencyNs:effective,baseLatencyNs:{...ns},
      cacheHits:Object.fromEntries(ABC.map(k=>[k,Boolean(hits[k])])),capacity,peakActive:peak,delayed,totalDelay,maxDelay,
      lastCompletion:last,truncated,issuedPayloadBytes:payload,contributingPayloadBytes:counted,excludedPayloadBytes:Math.max(0,payload-counted),issueRetiresOldest:retire,issueRetirementSelected:issueRetirement};
  }
  function model(data,ns,capacity,hits={},issueRetirement=data.fifoConfig?.defaultIssueRetirement??Boolean(data.fifoConfig?.issueRetiresOldest)) {
    if(!data.fifoConfig?.supportsIssueRetirement&&!data.fifoConfig?.issueRetiresOldest)return legacy.model(data,ns,capacity);
    const p=schedule(data,ns,capacity,hits,issueRetirement),end=data.begin+Math.ceil((Math.max(data.end,p.lastCompletion)-data.begin)/data.step)*data.step;
    const bins=(end-data.begin)/data.step;
    if(!Number.isInteger(bins)||bins<=0||bins>legacy.MAX_BINS)throw Error("模型bin数越界，未截断数据");
    const kinds=data.classes.length;
    const ce=data.classes.map(()=>new Float64Array(bins)),be=data.classes.map(()=>new Float64Array(bins));
    const cd=data.classes.map(()=>new Float64Array(bins+1)),bd=data.classes.map(()=>new Float64Array(bins+1));
    for(let i=0;i<data.events.length;i++){
      const e=data.events[i],a=(p.starts[i]-data.begin)/data.step,b=(p.expires[i]-data.begin)/data.step;
      const x=Math.max(0,a),y=Math.min(bins,b);if(y<=x)continue;
      const l=Math.floor(x),r=Math.ceil(y)-1,k=e[4],g=e[5]/(p.eventLatencies[i]*data.nsPerCycle);
      if(l===r){ce[k][l]+=y-x;be[k][l]+=(y-x)*g;}
      else{ce[k][l]+=l+1-x;ce[k][r]+=y-r;cd[k][l+1]++;cd[k][r]--;
        be[k][l]+=(l+1-x)*g;be[k][r]+=(y-r)*g;bd[k][l+1]+=g;bd[k][r]-=g;}
    }
    const count=new Float64Array(bins),bandwidth=new Float64Array(bins);
    for(let k=0;k<kinds;k++){
      let nc=0,nb=0,correction=0;
      for(let i=0;i<bins;i++){
        nc+=cd[k][i];const y=bd[k][i]-correction,t=nb+y;correction=(t-nb)-y;nb=t;
        ce[k][i]+=nc;be[k][i]+=nb;
        if(Math.abs(ce[k][i])<1e-10)ce[k][i]=0;if(Math.abs(be[k][i])<1e-9)be[k][i]=0;
        if(ce[k][i]<0||be[k][i]<0)throw Error("区间积分为负");
        count[i]+=ce[k][i];bandwidth[i]+=be[k][i]*256/1000;
      }
    }
    const waitingCount=new Float64Array(bins),waitingDelta=new Float64Array(bins+1);
    for(let i=0;i<data.events.length;i++){
      const a=Math.max(0,(data.events[i][0]-data.begin)/data.step),b=Math.min(bins,(p.starts[i]-data.begin)/data.step);
      if(b<=a)continue;const l=Math.floor(a),r=Math.ceil(b)-1;
      if(l===r)waitingCount[l]+=b-a;
      else{waitingCount[l]+=l+1-a;waitingCount[r]+=b-r;waitingDelta[l+1]++;waitingDelta[r]--;}
    }
    let queued=0,peakQueued=0;
    for(let i=0;i<bins;i++){queued+=waitingDelta[i];waitingCount[i]+=queued;peakQueued=Math.max(peakQueued,waitingCount[i]);}
    const {starts,expires,naturalExpires,cutBy,delays,eventLatencies,latencies,latencyNs,baseLatencyNs,cacheHits,...summary}=p;
    return {starts,expires,naturalExpires,cutBy,delays,eventLatencies,latencies,latencyNs,baseLatencyNs,cacheHits,
      end,count,bandwidth,classCount:ce,classGBs:be,waitingCount,
      fifo:{...summary,peakQueued,captureEnd:data.end,tailCycles:end-data.end,
        policy:summary.issueRetiresOldest?"New successful ATT issue retires the oldest contributing VMEM and releases its slot; S/L rate is not renormalized.":"Original finite FIFO: FCFS admission, natural completion releases slots; no issue-triggered retirement."}};
  }
  globalThis.VmemFifo={...legacy,schedule,model,latencyValues:values};
})();
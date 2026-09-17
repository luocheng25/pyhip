// SPDX-License-Identifier: MIT
// Dense CU count sweep and physical-XCD work selection. No hardware masks.
#define main preserved_matched_main
#include "vmem_matched_latency128.cpp"
#undef main

using V4=uint32_t __attribute__((ext_vector_type(4)));
constexpr uint32_t SliceBytes=512u<<20,LDSBytes=98304;
constexpr uint64_t Warm=20000000,BWTime=30000000,SlotTime=250000000,Gap=200000;
struct Resource {union{V4 words;struct{uint64_t base;uint32_t size,flags;};};__device__ Resource(void* p,uint32_t n):base(uint64_t(p)),size(n),flags(0x27000){}};
__device__ __forceinline__ uint32_t read_hw(){uint32_t v;asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)":"=s"(v));return v;}
__device__ __forceinline__ uint32_t read_xcd(){uint32_t v;asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)":"=s"(v));return v;}
__device__ __forceinline__ uint64_t walltime(){uint64_t v;asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)":"=s"(v)::"memory");return v;}
__device__ __forceinline__ uint64_t coretime(){uint64_t v;asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)":"=s"(v)::"memory");return v;}
__device__ __forceinline__ void waittime(uint64_t t){while(walltime()<t)asm volatile("s_sleep 15":::"memory");}
__host__ __device__ int local_rank(uint32_t hw){uint32_t se=(hw>>13)&7,c=(hw>>8)&15;if(se>3||c<(se&1)||c>=8+(se&1))return -1;return se*8+c-(se&1);}
__host__ __device__ int work_index(uint32_t mode,uint32_t count,uint32_t xcd,int rank,uint32_t target,uint32_t rotate,uint32_t block){
    if(mode==0)return block<count?int(block):-1;
    if(rank<0||xcd>=8)return -1;
    uint32_t r=(uint32_t(rank)+32-rotate)%32,x=(xcd+8-target)%8;
    if(mode==1)return x==0&&r<count?int(r):-1;
    uint32_t index=r*8+x;return index<count?int(index):-1;
}
#define DL(O,A,H) "buffer_load_dwordx4 %" #O ", %" #A ", %64, 0 offen" H "\n\t"
#define DL4(O0,O1,O2,O3,A0,A1,A2,A3,H) DL(O0,A0,H) DL(O1,A1,H) DL(O2,A2,H) DL(O3,A3,H)
#define OUT(I) "=&v"(v[I]),"=&v"(v[I+1]),"=&v"(v[I+2]),"=&v"(v[I+3])
#define ADDR(I) "v"(off[I]),"v"(off[I+1]),"v"(off[I+2]),"v"(off[I+3])
#define BATCH(H) asm volatile("; DOMAIN_BATCH_BEGIN\n\t" \
 DL4(0,1,2,3,32,33,34,35,H) DL4(4,5,6,7,36,37,38,39,H) DL4(8,9,10,11,40,41,42,43,H) DL4(12,13,14,15,44,45,46,47,H) \
 DL4(16,17,18,19,48,49,50,51,H) DL4(20,21,22,23,52,53,54,55,H) DL4(24,25,26,27,56,57,58,59,H) DL4(28,29,30,31,60,61,62,63,H) \
 "s_waitcnt vmcnt(0)\n\t; DOMAIN_BATCH_END":OUT(0),OUT(4),OUT(8),OUT(12),OUT(16),OUT(20),OUT(24),OUT(28):ADDR(0),ADDR(4),ADDR(8),ADDR(12),ADDR(16),ADDR(20),ADDR(24),ADDR(28),"s"(resource):"memory")
template<uint32_t P>__device__ __forceinline__ uint64_t bulk(uint64_t stop,V4 resource,uint32_t& index,uint32_t& last,V4* v){
    uint64_t n=0;do{
        #pragma nounroll
        for(uint32_t b=0;b<8;b++){
            uint32_t off[32];
            #pragma unroll
            for(uint32_t i=0;i<32;i++)off[i]=16*(index+i*256+threadIdx.x);
            if constexpr(P==0){BATCH("");}else{BATCH(" nt sc1");}
            last=index;index=(index+8192)&(SliceBytes/16-1);
        }
        ++n;
    }while(walltime()<stop);return n;
}
#undef BATCH
#undef ADDR
#undef OUT
#undef DL4
#undef DL
template<uint32_t P>__device__ __forceinline__ uint32_t step(uint32_t at,V4 resource,V4& last){
    if constexpr(P==0)asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)":"=&v"(last):"v"(at),"s"(resource):"memory");
    else asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen nt sc1\n\ts_waitcnt vmcnt(0)":"=&v"(last):"v"(at),"s"(resource):"memory");
    return last[0];
}
template<uint32_t P,uint32_t Phase>__device__ __forceinline__ Sample chase(V4 resource,uint32_t& at,uint32_t wave,uint64_t scheduled,uint64_t deadline){
    Sample s{};s.wave=wave;s.phase=Phase;s.scheduled=scheduled;s.deadline=deadline;auto& z=s.chase;V4 last{};
    z.hw0=read_hw();z.xcc0=read_xcd();z.start=at;z.steps=Phase?Nodes:0;z.active_lanes=__popcll(__ballot(1));
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)":::"memory");z.wall0=walltime();z.core0=coretime();
    if constexpr(Phase){
        if constexpr(Phase==1)asm volatile("; DOMAIN_CHASE_WARM_BEGIN":::"memory");else asm volatile("; DOMAIN_CHASE_MEASURE_BEGIN":::"memory");
        #pragma nounroll
        for(uint32_t group=0;group<16384;group++){
            #pragma unroll
            for(uint32_t i=0;i<128;i++)at=step<P>(at,resource,last);
        }
        at=step<P>(at,resource,last);
        if constexpr(Phase==1)asm volatile("; DOMAIN_CHASE_WARM_END":::"memory");else asm volatile("; DOMAIN_CHASE_MEASURE_END":::"memory");
    }
    z.core1=coretime();z.wall1=walltime();z.hw1=read_hw();z.xcc1=read_xcd();z.end=at;for(uint32_t i=0;i<4;i++)z.last[i]=last[i];return s;
}
struct DomainWorker {Worker clock;int32_t index,rank;uint32_t selected,pad;};
static_assert(sizeof(DomainWorker)==144);
struct DomainArgs {uint8_t *p0,*p1,*p2,*p3,*data;uint32_t count,mode,target,rotate,salt;uint64_t epoch;Sample* samples;DomainWorker* workers;uint32_t* sinks;};
template<uint32_t P,bool BW>__global__ __launch_bounds__(256) void xcd_domain_kernel(DomainArgs a){
    extern __shared__ uint8_t reserve[];const uint32_t tid=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(tid/64);
    DomainWorker rec{};auto& w=rec.clock;w.entry_wall=walltime();w.hw0=read_hw();w.xcc0=read_xcd();w.block=blockIdx.x;w.wave=wave;w.active_lanes=__popcll(__ballot(1));w.slice=SliceBytes;
    rec.rank=local_rank(w.hw0);rec.index=work_index(a.mode,a.count,w.xcc0,rec.rank,a.target,a.rotate,blockIdx.x);rec.selected=rec.index>=0;
    const bool probe=rec.selected&&!BW&&rec.index==0;w.role=rec.selected?(probe?1:2):0;
    reserve[tid]=tid;if(tid==0)reserve[LDSBytes-1]=0xa5;__syncthreads();
    const uint64_t warm=a.epoch+Warm+Gap,measure=warm+SlotTime,deadline=measure+SlotTime;
    const uint64_t stop=BW?a.epoch+Warm+BWTime:deadline+Gap,export_time=stop+Gap;
    if(!rec.selected){
        // No data accesses whileparked. Onlyonewave polls the boundedtimer;
        // theotherthree waitat thelocalbarrier. No inter-CTA synchronization.
        if(wave==0)waittime(export_time);
    }else{
        waittime(a.epoch);
        if(probe){
            if(tid%64==0){
                auto p=wave==0?a.p0:wave==1?a.p1:wave==2?a.p2:a.p3;Resource r(p,ProbeBytes);uint32_t at=0;
                waittime(warm);auto b=chase<P,0>(r.words,at,wave,warm,measure);b.export_wall=walltime();a.samples[wave*3]=b;
                auto x=chase<P,1>(r.words,at,wave,warm,measure);x.export_wall=walltime();a.samples[wave*3+1]=x;
                waittime(measure);auto z=chase<P,2>(r.words,at,wave,measure,deadline);waittime(export_time);z.export_wall=walltime();a.samples[wave*3+2]=z;
            }
        }else{
            uint32_t slice_id=BW?rec.index:rec.index-1;Resource r(a.data+uint64_t(slice_id)*SliceBytes,SliceBytes);uint32_t index=0,last=0;V4 v[32];
            w.warm0=walltime();asm volatile("; DOMAIN_BULK_WARM_BEGIN":::"memory");w.warm_chunks=bulk<P>(a.epoch+Warm,r.words,index,last,v);asm volatile("; DOMAIN_BULK_WARM_END":::"memory");
            w.warm1=walltime();w.warm_last_index=last;w.work0=walltime();w.core0=coretime();
            asm volatile("; DOMAIN_BULK_WORK_BEGIN":::"memory");w.work_chunks=bulk<P>(stop,r.words,index,last,v);asm volatile("; DOMAIN_BULK_WORK_END":::"memory");
            w.core1=coretime();w.work1=walltime();w.last_index=last;uint32_t sum=0x2468ace0;
            #pragma unroll
            for(uint32_t i=0;i<32;i++){
                #pragma unroll
                for(uint32_t j=0;j<4;j++)sum+=v[i][j];
            }
            w.sum=sum;waittime(export_time);
        }
    }
    __syncthreads();w.hw1=read_hw();w.xcc1=read_xcd();w.export_wall=walltime();
    if(w.role==2)a.sinks[blockIdx.x*256+tid]=w.sum;
    if(tid%64==0)a.workers[blockIdx.x*4+wave]=rec;
}
struct Config {std::string name;uint32_t mode,count,policy,bw;};
using Launch=void(*)(DomainArgs,uint32_t,hipStream_t);
struct Kernel{Launch launch;hipFuncAttributes attr;int occupancy;};
template<uint32_t P,bool B>void launch(DomainArgs a,uint32_t blocks,hipStream_t s){xcd_domain_kernel<P,B><<<blocks,256,LDSBytes,s>>>(a);HIP_OK(hipGetLastError());}
template<uint32_t P,bool B>Kernel describe(){Kernel k{};k.launch=launch<P,B>;auto f=reinterpret_cast<const void*>(xcd_domain_kernel<P,B>);HIP_OK(hipFuncGetAttributes(&k.attr,f));HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.occupancy,f,256,LDSBytes));if(k.attr.localSizeBytes||k.attr.sharedSizeBytes||k.occupancy!=1)throw std::runtime_error("resources");return k;}
Kernel select(const Config& c){if(c.policy==0)return c.bw?describe<0,true>():describe<0,false>();return c.bw?describe<6,true>():describe<6,false>();}
std::vector<Config> plan(const std::filesystem::path& p){std::ifstream f(p);std::string line;std::getline(f,line);std::vector<Config> cs;std::set<std::string> names;
    while(std::getline(f,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);Config c;if(!(in>>c.name>>c.mode>>c.count>>c.policy>>c.bw)||c.mode>2||!c.count||c.count>256||(c.mode&&c.count>32)||(c.policy!=0&&c.policy!=6)||c.bw>1||!names.insert(c.name).second||c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("plan");cs.push_back(c);}if(cs.empty())throw std::runtime_error("emptyplan");return cs;}
void run_domain(const std::filesystem::path& out,const std::vector<Config>& configs,uint32_t seed,uint32_t target,uint32_t rotate,const std::string& power_path){
    hipDeviceProp_t dev{};HIP_OK(hipGetDeviceProperties(&dev,0));char bdf[32];HIP_OK(hipDeviceGetPCIBusId(bdf,32,0));int wall;HIP_OK(hipDeviceGetAttribute(&wall,hipDeviceAttributeWallClockRate,0));if(std::string(bdf)!="0000:85:00.0"||dev.multiProcessorCount!=256||std::string(dev.gcnArchName).find("gfx950")!=0||dev.sharedMemPerMultiprocessor!=163840||wall!=100000||dev.warpSize!=64)throw std::runtime_error("device");
    uint32_t maximum=0;for(const auto& c:configs)maximum=std::max(maximum,c.count);uint64_t data_bytes=uint64_t(maximum)*SliceBytes;size_t free,total;HIP_OK(hipMemGetInfo(&free,&total));if(free<data_bytes+(3ull<<30))throw std::runtime_error("VRAM");
    hipStream_t s;hipEvent_t e0,e1;HIP_OK(hipStreamCreate(&s));HIP_OK(hipEventCreate(&e0));HIP_OK(hipEventCreate(&e1));std::array<uint8_t*,4> probes{};for(auto& p:probes)HIP_OK(hipMalloc(&p,uint64_t(ProbeBytes)+2*Guard));
    uint8_t* data;Sample* samples;DomainWorker* workers;uint32_t* sinks;uint64_t* epoch;uint32_t* next;uint4* eviction;uint32_t* evict_sink;
    HIP_OK(hipMalloc(&data,data_bytes+2*Guard));HIP_OK(hipMalloc(&samples,12*sizeof(Sample)));HIP_OK(hipMalloc(&workers,1024*sizeof(DomainWorker)));HIP_OK(hipMalloc(&sinks,65536*4));HIP_OK(hipMalloc(&epoch,8));HIP_OK(hipMalloc(&next,uint64_t(Nodes)*4));HIP_OK(hipMalloc(&eviction,1ull<<30));HIP_OK(hipMalloc(&evict_sink,65536*4));
    for(uint32_t i=0;i<4;i++){if(!disjoint(uint64_t(probes[i]),uint64_t(ProbeBytes)+2*Guard,uint64_t(data),data_bytes+2*Guard))throw std::runtime_error("dataoverlap");for(uint32_t j=0;j<i;j++)if(!disjoint(uint64_t(probes[i]),uint64_t(ProbeBytes)+2*Guard,uint64_t(probes[j]),uint64_t(ProbeBytes)+2*Guard))throw std::runtime_error("probeoverlap");}
    const uint32_t salt=seed^0x58264719;std::vector<uint32_t> links(Nodes);for(uint32_t i=0;i<Nodes;i++)links[i]=(i+1)%Nodes*128;HIP_OK(hipMemcpyAsync(next,links.data(),uint64_t(Nodes)*4,hipMemcpyHostToDevice,s));
    for(uint32_t w=0;w<4;w++){HIP_OK(hipMemsetAsync(probes[w],0xa5,uint64_t(ProbeBytes)+2*Guard,s));matched_initialize_probe<<<2048,256,0,s>>>(probes[w]+Guard,next,wave_salt(salt,w));HIP_OK(hipGetLastError());}
    HIP_OK(hipMemsetAsync(data,0xa5,data_bytes+2*Guard,s));matched_initialize_background<<<2048,256,0,s>>>(reinterpret_cast<uint32_t*>(data+Guard),data_bytes/4,salt);HIP_OK(hipGetLastError());HIP_OK(hipMemsetAsync(eviction,0x5a,1ull<<30,s));HIP_OK(hipStreamSynchronize(s));
    std::ofstream info(out/"device.json");info<<"{\"bdf\":\""<<bdf<<"\",\"arch\":\""<<dev.gcnArchName<<"\",\"CUs\":256,\"wave_size\":64,\"LDS\":98304,\"LDS_per_CU\":163840,\"wall_khz\":100000,\"sample_bytes\":112,\"worker_bytes\":144,\"nodes\":"<<Nodes<<",\"slice_bytes\":"<<SliceBytes<<",\"seed\":"<<seed<<",\"target_xcd\":"<<target<<",\"rotate\":"<<rotate<<",\"warm_ticks\":"<<Warm<<",\"slot_ticks\":"<<SlotTime<<",\"guard_ticks\":"<<Gap<<",\"BW_ticks\":"<<BWTime<<",\"data_allocation\":"<<uint64_t(data)<<",\"data_bytes\":"<<data_bytes<<",\"probe_allocations\":[";for(uint32_t i=0;i<4;i++){if(i)info<<',';info<<uint64_t(probes[i]);}info<<"],\"power_path\":\""<<power_path<<"\"}\n";info.close();
    std::ofstream meta(out/"cases.json");meta<<"[\n";uint32_t ordinal=0;
    for(const auto& c:configs){auto k=select(c);const uint32_t grid=c.mode?256:c.count;
        matched_eviction<<<256,256,0,s>>>(eviction,evict_sink);HIP_OK(hipGetLastError());HIP_OK(hipMemsetAsync(samples,0xff,12*sizeof(Sample),s));HIP_OK(hipMemsetAsync(workers,0xff,grid*4*sizeof(DomainWorker),s));HIP_OK(hipMemsetAsync(sinks,0xff,grid*256*4,s));HIP_OK(hipStreamSynchronize(s));uint64_t h0=host_ns();matched_epoch<<<1,64,0,s>>>(epoch);HIP_OK(hipGetLastError());uint64_t raw;HIP_OK(hipMemcpyAsync(&raw,epoch,8,hipMemcpyDeviceToHost,s));HIP_OK(hipStreamSynchronize(s));uint64_t h1=host_ns();
        DomainArgs a{probes[0]+Guard,probes[1]+Guard,probes[2]+Guard,probes[3]+Guard,data+Guard,c.count,c.mode,target,rotate,salt,raw+LeadTicks,samples,workers,sinks};
        PowerRecorder power(power_path);HIP_OK(hipEventRecord(e0,s));k.launch(a,grid,s);HIP_OK(hipEventRecord(e1,s));HIP_OK(hipEventSynchronize(e1));power.stop();float ms;HIP_OK(hipEventElapsedTime(&ms,e0,e1));
        std::vector<Sample> ss(c.bw?0:12);std::vector<DomainWorker> ws(grid*4);std::vector<uint32_t> sn(grid*256);if(!ss.empty())HIP_OK(hipMemcpy(ss.data(),samples,ss.size()*sizeof(Sample),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(ws.data(),workers,ws.size()*sizeof(DomainWorker),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(sn.data(),sinks,sn.size()*4,hipMemcpyDeviceToHost));binary(out/(c.name+".samples.bin"),ss);binary(out/(c.name+".workers.bin"),ws);binary(out/(c.name+".sinks.bin"),sn);binary(out/(c.name+".power.bin"),power.samples);
        bool topology=true,coverage=true;std::set<uint32_t> all,selected;std::set<int> indices;uint64_t first=~0ull,last=0;uint32_t probe_key=~0u;
        for(uint32_t block=0;block<grid;block++){auto key=cu(ws[block*4].clock.hw0,ws[block*4].clock.xcc0);if(!all.insert(key).second)topology=false;const int index=ws[block*4].index;if(index>=0){selected.insert(key);indices.insert(index);if(index==0)probe_key=key;}
            for(uint32_t wave=0;wave<4;wave++){const auto& r=ws[block*4+wave];const auto& w=r.clock;if(w.block!=block||w.wave!=wave||w.active_lanes!=64)throw std::runtime_error("workerrecord");if(cu(w.hw0,w.xcc0)!=key||cu(w.hw1,w.xcc1)!=key||r.rank<0||r.index!=index)topology=false;
                int expected=work_index(c.mode,c.count,w.xcc0,local_rank(w.hw0),target,rotate,block);if(r.index!=expected||r.selected!=(expected>=0)||w.role!=(expected>=0?(!c.bw&&expected==0?1:2):0))throw std::runtime_error("selection");
                if(w.role==2){if(!w.work_chunks||w.work1<=w.work0)throw std::runtime_error("work");uint32_t expected_last=uint32_t((w.warm_chunks+w.work_chunks)*65536-8192)%(SliceBytes/16);if(w.last_index!=expected_last||w.work_chunks*65536*16<SliceBytes)coverage=false;
                    const uint32_t id=c.bw?index:index-1;for(uint32_t lane=0;lane<64;lane++){uint32_t wanted=0x2468ace0;for(uint32_t j=0;j<32;j++)for(uint32_t p=0;p<4;p++)wanted+=word(uint64_t(id)*SliceBytes/4+uint64_t(w.last_index+j*256+wave*64+lane)*4+p,salt);if(sn[block*256+wave*64+lane]!=wanted||(lane==0&&w.sum!=wanted))throw std::runtime_error("sink");}}
            }
        }
        if(selected.size()!=c.count||indices.size()!=c.count)topology=false;else for(uint32_t i=0;i<c.count;i++)if(!indices.count(i))topology=false;
        for(uint32_t wave=0;wave<ss.size()/3;wave++)for(uint32_t phase=0;phase<3;phase++){const auto& z=ss[wave*3+phase];const auto& q=z.chase;if(z.wave!=wave||z.phase!=phase||q.start||q.end||q.active_lanes!=1||q.steps!=(phase?Nodes:0))throw std::runtime_error("chain");if(cu(q.hw0,q.xcc0)!=probe_key||cu(q.hw1,q.xcc1)!=probe_key)topology=false;if(q.wall0<z.scheduled||q.wall1>=z.deadline)coverage=false;if(phase){for(uint32_t p=1;p<4;p++)if(q.last[p]!=tag(Nodes-1,p,wave_salt(salt,wave)))throw std::runtime_error("tag");}if(phase==2){first=std::min(first,q.wall0);last=std::max(last,q.wall1);}}
        if(!ss.empty()){for(uint32_t wave=0;wave<4;wave++)if(ss[wave*3].export_wall>=first||ss[wave*3+1].export_wall>=first||ss[wave*3+2].export_wall<=last)coverage=false;for(const auto& r:ws){auto& w=r.clock;if(w.export_wall<=last||(w.role==2&&(w.work0>first||w.work1<last)))coverage=false;}}
        std::vector<uint8_t> guard(Guard);for(auto p:probes)for(uint64_t off:std::array<uint64_t,2>{0,Guard+ProbeBytes}){HIP_OK(hipMemcpy(guard.data(),p+off,Guard,hipMemcpyDeviceToHost));if(!std::all_of(guard.begin(),guard.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("probe guard");}for(uint64_t off:std::array<uint64_t,2>{0,Guard+data_bytes}){HIP_OK(hipMemcpy(guard.data(),data+off,Guard,hipMemcpyDeviceToHost));if(!std::all_of(guard.begin(),guard.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("data guard");}
        if(ordinal++)meta<<",\n";meta<<std::setprecision(15)<<"{\"name\":\""<<c.name<<"\",\"mode\":"<<c.mode<<",\"cus\":"<<c.count<<",\"policy\":"<<c.policy<<",\"bandwidth\":"<<c.bw<<",\"grid\":"<<grid<<",\"target\":"<<target<<",\"rotate\":"<<rotate<<",\"slice\":"<<SliceBytes<<",\"salt\":"<<salt<<",\"epoch\":"<<a.epoch<<",\"raw_epoch\":"<<raw<<",\"host_epoch0\":"<<h0<<",\"host_epoch1\":"<<h1<<",\"event_ms\":"<<ms<<",\"VGPR\":"<<k.attr.numRegs<<",\"private\":0,\"occupancy_max\":1,\"topology_valid\":"<<topology<<",\"coverage_valid\":"<<coverage<<",\"checks\":true}";meta.flush();std::cout<<"XCD_DENSE_CASE "<<c.name<<" selected="<<selected.size()<<" topology="<<topology<<" coverage="<<coverage<<std::endl;
    }
    meta<<"\n]\n";meta.close();for(auto p:probes)HIP_OK(hipFree(p));for(void* p:std::array<void*,8>{data,samples,workers,sinks,epoch,next,eviction,evict_sink})HIP_OK(hipFree(p));HIP_OK(hipEventDestroy(e0));HIP_OK(hipEventDestroy(e1));HIP_OK(hipStreamDestroy(s));
}
int main(int argc,char** argv){try{if(argc!=7)throw std::runtime_error("xcd_dense NEW_OUT PLAN.csv SEED TARGET_XCD ROTATE POWER_SYSFS");if(!std::getenv("HIP_VISIBLE_DEVICES")||std::string(std::getenv("HIP_VISIBLE_DEVICES"))!="5"||std::getenv("HSA_CU_MASK")||std::getenv("ROC_GLOBAL_CU_MASK"))throw std::runtime_error("environment");auto cs=plan(argv[2]);uint32_t target=std::stoul(argv[4]),rotate=std::stoul(argv[5]);if(target>=8||rotate>=32)throw std::runtime_error("selection");std::ifstream f(argv[6]);uint64_t p;if(!(f>>p)||!p)throw std::runtime_error("power");std::filesystem::path out=argv[1];if(std::filesystem::exists(out))throw std::runtime_error("existing output");HIP_OK(hipSetDevice(0));std::filesystem::create_directories(out);run_domain(out,cs,std::stoull(argv[3]),target,rotate,argv[6]);return 0;}catch(const std::exception& e){std::cerr<<"XCD_DENSE_FAILED "<<e.what()<<std::endl;return 1;}}
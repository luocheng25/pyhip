// SPDX-License-Identifier: MIT
// Reuse the immutable host utilities (allocation checks, power recorder,
// initialization, epoch), not the previous experiment's launch schedule.
#define main preserved_matched128_main
#include "vmem_matched_latency128.cpp"
#undef main

using Vec4=uint32_t __attribute__((ext_vector_type(4)));
constexpr uint32_t ScaleSlice=512u<<20, ScaleLDS=96*1024;
constexpr uint64_t ScaleWarm=20000000, ScaleGuard=200000, ScaleSlot=250000000, ScaleBW=30000000;
struct BufferResource {
    union {Vec4 words;struct{uint64_t base;uint32_t bytes,flags;};};
    __device__ BufferResource(void* p,uint32_t n):base(uint64_t(p)),bytes(n),flags(0x27000){}
};
__device__ __forceinline__ uint32_t scale_hw(){uint32_t x;asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)":"=s"(x));return x;}
__device__ __forceinline__ uint32_t scale_xcc(){uint32_t x;asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)":"=s"(x));return x;}
__device__ __forceinline__ uint64_t scale_wall(){uint64_t x;asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)":"=s"(x)::"memory");return x;}
__device__ __forceinline__ uint64_t scale_core(){uint64_t x;asm volatile("s_memtime %0\n\ts_waitcnt lgkmcnt(0)":"=s"(x)::"memory");return x;}
__device__ __forceinline__ void scale_wait(uint64_t until){while(scale_wall()<until)asm volatile("s_nop 7":::"memory");}
#define SCALE_POLICY(OP) \
 if constexpr(P==0){OP("");}else if constexpr(P==1){OP(" sc0");}else if constexpr(P==2){OP(" nt");}else if constexpr(P==3){OP(" sc1");} \
 else if constexpr(P==4){OP(" sc0 nt");}else if constexpr(P==5){OP(" sc0 sc1");}else if constexpr(P==6){OP(" nt sc1");}else{OP(" sc0 nt sc1");}
#define SCALE_ONE(H) asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen" H "\n\ts_waitcnt vmcnt(0)":"=&v"(last):"v"(at),"s"(resource):"memory")
template<uint32_t P>__device__ __forceinline__ uint32_t scale_step(uint32_t at,Vec4 resource,Vec4& last){SCALE_POLICY(SCALE_ONE);return last[0];}
#undef SCALE_ONE

#define SL(O,A,R,H) "buffer_load_dwordx4 %" #O ", %" #A ", %" #R ", 0 offen" H "\n\t"
#define SL4(O0,O1,O2,O3,A0,A1,A2,A3,R,H) SL(O0,A0,R,H) SL(O1,A1,R,H) SL(O2,A2,R,H) SL(O3,A3,R,H)
#define SO(I) "=&v"(values[I]),"=&v"(values[I+1]),"=&v"(values[I+2]),"=&v"(values[I+3])
#define SA(I) "v"(offset[I]),"v"(offset[I+1]),"v"(offset[I+2]),"v"(offset[I+3])
#define SB8(H) asm volatile("; SCALE_BATCH_BEGIN\n\t" \
 SL4(0,1,2,3,8,9,10,11,16,H) SL4(4,5,6,7,12,13,14,15,16,H) \
 "s_waitcnt vmcnt(0)\n\t; SCALE_BATCH_END":SO(0),SO(4):SA(0),SA(4),"s"(resource):"memory")
#define SB16(H) asm volatile("; SCALE_BATCH_BEGIN\n\t" \
 SL4(0,1,2,3,16,17,18,19,32,H) SL4(4,5,6,7,20,21,22,23,32,H) SL4(8,9,10,11,24,25,26,27,32,H) SL4(12,13,14,15,28,29,30,31,32,H) \
 "s_waitcnt vmcnt(0)\n\t; SCALE_BATCH_END":SO(0),SO(4),SO(8),SO(12):SA(0),SA(4),SA(8),SA(12),"s"(resource):"memory")
#define SB32(H) asm volatile("; SCALE_BATCH_BEGIN\n\t" \
 SL4(0,1,2,3,32,33,34,35,64,H) SL4(4,5,6,7,36,37,38,39,64,H) SL4(8,9,10,11,40,41,42,43,64,H) SL4(12,13,14,15,44,45,46,47,64,H) \
 SL4(16,17,18,19,48,49,50,51,64,H) SL4(20,21,22,23,52,53,54,55,64,H) SL4(24,25,26,27,56,57,58,59,64,H) SL4(28,29,30,31,60,61,62,63,64,H) \
 "s_waitcnt vmcnt(0)\n\t; SCALE_BATCH_END":SO(0),SO(4),SO(8),SO(12),SO(16),SO(20),SO(24),SO(28):SA(0),SA(4),SA(8),SA(12),SA(16),SA(20),SA(24),SA(28),"s"(resource):"memory")
template<uint32_t P,uint32_t D,uint32_t FixedThreads=0>
__device__ __forceinline__ uint64_t scale_stream(uint64_t stop,Vec4 resource,uint32_t& index,uint32_t& last,Vec4* values){
    const uint32_t threads=FixedThreads?FixedThreads:blockDim.x;uint64_t chunks=0;
    do{
        #pragma nounroll
        for(uint32_t batch=0;batch<8;batch++){
            uint32_t offset[D];
            #pragma unroll
            for(uint32_t i=0;i<D;i++)offset[i]=16*(index+i*threads+threadIdx.x);
            if constexpr(D==8){SCALE_POLICY(SB8);}else if constexpr(D==16){SCALE_POLICY(SB16);}else{SCALE_POLICY(SB32);}
            last=index;index=(index+D*threads)&(ScaleSlice/16-1);
        }
        ++chunks;
    }while(scale_wall()<stop);
    return chunks;
}
#undef SB32
#undef SB16
#undef SB8
#undef SA
#undef SO
#undef SL4
#undef SL
#undef SCALE_POLICY

struct ScaleArgs {
    uint8_t *probe0,*probe1,*probe2,*probe3,*data;
    uint32_t salt;uint64_t epoch;Sample* samples;Worker* workers;uint32_t* sinks;
};
template<uint32_t P,uint32_t D>
__global__ __launch_bounds__(512) void cu_scale_bw_kernel(ScaleArgs args){
    extern __shared__ uint8_t reserve[];
    const uint32_t tid=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(tid/64),waves=blockDim.x/64;
    Worker w{};w.entry_wall=scale_wall();w.hw0=scale_hw();w.xcc0=scale_xcc();w.block=blockIdx.x;w.wave=wave;w.role=1;w.active_lanes=__popcll(__ballot(1));w.slice=ScaleSlice;
    reserve[tid]=tid;if(tid==0)reserve[ScaleLDS-1]=0xa5;__syncthreads();
    scale_wait(args.epoch);BufferResource resource(args.data+uint64_t(blockIdx.x)*ScaleSlice,ScaleSlice);uint32_t index=0,last=0;Vec4 v[D];
    w.warm0=scale_wall();asm volatile("; SCALE_BW_WARM_BEGIN":::"memory");w.warm_chunks=scale_stream<P,D>(args.epoch+ScaleWarm,resource.words,index,last,v);asm volatile("; SCALE_BW_WARM_END":::"memory");
    w.warm1=scale_wall();w.warm_last_index=last;w.work0=scale_wall();w.core0=scale_core();
    asm volatile("; SCALE_BW_WORK_BEGIN":::"memory");w.work_chunks=scale_stream<P,D>(args.epoch+ScaleWarm+ScaleBW,resource.words,index,last,v);asm volatile("; SCALE_BW_WORK_END":::"memory");
    w.core1=scale_core();w.work1=scale_wall();w.last_index=last;uint32_t sum=0x2468ace0;
    #pragma unroll
    for(uint32_t i=0;i<D;i++){
        #pragma unroll
        for(uint32_t p=0;p<4;p++)sum+=v[i][p];
    }
    w.sum=sum;scale_wait(args.epoch+ScaleWarm+ScaleBW+ScaleGuard);__syncthreads();w.hw1=scale_hw();w.xcc1=scale_xcc();w.export_wall=scale_wall();
    args.sinks[blockIdx.x*blockDim.x+tid]=sum;
    if(tid%64==0)args.workers[blockIdx.x*waves+wave]=w;
}
template<uint32_t P,uint32_t Phase>
__device__ __forceinline__ Sample scale_chase(Vec4 resource,uint32_t& at,uint32_t wave,uint64_t scheduled,uint64_t deadline){
    Sample s{};s.wave=wave;s.phase=Phase;s.scheduled=scheduled;s.deadline=deadline;Record& z=s.chase;Vec4 last{};
    z.hw0=scale_hw();z.xcc0=scale_xcc();z.start=at;z.steps=Phase?Nodes:0;z.active_lanes=__popcll(__ballot(1));
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)":::"memory");z.wall0=scale_wall();z.core0=scale_core();
    if constexpr(Phase){
        if constexpr(Phase==1)asm volatile("; SCALE_CHASE_WARM_BEGIN":::"memory");else asm volatile("; SCALE_CHASE_MEASURE_BEGIN":::"memory");
        #pragma nounroll
        for(uint32_t group=0;group<(Nodes-1)/128;group++){
            #pragma unroll
            for(uint32_t i=0;i<128;i++)at=scale_step<P>(at,resource,last);
        }
        at=scale_step<P>(at,resource,last);
        if constexpr(Phase==1)asm volatile("; SCALE_CHASE_WARM_END":::"memory");else asm volatile("; SCALE_CHASE_MEASURE_END":::"memory");
    }
    z.core1=scale_core();z.wall1=scale_wall();z.hw1=scale_hw();z.xcc1=scale_xcc();z.end=at;
    for(uint32_t p=0;p<4;p++)z.last[p]=last[p];return s;
}
template<uint32_t P>
__global__ __launch_bounds__(256) void cu_scale_latency_kernel(ScaleArgs args){
    extern __shared__ uint8_t reserve[];
    const uint32_t tid=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(tid/64);
    Worker w{};w.entry_wall=scale_wall();w.hw0=scale_hw();w.xcc0=scale_xcc();w.block=blockIdx.x;w.wave=wave;w.role=blockIdx.x?1:0;w.active_lanes=__popcll(__ballot(1));w.slice=ScaleSlice;
    reserve[tid]=tid;if(tid==0)reserve[ScaleLDS-1]=0xa5;__syncthreads();
    const uint64_t warm=args.epoch+ScaleWarm+ScaleGuard,measured=warm+ScaleSlot,deadline=measured+ScaleSlot,stop=deadline+ScaleGuard,export_time=stop+ScaleGuard;
    scale_wait(args.epoch);
    if(blockIdx.x==0){
        if(tid%64==0){
            auto ptr=wave==0?args.probe0:wave==1?args.probe1:wave==2?args.probe2:args.probe3;BufferResource resource(ptr,ProbeBytes);uint32_t at=0;
            scale_wait(warm);auto b=scale_chase<P,0>(resource.words,at,wave,warm,measured);b.export_wall=scale_wall();args.samples[wave*3]=b;
            auto a=scale_chase<P,1>(resource.words,at,wave,warm,measured);a.export_wall=scale_wall();args.samples[wave*3+1]=a;
            scale_wait(measured);auto z=scale_chase<P,2>(resource.words,at,wave,measured,deadline);scale_wait(export_time);z.export_wall=scale_wall();args.samples[wave*3+2]=z;
        }
    }else{
        BufferResource resource(args.data+uint64_t(blockIdx.x-1)*ScaleSlice,ScaleSlice);uint32_t index=0,last=0;Vec4 v[32];
        w.warm0=scale_wall();asm volatile("; SCALE_LAT_BG_WARM_BEGIN":::"memory");w.warm_chunks=scale_stream<P,32,256>(warm-ScaleGuard,resource.words,index,last,v);asm volatile("; SCALE_LAT_BG_WARM_END":::"memory");
        w.warm1=scale_wall();w.warm_last_index=last;w.work0=scale_wall();w.core0=scale_core();
        asm volatile("; SCALE_LAT_BG_WORK_BEGIN":::"memory");w.work_chunks=scale_stream<P,32,256>(stop,resource.words,index,last,v);asm volatile("; SCALE_LAT_BG_WORK_END":::"memory");
        w.core1=scale_core();w.work1=scale_wall();w.last_index=last;uint32_t sum=0x2468ace0;
        #pragma unroll
        for(uint32_t i=0;i<32;i++){
            #pragma unroll
            for(uint32_t p=0;p<4;p++)sum+=v[i][p];
        }
        w.sum=sum;scale_wait(export_time);
    }
    __syncthreads();w.hw1=scale_hw();w.xcc1=scale_xcc();w.export_wall=scale_wall();
    if(blockIdx.x)args.sinks[blockIdx.x*256+tid]=w.sum;
    if(tid%64==0)args.workers[blockIdx.x*4+wave]=w;
}

struct ScaleCase {std::string name;uint32_t cus,policy,bandwidth,waves,depth;};
using ScaleLaunch=void(*)(ScaleArgs,uint32_t,uint32_t,hipStream_t);
struct ScaleKernel {ScaleLaunch launch;hipFuncAttributes attr;int occupancy;};
template<uint32_t P,uint32_t D>void scale_bw_launch(ScaleArgs a,uint32_t blocks,uint32_t waves,hipStream_t stream){cu_scale_bw_kernel<P,D><<<blocks,waves*64,ScaleLDS,stream>>>(a);HIP_OK(hipGetLastError());}
template<uint32_t P>void scale_lat_launch(ScaleArgs a,uint32_t blocks,uint32_t waves,hipStream_t stream){cu_scale_latency_kernel<P><<<blocks,256,ScaleLDS,stream>>>(a);HIP_OK(hipGetLastError());}
template<uint32_t P,uint32_t D>ScaleKernel scale_bw_desc(uint32_t waves){
    ScaleKernel k{};k.launch=scale_bw_launch<P,D>;auto f=reinterpret_cast<const void*>(cu_scale_bw_kernel<P,D>);
    HIP_OK(hipFuncGetAttributes(&k.attr,f));HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.occupancy,f,waves*64,ScaleLDS));return k;
}
template<uint32_t P>ScaleKernel scale_desc(const ScaleCase& c){
    if(c.bandwidth){
        if constexpr(P==0||P==6){if(c.depth==8)return scale_bw_desc<P,8>(c.waves);if(c.depth==16)return scale_bw_desc<P,16>(c.waves);}
        if(c.depth!=32)throw std::runtime_error("depthonly0/6controls");return scale_bw_desc<P,32>(c.waves);
    }
    ScaleKernel k{};k.launch=scale_lat_launch<P>;auto f=reinterpret_cast<const void*>(cu_scale_latency_kernel<P>);
    HIP_OK(hipFuncGetAttributes(&k.attr,f));HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.occupancy,f,256,ScaleLDS));return k;
}
ScaleKernel scale_select(const ScaleCase& c){switch(c.policy){case 0:return scale_desc<0>(c);case 1:return scale_desc<1>(c);case 2:return scale_desc<2>(c);case 3:return scale_desc<3>(c);case 4:return scale_desc<4>(c);case 5:return scale_desc<5>(c);case 6:return scale_desc<6>(c);case 7:return scale_desc<7>(c);}throw std::runtime_error("cache");}
std::vector<ScaleCase> scale_plan(const std::filesystem::path& path){
    std::ifstream f(path);std::string line;std::getline(f,line);std::vector<ScaleCase> result;std::set<std::string> seen;
    while(std::getline(f,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);ScaleCase c;
        if(!(in>>c.name>>c.cus>>c.policy>>c.bandwidth>>c.waves>>c.depth)||c.cus<1||c.cus>256||(c.cus&(c.cus-1))||c.policy>7||c.bandwidth>1||c.waves<1||c.waves>8||(c.waves&(c.waves-1))||(c.depth!=8&&c.depth!=16&&c.depth!=32)||(!c.bandwidth&&(c.waves!=4||c.depth!=32))||!seen.insert(c.name).second||c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("scaleplan");result.push_back(c);}
    if(result.empty())throw std::runtime_error("emptyplan");return result;
}
void scale_run(const std::filesystem::path& out,const std::vector<ScaleCase>& cases,uint32_t seed,const std::string& power_path){
    hipDeviceProp_t dev{};HIP_OK(hipGetDeviceProperties(&dev,0));char bdf[32];HIP_OK(hipDeviceGetPCIBusId(bdf,32,0));int wall;HIP_OK(hipDeviceGetAttribute(&wall,hipDeviceAttributeWallClockRate,0));
    if(std::string(bdf)!="0000:85:00.0"||std::string(dev.gcnArchName).find("gfx950")!=0||dev.multiProcessorCount!=256||dev.warpSize!=64||dev.sharedMemPerMultiprocessor!=163840||wall!=100000)throw std::runtime_error("device");
    uint64_t data_bytes=0;uint32_t max_threads=0;for(const auto& c:cases){data_bytes=std::max(data_bytes,uint64_t(c.cus)*ScaleSlice);max_threads=std::max(max_threads,c.cus*c.waves*64);}
    size_t free,total;HIP_OK(hipMemGetInfo(&free,&total));if(free<data_bytes+(3ull<<30))throw std::runtime_error("VRAM");
    hipStream_t stream;hipEvent_t begin,end;HIP_OK(hipStreamCreate(&stream));HIP_OK(hipEventCreate(&begin));HIP_OK(hipEventCreate(&end));
    std::array<uint8_t*,4> probes{};for(auto& p:probes)HIP_OK(hipMalloc(&p,uint64_t(ProbeBytes)+2*Guard));
    uint8_t* data;Sample* samples;Worker* workers;uint32_t* sinks;uint64_t* epoch;uint4* eviction;uint32_t* evict_sink;uint32_t* next_device;
    HIP_OK(hipMalloc(&data,data_bytes+2*Guard));HIP_OK(hipMalloc(&samples,12*sizeof(Sample)));HIP_OK(hipMalloc(&workers,256*8*sizeof(Worker)));HIP_OK(hipMalloc(&sinks,max_threads*4));HIP_OK(hipMalloc(&epoch,8));HIP_OK(hipMalloc(&eviction,1ull<<30));HIP_OK(hipMalloc(&evict_sink,65536*4));HIP_OK(hipMalloc(&next_device,uint64_t(Nodes)*4));HIP_OK(hipMemsetAsync(eviction,0x5a,1ull<<30,stream));
    for(uint32_t i=0;i<4;i++){if(!disjoint(uint64_t(probes[i]),uint64_t(ProbeBytes)+2*Guard,uint64_t(data),data_bytes+2*Guard))throw std::runtime_error("overlap");for(uint32_t j=0;j<i;j++)if(!disjoint(uint64_t(probes[i]),uint64_t(ProbeBytes)+2*Guard,uint64_t(probes[j]),uint64_t(ProbeBytes)+2*Guard))throw std::runtime_error("waveoverlap");}
    const uint32_t salt=seed^0x58264719u;std::vector<uint32_t> links(Nodes);for(uint32_t i=0;i<Nodes;i++)links[i]=(i+1)%Nodes*128;
    HIP_OK(hipMemcpyAsync(next_device,links.data(),uint64_t(Nodes)*4,hipMemcpyHostToDevice,stream));
    for(uint32_t wave=0;wave<4;wave++){HIP_OK(hipMemsetAsync(probes[wave],0xa5,uint64_t(ProbeBytes)+2*Guard,stream));matched_initialize_probe<<<2048,256,0,stream>>>(probes[wave]+Guard,next_device,wave_salt(salt,wave));HIP_OK(hipGetLastError());}
    HIP_OK(hipMemsetAsync(data,0xa5,data_bytes+2*Guard,stream));matched_initialize_background<<<2048,256,0,stream>>>(reinterpret_cast<uint32_t*>(data+Guard),data_bytes/4,salt);HIP_OK(hipGetLastError());HIP_OK(hipStreamSynchronize(stream));
    std::ofstream info(out/"device.json");info<<"{\"bdf\":\""<<bdf<<"\",\"arch\":\""<<dev.gcnArchName<<"\",\"CUs\":256,\"wave_size\":64,\"LDS\":98304,\"LDS_per_CU\":163840,\"wall_khz\":100000,\"sample_bytes\":112,\"worker_bytes\":128,\"seed\":"<<seed<<",\"node_spacing\":128,\"load_bytes\":16,\"nodes\":"<<Nodes<<",\"probe_bytes\":"<<ProbeBytes<<",\"slice_bytes\":"<<ScaleSlice<<",\"data_bytes\":"<<data_bytes<<",\"data_allocation\":"<<uint64_t(data)<<",\"probe_allocations\":[";
    for(uint32_t wave=0;wave<4;wave++){if(wave)info<<',';info<<uint64_t(probes[wave]);}info<<"],\"warm_ticks\":"<<ScaleWarm<<",\"slot_ticks\":"<<ScaleSlot<<",\"guard_ticks\":"<<ScaleGuard<<",\"BW_ticks\":"<<ScaleBW<<",\"power_path\":\""<<power_path<<"\"}\n";info.close();
    std::ofstream meta(out/"cases.json");meta<<"[\n";size_t ordinal=0;
    for(const auto& c:cases){
        auto kernel=scale_select(c);if(kernel.attr.localSizeBytes||kernel.attr.sharedSizeBytes||kernel.occupancy!=1)throw std::runtime_error("unsupportedresource:"+c.name);
        matched_eviction<<<256,256,0,stream>>>(eviction,evict_sink);HIP_OK(hipGetLastError());HIP_OK(hipMemsetAsync(samples,0xff,12*sizeof(Sample),stream));HIP_OK(hipMemsetAsync(workers,0xff,c.cus*c.waves*sizeof(Worker),stream));HIP_OK(hipMemsetAsync(sinks,0xff,c.cus*c.waves*64*4,stream));HIP_OK(hipStreamSynchronize(stream));
        uint64_t h0=host_ns();matched_epoch<<<1,64,0,stream>>>(epoch);HIP_OK(hipGetLastError());uint64_t raw;HIP_OK(hipMemcpyAsync(&raw,epoch,8,hipMemcpyDeviceToHost,stream));HIP_OK(hipStreamSynchronize(stream));uint64_t h1=host_ns();
        ScaleArgs args{probes[0]+Guard,probes[1]+Guard,probes[2]+Guard,probes[3]+Guard,data+Guard,salt,raw+LeadTicks,samples,workers,sinks};
        PowerRecorder power(power_path);HIP_OK(hipEventRecord(begin,stream));kernel.launch(args,c.cus,c.waves,stream);HIP_OK(hipEventRecord(end,stream));HIP_OK(hipEventSynchronize(end));power.stop();float ms;HIP_OK(hipEventElapsedTime(&ms,begin,end));
        std::vector<Sample> ss(c.bandwidth?0:12);std::vector<Worker> ww(c.cus*c.waves);std::vector<uint32_t> sink(c.cus*c.waves*64);
        if(!ss.empty())HIP_OK(hipMemcpy(ss.data(),samples,ss.size()*sizeof(Sample),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(ww.data(),workers,ww.size()*sizeof(Worker),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(sink.data(),sinks,sink.size()*4,hipMemcpyDeviceToHost));
        binary(out/(c.name+".samples.bin"),ss);binary(out/(c.name+".workers.bin"),ww);binary(out/(c.name+".sinks.bin"),sink);binary(out/(c.name+".power.bin"),power.samples);
        bool topology=true,overlap=true;std::set<uint32_t> seen;uint64_t lo=~0ull,hi=0;
        for(uint32_t wave=0;wave<ss.size()/3;wave++)for(uint32_t phase=0;phase<3;phase++){
            const auto& s=ss[wave*3+phase];const auto& z=s.chase;
            if(s.wave!=wave||s.phase!=phase||z.active_lanes!=1||z.start||z.end||z.steps!=(phase?Nodes:0)||z.wall1<=z.wall0)throw std::runtime_error("chainrecord");
            if(cu(z.hw0,z.xcc0)!=cu(ww[0].hw0,ww[0].xcc0)||cu(z.hw1,z.xcc1)!=cu(ww[0].hw0,ww[0].xcc0))topology=false;
            if(phase){if(z.last[0])throw std::runtime_error("endpointer");for(uint32_t p=1;p<4;p++)if(z.last[p]!=tag(Nodes-1,p,wave_salt(salt,wave)))throw std::runtime_error("payloadtag");}
            if(z.wall0<s.scheduled||z.wall1>=s.deadline)overlap=false;
            if(phase==2){lo=std::min(lo,z.wall0);hi=std::max(hi,z.wall1);}
        }
        if(!ss.empty())for(uint32_t wave=0;wave<4;wave++)if(ss[wave*3].export_wall>=lo||ss[wave*3+1].export_wall>=lo||ss[wave*3+2].export_wall<=hi)overlap=false;
        for(uint32_t block=0;block<c.cus;block++){
            uint32_t k=cu(ww[block*c.waves].hw0,ww[block*c.waves].xcc0);if(!seen.insert(k).second)topology=false;
            for(uint32_t wave=0;wave<c.waves;wave++){
                const auto& w=ww[block*c.waves+wave];bool probe=!c.bandwidth&&block==0;
                if(w.block!=block||w.wave!=wave||w.role!=(probe?0:1)||w.active_lanes!=64)throw std::runtime_error("workerrecord");
                if(cu(w.hw0,w.xcc0)!=k||cu(w.hw1,w.xcc1)!=k)topology=false;
                if(!ss.empty()&&(w.export_wall<=hi||(!probe&&(w.work0>lo||w.work1<hi))))overlap=false;
                if(probe)continue;
                if(!w.warm_chunks||!w.work_chunks||w.work1<=w.work0)throw std::runtime_error("workerwork");
                const uint32_t batch=c.depth*c.waves*64;uint32_t expected_index=uint32_t((w.warm_chunks+w.work_chunks)*8*batch-batch)%(ScaleSlice/16);
                if(w.last_index!=expected_index||w.work_chunks*8*batch*16<ScaleSlice)throw std::runtime_error("wholebuffercoverage");
                uint32_t slice_id=c.bandwidth?block:block-1;
                for(uint32_t lane=0;lane<64;lane++){
                    uint32_t sum=0x2468ace0;
                    for(uint32_t j=0;j<c.depth;j++)for(uint32_t p=0;p<4;p++)sum+=word(uint64_t(slice_id)*ScaleSlice/4+uint64_t(w.last_index+j*c.waves*64+wave*64+lane)*4+p,salt);
                    if(sink[block*c.waves*64+wave*64+lane]!=sum||(lane==0&&w.sum!=sum))throw std::runtime_error("sinkchecksum");
                }
            }
        }
        std::vector<uint8_t> g(Guard);for(auto p:probes)for(uint64_t off:std::array<uint64_t,2>{0,Guard+ProbeBytes}){HIP_OK(hipMemcpy(g.data(),p+off,Guard,hipMemcpyDeviceToHost));if(!std::all_of(g.begin(),g.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("probe guard");}
        for(uint64_t off:std::array<uint64_t,2>{0,Guard+data_bytes}){HIP_OK(hipMemcpy(g.data(),data+off,Guard,hipMemcpyDeviceToHost));if(!std::all_of(g.begin(),g.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("data guard");}
        if(ordinal++)meta<<",\n";meta<<std::setprecision(15)<<"{\"name\":\""<<c.name<<"\",\"cus\":"<<c.cus<<",\"policy\":"<<c.policy<<",\"bandwidth\":"<<c.bandwidth<<",\"waves\":"<<c.waves<<",\"depth\":"<<c.depth<<",\"slice\":"<<ScaleSlice<<",\"salt\":"<<salt<<",\"raw_epoch\":"<<raw<<",\"epoch\":"<<args.epoch<<",\"host_epoch0\":"<<h0<<",\"host_epoch1\":"<<h1<<",\"event_ms\":"<<ms<<",\"VGPR\":"<<kernel.attr.numRegs<<",\"private\":0,\"occupancy_max\":1,\"topology_valid\":"<<topology<<",\"overlap_valid\":"<<overlap<<",\"checks\":true}";meta.flush();
        std::cout<<"CU_SCALING_CASE "<<c.name<<" actual_CUs="<<seen.size()<<" topology="<<topology<<" overlap="<<overlap<<std::endl;
    }
    meta<<"\n]\n";meta.close();for(auto p:probes)HIP_OK(hipFree(p));for(void* p:std::array<void*,8>{data,samples,workers,sinks,epoch,eviction,evict_sink,next_device})HIP_OK(hipFree(p));HIP_OK(hipEventDestroy(begin));HIP_OK(hipEventDestroy(end));HIP_OK(hipStreamDestroy(stream));
}
int main(int argc,char** argv){try{
    if(argc!=5)throw std::runtime_error("read_cu_scaling NEW_OUT PLAN.csv SEED POWER_SYSFS");
    if(!std::getenv("HIP_VISIBLE_DEVICES")||std::string(std::getenv("HIP_VISIBLE_DEVICES"))!="5"||std::getenv("HSA_CU_MASK")||std::getenv("ROC_GLOBAL_CU_MASK"))throw std::runtime_error("environment");
    const auto cases=scale_plan(argv[2]);std::ifstream power(argv[4]);uint64_t v;if(!(power>>v)||!v)throw std::runtime_error("power");
    std::filesystem::path out=argv[1];if(std::filesystem::exists(out))throw std::runtime_error("existing output");HIP_OK(hipSetDevice(0));std::filesystem::create_directories(out);scale_run(out,cases,std::stoul(argv[3]),argv[4]);return 0;
}catch(const std::exception& e){std::cerr<<"CU_SCALING_FAILED "<<e.what()<<std::endl;return 1;}}
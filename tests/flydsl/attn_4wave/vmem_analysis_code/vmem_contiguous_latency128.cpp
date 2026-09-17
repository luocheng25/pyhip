// SPDX-License-Identifier: MIT
// Full-span dependent loads: four waves, lane0 only, four distinct buffers.
#define main preserved_pointer_chase_main
#include "vmem_pointer_chase.cpp"
#undef main
#include <atomic>
#include <chrono>
#include <thread>
#include <sys/timerfd.h>
#include <unistd.h>

constexpr uint32_t Threads128=256, Waves128=4, LDS128=96*1024;
constexpr uint32_t Span128=256u<<20, Stride128=128, Nodes128=Span128/Stride128+1;
constexpr uint32_t Bytes128=Nodes128*Stride128, BackgroundSlice128=64u<<20;
constexpr uint64_t Warm128=20000000, Slot128=250000000, Guard128=200000, Lead128=500000;
constexpr uint64_t BackgroundBytes128=uint64_t(255)*BackgroundSlice128;
using U8_128=uint32_t __attribute__((ext_vector_type(8)));
using F4_128=float __attribute__((ext_vector_type(4)));
struct Sample128 {Record chase;uint64_t scheduled,deadline,export_wall;uint32_t wave,phase;};
static_assert(sizeof(Sample128)==112);
struct Worker128 {
    uint64_t entry_wall,warm0,warm1,work0,work1,core0,core1,warm_chunks,work_chunks,export_wall;
    uint32_t hw0,hw1,xcc0,xcc1,block,wave,role,active_lanes,last_index,warm_last_index,sum,slice;
};
static_assert(sizeof(Worker128)==128);
struct Args128 {
    uint8_t *probe0,*probe1,*probe2,*probe3,*background;
    uint32_t scope,salt;uint64_t epoch;Sample128* samples;Worker128* workers;uint32_t* sinks;
};
__host__ __device__ uint32_t salt128(uint32_t salt,uint32_t wave){return salt^(wave*0x13579bdu);}
__host__ __device__ uint32_t word128(uint64_t i,uint32_t salt){return uint32_t(i)*0x9e3779b9u+salt;}
__global__ void initialize_chain128(uint8_t* data,uint32_t salt){
    for(uint32_t node=blockIdx.x*blockDim.x+threadIdx.x;node<Nodes128;node+=blockDim.x*gridDim.x){
        const uint32_t next=node+1==Nodes128?0:(node+1)*Stride128;
        *reinterpret_cast<uint4*>(data+uint64_t(node)*Stride128)=make_uint4(next,tag_word(node,1,salt),tag_word(node,2,salt),tag_word(node,3,salt));
    }
}
__global__ void initialize_background128(uint32_t* p,uint64_t words,uint32_t salt){
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=uint64_t(blockDim.x)*gridDim.x)p[i]=word128(i,salt);
}
__global__ void epoch128(uint64_t* p){if(threadIdx.x==0)*p=wall_clock();}
__device__ __forceinline__ void wait128(uint64_t until){while(wall_clock()<until)asm volatile("s_nop 7":::"memory");}

#define L128(O,A,H) "buffer_load_dwordx4 %" #O ", %" #A ", %64, 0 offen" H "\n\t"
#define L128_4(O0,O1,O2,O3,A0,A1,A2,A3,H) L128(O0,A0,H) L128(O1,A1,H) L128(O2,A2,H) L128(O3,A3,H)
#define O128(I) "=&v"(values[I]),"=&v"(values[I+1]),"=&v"(values[I+2]),"=&v"(values[I+3])
#define A128(I) "v"(offset[I]),"v"(offset[I+1]),"v"(offset[I+2]),"v"(offset[I+3])
#define B128(H) asm volatile("; BG128_BATCH_BEGIN\n\t" \
 L128_4(0,1,2,3,32,33,34,35,H) L128_4(4,5,6,7,36,37,38,39,H) L128_4(8,9,10,11,40,41,42,43,H) L128_4(12,13,14,15,44,45,46,47,H) \
 L128_4(16,17,18,19,48,49,50,51,H) L128_4(20,21,22,23,52,53,54,55,H) L128_4(24,25,26,27,56,57,58,59,H) L128_4(28,29,30,31,60,61,62,63,H) \
 "s_waitcnt vmcnt(0)\n\t; BG128_BATCH_END":O128(0),O128(4),O128(8),O128(12),O128(16),O128(20),O128(24),O128(28) \
 :A128(0),A128(4),A128(8),A128(12),A128(16),A128(20),A128(24),A128(28),"s"(resource):"memory")
template<uint32_t P>__device__ __forceinline__ uint64_t bulk128(uint64_t stop,U4 resource,uint32_t& index,uint32_t& last_index,U4* values){
    uint64_t chunks=0;
    do{
        #pragma nounroll
        for(uint32_t batch=0;batch<8;batch++){
            uint32_t offset[32];
            #pragma unroll
            for(uint32_t i=0;i<32;i++)offset[i]=16*(index+i*256+threadIdx.x);
            if constexpr(P==0){B128("");}else if constexpr(P==1){B128(" sc0");}else if constexpr(P==2){B128(" nt");}else if constexpr(P==3){B128(" sc1");}
            else if constexpr(P==4){B128(" sc0 nt");}else if constexpr(P==5){B128(" sc0 sc1");}else if constexpr(P==6){B128(" nt sc1");}else{B128(" sc0 nt sc1");}
            last_index=index;index=(index+8192)&(BackgroundSlice128/16-1);
        }
        ++chunks;
    }while(wall_clock()<stop);
    return chunks;
}
#undef B128
#undef A128
#undef O128
#undef L128_4
#undef L128
#define M128_4 \
 "v_mfma_f32_16x16x128_f8f6f4 %0, %4, %4, 0\n\t" "v_mfma_f32_16x16x128_f8f6f4 %1, %4, %4, 0\n\t" \
 "v_mfma_f32_16x16x128_f8f6f4 %2, %4, %4, 0\n\t" "v_mfma_f32_16x16x128_f8f6f4 %3, %4, %4, 0\n\t"
#define M128_16 M128_4 M128_4 M128_4 M128_4
#define M128_64 M128_16 M128_16 M128_16 M128_16
__device__ __forceinline__ uint64_t mfma128(uint64_t stop,F4_128& a,F4_128& b,F4_128& c,F4_128& d){
    U8_128 x={0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u};uint64_t n=0;
    do{asm volatile(M128_64 M128_64:"=&v"(a),"=&v"(b),"=&v"(c),"=&v"(d):"v"(x));++n;}while(wall_clock()<stop);return n;
}
#undef M128_64
#undef M128_16
#undef M128_4

template<uint32_t P,uint32_t Phase>
__device__ __forceinline__ Sample128 chase128(U4 resource,uint32_t& at,uint32_t wave,uint64_t scheduled,uint64_t deadline){
    Sample128 s{};s.wave=wave;s.phase=Phase;s.scheduled=scheduled;s.deadline=deadline;
    Record& z=s.chase;U4 last{};z.hw0=hwid();z.xcc0=xccid();z.start=at;z.steps=Phase?Nodes128:0;z.active_lanes=__popcll(__ballot(1));
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)":::"memory");z.wall0=wall_clock();z.core0=core_clock();
    if constexpr(Phase){
        if constexpr(Phase==1)asm volatile("; CHASE128_WARM_BEGIN":::"memory");else asm volatile("; CHASE128_MEASURE_BEGIN":::"memory");
        #pragma nounroll
        for(uint32_t group=0;group<(Nodes128-1)/128;group++){
            #pragma unroll
            for(uint32_t i=0;i<128;i++)at=step<Vgpr,16,P>(at,resource,0,last);
        }
        at=step<Vgpr,16,P>(at,resource,0,last);
        if constexpr(Phase==1)asm volatile("; CHASE128_WARM_END":::"memory");else asm volatile("; CHASE128_MEASURE_END":::"memory");
    }
    z.core1=core_clock();z.wall1=wall_clock();z.hw1=hwid();z.xcc1=xccid();z.end=at;
    for(uint32_t p=0;p<4;p++)z.last[p]=last[p];return s;
}

template<uint32_t P>__global__ __launch_bounds__(Threads128) void contiguous_latency128_kernel(Args128 args){
    extern __shared__ __attribute__((aligned(16))) uint8_t reserve[];
    const uint32_t tid=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(tid/64);
    Worker128 wr{};wr.entry_wall=wall_clock();wr.hw0=hwid();wr.xcc0=xccid();wr.block=blockIdx.x;wr.wave=wave;wr.role=blockIdx.x?args.scope:0;wr.active_lanes=__popcll(__ballot(1));wr.slice=BackgroundSlice128;
    reserve[tid]=uint8_t(tid);if(tid==0)reserve[LDS128-1]=0xa5;__syncthreads();
    const uint64_t warm_start=args.epoch+Warm128+Guard128,measure_start=warm_start+Slot128;
    const uint64_t deadline=measure_start+Slot128,stop=deadline+Guard128,export_time=stop+Guard128;
    wait128(args.epoch);
    if(blockIdx.x==0){
        // Four independent wave leaders. Other lanes do not issue loads.
        if(tid%64==0){
            uint8_t* ptr=wave==0?args.probe0:wave==1?args.probe1:wave==2?args.probe2:args.probe3;
            Resource resource(ptr,Bytes128);uint32_t at=0;
            wait128(warm_start);
            auto blank=chase128<P,0>(resource.words,at,wave,warm_start,measure_start);
            blank.export_wall=wall_clock();args.samples[wave*3]=blank;
            auto warm=chase128<P,1>(resource.words,at,wave,warm_start,measure_start);
            warm.export_wall=wall_clock();args.samples[wave*3+1]=warm;
            wait128(measure_start);
            auto measured=chase128<P,2>(resource.words,at,wave,measure_start,deadline);
            // Never let a finished wave's record write overlap another leader's
            // measured chain. Overruns are rejected, not silently shortened.
            wait128(export_time);measured.export_wall=wall_clock();args.samples[wave*3+2]=measured;
        }
    }else if(args.scope==1){
        F4_128 a{},b{},c{},d{};wr.warm0=wall_clock();
        asm volatile("; MFMA128_WARM_BEGIN":::"memory");wr.warm_chunks=mfma128(warm_start-Guard128,a,b,c,d);asm volatile("; MFMA128_WARM_END":::"memory");
        wr.warm1=wall_clock();wr.work0=wall_clock();wr.core0=core_clock();
        asm volatile("; MFMA128_WORK_BEGIN":::"memory");wr.work_chunks=mfma128(stop,a,b,c,d);asm volatile("; MFMA128_WORK_END":::"memory");
        wr.core1=core_clock();wr.work1=wall_clock();wr.sum=uint32_t(a[0]+a[1]+a[2]+a[3]+b[0]+b[1]+b[2]+b[3]+c[0]+c[1]+c[2]+c[3]+d[0]+d[1]+d[2]+d[3]);
        wait128(export_time);
    }else{
        Resource resource(args.background+uint64_t(blockIdx.x-1)*BackgroundSlice128,BackgroundSlice128);uint32_t index=0,last=0;U4 values[32];
        wr.warm0=wall_clock();asm volatile("; BULK128_WARM_BEGIN":::"memory");wr.warm_chunks=bulk128<P>(warm_start-Guard128,resource.words,index,last,values);asm volatile("; BULK128_WARM_END":::"memory");
        wr.warm1=wall_clock();wr.warm_last_index=last;wr.work0=wall_clock();wr.core0=core_clock();
        asm volatile("; BULK128_WORK_BEGIN":::"memory");wr.work_chunks=bulk128<P>(stop,resource.words,index,last,values);asm volatile("; BULK128_WORK_END":::"memory");
        wr.core1=core_clock();wr.work1=wall_clock();wr.last_index=last;uint32_t sum=0x2468ace0u;
        #pragma unroll
        for(uint32_t i=0;i<32;i++){
            #pragma unroll
            for(uint32_t p=0;p<4;p++)sum+=values[i][p];
        }
        wr.sum=sum;wait128(export_time);
    }
    __syncthreads();wr.hw1=hwid();wr.xcc1=xccid();wr.export_wall=wall_clock();
    if(blockIdx.x)args.sinks[blockIdx.x*256+tid]=wr.sum;
    if(tid%64==0)args.workers[blockIdx.x*4+wave]=wr;
}
using Launch128=void(*)(Args128,uint32_t,hipStream_t);
struct Kernel128{Launch128 launch;hipFuncAttributes attr;int occupancy;};
template<uint32_t P>void launch128(Args128 a,uint32_t blocks,hipStream_t stream){contiguous_latency128_kernel<P><<<blocks,256,LDS128,stream>>>(a);HIP_OK(hipGetLastError());}
template<uint32_t P>Kernel128 describe128(){
    Kernel128 k{};k.launch=launch128<P>;const void* f=reinterpret_cast<const void*>(contiguous_latency128_kernel<P>);
    HIP_OK(hipFuncGetAttributes(&k.attr,f));HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.occupancy,f,256,LDS128));
    if(k.attr.localSizeBytes||k.attr.sharedSizeBytes||k.occupancy!=1)throw std::runtime_error("scratch or1CTA/CU occupancy");return k;
}
Kernel128 select128(uint32_t p){switch(p){case 0:return describe128<0>();case 1:return describe128<1>();case 2:return describe128<2>();case 3:return describe128<3>();case 4:return describe128<4>();case 5:return describe128<5>();case 6:return describe128<6>();case 7:return describe128<7>();}throw std::runtime_error("policy");}
uint64_t host128(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Power128{uint64_t begin_ns,end_ns,microwatts;};
struct Recorder128{
    std::atomic<bool> done{false};std::vector<Power128> records;std::thread worker;
    explicit Recorder128(std::string path):worker([this,path]{
        int fd=timerfd_create(CLOCK_MONOTONIC,TFD_CLOEXEC);if(fd<0)return;
        itimerspec timer{};timer.it_value.tv_nsec=1000000;timer.it_interval.tv_nsec=20000000;
        if(timerfd_settime(fd,0,&timer,nullptr)){close(fd);return;}
        while(!done.load()){uint64_t ticks;if(read(fd,&ticks,8)!=8)break;Power128 p{};p.begin_ns=host128();std::ifstream f(path);if(f>>p.microwatts){p.end_ns=host128();records.push_back(p);}}
        close(fd);
    }){}
    void stop(){done.store(true);if(worker.joinable())worker.join();}
    ~Recorder128(){stop();}
};
struct Case128{std::string name;uint32_t policy,scope;};
std::vector<Case128> plan128(const std::filesystem::path& p){
    std::ifstream file(p);std::string line;std::getline(file,line);std::vector<Case128> result;std::set<std::string> names;
    while(std::getline(file,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);Case128 c;
        if(!(in>>c.name>>c.policy>>c.scope)||c.policy>7||c.scope>2||!names.insert(c.name).second||c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("plan");result.push_back(c);}
    if(result.empty())throw std::runtime_error("empty plan");return result;
}
bool disjoint128(uint64_t a,uint64_t an,uint64_t b,uint64_t bn){return a+an<=b||b+bn<=a;}
void run128(const std::filesystem::path& out,const std::vector<Case128>& configs,uint32_t seed,const std::string& power_path){
    hipDeviceProp_t dev{};HIP_OK(hipGetDeviceProperties(&dev,0));char bdf[32];HIP_OK(hipDeviceGetPCIBusId(bdf,32,0));int wall;HIP_OK(hipDeviceGetAttribute(&wall,hipDeviceAttributeWallClockRate,0));
    if(std::string(bdf)!="0000:85:00.0"||std::string(dev.gcnArchName).find("gfx950")!=0||dev.multiProcessorCount!=256||wall!=100000||dev.sharedMemPerMultiprocessor!=163840||dev.warpSize!=64)throw std::runtime_error("device");
    hipStream_t stream;hipEvent_t begin,end;HIP_OK(hipStreamCreate(&stream));HIP_OK(hipEventCreate(&begin));HIP_OK(hipEventCreate(&end));
    std::array<uint8_t*,4> probes{};for(auto& p:probes)HIP_OK(hipMalloc(&p,uint64_t(Bytes128)+2*Guard));
    uint8_t* background;Sample128* samples;Worker128* workers;uint32_t* sinks;uint64_t* epoch_device;uint4* eviction;uint32_t* evict_sink;
    HIP_OK(hipMalloc(&background,BackgroundBytes128+2*Guard));HIP_OK(hipMalloc(&samples,12*sizeof(Sample128)));HIP_OK(hipMalloc(&workers,1024*sizeof(Worker128)));HIP_OK(hipMalloc(&sinks,65536*4));HIP_OK(hipMalloc(&epoch_device,8));HIP_OK(hipMalloc(&eviction,EvictionBytes));HIP_OK(hipMalloc(&evict_sink,65536*4));HIP_OK(hipMemsetAsync(eviction,0x5a,EvictionBytes,stream));
    for(uint32_t i=0;i<4;i++){
        if(!disjoint128(uint64_t(probes[i]),uint64_t(Bytes128)+2*Guard,uint64_t(background),BackgroundBytes128+2*Guard))throw std::runtime_error("background overlap");
        for(uint32_t j=0;j<i;j++)if(!disjoint128(uint64_t(probes[i]),uint64_t(Bytes128)+2*Guard,uint64_t(probes[j]),uint64_t(Bytes128)+2*Guard))throw std::runtime_error("wave overlap");
    }
    std::ofstream info(out/"device.json");info<<"{\"bdf\":\""<<bdf<<"\",\"arch\":\""<<dev.gcnArchName<<"\",\"CUs\":256,\"threads\":256,\"wave_size\":64,\"LDS\":98304,\"LDS_per_CU\":163840,\"wall_khz\":100000,\"sample_bytes\":112,\"worker_bytes\":128,\"seed\":"<<seed<<",\"stride\":128,\"load_bytes\":16,\"nodes\":"<<Nodes128<<",\"probe_bytes_per_wave\":"<<Bytes128<<",\"address_span_per_sweep\":"<<Span128+16<<",\"warm_ticks\":"<<Warm128<<",\"slot_ticks\":"<<Slot128<<",\"guard_ticks\":"<<Guard128<<",\"background_bytes\":"<<BackgroundBytes128<<",\"background_allocation\":"<<uint64_t(background)<<",\"probe_allocations\":[";
    for(uint32_t i=0;i<4;i++){if(i)info<<',';info<<uint64_t(probes[i]);}info<<"],\"power_path\":\""<<power_path<<"\"}\n";info.close();
    std::ofstream meta(out/"cases.json");meta<<"[\n";uint32_t ordinal=0;
    for(const auto& c:configs){
        auto kernel=select128(c.policy);uint32_t blocks=c.scope?256:1,salt=seed^0x58264719u;
        for(uint32_t wave=0;wave<4;wave++){HIP_OK(hipMemsetAsync(probes[wave],0xa5,uint64_t(Bytes128)+2*Guard,stream));initialize_chain128<<<2048,256,0,stream>>>(probes[wave]+Guard,salt128(salt,wave));HIP_OK(hipGetLastError());}
        HIP_OK(hipMemsetAsync(background,0xa5,BackgroundBytes128+2*Guard,stream));
        if(c.scope==2){initialize_background128<<<2048,256,0,stream>>>(reinterpret_cast<uint32_t*>(background+Guard),BackgroundBytes128/4,salt);HIP_OK(hipGetLastError());}
        chase_eviction<<<256,256,0,stream>>>(eviction,EvictionBytes/16,evict_sink);HIP_OK(hipGetLastError());
        HIP_OK(hipMemsetAsync(samples,0xff,12*sizeof(Sample128),stream));HIP_OK(hipMemsetAsync(workers,0xff,blocks*4*sizeof(Worker128),stream));HIP_OK(hipMemsetAsync(sinks,0xff,blocks*256*4,stream));HIP_OK(hipStreamSynchronize(stream));
        const uint64_t host_epoch0=host128();epoch128<<<1,64,0,stream>>>(epoch_device);HIP_OK(hipGetLastError());uint64_t raw_epoch;HIP_OK(hipMemcpyAsync(&raw_epoch,epoch_device,8,hipMemcpyDeviceToHost,stream));HIP_OK(hipStreamSynchronize(stream));const uint64_t host_epoch1=host128(),epoch=raw_epoch+Lead128;
        Args128 args{probes[0]+Guard,probes[1]+Guard,probes[2]+Guard,probes[3]+Guard,background+Guard,c.scope,salt,epoch,samples,workers,sinks};
        Recorder128 power(power_path);HIP_OK(hipEventRecord(begin,stream));kernel.launch(args,blocks,stream);HIP_OK(hipEventRecord(end,stream));HIP_OK(hipEventSynchronize(end));power.stop();float ms;HIP_OK(hipEventElapsedTime(&ms,begin,end));
        std::vector<Sample128> sp(12);std::vector<Worker128> wr(blocks*4);std::vector<uint32_t> sums(blocks*256);
        HIP_OK(hipMemcpy(sp.data(),samples,sp.size()*sizeof(Sample128),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(wr.data(),workers,wr.size()*sizeof(Worker128),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(sums.data(),sinks,sums.size()*4,hipMemcpyDeviceToHost));
        binary(out/(c.name+".samples.bin"),sp);binary(out/(c.name+".workers.bin"),wr);binary(out/(c.name+".sinks.bin"),sums);binary(out/(c.name+".power.bin"),power.records);
        bool topology=true,overlap=true;std::set<uint32_t> seen;uint64_t first=~0ull,last=0;
        const uint32_t target=cu(wr[0].hw0,wr[0].xcc0);
        for(uint32_t wave=0;wave<4;wave++)for(uint32_t phase=0;phase<3;phase++){
            const auto& s=sp[wave*3+phase];const auto& z=s.chase;
            if(s.wave!=wave||s.phase!=phase||z.steps!=(phase?Nodes128:0)||z.active_lanes!=1||z.start!=0||z.end!=0||z.wall1<=z.wall0)throw std::runtime_error("full-span chain record");
            if(cu(z.hw0,z.xcc0)!=target||cu(z.hw1,z.xcc1)!=target)topology=false;else if(z.core1<=z.core0)throw std::runtime_error("core clock");
            if(phase){if(z.last[0]!=0)throw std::runtime_error("endpoint");for(uint32_t p=1;p<4;p++)if(z.last[p]!=tag_word(Nodes128-1,p,salt128(salt,wave)))throw std::runtime_error("per-wave payload tags");}
            if(z.wall0<s.scheduled||z.wall1>=s.deadline)overlap=false;
            if(phase==2){first=std::min(first,z.wall0);last=std::max(last,z.wall1);}
        }
        for(uint32_t wave=0;wave<4;wave++){
            if(sp[wave*3].export_wall>=first||sp[wave*3+1].export_wall>=first||sp[wave*3+2].export_wall<=last)overlap=false;
        }
        for(uint32_t block=0;block<blocks;block++){
            uint32_t key=cu(wr[block*4].hw0,wr[block*4].xcc0);if(!seen.insert(key).second)topology=false;
            for(uint32_t wave=0;wave<4;wave++){
                const auto& w=wr[block*4+wave];if(w.block!=block||w.wave!=wave||w.role!=(block?c.scope:0)||w.active_lanes!=64)throw std::runtime_error("worker record");
                if(cu(w.hw0,w.xcc0)!=key||cu(w.hw1,w.xcc1)!=key)topology=false;
                if(w.export_wall<=last)overlap=false;
                if(block){
                    if(!w.warm_chunks||!w.work_chunks||w.work1<=w.work0)throw std::runtime_error("worker count/time");
                    if(w.work0>first||w.work1<last)overlap=false;
                    if(c.scope==2&&w.last_index!=uint32_t((w.warm_chunks+w.work_chunks)*65536-8192)%(BackgroundSlice128/16))throw std::runtime_error("bulk offset");
                    for(uint32_t lane=0;lane<64;lane++){
                        uint32_t expected=32;
                        if(c.scope==2){expected=0x2468ace0u;for(uint32_t slot=0;slot<32;slot++)for(uint32_t p=0;p<4;p++)expected+=word128(uint64_t(block-1)*BackgroundSlice128/4+uint64_t(w.last_index+slot*256+wave*64+lane)*4+p,salt);}
                        if(sums[block*256+wave*64+lane]!=expected||(lane==0&&w.sum!=expected))throw std::runtime_error("background checksum");
                    }
                }
            }
        }
        std::vector<uint8_t> guard(Guard);for(auto p:probes)for(uint64_t offset:std::array<uint64_t,2>{0,Guard+Bytes128}){HIP_OK(hipMemcpy(guard.data(),p+offset,Guard,hipMemcpyDeviceToHost));if(!std::all_of(guard.begin(),guard.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("probe guard");}
        for(uint64_t offset:std::array<uint64_t,2>{0,Guard+BackgroundBytes128}){HIP_OK(hipMemcpy(guard.data(),background+offset,Guard,hipMemcpyDeviceToHost));if(!std::all_of(guard.begin(),guard.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("background guard");}
        if(ordinal++)meta<<",\n";
        meta<<std::setprecision(15)<<"{\"name\":\""<<c.name<<"\",\"policy\":"<<c.policy<<",\"scope\":"<<c.scope<<",\"blocks\":"<<blocks<<",\"salt\":"<<salt<<",\"raw_epoch\":"<<raw_epoch<<",\"epoch\":"<<epoch<<",\"host_epoch0\":"<<host_epoch0<<",\"host_epoch1\":"<<host_epoch1<<",\"event_ms\":"<<ms<<",\"VGPR\":"<<kernel.attr.numRegs<<",\"private\":0,\"occupancy_max\":1,\"topology_valid\":"<<topology<<",\"overlap_valid\":"<<overlap<<",\"checks\":true}";meta.flush();
        std::cout<<"CONTIGUOUS128_CASE "<<c.name<<" topology="<<topology<<" overlap="<<overlap<<" four_buffers_span="<<Span128+16<<" power_records="<<power.records.size()<<std::endl;
    }
    meta<<"\n]\n";meta.close();for(auto p:probes)HIP_OK(hipFree(p));for(void* p:std::array<void*,7>{background,samples,workers,sinks,epoch_device,eviction,evict_sink})HIP_OK(hipFree(p));HIP_OK(hipEventDestroy(begin));HIP_OK(hipEventDestroy(end));HIP_OK(hipStreamDestroy(stream));
}
int main(int argc,char** argv){try{
    if(argc!=5)throw std::runtime_error("contiguous128 NEW_OUT PLAN.csv SEED POWER_SYSFS");
    if(!std::getenv("HIP_VISIBLE_DEVICES")||std::string(std::getenv("HIP_VISIBLE_DEVICES"))!="5"||std::getenv("HSA_CU_MASK")||std::getenv("ROC_GLOBAL_CU_MASK"))throw std::runtime_error("environment");
    auto configs=plan128(argv[2]);std::ifstream power(argv[4]);uint64_t watts;if(!(power>>watts)||!watts)throw std::runtime_error("power sensor");
    std::filesystem::path out(argv[1]);if(std::filesystem::exists(out))throw std::runtime_error("existing output");HIP_OK(hipSetDevice(0));std::filesystem::create_directories(out);run128(out,configs,std::stoul(argv[3]),argv[4]);return 0;
}catch(const std::exception& e){std::cerr<<"CONTIGUOUS128_FAILED "<<e.what()<<std::endl;return 1;}}
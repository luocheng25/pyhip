// SPDX-License-Identifier: MIT
// Fixed-width sequential/random hardware characterization. Old probes remain
// immutable; only their resource descriptor, clocks and dependent-load asm are reused.
#define main preserved_pointer_chase_main
#include "vmem_pointer_chase.cpp"
#undef main
#include <atomic>
#include <chrono>
#include <thread>
#include <sys/timerfd.h>
#include <unistd.h>

constexpr uint32_t TableThreads=256, TableLDS=96*1024, TableDepth=32, BatchVectors=TableThreads*TableDepth;
constexpr uint32_t WorkBatches=8, MaxSamples=32;
constexpr uint64_t TableWarm=20000000, TableSlot=2000000, TableGuard=200000, TableLead=500000, BandwidthTicks=30000000;
constexpr uint32_t ProbeBytes=1u<<30, FullSlice=64u<<20;
using T8=uint32_t __attribute__((ext_vector_type(8)));
using TF4=float __attribute__((ext_vector_type(4)));
struct TableWorker {
    uint64_t entry_wall,warm0,warm1,work0,work1,core0,core1,warm_chunks,work_chunks,store_wall;
    uint32_t hw0,hw1,xcc0,xcc1,block,wave,role,active_lanes,last_index,warm_last_index,sum,slice;
};
static_assert(sizeof(TableWorker)==128);
struct TableSample {Record chase;uint64_t deadline,ready;uint32_t iteration,late;};
static_assert(sizeof(TableSample)==104);
struct TableCase {std::string name;uint32_t random,store,policy,scope,bandwidth,blank;};
struct TableArgs {
    uint8_t *probe,*data;uint32_t slice,random,scope,bandwidth,blank,chains,salt;
    uint64_t epoch;TableSample* samples;TableWorker* workers;uint32_t* sinks;
};
__host__ __device__ uint32_t permute_vector(uint32_t x,uint32_t mask,uint32_t salt){
    // Each xor-shift, odd multiplication and xor is invertible on this2^n domain.
    // Thus every16B vector occurs once per cycle; no index-buffer traffic.
    x=(x^(salt&mask))&mask;
    x=((x^(x>>13))*0x7feb352du)&mask;
    x=((x^(x>>9))*0x846ca68bu)&mask;
    return (x^(x>>16))&mask;
}
__host__ __device__ uint32_t vector_at(uint32_t rank,uint32_t mask,uint32_t salt,uint32_t random){
    return random?permute_vector(rank&mask,mask,salt):(rank&mask);
}
__host__ __device__ uint32_t table_word(uint64_t i,uint32_t salt){return uint32_t(i)*0x9e3779b9u+salt;}
__host__ __device__ uint32_t store_value(uint32_t part,uint32_t salt){return salt+part*0x13579bdu;}
__device__ __forceinline__ void table_wait(uint64_t until){while(wall_clock()<until)asm volatile("s_nop 7":::"memory");}
__global__ void table_epoch(uint64_t* t){if(threadIdx.x==0)*t=wall_clock();}
__global__ void initialize_table_data(uint32_t* p,uint64_t n,uint32_t salt){
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(blockDim.x)*gridDim.x)p[i]=table_word(i,salt);
}
__global__ void initialize_table_chain(uint4* p,uint32_t vectors,uint32_t salt,uint32_t random){
    for(uint32_t r=blockIdx.x*blockDim.x+threadIdx.x;r<vectors;r+=blockDim.x*gridDim.x){
        uint32_t at=vector_at(r,vectors-1,salt,random),next=vector_at(r+1,vectors-1,salt,random);
        p[at]=make_uint4(next*16,tag_word(at,1,salt),tag_word(at,2,salt),tag_word(at,3,salt));
    }
}
__global__ void verify_table_store(const uint32_t* p,uint64_t words,uint32_t salt,unsigned long long* bad){
    unsigned long long n=0;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=uint64_t(blockDim.x)*gridDim.x)
        n+=p[i]!=store_value(i%4,salt);
    if(n)atomicAdd(bad,n);
}
__global__ void verify_table_probe(const uint4* p,uint32_t hops,uint32_t salt,uint32_t random,unsigned long long* bad){
    unsigned long long n=0;
    for(uint32_t r=blockIdx.x*blockDim.x+threadIdx.x;r<hops;r+=blockDim.x*gridDim.x){
        const auto x=p[vector_at(r,ProbeBytes/16-1,salt,random)];
        n+=(x.x!=store_value(0,salt))+(x.y!=store_value(1,salt))+(x.z!=store_value(2,salt))+(x.w!=store_value(3,salt));
    }
    if(n)atomicAdd(bad,n);
}

#define TABLE_POLICY(OP) \
 if constexpr(P==0){OP("");}else if constexpr(P==1){OP(" sc0");}else if constexpr(P==2){OP(" nt");}else if constexpr(P==3){OP(" sc1");} \
 else if constexpr(P==4){OP(" sc0 nt");}else if constexpr(P==5){OP(" sc0 sc1");}else if constexpr(P==6){OP(" nt sc1");}else{OP(" sc0 nt sc1");}
#define TL(O,A,H) "buffer_load_dwordx4 %" #O ", %" #A ", %64, 0 offen" H "\n\t"
#define TL4(O0,O1,O2,O3,A0,A1,A2,A3,H) TL(O0,A0,H) TL(O1,A1,H) TL(O2,A2,H) TL(O3,A3,H)
#define TOUT(I) "=&v"(v[I]),"=&v"(v[I+1]),"=&v"(v[I+2]),"=&v"(v[I+3])
#define TADDR(I) "v"(off[I]),"v"(off[I+1]),"v"(off[I+2]),"v"(off[I+3])
#define TBLOAD(H) asm volatile("; TABLE_LOAD_BATCH_BEGIN\n\t" \
 TL4(0,1,2,3,32,33,34,35,H) TL4(4,5,6,7,36,37,38,39,H) TL4(8,9,10,11,40,41,42,43,H) TL4(12,13,14,15,44,45,46,47,H) \
 TL4(16,17,18,19,48,49,50,51,H) TL4(20,21,22,23,52,53,54,55,H) TL4(24,25,26,27,56,57,58,59,H) TL4(28,29,30,31,60,61,62,63,H) \
 "s_waitcnt vmcnt(0)\n\t; TABLE_LOAD_BATCH_END":TOUT(0),TOUT(4),TOUT(8),TOUT(12),TOUT(16),TOUT(20),TOUT(24),TOUT(28) \
 :TADDR(0),TADDR(4),TADDR(8),TADDR(12),TADDR(16),TADDR(20),TADDR(24),TADDR(28),"s"(rs):"memory")
#define TS(A,H) "buffer_store_dwordx4 %0, %" #A ", %33, 0 offen" H "\n\t"
#define TS4(A,B,C,D,H) TS(A,H) TS(B,H) TS(C,H) TS(D,H)
#define TBSTORE(H) asm volatile("; TABLE_STORE_BATCH_BEGIN\n\t" \
 TS4(1,2,3,4,H) TS4(5,6,7,8,H) TS4(9,10,11,12,H) TS4(13,14,15,16,H) TS4(17,18,19,20,H) TS4(21,22,23,24,H) TS4(25,26,27,28,H) TS4(29,30,31,32,H) \
 "; TABLE_STORE_BATCH_END"::"v"(value),TADDR(0),TADDR(4),TADDR(8),TADDR(12),TADDR(16),TADDR(20),TADDR(24),TADDR(28),"s"(rs):"memory")
template<bool Store,uint32_t P>
__device__ __forceinline__ uint64_t table_stream(uint64_t until,U4 rs,uint32_t slice,uint32_t random,uint32_t salt,
        uint32_t& index,uint32_t& last_index,U4* v){
    const uint32_t mask=slice/16-1;const U4 value={store_value(0,salt),store_value(1,salt),store_value(2,salt),store_value(3,salt)};
    uint64_t chunks=0;
    do{
        #pragma nounroll
        for(uint32_t batch=0;batch<WorkBatches;batch++){
            uint32_t off[TableDepth];
            #pragma unroll
            for(uint32_t j=0;j<TableDepth;j++)off[j]=16*vector_at(index+j*TableThreads+threadIdx.x,mask,salt,random);
            if constexpr(Store){TABLE_POLICY(TBSTORE);}else{TABLE_POLICY(TBLOAD);}
            last_index=index;index=(index+BatchVectors)&mask;
        }
        // Store batches have no per-batch wait; drain once per256 issued stores
        // before timing/control and before any outputs can be reused.
        if constexpr(Store)asm volatile("s_waitcnt vmcnt(0)":::"memory");
        ++chunks;
    }while(wall_clock()<until);
    return chunks;
}
#undef TBSTORE
#undef TS4
#undef TS
#undef TBLOAD
#undef TADDR
#undef TOUT
#undef TL4
#undef TL
#define TSTORE1(H) asm volatile("buffer_store_dwordx4 %0, %1, %2, 0 offen" H "\n\ts_waitcnt vmcnt(0)"::"v"(value),"v"(offset),"s"(rs):"memory")
template<uint32_t P>__device__ __forceinline__ void table_store_one(U4 value,uint32_t offset,U4 rs){TABLE_POLICY(TSTORE1);}
#undef TSTORE1
#undef TABLE_POLICY

#define TM4 \
 "v_mfma_f32_16x16x128_f8f6f4 %0, %4, %4, 0\n\t" "v_mfma_f32_16x16x128_f8f6f4 %1, %4, %4, 0\n\t" \
 "v_mfma_f32_16x16x128_f8f6f4 %2, %4, %4, 0\n\t" "v_mfma_f32_16x16x128_f8f6f4 %3, %4, %4, 0\n\t"
#define TM16 TM4 TM4 TM4 TM4
#define TM64 TM16 TM16 TM16 TM16
__device__ __forceinline__ uint64_t table_mfma(uint64_t stop,TF4& a,TF4& b,TF4& c,TF4& d){
    const T8 x={0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u,0x20202020u};uint64_t n=0;
    do{asm volatile(TM64 TM64:"=&v"(a),"=&v"(b),"=&v"(c),"=&v"(d):"v"(x));++n;}while(wall_clock()<stop);return n;
}
#undef TM64
#undef TM16
#undef TM4

template<bool Store,uint32_t P>
__global__ __launch_bounds__(TableThreads) void hardware_table_kernel(TableArgs args){
    extern __shared__ __attribute__((aligned(16))) uint8_t reserve[];
    const uint32_t tid=threadIdx.x,wave=__builtin_amdgcn_readfirstlane(tid/64);
    TableWorker wr{};wr.entry_wall=wall_clock();wr.hw0=hwid();wr.xcc0=xccid();wr.block=blockIdx.x;wr.wave=wave;wr.active_lanes=__popcll(__ballot(1));wr.slice=args.slice;
    const bool probe=!args.bandwidth&&blockIdx.x==0;
    const bool compute=args.scope==1&&blockIdx.x!=0;
    wr.role=probe?0:compute?2:1;
    reserve[tid]=tid;if(tid==0)reserve[TableLDS-1]=0xa5;__syncthreads();
    const uint64_t first=args.epoch+TableWarm+TableGuard;
    const uint64_t end=first+(args.bandwidth?BandwidthTicks:uint64_t(args.chains)*TableSlot)+TableGuard;
    table_wait(args.epoch);
    if(probe){
        if(tid==0){
            Resource resource(args.probe,ProbeBytes);U4 last{},value={store_value(0,args.salt),store_value(1,args.salt),store_value(2,args.salt),store_value(3,args.salt)};
            uint32_t rank=0,at=16*vector_at(0,ProbeBytes/16-1,args.salt,args.random);
            for(uint32_t sample=0;sample<args.chains;sample++){
                uint64_t scheduled=first+uint64_t(sample)*TableSlot;table_wait(scheduled);
                TableSample s{};s.deadline=scheduled+TableSlot;s.ready=wall_clock();s.iteration=sample;
                Record& z=s.chase;z.hw0=hwid();z.xcc0=xccid();z.start=at;z.steps=args.blank?0:4097;z.active_lanes=__popcll(__ballot(1));
                asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)":::"memory");z.wall0=wall_clock();z.core0=core_clock();
                asm volatile("; TABLE_SERIAL_BEGIN":::"memory");
                if(!args.blank){
                    #pragma nounroll
                    for(uint32_t group=0;group<32;group++){
                        #pragma unroll
                        for(uint32_t j=0;j<128;j++){
                            if constexpr(Store){table_store_one<P>(value,at,resource.words);++rank;at=16*vector_at(rank,ProbeBytes/16-1,args.salt,args.random);}
                            else at=step<Vgpr,16,P>(at,resource.words,0,last);
                        }
                    }
                    if constexpr(Store){table_store_one<P>(value,at,resource.words);++rank;at=16*vector_at(rank,ProbeBytes/16-1,args.salt,args.random);}
                    else at=step<Vgpr,16,P>(at,resource.words,0,last);
                }
                asm volatile("; TABLE_SERIAL_END":::"memory");
                z.core1=core_clock();z.wall1=wall_clock();z.hw1=hwid();z.xcc1=xccid();z.end=at;
                for(uint32_t p=0;p<4;p++)z.last[p]=Store?value[p]:last[p];s.late=z.wall1>=s.deadline;args.samples[sample]=s;
            }
        }
    }else if(compute){
        TF4 a{},b{},c{},d{};wr.warm0=wall_clock();
        asm volatile("; TABLE_MFMA_WARM_BEGIN":::"memory");wr.warm_chunks=table_mfma(first-TableGuard,a,b,c,d);asm volatile("; TABLE_MFMA_WARM_END":::"memory");
        wr.warm1=wall_clock();wr.work0=wall_clock();wr.core0=core_clock();
        asm volatile("; TABLE_MFMA_WORK_BEGIN":::"memory");wr.work_chunks=table_mfma(end,a,b,c,d);asm volatile("; TABLE_MFMA_WORK_END":::"memory");
        wr.core1=core_clock();wr.work1=wall_clock();
        wr.sum=uint32_t(a[0]+a[1]+a[2]+a[3]+b[0]+b[1]+b[2]+b[3]+c[0]+c[1]+c[2]+c[3]+d[0]+d[1]+d[2]+d[3]);
    }else{
        const uint32_t slice_id=args.bandwidth?blockIdx.x:blockIdx.x-1;
        Resource resource(args.data+uint64_t(slice_id)*args.slice,args.slice);uint32_t index=0,last=0;U4 values[TableDepth];
        wr.warm0=wall_clock();
        asm volatile("; TABLE_STREAM_WARM_BEGIN":::"memory");
        wr.warm_chunks=table_stream<Store,P>(first-TableGuard,resource.words,args.slice,args.random,args.salt,index,last,values);
        asm volatile("; TABLE_STREAM_WARM_END":::"memory");
        wr.warm1=wall_clock();wr.warm_last_index=last;wr.work0=wall_clock();wr.core0=core_clock();
        asm volatile("; TABLE_STREAM_WORK_BEGIN":::"memory");
        wr.work_chunks=table_stream<Store,P>(end,resource.words,args.slice,args.random,args.salt,index,last,values);
        asm volatile("; TABLE_STREAM_WORK_END":::"memory");
        wr.core1=core_clock();wr.work1=wall_clock();wr.last_index=last;
        uint32_t sum=0x2468ace0u;
        if constexpr(!Store){
            #pragma unroll
            for(uint32_t i=0;i<TableDepth;i++){
                #pragma unroll
                for(uint32_t p=0;p<4;p++)sum+=values[i][p];
            }
        }
        wr.sum=sum;
    }
    if(!probe||tid==0)table_wait(end+TableGuard);
    __syncthreads();wr.hw1=hwid();wr.xcc1=xccid();wr.store_wall=wall_clock();
    if(!probe)args.sinks[blockIdx.x*256+tid]=wr.sum;
    if(tid%64==0)args.workers[blockIdx.x*4+wave]=wr;
}
using TableLaunch=void(*)(TableArgs,uint32_t,hipStream_t);
struct TableKernel {TableLaunch launch;hipFuncAttributes attr;int occupancy;};
template<bool S,uint32_t P>void table_launch(TableArgs a,uint32_t blocks,hipStream_t s){hardware_table_kernel<S,P><<<blocks,256,TableLDS,s>>>(a);HIP_OK(hipGetLastError());}
template<bool S,uint32_t P>TableKernel table_describe(){
    TableKernel k{};k.launch=table_launch<S,P>;const void* fn=reinterpret_cast<const void*>(hardware_table_kernel<S,P>);
    HIP_OK(hipFuncGetAttributes(&k.attr,fn));HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.occupancy,fn,256,TableLDS));
    if(k.attr.localSizeBytes||k.attr.sharedSizeBytes||k.occupancy!=1)throw std::runtime_error("scratch/1CTA occupancy");return k;
}
template<bool S>TableKernel table_policy(uint32_t p){
    switch(p){case 0:return table_describe<S,0>();case 1:return table_describe<S,1>();case 2:return table_describe<S,2>();case 3:return table_describe<S,3>();case 4:return table_describe<S,4>();case 5:return table_describe<S,5>();case 6:return table_describe<S,6>();case 7:return table_describe<S,7>();}throw std::runtime_error("policy");
}
std::vector<TableCase> table_plan(const std::filesystem::path& p){
    std::ifstream f(p);std::string line;std::getline(f,line);std::vector<TableCase> result;std::set<std::string> seen;
    while(std::getline(f,line)){
        std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);TableCase c;
        if(!(s>>c.name>>c.random>>c.store>>c.policy>>c.scope>>c.bandwidth>>c.blank)||c.random>1||c.store>1||c.policy>7||c.scope>2||c.bandwidth>1||c.blank>1||(c.blank&&c.bandwidth)||!seen.insert(c.name).second||c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("plan");result.push_back(c);
    }
    if(result.empty())throw std::runtime_error("empty plan");return result;
}
uint64_t host_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct PowerSample{uint64_t begin_ns,end_ns,microwatts;};
struct PowerRecorder {
    std::atomic<bool> done{false};std::vector<PowerSample> records;std::thread thread;
    explicit PowerRecorder(std::string path):thread([this,path]{
        int timer=timerfd_create(CLOCK_MONOTONIC,TFD_CLOEXEC);if(timer<0)return;
        itimerspec spec{};spec.it_value.tv_nsec=1000000;spec.it_interval.tv_nsec=20000000;
        if(timerfd_settime(timer,0,&spec,nullptr)){close(timer);return;}
        while(!done.load()){
            uint64_t ticks;if(read(timer,&ticks,8)!=8)break;
            PowerSample p{};p.begin_ns=host_ns();std::ifstream f(path);if(f>>p.microwatts){p.end_ns=host_ns();records.push_back(p);}
        }
        close(timer);
    }){}
    void stop(){done.store(true);if(thread.joinable())thread.join();}
    ~PowerRecorder(){stop();}
};

void run_tables(const std::filesystem::path& out,const std::vector<TableCase>& configs,uint32_t seed,uint32_t count,const std::string& power_path){
    hipDeviceProp_t dev{};HIP_OK(hipGetDeviceProperties(&dev,0));char bdf[32];HIP_OK(hipDeviceGetPCIBusId(bdf,32,0));int wall;HIP_OK(hipDeviceGetAttribute(&wall,hipDeviceAttributeWallClockRate,0));
    if(std::string(bdf)!="0000:85:00.0"||std::string(dev.gcnArchName).find("gfx950")!=0||dev.multiProcessorCount!=256||wall!=100000||dev.sharedMemPerMultiprocessor!=163840||dev.warpSize!=64)throw std::runtime_error("device");
    const uint64_t data_max=uint64_t(256)*FullSlice;
    size_t free,total;HIP_OK(hipMemGetInfo(&free,&total));if(free<data_max+4ull*ProbeBytes)throw std::runtime_error("memory");
    hipStream_t stream;HIP_OK(hipStreamCreate(&stream));hipEvent_t start,stop;HIP_OK(hipEventCreate(&start));HIP_OK(hipEventCreate(&stop));
    uint8_t *data,*probe;uint4* eviction;uint32_t* evict_sink;TableSample* samples;TableWorker* workers;uint32_t* sinks;uint64_t* epoch_device;unsigned long long* errors;
    HIP_OK(hipMalloc(&data,data_max+2*Guard));HIP_OK(hipMalloc(&probe,uint64_t(ProbeBytes)+2*Guard));HIP_OK(hipMalloc(&eviction,EvictionBytes));HIP_OK(hipMalloc(&evict_sink,256*256*4));
    HIP_OK(hipMalloc(&samples,MaxSamples*sizeof(TableSample)));HIP_OK(hipMalloc(&workers,1024*sizeof(TableWorker)));HIP_OK(hipMalloc(&sinks,65536*4));HIP_OK(hipMalloc(&epoch_device,8));HIP_OK(hipMalloc(&errors,8));
    if(!((uint64_t(data)+data_max+2*Guard<=uint64_t(probe))||(uint64_t(probe)+ProbeBytes+2*Guard<=uint64_t(data))))throw std::runtime_error("allocation overlap");
    HIP_OK(hipMemsetAsync(eviction,0x5a,EvictionBytes,stream));
    std::ofstream info(out/"device.json");info<<"{\"bdf\":\""<<bdf<<"\",\"arch\":\""<<dev.gcnArchName<<"\",\"CUs\":256,\"wave_size\":64,\"threads\":256,\"LDS\":98304,\"LDS_per_CU\":163840,\"wall_khz\":100000,\"sample_bytes\":104,\"worker_bytes\":128,\"seed\":"<<seed<<",\"probe_allocation\":"<<uint64_t(probe)<<",\"probe_bytes\":"<<ProbeBytes<<",\"data_allocation\":"<<uint64_t(data)<<",\"data_allocation_bytes\":"<<data_max+2*Guard<<",\"node_bytes\":16,\"warm_ticks\":"<<TableWarm<<",\"slot_ticks\":"<<TableSlot<<",\"guard_ticks\":"<<TableGuard<<",\"bandwidth_ticks\":"<<BandwidthTicks<<",\"power_path\":\""<<power_path<<"\",\"power_unit\":\"microwatts; whole GPU socket\"}\n";info.close();
    std::ofstream meta(out/"cases.json");meta<<"[\n";uint32_t ordinal=0;
    for(const auto& c:configs){
        const auto k=c.store?table_policy<true>(c.policy):table_policy<false>(c.policy);
        const uint32_t blocks=c.scope==0?1:256,slice=c.scope==2?FullSlice:ProbeBytes;
        const uint32_t memory_ctas=c.bandwidth?(c.scope==2?256:1):(c.scope==2?255:0);
        const uint64_t bytes=uint64_t(memory_ctas)*slice;const uint32_t salt=seed^(c.random*0x29f710adu)^(c.store*0x84926153u);
        HIP_OK(hipMemsetAsync(data,0xa5,data_max+2*Guard,stream));HIP_OK(hipMemsetAsync(probe,0xa5,uint64_t(ProbeBytes)+2*Guard,stream));
        if(!c.store){
            if(bytes){initialize_table_data<<<2048,256,0,stream>>>(reinterpret_cast<uint32_t*>(data+Guard),bytes/4,salt);HIP_OK(hipGetLastError());}
            if(!c.bandwidth){initialize_table_chain<<<2048,256,0,stream>>>(reinterpret_cast<uint4*>(probe+Guard),ProbeBytes/16,salt,c.random);HIP_OK(hipGetLastError());}
        }
        chase_eviction<<<256,256,0,stream>>>(eviction,EvictionBytes/16,evict_sink);HIP_OK(hipGetLastError());
        HIP_OK(hipMemsetAsync(samples,0xff,MaxSamples*sizeof(TableSample),stream));HIP_OK(hipMemsetAsync(workers,0xff,blocks*4*sizeof(TableWorker),stream));HIP_OK(hipMemsetAsync(sinks,0xff,blocks*256*4,stream));HIP_OK(hipStreamSynchronize(stream));
        table_epoch<<<1,64,0,stream>>>(epoch_device);HIP_OK(hipGetLastError());uint64_t epoch;HIP_OK(hipMemcpyAsync(&epoch,epoch_device,8,hipMemcpyDeviceToHost,stream));HIP_OK(hipStreamSynchronize(stream));epoch+=TableLead;
        TableArgs args{probe+Guard,data+Guard,slice,c.random,c.scope,c.bandwidth,c.blank,count,salt,epoch,samples,workers,sinks};
        PowerRecorder power(power_path);uint64_t host_begin=host_ns();
        HIP_OK(hipEventRecord(start,stream));k.launch(args,blocks,stream);HIP_OK(hipEventRecord(stop,stream));HIP_OK(hipEventSynchronize(stop));uint64_t host_end=host_ns();power.stop();float ms;HIP_OK(hipEventElapsedTime(&ms,start,stop));
        std::vector<TableWorker> hw(blocks*4);std::vector<uint32_t> hs(blocks*256);std::vector<TableSample> hp(c.bandwidth?0:count);
        HIP_OK(hipMemcpy(hw.data(),workers,hw.size()*sizeof(TableWorker),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(hs.data(),sinks,hs.size()*4,hipMemcpyDeviceToHost));
        if(!hp.empty())HIP_OK(hipMemcpy(hp.data(),samples,hp.size()*sizeof(TableSample),hipMemcpyDeviceToHost));
        binary(out/(c.name+".workers.bin"),hw);binary(out/(c.name+".sinks.bin"),hs);binary(out/(c.name+".samples.bin"),hp);binary(out/(c.name+".power.bin"),power.records);
        bool topology=true,overlap=true;std::set<uint32_t> cus;
        for(uint32_t block=0;block<blocks;block++){
            uint32_t key=cu(hw[4*block].hw0,hw[4*block].xcc0);if(!cus.insert(key).second)topology=false;
            for(uint32_t w=0;w<4;w++){
                auto v=hw[block*4+w];if(v.block!=block||v.wave!=w||v.active_lanes!=64)throw std::runtime_error("worker fields");
                if(cu(v.hw0,v.xcc0)!=key||cu(v.hw1,v.xcc1)!=key)topology=false;
                if(!hp.empty()&&(v.store_wall<=hp.back().chase.wall1||(block&&(v.work0>hp.front().chase.wall0||v.work1<hp.back().chase.wall1))))overlap=false;
                if(v.role&&(!v.warm_chunks||!v.work_chunks||v.work1<=v.work0))throw std::runtime_error("background count/time");
                if(v.role==1){
                    uint32_t expected=uint32_t((v.warm_chunks+v.work_chunks)*WorkBatches*BatchVectors-BatchVectors)&(slice/16-1);
                    if(v.last_index!=expected)throw std::runtime_error("bulk cursor");
                    if(v.work_chunks*WorkBatches*BatchVectors*16<slice)throw std::runtime_error("incomplete bulk slice");
                    for(uint32_t lane=0;lane<64;lane++){
                        uint32_t sum=0x2468ace0u;
                        if(!c.store){
                            uint32_t tid=w*64+lane,slice_id=c.bandwidth?block:block-1;
                            for(uint32_t j=0;j<TableDepth;j++){
                                uint32_t at=vector_at(v.last_index+j*256+tid,slice/16-1,salt,c.random);
                                for(uint32_t p=0;p<4;p++)sum+=table_word(uint64_t(slice_id)*slice/4+uint64_t(at)*4+p,salt);
                            }
                        }
                        if(hs[block*256+w*64+lane]!=sum||(lane==0&&v.sum!=sum))throw std::runtime_error("bulk last batch checksum");
                    }
                }else if(v.role==2){for(uint32_t lane=0;lane<64;lane++)if(hs[block*256+w*64+lane]!=32)throw std::runtime_error("MFMA result");}
            }
        }
        for(uint32_t i=0;i<hp.size();i++){
            auto s=hp[i];auto z=s.chase;uint32_t hops=c.blank?0:4097;
            if(z.steps!=hops||z.active_lanes!=1||s.iteration!=i||z.start!=16*vector_at(i*hops,ProbeBytes/16-1,salt,c.random)||z.end!=16*vector_at((i+1)*hops,ProbeBytes/16-1,salt,c.random))throw std::runtime_error("serial endpoint");
            if(!c.store&&hops){if(z.last[0]!=z.end)throw std::runtime_error("load result");uint32_t prev=vector_at((i+1)*hops-1,ProbeBytes/16-1,salt,c.random);for(uint32_t p=1;p<4;p++)if(z.last[p]!=tag_word(prev,p,salt))throw std::runtime_error("chain tags");}
            if(cu(z.hw0,z.xcc0)!=cu(hw[0].hw0,hw[0].xcc0)||cu(z.hw1,z.xcc1)!=cu(hw[0].hw0,hw[0].xcc0))topology=false;
            if(s.late||z.wall1>=s.deadline||z.wall0<epoch+TableWarm+TableGuard+uint64_t(i)*TableSlot)overlap=false;
        }
        HIP_OK(hipMemsetAsync(errors,0,8,stream));
        if(c.store){
            if(bytes){verify_table_store<<<2048,256,0,stream>>>(reinterpret_cast<const uint32_t*>(data+Guard),bytes/4,salt,errors);HIP_OK(hipGetLastError());}
            if(!c.bandwidth&&!c.blank){verify_table_probe<<<1024,256,0,stream>>>(reinterpret_cast<const uint4*>(probe+Guard),count*4097,salt,c.random,errors);HIP_OK(hipGetLastError());}
        }
        HIP_OK(hipStreamSynchronize(stream));unsigned long long bad;HIP_OK(hipMemcpy(&bad,errors,8,hipMemcpyDeviceToHost));if(bad)throw std::runtime_error("written value validation");
        std::vector<uint8_t> g(Guard);for(auto [p,n]:std::array<std::pair<uint8_t*,uint64_t>,2>{{{probe,ProbeBytes},{data,data_max}}})for(uint64_t at:std::array<uint64_t,2>{0,Guard+n}){HIP_OK(hipMemcpy(g.data(),p+at,Guard,hipMemcpyDeviceToHost));if(!std::all_of(g.begin(),g.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("guard");}
        if(ordinal++)meta<<",\n";
        meta<<std::setprecision(15)<<"{\"name\":\""<<c.name<<"\",\"random\":"<<c.random<<",\"store\":"<<c.store<<",\"policy\":"<<c.policy<<",\"scope\":"<<c.scope<<",\"bandwidth\":"<<c.bandwidth<<",\"blank\":"<<c.blank<<",\"blocks\":"<<blocks<<",\"memory_ctas\":"<<memory_ctas<<",\"slice\":"<<slice<<",\"data_bytes\":"<<bytes<<",\"salt\":"<<salt<<",\"epoch\":"<<epoch<<",\"chains\":"<<(c.bandwidth?0:count)<<",\"event_ms\":"<<ms<<",\"host_begin_ns\":"<<host_begin<<",\"host_end_ns\":"<<host_end<<",\"VGPR\":"<<k.attr.numRegs<<",\"private\":0,\"occupancy_max\":1,\"topology_valid\":"<<topology<<",\"overlap_valid\":"<<overlap<<",\"store_validation_errors\":0,\"checks\":true}";meta.flush();
        std::cout<<"HARDWARE_TABLE_CASE "<<c.name<<" topology="<<topology<<" overlap="<<overlap<<" CUs="<<cus.size()<<" power_samples="<<power.records.size()<<std::endl;
    }
    meta<<"\n]\n";meta.close();for(auto p:std::array<void*,9>{data,probe,eviction,evict_sink,samples,workers,sinks,epoch_device,errors})HIP_OK(hipFree(p));HIP_OK(hipEventDestroy(start));HIP_OK(hipEventDestroy(stop));HIP_OK(hipStreamDestroy(stream));
}
int main(int argc,char** argv){try{
    if(argc!=6)throw std::runtime_error("hardware_tables NEW_OUT PLAN.csv SEED CHAINS(29main) POWER_SYSFS");
    if(!std::getenv("HIP_VISIBLE_DEVICES")||std::string(std::getenv("HIP_VISIBLE_DEVICES"))!="5"||std::getenv("HSA_CU_MASK")||std::getenv("ROC_GLOBAL_CU_MASK"))throw std::runtime_error("environment");
    auto configs=table_plan(argv[2]);uint32_t count=std::stoul(argv[4]);if(count<2||count>MaxSamples)throw std::runtime_error("chains");
    std::ifstream power(argv[5]);uint64_t microwatts;if(!(power>>microwatts)||!microwatts)throw std::runtime_error("power sensor");
    std::filesystem::path out(argv[1]);if(std::filesystem::exists(out))throw std::runtime_error("existing output");HIP_OK(hipSetDevice(0));std::filesystem::create_directories(out);
    run_tables(out,configs,std::stoul(argv[3]),count,argv[5]);return 0;
}catch(const std::exception& e){std::cerr<<"HARDWARE_TABLE_FAILED "<<e.what()<<std::endl;return 1;}}
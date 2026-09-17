// SPDX-License-Identifier: MIT
// Paired sequential/random chains, using the EXACT previous timedcodeobject.
// Only host initialization changes: one random order of128B nodes perwave.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/timerfd.h>
#include <unistd.h>

#define HIP_OK(x) do{auto e=(x);if(e!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(e));}while(0)
constexpr uint32_t Nodes=2097153, Stride=128, ProbeBytes=Nodes*Stride, Guard=65536, Slice=64u<<20;
constexpr uint64_t BackgroundBytes=uint64_t(255)*Slice, LeadTicks=500000;
struct Record {uint64_t core0,core1,wall0,wall1;uint32_t hw0,hw1,xcc0,xcc1,start,end,steps,active_lanes,last[4];};
struct Sample {Record chase;uint64_t scheduled,deadline,export_wall;uint32_t wave,phase;};
struct Worker {
    uint64_t entry_wall,warm0,warm1,work0,work1,core0,core1,warm_chunks,work_chunks,export_wall;
    uint32_t hw0,hw1,xcc0,xcc1,block,wave,role,active_lanes,last_index,warm_last_index,sum,slice;
};
// Binary-compatible with the preservedArgs128 kernelarg (80bytes).
struct Args {
    uint8_t *probe0,*probe1,*probe2,*probe3,*background;
    uint32_t scope,salt;uint64_t epoch;Sample* samples;Worker* workers;uint32_t* sinks;
};
static_assert(sizeof(Record)==80&&sizeof(Sample)==112&&sizeof(Worker)==128&&sizeof(Args)==80);
uint32_t cu(uint32_t hw,uint32_t xcc){return (xcc<<8)|(((hw>>13)&7)<<4)|((hw>>8)&15);}
__host__ __device__ uint32_t wave_salt(uint32_t salt,uint32_t wave){return salt^(wave*0x13579bdu);}
__host__ __device__ uint32_t tag(uint32_t node,uint32_t p,uint32_t salt){return node*(0x9e3779b9u+p*2u)^(salt+p*0x13579bdu);}
__host__ __device__ uint32_t word(uint64_t i,uint32_t salt){return uint32_t(i)*0x9e3779b9u+salt;}
__global__ void matched_initialize_probe(uint8_t* p,const uint32_t* next,uint32_t salt){
    for(uint32_t node=blockIdx.x*blockDim.x+threadIdx.x;node<Nodes;node+=blockDim.x*gridDim.x)
        *reinterpret_cast<uint4*>(p+uint64_t(node)*Stride)=make_uint4(next[node],tag(node,1,salt),tag(node,2,salt),tag(node,3,salt));
}
__global__ void matched_initialize_background(uint32_t* p,uint64_t n,uint32_t salt){
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=uint64_t(blockDim.x)*gridDim.x)p[i]=word(i,salt);
}
__global__ void matched_eviction(const uint4* p,uint32_t* sink){
    uint32_t sum=0;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<(1ull<<30)/16;i+=uint64_t(blockDim.x)*gridDim.x){
        uint32_t __attribute__((ext_vector_type(4))) x;
        asm volatile("global_load_dwordx4 %0, %1, off\n\ts_waitcnt vmcnt(0)":"=&v"(x):"v"(p+i):"memory");sum+=x[0]+x[1]+x[2]+x[3];
    }
    sink[blockIdx.x*blockDim.x+threadIdx.x]=sum;
}
__global__ void matched_epoch(uint64_t* out){
    if(threadIdx.x==0){uint64_t x;asm volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)":"=s"(x)::"memory");*out=x;}
}
template<class T>void binary(const std::filesystem::path& path,const std::vector<T>& data){
    if(std::filesystem::exists(path))throw std::runtime_error("existing binary");
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));if(!f)throw std::runtime_error("binary write");
}
struct Case {std::string name;uint32_t policy,scope;};
std::vector<Case> read_plan(const std::filesystem::path& p){
    std::ifstream f(p);std::string line;std::getline(f,line);std::vector<Case> result;std::set<std::string> seen;
    while(std::getline(f,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);Case c;
        if(!(in>>c.name>>c.policy>>c.scope)||c.policy>7||c.scope>2||!seen.insert(c.name).second||c.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("plan");result.push_back(c);}
    if(result.empty())throw std::runtime_error("empty plan");return result;
}
uint64_t host_ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct PowerSample{uint64_t begin_ns,end_ns,microwatts;};
struct PowerRecorder {
    std::atomic<bool> done{false};std::vector<PowerSample> samples;std::thread thread;
    explicit PowerRecorder(std::string path):thread([this,path]{
        int fd=timerfd_create(CLOCK_MONOTONIC,TFD_CLOEXEC);if(fd<0)return;
        itimerspec spec{};spec.it_value.tv_nsec=1000000;spec.it_interval.tv_nsec=20000000;
        if(timerfd_settime(fd,0,&spec,nullptr)){close(fd);return;}
        while(!done.load()){uint64_t ticks;if(read(fd,&ticks,8)!=8)break;PowerSample p{};p.begin_ns=host_ns();std::ifstream f(path);if(f>>p.microwatts){p.end_ns=host_ns();samples.push_back(p);}}
        close(fd);
    }){}
    void stop(){done.store(true);if(thread.joinable())thread.join();}
    ~PowerRecorder(){stop();}
};
bool disjoint(uint64_t a,uint64_t an,uint64_t b,uint64_t bn){return a+an<=b||b+bn<=a;}

void run(const std::filesystem::path& out,const std::vector<Case>& cases,uint32_t seed,uint32_t pair_order,const std::string& power_path,const std::string& codeobject){
    hipDeviceProp_t dev{};HIP_OK(hipGetDeviceProperties(&dev,0));char bdf[32];HIP_OK(hipDeviceGetPCIBusId(bdf,32,0));int wall;HIP_OK(hipDeviceGetAttribute(&wall,hipDeviceAttributeWallClockRate,0));
    if(std::string(bdf)!="0000:85:00.0"||std::string(dev.gcnArchName).find("gfx950")!=0||dev.multiProcessorCount!=256||wall!=100000||dev.sharedMemPerMultiprocessor!=163840||dev.warpSize!=64)throw std::runtime_error("device");
    size_t free,total;HIP_OK(hipMemGetInfo(&free,&total));if(free<BackgroundBytes+(3ull<<30))throw std::runtime_error("VRAM capacity");
    hipModule_t module;HIP_OK(hipModuleLoad(&module,codeobject.c_str()));
    std::array<hipFunction_t,8> kernels;std::array<int,8> regs;
    for(uint32_t p=0;p<8;p++){
        const std::string name="_Z28contiguous_latency128_kernelILj"+std::to_string(p)+"EEv7Args128";
        HIP_OK(hipModuleGetFunction(&kernels[p],module,name.c_str()));int local,shared,occ;
        HIP_OK(hipFuncGetAttribute(&regs[p],HIP_FUNC_ATTRIBUTE_NUM_REGS,kernels[p]));
        HIP_OK(hipFuncGetAttribute(&local,HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,kernels[p]));
        HIP_OK(hipFuncGetAttribute(&shared,HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,kernels[p]));
        HIP_OK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&occ,kernels[p],256,98304));
        if(local||shared||occ!=1)throw std::runtime_error("module resource/occupancy");
    }
    hipStream_t stream;hipEvent_t begin,end;HIP_OK(hipStreamCreate(&stream));HIP_OK(hipEventCreate(&begin));HIP_OK(hipEventCreate(&end));
    std::array<uint8_t*,4> probes{};for(auto& p:probes)HIP_OK(hipMalloc(&p,uint64_t(ProbeBytes)+2*Guard));
    uint8_t* background;Sample* samples;Worker* workers;uint32_t* sinks;uint64_t* epoch;uint4* eviction;uint32_t* evict_sink;uint32_t* next_device;
    HIP_OK(hipMalloc(&background,BackgroundBytes+2*Guard));HIP_OK(hipMalloc(&samples,12*sizeof(Sample)));HIP_OK(hipMalloc(&workers,1024*sizeof(Worker)));HIP_OK(hipMalloc(&sinks,65536*4));HIP_OK(hipMalloc(&epoch,8));HIP_OK(hipMalloc(&eviction,1ull<<30));HIP_OK(hipMalloc(&evict_sink,65536*4));HIP_OK(hipMalloc(&next_device,uint64_t(Nodes)*4));HIP_OK(hipMemsetAsync(eviction,0x5a,1ull<<30,stream));
    for(uint32_t i=0;i<4;i++){
        if(uint64_t(probes[i]+Guard)%128||!disjoint(uint64_t(probes[i]),uint64_t(ProbeBytes)+2*Guard,uint64_t(background),BackgroundBytes+2*Guard))throw std::runtime_error("probe/background overlap or alignment");
        for(uint32_t j=0;j<i;j++)if(!disjoint(uint64_t(probes[i]),uint64_t(ProbeBytes)+2*Guard,uint64_t(probes[j]),uint64_t(ProbeBytes)+2*Guard))throw std::runtime_error("wave buffer overlap");
    }
    std::array<std::vector<uint32_t>,4> orders,random_next;
    for(uint32_t wave=0;wave<4;wave++){
        auto& order=orders[wave];order.resize(Nodes);std::iota(order.begin(),order.end(),0);
        std::mt19937_64 rng(uint64_t(seed)^(0x9e3779b97f4a7c15ull*(wave+1)));
        std::shuffle(order.begin()+1,order.end(),rng); // anchor0;uniformshuffleofallothernodes
        auto& next=random_next[wave];next.resize(Nodes);std::vector<uint8_t> seen(Nodes);
        for(uint32_t i=0;i<Nodes;i++){if(order[i]>=Nodes||seen[order[i]]++)throw std::runtime_error("permutation");next[order[i]]=order[(i+1)%Nodes]*Stride;}
        uint32_t at=0;for(uint32_t i=0;i<Nodes;i++){if(at!=order[i]*Stride)throw std::runtime_error("host dependent cycle");at=next[at/Stride];}if(at)throw std::runtime_error("cycle closure");
        binary(out/("wave_"+std::to_string(wave)+".order.bin"),order);
    }
    std::vector<uint32_t> sequential_next(Nodes);for(uint32_t i=0;i<Nodes;i++)sequential_next[i]=(i+1)%Nodes*Stride;
    std::ofstream info(out/"device.json");info<<"{\"bdf\":\""<<bdf<<"\",\"arch\":\""<<dev.gcnArchName<<"\",\"CUs\":256,\"threads\":256,\"wave_size\":64,\"LDS\":98304,\"LDS_per_CU\":163840,\"wall_khz\":100000,\"sample_bytes\":112,\"worker_bytes\":128,\"args_bytes\":80,\"seed\":"<<seed<<",\"pair_order\":"<<pair_order<<",\"node_spacing\":128,\"load_bytes\":16,\"nodes\":"<<Nodes<<",\"probe_bytes_per_wave\":"<<ProbeBytes<<",\"address_span_per_sweep\":"<<((Nodes-1)*Stride+16)<<",\"warm_ticks\":20000000,\"slot_ticks\":250000000,\"guard_ticks\":200000,\"background_bytes\":"<<BackgroundBytes<<",\"background_allocation\":"<<uint64_t(background)<<",\"probe_allocations\":[";
    for(uint32_t wave=0;wave<4;wave++){if(wave)info<<',';info<<uint64_t(probes[wave]);}info<<"],\"code_object\":\""<<codeobject<<"\",\"power_path\":\""<<power_path<<"\",\"pair_difference\":\"onlyprobechainlinkorder;identicalkernelbinary/buffers/salts/cache/background/schedule\"}\n";info.close();
    std::ofstream meta(out/"cases.json");meta<<"[\n";size_t ordinal=0,pair_index=0;
    for(const auto& c:cases){
        const uint32_t salt=seed^0x58264719u,blocks=c.scope?256:1;
        // Counterbalancepatternorderacrosspairsandseeds;do notchoosebylatency.
        const uint32_t first=(pair_order+pair_index++)%2;
        for(uint32_t iteration=0;iteration<2;iteration++){
            const uint32_t random=(first+iteration)%2;const std::string name=c.name+(random?"_random":"_sequential");
            for(uint32_t wave=0;wave<4;wave++){
                const auto& links=random?random_next[wave]:sequential_next;
                HIP_OK(hipMemsetAsync(probes[wave],0xa5,uint64_t(ProbeBytes)+2*Guard,stream));HIP_OK(hipMemcpyAsync(next_device,links.data(),uint64_t(Nodes)*4,hipMemcpyHostToDevice,stream));
                matched_initialize_probe<<<2048,256,0,stream>>>(probes[wave]+Guard,next_device,wave_salt(salt,wave));HIP_OK(hipGetLastError());
            }
            HIP_OK(hipMemsetAsync(background,0xa5,BackgroundBytes+2*Guard,stream));
            if(c.scope==2){matched_initialize_background<<<2048,256,0,stream>>>(reinterpret_cast<uint32_t*>(background+Guard),BackgroundBytes/4,salt);HIP_OK(hipGetLastError());}
            matched_eviction<<<256,256,0,stream>>>(eviction,evict_sink);HIP_OK(hipGetLastError());
            HIP_OK(hipMemsetAsync(samples,0xff,12*sizeof(Sample),stream));HIP_OK(hipMemsetAsync(workers,0xff,blocks*4*sizeof(Worker),stream));HIP_OK(hipMemsetAsync(sinks,0xff,blocks*256*4,stream));HIP_OK(hipStreamSynchronize(stream));
            const uint64_t host_epoch0=host_ns();matched_epoch<<<1,64,0,stream>>>(epoch);HIP_OK(hipGetLastError());uint64_t raw_epoch;HIP_OK(hipMemcpyAsync(&raw_epoch,epoch,8,hipMemcpyDeviceToHost,stream));HIP_OK(hipStreamSynchronize(stream));const uint64_t host_epoch1=host_ns();
            Args args{probes[0]+Guard,probes[1]+Guard,probes[2]+Guard,probes[3]+Guard,background+Guard,c.scope,salt,raw_epoch+LeadTicks,samples,workers,sinks};void* kernel_args[]={&args};
            PowerRecorder power(power_path);HIP_OK(hipEventRecord(begin,stream));
            HIP_OK(hipModuleLaunchKernel(kernels[c.policy],blocks,1,1,256,1,1,98304,stream,kernel_args,nullptr));
            HIP_OK(hipEventRecord(end,stream));HIP_OK(hipEventSynchronize(end));power.stop();float ms;HIP_OK(hipEventElapsedTime(&ms,begin,end));
            std::vector<Sample> hs(12);std::vector<Worker> hw(blocks*4);std::vector<uint32_t> sums(blocks*256);
            HIP_OK(hipMemcpy(hs.data(),samples,hs.size()*sizeof(Sample),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(hw.data(),workers,hw.size()*sizeof(Worker),hipMemcpyDeviceToHost));HIP_OK(hipMemcpy(sums.data(),sinks,sums.size()*4,hipMemcpyDeviceToHost));
            binary(out/(name+".samples.bin"),hs);binary(out/(name+".workers.bin"),hw);binary(out/(name+".sinks.bin"),sums);binary(out/(name+".power.bin"),power.samples);
            bool topology=true,overlap=true;uint32_t target=cu(hw[0].hw0,hw[0].xcc0);uint64_t first_time=~0ull,last_time=0;std::set<uint32_t> seen;
            for(uint32_t wave=0;wave<4;wave++)for(uint32_t phase=0;phase<3;phase++){
                const auto& s=hs[wave*3+phase];const auto& z=s.chase;
                if(s.wave!=wave||s.phase!=phase||z.active_lanes!=1||z.steps!=(phase?Nodes:0)||z.start||z.end||z.wall1<=z.wall0)throw std::runtime_error("fullcycle record");
                if(cu(z.hw0,z.xcc0)!=target||cu(z.hw1,z.xcc1)!=target)topology=false;else if(z.core1<=z.core0)throw std::runtime_error("sameCU clock");
                if(phase){const uint32_t tail=random?orders[wave].back():Nodes-1;if(z.last[0])throw std::runtime_error("cycle endpoint");for(uint32_t p=1;p<4;p++)if(z.last[p]!=tag(tail,p,wave_salt(salt,wave)))throw std::runtime_error("lastnodetag");}
                if(z.wall0<s.scheduled||z.wall1>=s.deadline)overlap=false;
                if(phase==2){first_time=std::min(first_time,z.wall0);last_time=std::max(last_time,z.wall1);}
            }
            for(uint32_t wave=0;wave<4;wave++)if(hs[wave*3].export_wall>=first_time||hs[wave*3+1].export_wall>=first_time||hs[wave*3+2].export_wall<=last_time)overlap=false;
            for(uint32_t block=0;block<blocks;block++){
                const uint32_t k=cu(hw[block*4].hw0,hw[block*4].xcc0);if(!seen.insert(k).second)topology=false;
                for(uint32_t wave=0;wave<4;wave++){
                    const auto& wr=hw[block*4+wave];
                    if(wr.block!=block||wr.wave!=wave||wr.role!=(block?c.scope:0)||wr.active_lanes!=64)throw std::runtime_error("worker record");
                    if(cu(wr.hw0,wr.xcc0)!=k||cu(wr.hw1,wr.xcc1)!=k)topology=false;
                    if(wr.export_wall<=last_time)overlap=false;
                    if(block){
                        if(!wr.warm_chunks||!wr.work_chunks||wr.work1<=wr.work0)throw std::runtime_error("work count/time");
                        if(wr.work0>first_time||wr.work1<last_time)overlap=false;
                        if(c.scope==2&&wr.last_index!=uint32_t((wr.warm_chunks+wr.work_chunks)*65536-8192)%(Slice/16))throw std::runtime_error("bulkcursor");
                        for(uint32_t lane=0;lane<64;lane++){
                            uint32_t expected=32;
                            if(c.scope==2){expected=0x2468ace0;for(uint32_t j=0;j<32;j++)for(uint32_t p=0;p<4;p++)expected+=word(uint64_t(block-1)*Slice/4+uint64_t(wr.last_index+j*256+wave*64+lane)*4+p,salt);}
                            if(sums[block*256+wave*64+lane]!=expected||(lane==0&&wr.sum!=expected))throw std::runtime_error("background data");
                        }
                    }
                }
            }
            std::vector<uint8_t> g(Guard);for(auto ptr:probes)for(uint64_t off:std::array<uint64_t,2>{0,Guard+ProbeBytes}){HIP_OK(hipMemcpy(g.data(),ptr+off,Guard,hipMemcpyDeviceToHost));if(!std::all_of(g.begin(),g.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("probeguard");}
            for(uint64_t off:std::array<uint64_t,2>{0,Guard+BackgroundBytes}){HIP_OK(hipMemcpy(g.data(),background+off,Guard,hipMemcpyDeviceToHost));if(!std::all_of(g.begin(),g.end(),[](uint8_t x){return x==0xa5;}))throw std::runtime_error("backgroundguard");}
            if(ordinal++)meta<<",\n";
            meta<<std::setprecision(15)<<"{\"name\":\""<<name<<"\",\"pair\":\""<<c.name<<"\",\"random\":"<<random<<",\"pattern_order\":"<<iteration<<",\"policy\":"<<c.policy<<",\"scope\":"<<c.scope<<",\"blocks\":"<<blocks<<",\"salt\":"<<salt<<",\"raw_epoch\":"<<raw_epoch<<",\"epoch\":"<<args.epoch<<",\"host_epoch0\":"<<host_epoch0<<",\"host_epoch1\":"<<host_epoch1<<",\"event_ms\":"<<ms<<",\"VGPR\":"<<regs[c.policy]<<",\"private\":0,\"occupancy_max\":1,\"topology_valid\":"<<topology<<",\"overlap_valid\":"<<overlap<<",\"tail_nodes\":[";
            for(uint32_t wave=0;wave<4;wave++){if(wave)meta<<',';meta<<(random?orders[wave].back():Nodes-1);}meta<<"],\"checks\":true}";meta.flush();
            std::cout<<"MATCHED128_CASE "<<name<<" topology="<<topology<<" overlap="<<overlap<<" CUs="<<seen.size()<<" power="<<power.samples.size()<<std::endl;
        }
    }
    meta<<"\n]\n";meta.close();for(auto p:probes)HIP_OK(hipFree(p));for(void* p:std::array<void*,8>{background,samples,workers,sinks,epoch,eviction,evict_sink,next_device})HIP_OK(hipFree(p));
    HIP_OK(hipEventDestroy(begin));HIP_OK(hipEventDestroy(end));HIP_OK(hipStreamDestroy(stream));HIP_OK(hipModuleUnload(module));
}
int main(int argc,char** argv){try{
    if(argc!=7)throw std::runtime_error("matched128 NEW_OUT PLAN.csv SEED PAIR_ORDER(0/1) POWER_SYSFS PRESERVED_CODE_OBJECT");
    if(!std::getenv("HIP_VISIBLE_DEVICES")||std::string(std::getenv("HIP_VISIBLE_DEVICES"))!="5"||std::getenv("HSA_CU_MASK")||std::getenv("ROC_GLOBAL_CU_MASK"))throw std::runtime_error("environment");
    const auto cases=read_plan(argv[2]);uint32_t order=std::stoul(argv[4]);if(order>1)throw std::runtime_error("patternorder");
    std::ifstream power(argv[5]);uint64_t value;if(!(power>>value)||!value)throw std::runtime_error("power sensor");
    std::filesystem::path out=argv[1];if(std::filesystem::exists(out))throw std::runtime_error("existing output");
    HIP_OK(hipSetDevice(0));std::filesystem::create_directories(out);run(out,cases,std::stoul(argv[3]),order,argv[5],argv[6]);return 0;
}catch(const std::exception& e){std::cerr<<"MATCHED128_FAILED "<<e.what()<<std::endl;return 1;}}
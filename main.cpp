#include "CL/cl.h"
#include <iostream>
#include<vector>
#include <cstdlib>
#include <ctime>

using namespace std;
const int sizze=8;
int arr[sizze]={7,3,1,8,2,0,1,4};
int arr1[sizze];
int arr2[sizze];
int main() {
  int m=1;
      while(m<sizze){
        m*=2;
      }
   size_t localSize=m;
  size_t numgroup=(sizze+localSize-1)/localSize;
   cout<<numgroup<<endl;
   size_t globalSize = m;

const char* kernelSource1=R"CLC(
  __kernel void reduction(__global int*input,__global int* output,const int n,const int m, __local int*temp){
        int global_id=get_global_id(0);
        int local_id=get_local_id(0);
       temp[local_id]=input[global_id];
      barrier(CLK_LOCAL_MEM_FENCE);
      
      for(int i=1;i<=m;i*=2){
        int idx=(local_id+1)*i*2-1;
        if(idx<m){
              temp[idx]=temp[idx]+temp[idx-i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if(local_id==m-1){
        temp[local_id]=0;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for(int i=m/2;i>=1;i/=2){
        int idx=(local_id+1)*i*2-1;
        if(idx<m){
            int t=temp[idx-i];
            temp[idx-i]=temp[idx];
            temp[idx]+=t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
        if(global_id<n){
             output[global_id]=temp[local_id];
        }
      

  }
)CLC";

const char* extractLSB_kernelSource=R"CLC(
  __kernel void extractLSB(__global int*input,__global int*output,__local int*temp,int bitNumberK){
      int global_id=get_global_id(0);
      int local_id=get_local_id(0);
      temp[local_id]=input[global_id];

      barrier(CLK_LOCAL_MEM_FENCE);
      for(int i=1;i<bitNumberK;i++){
        temp[local_id]=temp[local_id]/2;
      }
      if(temp[local_id]%2==0){
        temp[local_id]=1;
      }
      else{
        temp[local_id]=0;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      output[global_id]=temp[local_id];

  }
)CLC";

const char* scatter_kernelSource=R"CLC(
  __kernel void scatter_phase(__global int*input,__global int*LSB0_check,__global int*scan,__global int*output,int total0){
      int global_id=get_global_id(0);
      if(LSB0_check[global_id]==1){
        int pos=scan[global_id];
        output[pos]=input[global_id];
      }
      else{
        int pos=total0+(global_id-scan[global_id]);
         output[pos]=input[global_id];
      }
     barrier(CLK_GLOBAL_MEM_FENCE);
     
  }
)CLC";

//kernel source///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cl_int status;

   cl_uint platformCount;
      status = clGetPlatformIDs(0, nullptr, &platformCount);
     

      vector<cl_platform_id> platform(platformCount);
     status = clGetPlatformIDs(platformCount, platform.data(), nullptr);
        

        cl_uint deviceCount;
        cl_device_id gpu;
        int platform_container;
           status =  clGetDeviceIDs(platform[0],CL_DEVICE_TYPE_GPU,1,&gpu,nullptr);
          
            if(status!=CL_SUCCESS){
              cout<<status<<endl;
            }
        
           
  int res=0;      

   cl_context context =clCreateContext(nullptr,1,&gpu,nullptr,nullptr,&status);
   cl_command_queue queue=clCreateCommandQueue(context,gpu,0,&status);

   cl_mem buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(arr),nullptr,&status);
   cl_mem buffer1=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(arr2),nullptr,&status);
  cl_mem LSB_buffer_store=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(arr1),nullptr,&status);
  cl_mem final_res_buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(arr2),nullptr,&status);

   clEnqueueWriteBuffer(queue,buffer,CL_TRUE,0,sizeof(arr),arr,0,nullptr,nullptr);
  

   //program
    cl_program LSB_extract_program = clCreateProgramWithSource(context, 1, &extractLSB_kernelSource, nullptr, &status);
    cl_int build_status2=   clBuildProgram(LSB_extract_program, 1, &gpu, nullptr, nullptr, nullptr);

   cl_program program = clCreateProgramWithSource(context, 1, &kernelSource1, nullptr, &status);
    cl_int build_status=   clBuildProgram(program, 1, &gpu, nullptr, nullptr, nullptr);

    cl_program scatter_program = clCreateProgramWithSource(context,1,&scatter_kernelSource,nullptr,&status);
    cl_int build_status3= clBuildProgram(scatter_program,1,&gpu,nullptr,nullptr,nullptr);

  if (build_status3 != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(scatter_program, gpu, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      vector<char> log(log_size);
      clGetProgramBuildInfo(scatter_program, gpu, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      cout << "Build Log:\n" << log.data() << endl;
  }
  else{
    cout<<"build success"<<endl;
  }

   //kernel management, compile and run

   cl_kernel kernel= clCreateKernel(program,"reduction",nullptr);
   cl_kernel kernelLSB= clCreateKernel(LSB_extract_program,"extractLSB",nullptr);
   cl_kernel KernelScatter= clCreateKernel(scatter_program,"scatter_phase",nullptr);

    

   
   //clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
   

for(int i=1;i<=8;i++){

  // extract number i bit

    clSetKernelArg(kernelLSB,0,sizeof(cl_mem),&buffer);
    clSetKernelArg(kernelLSB,1,sizeof(cl_mem),&LSB_buffer_store);
    clSetKernelArg(kernelLSB,2,sizeof(int)*sizze,nullptr);
    clSetKernelArg(kernelLSB,3,sizeof(int),&i);
  clEnqueueNDRangeKernel(queue, kernelLSB, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
  clEnqueueReadBuffer(queue,LSB_buffer_store,CL_TRUE ,0,sizeof(int)*sizze,arr1,0,nullptr,nullptr);

  //reduction and scan

    clSetKernelArg(kernel,0,sizeof(cl_mem),&LSB_buffer_store);
    clSetKernelArg(kernel,1,sizeof(cl_mem),&buffer1);
    clSetKernelArg(kernel,2,sizeof(int),&sizze);
    clSetKernelArg(kernel,3,sizeof(int),&m);
    clSetKernelArg(kernel,4,sizeof(int)*m,nullptr);

  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
  clEnqueueReadBuffer(queue,buffer1,CL_TRUE ,0,sizeof(int)*sizze,arr2,0,nullptr,nullptr);
  int CountTotal0=arr2[sizze-1]+arr1[sizze-1];

  // sort by LSB
    clSetKernelArg(KernelScatter,0,sizeof(cl_mem),&buffer);
    clSetKernelArg(KernelScatter,1,sizeof(cl_mem),&LSB_buffer_store);
    clSetKernelArg(KernelScatter,2,sizeof(cl_mem),&buffer1);
    clSetKernelArg(KernelScatter,3,sizeof(cl_mem),&final_res_buffer);
    clSetKernelArg(KernelScatter,4,sizeof(int),&CountTotal0);

  clEnqueueNDRangeKernel(queue, KernelScatter, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

  // swap in/out buffer to continue caculate

  cl_mem temp=buffer;
  buffer=final_res_buffer;
  final_res_buffer=temp;
}
status = clEnqueueReadBuffer(queue,buffer,CL_TRUE ,0,sizeof(int)*sizze,arr2,0,nullptr,nullptr);
for(int i=0;i<sizze;i++){
  cout<<arr2[i]<<" ";
}
    clFinish(queue);
    

  clReleaseMemObject(buffer);
  clReleaseMemObject(buffer1);
  clReleaseMemObject(LSB_buffer_store);
  clReleaseMemObject(final_res_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    //cd D:\Nhan\it\OpenCl_project
    //cl main.cpp /EHsc /I "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include" /link /LIBPATH:"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib" OpenCL.lib

}



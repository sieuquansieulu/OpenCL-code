#include "CL/cl.h"
#include <iostream>
#include<vector>
#include <cstdlib>
#include <ctime>

using namespace std;
const int sizze=8;
int arr[sizze]={7,3,1,8,2,0,1,4};
int flag[sizze];
int segmented_id[sizze];
int arr2[sizze]={7,3,1,8,2,0,1,4};
int main() {
  srand((unsigned)time(NULL));
  int m=1;
      while(m<sizze){
        m*=2;
      }
   size_t localSize=m;
  size_t numgroup=(sizze+localSize-1)/localSize;
   cout<<numgroup<<endl;
   size_t globalSize = m;
int pos=0;
while (pos<sizze){
   flag[pos]=1;
   int valid_length=sizze-pos;
   int random_length=1+(rand()%valid_length);
   pos+=random_length;
}
for(int i=0;i<sizze;i++){
  cout<<arr[i]<<" ";
}
cout<<endl;



const char* kernelSource1=R"CLC(
  __kernel void reduction(__global int*input,__global int* output,const int n,const int m, __local int*temp,__global int*flag,__local int*flagSave,__local int*flagSave2){
    int global_id=get_global_id(0);
    int local_id=get_local_id(0);

      int x=(global_id<n) ? input[global_id]:0;
      int y=(global_id<n) ? flag[global_id]:1;
      temp[local_id]=x;
      flagSave[local_id]=y;
     flagSave2[local_id]=y;
      barrier(CLK_LOCAL_MEM_FENCE);


      for(int i=1;i<m;i*=2){
        int idx=(local_id+1)*i*2-1;
        if(idx<m){
          int LFlag=flagSave[idx-i];
          int RFlag=flagSave[idx];

          if(RFlag==0){
            temp[idx]=temp[idx]+temp[idx-i];
          }
          flagSave[idx]= LFlag | RFlag;
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
          int L=idx-i;
          int R=idx;

          int carry=temp[idx];
          int t=temp[L];
          int rightFirs=R-i+1;
          int HeadFlag=flagSave2[rightFirs];

          int carryLeft=carry;
          int carrtRight= (HeadFlag == 0) ? (carry+t):0;

          temp[L]=carryLeft;
          temp[R]=carrtRight;

        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if(global_id<n){
          output[global_id]=temp[local_id];
      }

     
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
   cl_mem flag_save=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(flag),nullptr,&status);


   clEnqueueWriteBuffer(queue,buffer,CL_TRUE,0,sizeof(arr),arr,0,nullptr,nullptr);
    clEnqueueWriteBuffer(queue,flag_save,CL_TRUE,0,sizeof(flag),flag,0,nullptr,nullptr);
    

   //cout<<write_status;
  

   //program
 
   cl_program program = clCreateProgramWithSource(context, 1, &kernelSource1, nullptr, &status);
   cl_int build_status=   clBuildProgram(program, 1, &gpu, nullptr, nullptr, nullptr);




  if (build_status != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(program, gpu, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
      vector<char> log(log_size);
      clGetProgramBuildInfo(program, gpu, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
      cout << "Build Log:\n" << log.data() << endl;
  }
  else{
    cout<<"build success"<<endl;
  }

   //kernel management, compile and run

   cl_kernel kernel= clCreateKernel(program,"reduction",nullptr);

   

  

for(int i=0;i<sizze;i++){
  cout<<flag[i]<<" ";
}
cout<<endl;
// for(int i=0;i<sizze;i++){
//   if(flag[i]==1){
//   segmented_id[i]++;
//   }
//    cout<<segmented_id[i]<<" ";
// }
// cout<<endl;



  clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer);
  clSetKernelArg(kernel,1,sizeof(cl_mem),&buffer1);
  clSetKernelArg(kernel,2,sizeof(int),&sizze);
  clSetKernelArg(kernel,3,sizeof(int),&m);
  clSetKernelArg(kernel,4,sizeof(int)*m,nullptr);
  clSetKernelArg(kernel,5,sizeof(cl_mem),&flag_save);
  clSetKernelArg(kernel,6,sizeof(int)*m,nullptr);
  clSetKernelArg(kernel,7,sizeof(int)*m,nullptr);


   clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
   status = clEnqueueReadBuffer(queue,buffer1,CL_TRUE ,0,sizeof(int)*sizze,arr2,0,nullptr,nullptr);
for(int i=0;i<sizze;i++){
  cout<<arr2[i]<<" ";
}


    clFinish(queue);
    

  clReleaseMemObject(buffer);
  clReleaseMemObject(buffer1);
  clReleaseMemObject(flag_save);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
    //cd D:\Nhan\it\OpenCl_project
    //cl main.cpp /EHsc /I "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include" /link /LIBPATH:"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib" OpenCL.lib

}



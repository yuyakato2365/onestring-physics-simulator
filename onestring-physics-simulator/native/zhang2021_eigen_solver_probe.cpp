// Minimal ABI probe for independently compiled Eigen solver; no author code.
#include <windows.h>
#include <cmath>
#include <iostream>
using INIT=void (*)(void*,int*,int*,int*,double*,int*);
using SOLVE=void (*)(void*,int*,int*,int*,int*,int*,double*,int*,int*,int*,int*,int*,int*,double*,double*,int*,double*);
int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"Give alternative solver DLL path\n";return 2;}
    HMODULE h=LoadLibraryA(argv[1]);
    if(!h){std::cerr<<"Could not load DLL "<<GetLastError()<<"\n";return 3;}
    auto init=reinterpret_cast<INIT>(GetProcAddress(h,"pardisoinit"));
    auto solve=reinterpret_cast<SOLVE>(GetProcAddress(h,"pardiso"));
    if(!init||!solve){std::cerr<<"ABI exports missing\n";return 4;}
    void *pt[64]={};
    int mtype=-2,sol=0,iparm[64]={},err=0;
    double dparm[64]={};
    init(pt,&mtype,&sol,iparm,dparm,&err);
    if(err){std::cerr<<"init="<<err<<"\n";return 5;}
    int n=2,maxfct=1,mnum=1,phase=11,nrhs=1,msglvl=0,perm=0;
    int ia[3]={1,3,4}, ja[3]={1,2,2};
    double a[3]={2,1,3},b[2]={1,2},x[2]={0,0};
    auto call=[&](int operation){
       phase=operation;
       solve(pt,&maxfct,&mnum,&mtype,&phase,&n,a,ia,ja,&perm,
             &nrhs,iparm,&msglvl,b,x,&err,dparm);
       std::cout<<"phase="<<operation<<" error="<<err<<"\n";
       return err==0;
    };
    if(!call(11)||!call(22)||!call(33))return 6;
    if(std::fabs(x[0]-0.2)>1e-8 || std::fabs(x[1]-0.6)>1e-8) {
        std::cerr<<"WRONG_NUMERICAL_ANSWER x0="<<x[0]<<" x1="<<x[1]<<"\n";
        return 7;
    }
    if(!call(-1))return 8;
    std::cout<<"LEGAL_SOLVER_EQUIVALENT_LINEAR_SYSTEM_OK\n";
    FreeLibrary(h);
    return 0;
}

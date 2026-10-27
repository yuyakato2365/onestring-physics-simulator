// Clean-room independent sparse solver, built with Eigen (MPL-2.0).
// Only the numeric Ax=b backend changes; original Zhang 2021 binaries
// Straighten.exe, Segment.exe, QMSim.exe are kept unchanged.
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <cmath>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <exception>
#ifdef _WIN32
#define EXPORTED extern "C" __declspec(dllexport)
#else
#define EXPORTED extern "C"
#endif

using Matrix=Eigen::SparseMatrix<double,Eigen::ColMajor,int>;
using Vector=Eigen::VectorXd;
struct Context {
  Matrix a;
  Eigen::SparseLU<Matrix,Eigen::COLAMDOrdering<int>> lu;
  int n=0;
  bool ready=false;
};
static uint64_t key(int i,int j) {
  return (uint64_t(uint32_t(i))<<32)|uint32_t(j);
}
static bool load(Context& c,int n,const double* a,const int* ia,
                 const int* ja,int base) {
  if(!a||!ia||!ja||n<=0||n>200000||ia[0]!=base)return false;
  auto nnz=int64_t(ia[n])-base;
  if(nnz<n||nnz>50000000)return false;
  std::vector<Eigen::Triplet<double,int>> ts;
  ts.reserve(size_t(nnz)*2);
  std::unordered_set<uint64_t> given;
  for(int i=0;i<n;++i) {
    int64_t first=int64_t(ia[i])-base,last=int64_t(ia[i+1])-base;
    if(first<0||last<first||last>nnz)return false;
    for(int64_t k=first;k<last;++k) {
      int j=ja[k]-base;
      if(j<0||j>=n||!std::isfinite(a[k]))return false;
      ts.emplace_back(i,j,a[k]);
      given.insert(key(i,j));
    }
  }
  size_t orig=ts.size();
  for(size_t k=0;k<orig;++k) {
    auto& t=ts[k];
    if(t.row()!=t.col()&&!given.count(key(t.col(),t.row())))
      ts.emplace_back(t.col(),t.row(),t.value());
  }
  Matrix m(n,n);
  m.setFromTriplets(ts.begin(),ts.end());
  m.makeCompressed();
  Matrix diff=m-Matrix(m.transpose());
  if(diff.norm()>1e-8*std::max(1.0,m.norm()))return false;
  c.a=std::move(m);c.n=n;
  return true;
}
EXPORTED void pardisoinit(void** pt,int* mtype,int* solver,int* iparm,
                         double* dparm,int* error) noexcept {
  if(!error)return;
  *error=-100;
  if(!pt||!mtype||!solver||!iparm||!dparm||*mtype!=-2||*solver!=0)return;
  for(int i=0;i<64;++i){iparm[i]=0;dparm[i]=0;}
  iparm[0]=1;iparm[2]=1;iparm[34]=0; // original author uses 1-based CSR
  *error=0;
}
EXPORTED void pardiso(void** pt,int* maxfct,int* mnum,int* mtype,
   int* phase,int* n,double* a,int* ia,int* ja,int* perm,
   int* nrhs,int* iparm,int* msglvl,double* b,double* x,
   int* error,double* dparm) noexcept {
  (void)maxfct;(void)mnum;(void)perm;(void)msglvl;(void)dparm;
  if(!error)return;
  *error=-100;
  if(!pt||!mtype||!phase||!n||!nrhs||!iparm||
     *mtype!=-2||*n<=0||*nrhs<=0||*nrhs>16)return;
  try {
    if(*phase==-1||*phase==0){
      delete static_cast<Context*>(pt[0]);pt[0]=nullptr;*error=0;return;
    }
    if(*phase==11){
      delete static_cast<Context*>(pt[0]);pt[0]=new Context();
      *error=0;return;
    }
    auto* ctx=static_cast<Context*>(pt[0]);
    if(!ctx)return;
    if(*phase==22){
      ctx->ready=false;
      if(!load(*ctx,*n,a,ia,ja,iparm[34]==1?0:1)){
        *error=-101;return;
      }
      ctx->lu.analyzePattern(ctx->a);
      if(ctx->lu.info()!=Eigen::Success){*error=-102;return;}
      ctx->lu.factorize(ctx->a);
      if(ctx->lu.info()!=Eigen::Success){*error=-103;return;}
      ctx->ready=true;*error=0;return;
    }
    if(*phase==33&&ctx->ready&&ctx->n==*n&&b&&x){
      for(int col=0;col<*nrhs;++col){
        Eigen::Map<const Vector> rhs(b+size_t(col)*(*n),*n);
        Vector v=ctx->lu.solve(rhs);
        if(ctx->lu.info()!=Eigen::Success||!v.allFinite()){
          *error=-104;return;
        }
        for(int iter=0;iter<3;++iter){
          Vector residual=rhs-ctx->a*v;
          double scaled=residual.norm()/
            std::max(1.,rhs.norm()+ctx->a.norm()*v.norm());
          if(scaled<1e-9)break;
          Vector dv=ctx->lu.solve(residual);
          if(ctx->lu.info()!=Eigen::Success||!dv.allFinite()){
            *error=-105;return;
          }
          v+=dv;
        }
        Vector r=rhs-ctx->a*v;
        if(!r.allFinite()||r.norm()>1e-7*
           std::max(1.,rhs.norm()+ctx->a.norm()*v.norm())){
          *error=-106;return;
        }
        Eigen::Map<Vector>(x+size_t(col)*(*n),*n)=v;
      }
      *error=0;return;
    }
  }catch(const std::exception&){*error=-190;return;}
   catch(...){*error=-191;return;}
}
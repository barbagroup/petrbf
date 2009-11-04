struct PARAMETER{
int nt;
int memory;
double vis;
double t;
double dt;
double u_inf;
};

struct PARTICLE{
int n;
int no;
int ni;
int nj;
int nilocal;
int njlocal;
int ista;
int iend;
int jsta;
int jend;
double sigma;
double sigma0;
double overlap;
double h;
double xmin;
double xmax;
double ymin;
double ymax;
double zmin;
double zmax;
double r_grid;
Vec i;
Vec j;
Vec ii;
Vec jj;
Vec xi;
Vec yi;
Vec zi;
Vec gi;
Vec ei;
Vec wi;
Vec xj;
Vec yj;
Vec zj;
Vec gj;
PetscScalar *il;
PetscScalar *jl;
PetscScalar *xil;
PetscScalar *yil;
PetscScalar *zil;
PetscScalar *gil;
PetscScalar *eil;
PetscScalar *wil;
PetscScalar *xjl;
PetscScalar *yjl;
PetscScalar *zjl;
PetscScalar *gjl;
};

struct BOUNDARY{
int n;
double r;
double *x;
double *y;
double *z;
double *g;
double *ut;
double *vt;
double *vnx;
double *vny;
double *vtx;
double *vty;
};

struct CLUSTER{
int nsigma_box;
int sigma_buffer;
int sigma_trunc;
int n;
int nx;
int ny;
int nz;
int neighbor_buffer;
int neighbor_trunc;
int neighbor_ghost;
int niperbox;
int njperbox;
int nclocal;
int ncghost;
int nighost;
int njghost;
int npbufferi;
int nptruncj;
int maxbuffer;
int maxtrunc;
int maxghost;
int maxlocal;
int file;
int icsta;
int icend;
int *ista;
int *iend;
int *jsta;
int *jend;
int *ix;
int *iy;
int *iz;
int *ilocal;
int *jlocal;
int *ighost;
int *jghost;
int *idx;
int *jdx;
double xmin;
double xmax;
double ymin;
double ymax;
double zmin;
double zmax;
double box_length;
double buffer_length;
double trunc_length;
double *xc;
double *yc;
double *zc;
double *xib;
double *yib;
double *zib;
double *gib;
double *eib;
double *wib;
double *xjt;
double *yjt;
double *zjt;
double *gjt;
};

struct GRID{
int nx;
int ny;
int nz;
};

struct HIERARCHICAL{
int mp;
};

struct MPI2{
int nprocs;
int myrank;
int nsta;
int nend;
int ista;
int iend;
double *sendi;
double *recvi;
double *sendj;
double *recvj;
};

struct BOTH{
PARTICLE *p;
CLUSTER *c;
};

double epsf=1e-6;

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
double r_grid;
Vec i;
Vec j;
Vec ii;
Vec jj;
Vec xi;
Vec yi;
Vec gi;
Vec ei;
Vec wi;
Vec xj;
Vec yj;
Vec gj;
PetscScalar *il;
PetscScalar *jl;
PetscScalar *xil;
PetscScalar *yil;
PetscScalar *gil;
PetscScalar *eil;
PetscScalar *wil;
PetscScalar *xjl;
PetscScalar *yjl;
PetscScalar *gjl;
};

struct BOUNDARY{
int n;
double r;
double *x;
double *y;
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
double box_length;
double buffer_length;
double trunc_length;
double *xc;
double *yc;
double *xib;
double *yib;
double *gib;
double *eib;
double *wib;
double *xjt;
double *yjt;
double *gjt;
};

struct GRID{
int nx;
int ny;
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

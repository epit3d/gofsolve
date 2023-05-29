package fsolve

/*
#cgo LDFLAGS: -lm

#include "fsolve.h"
#include "callback.h"
*/
import "C"
import (
	"runtime/cgo"
	"unsafe"
)

type Callback func(n int, x []float64, fvec []float64)

//export gocallback
func gocallback(
	h C.uintptr_t,
	n C.int,
	x *C.double,
	fvec *C.double,
) {
	fn := cgo.Handle(h).Value().(Callback)

	xarray := unsafe.Slice(x, n)
	fvecarray := unsafe.Slice(fvec, n)

	xgo := []float64{}
	fvecgo := []float64{}

	// convert to golang slices
	for i := 0; i < int(n); i++ {
		xgo = append(xgo, float64(xarray[i]))
		fvecgo = append(fvecgo, float64(fvecarray[i]))
	}

	fn(int(n), xgo, fvecgo)

	// convert back to C arrays
	for i := 0; i < int(n); i++ {
		xarray[i] = C.double(xgo[i])
		fvecarray[i] = C.double(fvecgo[i])
	}
}

/*
	fdjac1() estimates an N by N Jacobian matrix using forward differences.

Discussion:

	This function computes a forward-difference approximation
	to the N by N jacobian matrix associated with a specified
	problem of N functions in N variables.

	If the jacobian has a banded form, then function evaluations are saved
	by only approximating the nonzero terms.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	07 April 2021

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Parameters:

	Input, void FCN(int n,double x[], double fx[]): the name of the
	C routine which returns in fx[] the function value at the n-dimensional
	vector x[].

	Input, int N, the number of functions and variables.

	Input, double X[N], the evaluation point.

	Input, double FVEC[N], the functions evaluated at X.

	Output, double FJAC[N*N], the approximate jacobian matrix at X.

	Input, int LDFJAC, specifies the leading dimension of the array fjac,
	not less than N.

	   ml is a nonnegative integer input variable which specifies
	     the number of subdiagonals within the band of the
	     jacobian matrix. if the jacobian is not banded, set
	     ml to at least n - 1.

	   epsfcn is an input variable used in determining a suitable
	     step length for the forward-difference approximation. this
	     approximation assumes that the relative errors in the
	     functions are of the order of epsfcn. if epsfcn is less
	     than the machine precision, it is assumed that the relative
	     errors in the functions are of the order of the machine
	     precision.

	   mu is a nonnegative integer input variable which specifies
	     the number of superdiagonals within the band of the
	     jacobian matrix. if the jacobian is not banded, set
	     mu to at least n - 1.

	   wa1 and wa2 are work arrays of length n. if ml + mu + 1 is at
	     least n, then the jacobian is considered dense, and wa2 is
	     not referenced.
*/
func Fdjac1(cb Callback, n int, x []float64, fvec []float64, fjac []float64, ldfjac int, ml int, mu int, epsfcn float64, wa1 []float64, wa2 []float64) {
	h := cgo.NewHandle(cb)
	defer h.Delete()

	C.fdjac1_wrapped(
		C.uintptr_t(h),
		C.int(n),
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&fvec[0])),
		(*C.double)(unsafe.Pointer(&fjac[0])),
		C.int(ldfjac),
		C.int(ml),
		C.int(mu),
		C.double(epsfcn),
		(*C.double)(unsafe.Pointer(&wa1[0])),
		(*C.double)(unsafe.Pointer(&wa2[0])),
	)
}

/*
	fsolve() finds a zero of a system of N nonlinear equations.

Discussion:

	fsolve() finds a zero of a system of N nonlinear functions in N variables
	by a modification of the Powell hybrid method.

	This is done by using the more general nonlinear equation solver HYBRD.

	The user must provide FCN, which calculates the functions.

	The jacobian is calculated by a forward-difference approximation.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	09 April 2021

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Parameters:

	Input, void FCN(int n,double x[], double fx[]): the name of the
	C routine which returns in fx[] the function value at the n-dimensional
	vector x[].

	Input, int N, the number of functions and variables.

	Input/output, double X[N].  On input, an initial estimate of the solution.
	On output, the final estimate of the solution.

	Output, double FVEC[N], the functions evaluated at the output X.

	Input, double TOL, a nonnegative variable. tTermination occurs when the
	algorithm estimates that the relative error between X and the solution
	is at most TOL.

	Output, int INFO:
	0: improper input parameters.
	1: algorithm estimates that the relative error
	   between x and the solution is at most tol.
	2: number of calls to fcn has reached or exceeded 200*(n+1).
	3: tol is too small. no further improvement in
	   the approximate solution x is possible.
	4: iteration is not making good progress.
*/
func Fsolve(cb Callback, n int, x []float64, fvec []float64, tol float64) int {
	h := cgo.NewHandle(cb)
	defer h.Delete()

	result := C.fsolve_wrapped(
		C.uintptr_t(h),
		C.int(n),
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&fvec[0])),
		C.double(tol),
	)

	return int(result)
}

/*
	hybrd() finds a zero of a system of N nonlinear equations.

Discussion:

	The purpose of HYBRD is to find a zero of a system of
	N nonlinear functions in N variables by a modification
	of the Powell hybrid method.

	The user must provide FCN, which calculates the functions.

	The jacobian is calculated by a forward-difference approximation.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	07 April 2021

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Parameters:

	Input, void FCN(int n,double x[], double fx[]): the name of the
	C routine which returns in fx[] the function value at the n-dimensional
	vector x[].

	Input, int N, the number of functions and variables.

	Input/output, double X[N].  On input an initial estimate of the solution.
	On output, the final estimate of the solution.

	Output, double FVEC[N], the functions evaluated at the output value of X.

	Input, double XTOL, a nonnegative value.  Termination occurs when the
	relative error between two consecutive iterates is at most XTOL.

	Input, int MAXFEV.  Termination occurs when the number of calls to FCN
	is at least MAXFEV by the end of an iteration.

	Input, int ML, specifies the number of subdiagonals within the band of
	the jacobian matrix.  If the jacobian is not banded, set
	ml to at least n - 1.

	Input, int MU, specifies the number of superdiagonals within the band of
	the jacobian matrix. if the jacobian is not banded, set
	mu to at least n - 1.

	   epsfcn is an input variable used in determining a suitable
	     step length for the forward-difference approximation. this
	     approximation assumes that the relative errors in the
	     functions are of the order of epsfcn. if epsfcn is less
	     than the machine precision, it is assumed that the relative
	     errors in the functions are of the order of the machine
	     precision.

	   diag is an array of length n. if mode = 1 (see
	     below), diag is internally set. if mode = 2, diag
	     must contain positive entries that serve as
	     multiplicative scale factors for the variables.

	   mode is an integer input variable. if mode = 1, the
	     variables will be scaled internally. if mode = 2,
	     the scaling is specified by the input diag. other
	     values of mode are equivalent to mode = 1.

	   factor is a positive input variable used in determining the
	     initial step bound. this bound is set to the product of
	     factor and the euclidean norm of diag*x if nonzero, or else
	     to factor itself. in most cases factor should lie in the
	     interval (.1,100.). 100. is a generally recommended value.

	Output, int INFO:
	0: improper input parameters.
	1: algorithm estimates that the relative error
	   between x and the solution is at most tol.
	2: number of calls to fcn has reached or exceeded 200*(n+1).
	3: tol is too small. no further improvement in
	   the approximate solution x is possible.
	4: iteration is not making good progress.

	Output, int NFEV, the number of calls to fcn.

	   fjac is an output n by n array which contains the
	     orthogonal matrix q produced by the qr factorization
	     of the final approximate jacobian.

	   ldfjac is a positive integer input variable not less than n
	     which specifies the leading dimension of the array fjac.

	   r is an output array of length lr which contains the
	     upper triangular matrix produced by the qr factorization
	     of the final approximate jacobian, stored rowwise.

	   lr is a positive integer input variable not less than
	     (n*(n+1))/2.

	   qtf is an output array of length n which contains
	     the vector (q transpose)*fvec.

	   wa1, wa2, wa3, and wa4 are work arrays of length n.
*/
func Hybrd(cb Callback, n int, x []float64, fvec []float64, xtol float64, maxfev int, ml int, mu int, epsfcn float64, diag []float64, mode int, factor float64, nfev int, fjac []float64, ldfjac int, r []float64, lr int, qtf []float64, wa1 []float64, wa2 []float64, wa3 []float64, wa4 []float64) int {
	h := cgo.NewHandle(cb)
	defer h.Delete()

	result := C.hybrd_wrapped(
		C.uintptr_t(h),
		C.int(n),
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&fvec[0])),
		C.double(xtol),
		C.int(maxfev),
		C.int(ml),
		C.int(mu),
		C.double(epsfcn),
		(*C.double)(unsafe.Pointer(&diag[0])),
		C.int(mode),
		C.double(factor),
		C.int(nfev),
		(*C.double)(unsafe.Pointer(&fjac[0])),
		C.int(ldfjac),
		(*C.double)(unsafe.Pointer(&r[0])),
		C.int(lr),
		(*C.double)(unsafe.Pointer(&qtf[0])),
		(*C.double)(unsafe.Pointer(&wa1[0])),
		(*C.double)(unsafe.Pointer(&wa2[0])),
		(*C.double)(unsafe.Pointer(&wa3[0])),
		(*C.double)(unsafe.Pointer(&wa4[0])),
	)

	return int(result)
}

/*
	dogleg() combines Gauss-Newton and gradient for a minimizing step.

Discussion:

	Given an M by N matrix A, an n by n nonsingular diagonal
	matrix d, an m-vector b, and a positive number delta, the
	problem is to determine the convex combination x of the
	gauss-newton and scaled gradient directions that minimizes
	(a*x - b) in the least squares sense, subject to the
	restriction that the euclidean norm of d*x be at most delta.

	This function completes the solution of the problem
	if it is provided with the necessary information from the
	qr factorization of a.

	That is, if a = q*r, where q has orthogonal columns and r is an upper
	triangular matrix, then dogleg expects the full upper triangle of r and
	the first n components of Q'*b.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	07 April 2021

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Parameters:

	Input, int N, the order of R.

	Input, double R[LR], the upper triangular matrix R stored by rows.

	Input, int LR, the size of the storage for R, which should be at
	least (n*(n+1))/2.

	Input, double DIAG[N], the diagonal elements of the matrix D.

	Input, double QTB[N], the first n elements of the vector
	(q transpose)*b.

	Input, double DELTA, an upper bound on the euclidean norm of d*x.

	Output, double X[N], contains the desired convex combination of the
	gauss-newton direction and the scaled gradient direction.

	Workspace, WA1[N].

	Workspace, WA2[N].
*/
func Dogleg(n int, r []float64, lr int, diag []float64, qtb []float64,
	delta float64, x []float64, wa1, wa2 []float64) {
	c_n := C.int(n)
	c_lr := C.int(lr)
	c_delta := C.double(delta)

	c_r := (*C.double)(unsafe.Pointer(&r[0]))
	c_diag := (*C.double)(unsafe.Pointer(&diag[0]))
	c_qtb := (*C.double)(unsafe.Pointer(&qtb[0]))
	c_x := (*C.double)(unsafe.Pointer(&x[0]))
	c_wa1 := (*C.double)(unsafe.Pointer(&wa1[0]))
	c_wa2 := (*C.double)(unsafe.Pointer(&wa2[0]))

	C.dogleg(c_n, c_r, c_lr, c_diag, c_qtb, c_delta, c_x, c_wa1, c_wa2)
}

/*
	enorm() returns the Euclidean norm of a vector.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	07 April 2021

Author:

	John Burkardt

Input:

	int N, the number of entries in A.

	double X[N], the vector whose norm is desired.

Output:

	double ENORM, the norm of X.
*/
func Enorm(n int, x []float64) float64 {
	c_n := C.int(n)
	c_x := (*C.double)(unsafe.Pointer(&x[0]))

	result := C.enorm(c_n, c_x)
	return float64(result)
}

/*
	qform() constructs the standard form of Q from its factored form.

Discussion:

	This function proceeds from the computed QR factorization of
	an M by N matrix A to accumulate the M by M orthogonal matrix
	Q from its factored form.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	02 January 2018

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Parameters:

	Input, int M, the number of rows of A, and the order of Q.

	Input, int N, the number of columns of A.

	Input/output, double Q[LDQ*N].  On input, the full lower trapezoid in
	the first min(M,N) columns of Q contains the factored form.
	On output Q has been accumulated into a square matrix.

	Input, int LDQ, the leading dimension of the array Q.
*/
func Qform(m, n int, q []float64, ldq int) {
	c_m := C.int(m)
	c_n := C.int(n)
	c_q := (*C.double)(unsafe.Pointer(&q[0]))
	c_ldq := C.int(ldq)

	C.qform(c_m, c_n, c_q, c_ldq)
}

/*
	qrfac() computes the QR factorization of an M by N matrix.

Discussion:

	This function uses Householder transformations with optional column
	pivoting to compute a QR factorization of the M by N matrix A.

	That is, QRFAC determines an orthogonal
	matrix Q, a permutation matrix P, and an upper trapezoidal
	matrix R with diagonal elements of nonincreasing magnitude,
	such that A*P = Q*R.

	The Householder transformation for
	column k, k = 1,2,...,min(m,n), is of the form

	  i - (1/u(k))*u*u'

	where U has zeros in the first K-1 positions.

	The form of this transformation and the method of pivoting first
	appeared in the corresponding LINPACK function.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	02 January 2017

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Parameters:

	Input, int M, the number of rows of A.

	Input, int N, the number of columns of A.

	Input/output, double A[M*N].  On input, the matrix for which the QR
	factorization is to be computed.  On output, the strict upper trapezoidal
	part contains the strict upper trapezoidal part of the R factor, and
	the lower trapezoidal part contains a factored form of Q, the non-trivial
	elements of the U vectors described above.

	Input, int LDA, a positive value not less than M which specifies the
	leading dimension of the array A.

	Input, bool PIVOT.  If true, then column pivoting is enforced.

	Output, integer IPVT[LIPVT].  If PIVOT is true, then on output IPVT
	defines the permutation matrix P such that A*P = Q*R.  Column J of P
	is column IPVT[J] of the identity matrix.

	   lipvt is a positive integer input variable. if pivot is false,
	     then lipvt may be as small as 1. if pivot is true, then
	     lipvt must be at least n.

	Output, double RDIAG[N], the diagonal elements of r.

	   acnorm is an output array of length n which contains the
	     norms of the corresponding columns of the input matrix a.
	     if this information is not needed, then acnorm can coincide
	     with rdiag.
*/
func Qrfac(m, n int, a []float64, lda int, pivot bool, ipvt []int,
	lipvt int, rdiag, acnorm []float64) {
	c_m := C.int(m)
	c_n := C.int(n)
	c_a := (*C.double)(unsafe.Pointer(&a[0]))
	c_lda := C.int(lda)
	c_pivot := C.bool(pivot)
	c_ipvt := (*C.int)(unsafe.Pointer(&ipvt[0]))
	c_lipvt := C.int(lipvt)
	c_rdiag := (*C.double)(unsafe.Pointer(&rdiag[0]))
	c_acnorm := (*C.double)(unsafe.Pointer(&acnorm[0]))

	C.qrfac(c_m, c_n, c_a, c_lda, c_pivot, c_ipvt, c_lipvt, c_rdiag, c_acnorm)
}

/*
	r1mpyq() multiplies an M by N matrix A by the Q factor.

Discussion:

	Given an M by N matrix A, this function computes a*q where
	q is the product of 2*(n - 1) transformations

	  gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)

	and gv(i), gw(i) are givens rotations in the (i,n) plane which
	eliminate elements in the i-th and n-th planes, respectively.

	Q itself is not given, rather the information to recover the
	GV and GW rotations is supplied.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	07 April 2021

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Input:

	int M, the number of rows of A.

	int N, the number of columns of A.

	double A[M*N], the matrix to be postmultiplied
	by the orthogonal matrix Q described above.

	int LDA, a positive value not less than M
	which specifies the leading dimension of the array A.

	double V[N].  V(I) must contain the information necessary to
	recover the givens rotation GV(I) described above.

	double W[N], contains the information necessary to recover the
	Givens rotation gw(i) described above.

Output:

	double A[M*N], the value of A*Q.
*/
func R1mpyq(m, n int, a []float64, lda int, v, w []float64) {
	c_m := C.int(m)
	c_n := C.int(n)
	c_a := (*C.double)(unsafe.Pointer(&a[0]))
	c_lda := C.int(lda)
	c_v := (*C.double)(unsafe.Pointer(&v[0]))
	c_w := (*C.double)(unsafe.Pointer(&w[0]))

	C.r1mpyq(c_m, c_n, c_a, c_lda, c_v, c_w)
}

/*
	r1updt() updates the Q factor after a rank one update of the matrix.

Discussion:

	Given an M by N lower trapezoidal matrix S, an M-vector U,
	and an N-vector V, the problem is to determine an
	orthogonal matrix Q such that

	  (S + U*V') * Q

	is again lower trapezoidal.

	This function determines q as the product of 2*(n - 1) transformations

	  gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)

	where gv(i), gw(i) are givens rotations in the (i,n) plane
	which eliminate elements in the i-th and n-th planes,
	respectively.

	Q itself is not accumulated, rather the
	information to recover the gv, gw rotations is returned.

Licensing:

	This code is distributed under the GNU LGPL license.

Modified:

	07 April 2021

Author:

	Original FORTRAN77 version by Jorge More, Burt Garbow, Ken Hillstrom.
	C version by John Burkardt.

Reference:

	Jorge More, Burton Garbow, Kenneth Hillstrom,
	User Guide for MINPACK-1,
	Technical Report ANL-80-74,
	Argonne National Laboratory, 1980.

Input:

	int M, the number of rows of S.

	int N, the number of columns of S.  N must not exceed M.

	double S[LS], the lower trapezoidal matrix S stored by columns.

	int LS, a positive value not less than (N*(2*M-N+1))/2.

	double U[M], the vector U.

	double V[N], the vector v.

Output:

	double S[LS], the lower trapezoidal matrix produced as described above.

	double V[N], information necessary to recover the givens rotation gv(i)
	described above.

	double W[M], information necessary to recover the givens
	rotation gw(i) described above.

	bool SING, is true if any of the diagonal elements of s are zero.
*/
func R1updt(m, n int, s []float64, ls int, u, v, w []float64) bool {
	c_m := C.int(m)
	c_n := C.int(n)
	c_s := (*C.double)(unsafe.Pointer(&s[0]))
	c_ls := C.int(ls)
	c_u := (*C.double)(unsafe.Pointer(&u[0]))
	c_v := (*C.double)(unsafe.Pointer(&v[0]))
	c_w := (*C.double)(unsafe.Pointer(&w[0]))

	result := C.r1updt(c_m, c_n, c_s, c_ls, c_u, c_v, c_w)
	return bool(result)
}

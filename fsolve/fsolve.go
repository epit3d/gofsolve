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

// fdjac1 function
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

// fsolve function
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

// hybrd function
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

// Dogleg function
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

// Enorm function
func Enorm(n int, x []float64) float64 {
	c_n := C.int(n)
	c_x := (*C.double)(unsafe.Pointer(&x[0]))

	result := C.enorm(c_n, c_x)
	return float64(result)
}

// Qform function
func Qform(m, n int, q []float64, ldq int) {
	c_m := C.int(m)
	c_n := C.int(n)
	c_q := (*C.double)(unsafe.Pointer(&q[0]))
	c_ldq := C.int(ldq)

	C.qform(c_m, c_n, c_q, c_ldq)
}

// Qrfac function
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

// R1mpyq function
func R1mpyq(m, n int, a []float64, lda int, v, w []float64) {
	c_m := C.int(m)
	c_n := C.int(n)
	c_a := (*C.double)(unsafe.Pointer(&a[0]))
	c_lda := C.int(lda)
	c_v := (*C.double)(unsafe.Pointer(&v[0]))
	c_w := (*C.double)(unsafe.Pointer(&w[0]))

	C.r1mpyq(c_m, c_n, c_a, c_lda, c_v, c_w)
}

// R1updt function
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

package main

import (
	"log"
	"math"

	"github.com/epit3d/gofsolve/fsolve"
)

func main() {
	// dimension of the system
	n := 1
	x := make([]float64, n)
	fvec := make([]float64, n)

	// tolerance of the solution
	tol := 1e-5

	// initial guess
	x[0] = 0.0

	// define the system of equations
	cb := func(n int, x, fvec []float64) {
		// solve x = 5.0 + 0.8 * sin(x)

		e := 0.8
		m := 5.0

		fvec[0] = x[0] - m - e*math.Sin(x[0])
	}

	// solve the system
	info := fsolve.Fsolve(cb, n, x, fvec, tol)
	if info != 1 {
		panic("fsolve failed")
	}

	// print the solution (approximate solution should be 4.27523)
	log.Printf("x = %v\n", x[0])
}

# gofsolve - golang bind for fsolve library

## What is this?

```
fsolve() finds a zero of a system of N nonlinear functions in N variables
by a modification of the Powell hybrid method.
```

## Source

Source of library is [here](https://people.sc.fsu.edu/~jburkardt/c_src/fsolve/fsolve.html)

## Example

Solve equation $x = 5.0 + 0.8 \cdot \sin{x}$

```go
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

```

## Tests

Tests are taken from [here](https://people.sc.fsu.edu/~jburkardt/c_src/fsolve_test/fsolve_test.html). They are implemented at [fsolve_test.go](fsolve_test.go) and can be run with `go test -v`

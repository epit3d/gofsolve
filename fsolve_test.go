package gofsolve_test

import (
	"math"
	"testing"

	"github.com/epit3d/gofsolve/fsolve"
)

func TestFsolve(t *testing.T) {
	type args struct {
		cb   fsolve.Callback
		n    int
		x    []float64
		fvec []float64
		tol  float64
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			name: "test 1",
			want: 1, // ok, less than tol
			args: args{
				cb: func(n int, x, fvec []float64) {
					e := 0.8
					m := 5.0

					fvec[0] = x[0] - m - e*math.Sin(x[0])
				},
				n:    1,
				x:    []float64{0.0},
				fvec: []float64{0.0},
				tol:  1e-5,
			},
		},
		{
			name: "test 2",
			want: 1, // ok, less than tol
			args: args{
				cb: func(n int, x, fvec []float64) {
					fvec[0] = x[0]*x[0] - 10.0*x[0] + x[1]*x[1] + 8.0
					fvec[1] = x[0]*x[1]*x[1] + x[0] - 10.0*x[1] + 8.0
				},
				n:    2,
				x:    []float64{0.0, 0.0},
				fvec: []float64{0.0, 0.0},
				tol:  1e-5,
			},
		},
		{
			name: "test 3",
			want: 4, // iteration is not making good progress.
			args: args{
				n: 4,
				cb: func(n int, x, fvec []float64) {
					for i := 0; i < n; i++ {
						fvec[i] = math.Pow(x[i]-float64(i+1), 2)
					}
				},
				x:    make([]float64, 4),
				fvec: make([]float64, 4),
				tol:  1e-5,
			},
		},
		{
			name: "test 4",
			want: 1, // ok, less than tol
			args: args{
				n:    8,
				x:    make([]float64, 8),
				fvec: make([]float64, 8),
				cb: func(n int, x, fx []float64) {
					for i := 0; i < n; i++ {
						fx[i] = (3.0-2.0*x[i])*x[i] + 1.0

						if 0 < i {
							fx[i] = fx[i] - x[i-1]
						}

						if i < n-1 {
							fx[i] = fx[i] - 2.0*x[i+1]
						}
					}
				},
				tol: 1e-5,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := fsolve.Fsolve(tt.args.cb, tt.args.n, tt.args.x, tt.args.fvec, tt.args.tol); got != tt.want {
				t.Errorf("Fsolve() = %v, want %v", got, tt.want)
			}
		})
	}
}

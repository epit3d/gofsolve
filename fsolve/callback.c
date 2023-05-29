#include "callback.h"

// define callback
void callback(uintptr_t h, int n, double x[], double f[])
{
    // call go callback
    gocallback(h, n, x, f);
}
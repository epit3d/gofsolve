#ifndef CLIBRARY_H
#define CLIBRARY_H

#include <stddef.h>
#include <stdint.h>

typedef void *callbackfunc(int n, double x[], double f[]);

void callback(uintptr_t h, int n, double x[], double f[]);

extern void gocallback(uintptr_t h, int n, double x[], double f[]);

#endif
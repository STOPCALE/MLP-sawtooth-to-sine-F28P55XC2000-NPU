/* Diagnostic pre-flight: verify quant_params.h compiles with gcc and reads back correctly.
   Not part of the teaching material - just an environment smoke test. */
#include <stdio.h>
#include "quant_params.h"

int main(void) {
    printf("N_LAYERS = %d | sizeof(qlayer_t) = %d bytes\n",
           N_LAYERS, (int)sizeof(qlayer_t));
    for (int i = 0; i < N_LAYERS; i++) {
        const qlayer_t *L = &LAYERS[i];
        printf("layer %d: %2d -> %2d | S_in=%.9g z_in=%4d | S_out=%.9g z_out=%4d | "
               "W[0]=%4d SW[0]=%.9g B[0]=%d\n",
               i, L->in_dim, L->out_dim, L->S_in, L->z_in,
               L->S_out, L->z_out, (int)L->W_q[0], L->S_w[0], (int)L->b_int[0]);
    }
    return 0;
}

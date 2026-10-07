/* Tool (NOT teaching material): PC pre-flight for the MCU hand-written inference.
 *
 * Compiles hand_infer.c with the host compiler and runs its selftest, so we know
 * what the board MUST report before we ever flash it.
 *
 * Expected: maxerr ~ 3.14e-2  (PC PTQ int8 measurement)
 *           if it shows 1e-1 ~ 1.0 -> the rounding / magic number is broken.
 *
 * Build: gcc -O2 -Wall hand_infer.c test_hand_infer.c -o test_hand_infer.exe
 */
#include <math.h>
#include <stdio.h>

#include "hand_infer.h"

int main(void)
{
    int k;
    hand_selftest();

    printf("hand_selftest: maxerr = %.6e  at s = %+.4f\n",
           (double)g_st_maxerr, (double)g_st_maxerr_s);
    printf("expected     : maxerr ~ 3.14e-2  (PC PTQ measurement)\n\n");

    printf("spot check: s -> y_hand vs y_true = -sin(pi*s)\n");
    for (k = 0; k <= 8; k++)
    {
        float s = -1.0f + 0.25f * (float)k;
        float y = hand_infer_one(s);
        float t = -sinf(3.14159265f * s);
        printf("  s=%+.2f   y=%+.6f   true=%+.6f   err=%.2e\n",
               (double)s, (double)y, (double)t, (double)fabsf(y - t));
    }
    return 0;
}

// https://github.com/leerichardson/tree-swdft-2D

#pragma once

#include <math.h>
#include <complex>
#include <vector>
#include <stdexcept>


template<typename T, typename V>
std::vector<std::complex<T>> tswdft2d(const V *x, int n0, int n1, int N0, int N1) {

    static const T M_PI = 3.14159265358979323846;
    static const std::complex<T> I(0.0, 1.0);

    /*  Gets the 1D node index based on the 2D binary tree position used by the tswdft2d function */
    auto NODE = [n1](auto node0, auto node1) { return ((node0) * (n1)) + (node1); };

    /* Calculates the node index for a particular level of a tree */
    auto IMOD = [](auto node, auto level) { return(node) % (int)(pow((2), (level))); };

    /* Get index of tree inside the level_prev or level_cur arrays in the tswdft2d function */
    auto TREE_2D = [n0, n1, N1](auto x, auto y) { return (((x) * (N1)) + (y)) * ((n0) * (n1)); };

    /* Used to initialize level 0 of binary trees to the data  */
    auto X_LOOKUP = [N1](auto x, auto y) { return ((x) * (N1)) + (y); };

    /* Lookups used in 2D SWFFT and 2D SWDFT sliding window functions  */
    auto WINDOW_LOOKUP = [n1](auto x, auto y) { return ((x) * (n1)) + (y); };


    /* Initialize values integers for indexing, and complex-numbers for calculation inside the 6-loops */
    int shift0, shift1, level_min, level_max, nodes0, nodes1, min0, min1;
    std::complex<T> T_prev, T_cur, twid;

    /* Verify n0 and n1 are powers of two */
    if ((int)ceil(log2(n0)) != (int)floor(log2(n0))) {
        //printf("n0 is not a power of two! \n");
        //exit(-1);
        throw std::runtime_error("n0 is not a power of two!");
    }

    if ((int)ceil(log2(n1)) != (int)floor(log2(n1))) {
        //printf("n1 is not a power of two! \n");
        //exit(-1);
        throw std::runtime_error("n1 is not a power of two!");
    }

    /* Calculate the number of levels in the binary tree */
    int m0 = (int)log2(n0);
    int m1 = (int)log2(n1);

    /* Initialize twiddle-factor vectors for both dimensions */
    //std::complex<double> *current_twiddle;
    std::vector<std::complex<T>> twiddle0(n0);
    std::vector<std::complex<T>> twiddle1(n1);

    for (int init_tf0 = 0; init_tf0 < n0; init_tf0++) {
        twiddle0[init_tf0] = pow(exp((2 * M_PI * I) / static_cast<T>(n0)), -init_tf0);
    }

    for (int init_tf1 = 0; init_tf1 < n1; init_tf1++) {
        twiddle1[init_tf1] = pow(exp((2 * M_PI * I) / static_cast<T>(n1)), -init_tf1);
    }

    /* Allocate memory for two levels of tree data-structure */
    std::vector<std::complex<T>> level_prev(N0 * N1 * n0 * n1);
    std::vector<std::complex<T>> level_cur(N0 * N1 * n0 * n1);

    /* Initialize level 0 of the tree data-structure to the data  */
    for (int row = 0; row < N0; row++) {
        for (int col = 0; col < N1; col++) {
            level_prev[TREE_2D(row, col)] = x[X_LOOKUP(row, col)];
        }
    }

    /*
      The six loops. The first two loops (dim and levels) are over levels of the tree data-structure. When dim = 1,
      the algorithm corresponds to the row FFT potion of the 2D FFT algorithm, and when dim = 0 the algorithm
      corresponds to the column FFT portion.The next two loops (N0 and N1) are over trees in both directions,
      and the final two loops (nodes0 and nodes1) are over nodes of a particular level of a particular tree. The
      calculation inside the six-loops uses macros TREE_2D, NODE, and IMOD to access the correct indices of
      the tree data-structure.
    */
    for (int dim = 1; dim >= 0; dim--) {
        /* Get levels of trees corresponding to this dimension */
        level_min = (dim == 1) ? 1 : (m1 + 1);
        level_max = (dim == 1) ? m1 : (m1 + m0);

        /* Set the twiddle-factor vector corresponding to the current dimension */
        auto& current_twiddle = (dim == 1) ? twiddle1 : twiddle0;

        for (int level = level_min; level <= level_max; level++) {
            /* Number of nodes for this level of the binary tree in both direction*/
            nodes0 = (dim == 1) ? 1 : ((int)pow(2, level - m1));
            nodes1 = (dim == 1) ? ((int)pow(2, level)) : n1;

            /* Get shift distance between the current and previous tree with the repeated calculation */
            shift0 = (dim == 1) ? 0 : ((int)pow(2, m1 + m0 - level));
            shift1 = (dim == 1) ? ((int)pow(2, m1 - level)) : 0;

            /* Get first trees requiring this level for the row (min0) or column (min1) */
            min0 = (dim == 1) ? 0 : (n0 - shift0);
            min1 = (dim == 1) ? (n1 - shift1) : (n1 - 1);

            for (int p0 = min0; p0 < N0; p0++) {
                for (int p1 = min1; p1 < N1; p1++) {
                    for (int node0 = 0; node0 < nodes0; node0++) {
                        for (int node1 = 0; node1 < nodes1; node1++) {
                            T_prev = level_prev[TREE_2D(p0 - shift0, p1 - shift1) + IMOD(NODE(node0, node1), level - 1)];
                            T_cur = level_prev[TREE_2D(p0, p1) + IMOD(NODE(node0, node1), level - 1)];
                            twid = current_twiddle[(dim == 1) ? (node1 * shift1) : (node0 * shift0)];

                            level_cur[TREE_2D(p0, p1) + NODE(node0, node1)] = T_prev + (twid * T_cur);
                        }
                    }
                }
            }

            std::swap(level_prev, level_cur);
        }
    }

    /* Subset the final DFT coefficients into the array a in row-major order */
    int result_ind = 0;
    for (int p0_result = (n0 - 1); p0_result < N0; p0_result++) {
        for (int p1_result = (n1 - 1); p1_result < N1; p1_result++) {
            memmove(&level_cur[result_ind], &level_prev[TREE_2D(p0_result, p1_result)], sizeof(std::complex<T>) * (n0 * n1));
            result_ind += (n0 * n1);
        }
    }
    //delete[] level_prev;

    //a = realloc(level_cur, sizeof(double complex) * (N0 - n0 + 1) * (N1 - n1 + 1) * n0 * n1);
    //return a;
    return level_cur;
}

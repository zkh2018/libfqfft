/** @file
 *****************************************************************************

 Implementation of interfaces for auxiliary functions for the "recursive" evaluation domain.

 See recursive_domain_aux.hpp .

 *****************************************************************************
 * @author     This file is part of libfqfft, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef RECURSIVE_DOMAIN_AUX_TCC_
#define RECURSIVE_DOMAIN_AUX_TCC_

#include <algorithm>
#include <vector>
#include "prover_config.hpp"

#ifdef MULTICORE
#include <omp.h>
#endif

#include <libff/algebra/fields/field_utils.hpp>

#include <libfqfft/tools/exceptions.hpp>

#ifdef DEBUG
#include <libff/common/profiling.hpp>
#endif

namespace libfqfft {

static inline std::vector<fft_stage> get_stages(unsigned int n, const std::vector<unsigned int>& radixes)
{
    std::vector<fft_stage> stages;

    // Use the specified radices
    for (unsigned int i = 0; i < radixes.size(); i++)
    {
        n /= radixes[i];
        stages.push_back(fft_stage(radixes[i], n));
    }

    // Fill in the rest of the tree if needed
    unsigned int p = 4;
    while (n > 1)
    {
        while (n % p)
        {
            switch (p)
            {
                case 4: p = 2; break;
            }
        }
        n /= p;
        stages.push_back(fft_stage(p, n));
    };

    for (unsigned int i = 0; i < stages.size(); i++)
    {
        std::cout << "Stage " << i << ": " << stages[i].radix << ", " << stages[i].length << std::endl;
    }

    return stages;
}

template<typename FieldT>
static void butterfly_2(std::vector<FieldT>& out, const std::vector<FieldT>& twiddles, unsigned int stride, unsigned int stage_length, unsigned int out_offset)
{
    unsigned int out_offset2 = out_offset + stage_length;

    FieldT t = out[out_offset2];
    out[out_offset2] = out[out_offset] - t;
    out[out_offset] += t;
    out_offset2++;
    out_offset++;

    for (unsigned int k = 1; k < stage_length; k++)
    {
        FieldT t = twiddles[k] * out[out_offset2];
        out[out_offset2] = out[out_offset] - t;
        out[out_offset] += t;
        out_offset2++;
        out_offset++;
    }
}

template<typename FieldT>
static void butterfly_2_parallel(std::vector<FieldT>& out, const std::vector<FieldT>& twiddles, unsigned int stride, unsigned int stage_length, unsigned int out_offset, unsigned int num_threads)
{
    unsigned int out_offset2 = out_offset + stage_length;

    auto ranges = libsnark::get_cpu_ranges(0, stage_length, num_threads);

#ifdef MULTICORE
    #pragma omp parallel for num_threads(num_threads)
#endif
    for (unsigned int c = 0; c < ranges.size(); c++)
    {
        unsigned int offset1 = out_offset + ranges[c].first;
        unsigned int offset2 = out_offset2 + ranges[c].first;
        for (unsigned int k = ranges[c].first; k < ranges[c].second; k++)
        {
            FieldT t = twiddles[k] * out[offset2];
            out[offset2] = out[offset1] - t;
            out[offset1] += t;
            offset2++;
            offset1++;
        }
    }
}

template<typename FieldT>
static void butterfly_4(std::vector<FieldT>& out, const std::vector<FieldT>& twiddles, unsigned int stride, unsigned int stage_length, unsigned int out_offset)
{
    const FieldT j = twiddles[twiddles.size() - 1];
    unsigned int tw = 0;

    /* Case twiddle == one */
    {
        const unsigned i0  = out_offset;
        const unsigned i1  = out_offset + stage_length;
        const unsigned i2  = out_offset + stage_length*2;
        const unsigned i3  = out_offset + stage_length*3;

        const FieldT z0  = out[i0];
        const FieldT z1  = out[i1];
        const FieldT z2  = out[i2];
        const FieldT z3  = out[i3];

        const FieldT t1  = z0 + z2;
        const FieldT t2  = z1 + z3;
        const FieldT t3  = z0 - z2;
        const FieldT t4j = j * (z1 - z3);

        out[i0] = t1 + t2;
        out[i1] = t3 - t4j;
        out[i2] = t1 - t2;
        out[i3] = t3 + t4j;

        out_offset++;
        tw += 3;
    }

    for (unsigned int k = 1; k < stage_length; k++)
    {
        const unsigned i0  = out_offset;
        const unsigned i1  = out_offset + stage_length;
        const unsigned i2  = out_offset + stage_length*2;
        const unsigned i3  = out_offset + stage_length*3;

        const FieldT z0  = out[i0];
        const FieldT z1  = out[i1] * twiddles[tw];
        const FieldT z2  = out[i2] * twiddles[tw+1];
        const FieldT z3  = out[i3] * twiddles[tw+2];

        const FieldT t1  = z0 + z2;
        const FieldT t2  = z1 + z3;
        const FieldT t3  = z0 - z2;
        const FieldT t4j = j * (z1 - z3);

        out[i0] = t1 + t2;
        out[i1] = t3 - t4j;
        out[i2] = t1 - t2;
        out[i3] = t3 + t4j;

        out_offset++;
        tw += 3;
    }
}

template<typename FieldT>
static void butterfly_4_parallel(std::vector<FieldT>& out, const std::vector<FieldT>& twiddles, unsigned int stride, unsigned int stage_length, unsigned int out_offset, unsigned int num_threads)
{
    const FieldT j = twiddles[twiddles.size() - 1];
    const auto ranges = libsnark::get_cpu_ranges(0, stage_length, num_threads);

#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (unsigned int c = 0; c < ranges.size(); c++)
    {
        unsigned int offset = out_offset + ranges[c].first;
        unsigned int tw = 3 * ranges[c].first;
        for (unsigned int k = ranges[c].first; k < ranges[c].second; k++)
        {
            const unsigned i0  = offset;
            const unsigned i1  = offset + stage_length;
            const unsigned i2  = offset + stage_length*2;
            const unsigned i3  = offset + stage_length*3;

            const FieldT z0  = out[i0];
            const FieldT z1  = out[i1] * twiddles[tw];
            const FieldT z2  = out[i2] * twiddles[tw+1];
            const FieldT z3  = out[i3] * twiddles[tw+2];

            const FieldT t1  = z0 + z2;
            const FieldT t2  = z1 + z3;
            const FieldT t3  = z0 - z2;
            const FieldT t4j = j * (z1 - z3);

            out[i0] = t1 + t2;
            out[i1] = t3 - t4j;
            out[i2] = t1 - t2;
            out[i3] = t3 + t4j;

            offset++;
            tw += 3;
        }
    }
}


#ifdef USE_GPU
template<typename FieldT, bool smt>
void _recursive_FFT_inner(
    std::vector<FieldT>& in,
    std::vector<FieldT>& out,
    std::vector<std::vector<Info>>& infos,
    const std::vector<std::vector<FieldT>>& twiddles,
    const std::vector<fft_stage>& stages,
    unsigned int in_offset,
    unsigned int out_offset,
    unsigned int stride,
    unsigned int level,
    unsigned int num_threads)
{
    const unsigned int radix = stages[level].radix;
    const unsigned int stage_length = stages[level].length;

    if (num_threads > 1)
    {
        if (stage_length == 1)
        {
            for (unsigned int i = 0; i < radix; i++)
            {
                out[out_offset + i] = in[in_offset + i * stride];
            }
        }
        else
        {
#ifdef MULTICORE
            unsigned int num_threads_recursive = (num_threads >= radix) ? radix : num_threads;
            #pragma omp parallel for num_threads(num_threads_recursive)
#endif
            for (unsigned int i = 0; i < radix; i++)
            {
                unsigned int num_threads_in_recursion = (num_threads < radix) ? 1 : (num_threads + i) / radix;
                if (smt)
                {
#ifdef MULTICORE
                    omp_set_num_threads(num_threads_in_recursion * 2);
#endif
                }
                // std::cout << "Start thread on " << level << ": " << num_threads_in_recursion << std::endl;
                _recursive_FFT_inner<FieldT, smt>(in, out, infos, twiddles, stages, in_offset + i*stride, out_offset + i*stage_length, stride*radix, level+1, num_threads_in_recursion);
            }
        }

        Info info(in_offset, out_offset, radix, stage_length, stride, level);
        if(level > infos.size()){
            infos.resize(level+1);
        }
        infos[level].push_back(info);
        //infos.push_back(info);

        //switch (radix)
        //{
        //    case 2: butterfly_2(out, twiddles[level], stride, stage_length, out_offset); break;
        //    case 4: butterfly_4(out, twiddles[level], stride, stage_length, out_offset); break;
        //    default: std::cout << "error" << std::endl; assert(false);
        //}
    }
    else
    {
        if (stage_length == 1)
        {
            for (unsigned int i = 0; i < radix; i++)
            {
                //out[out_offset + i] = in[in_offset + i * stride];
            }
        }
        else
        {
            for (unsigned int i = 0; i < radix; i++)
            {
                _recursive_FFT_inner<FieldT, smt>(in, out, infos, twiddles, stages, in_offset + i*stride, out_offset + i*stage_length, stride*radix, level+1, num_threads);
            }
        }

        /*if (smt)
        {
            switch (radix)
            {
                case 2: butterfly_2_parallel(out, twiddles[level], stride, stage_length, out_offset, 2); break;
                case 4: butterfly_4_parallel(out, twiddles[level], stride, stage_length, out_offset, 2); break;
                default: std::cout << "error" << std::endl; assert(false);
            }
        }
        else*/
        {
            Info info(in_offset, out_offset, radix, stage_length, stride, level);
            if(level > infos.size()){
                infos.resize(level+1);
            }
            infos[level].push_back(info);
            //infos.push_back(info);
            //switch (radix)
            //{
            //    case 2: butterfly_2(out, twiddles[level], stride, stage_length, out_offset); break;
            //    case 4: butterfly_4(out, twiddles[level], stride, stage_length, out_offset); break;
            //    default: std::cout << "error" << std::endl; assert(false);
            //}
        }
    }
}

template<typename FieldT>
void _recursive_FFT(fft_data<FieldT>& data, std::vector<FieldT>& in, bool inverse, std::vector<std::vector<Info>>& infos, bool use_gpu)
{
    double t0 = omp_get_wtime();
#ifdef MULTICORE
    size_t num_threads = omp_get_max_threads();
    if (data.smt)
    {
        num_threads /= 2;
    }
#else
    size_t num_threads = 1;
#endif
    num_threads = 1;
    //std::vector<std::vector<Info>> infos;//(data.scratch.size());
    if (data.smt)
    {
        _recursive_FFT_inner<FieldT, true>(in, data.scratch, infos, inverse? data.iTwiddles : data.fTwiddles, data.stages, 0, 0, 1, 0, num_threads);
    }
    else
    {
        _recursive_FFT_inner<FieldT, false>(in, data.scratch, infos, inverse? data.iTwiddles : data.fTwiddles, data.stages, 0, 0, 1, 0, num_threads);
    }
    double t1 = omp_get_wtime();
    int n = 0;
    for(int i = 0; i < infos.size(); i++){
        n += infos[i].size();
    }

    if(!use_gpu){
        for(int i = infos.size()-1; i >= 0; i--){
#pragma omp parallel for
            for(int j = 0; j < infos[i].size(); j++){ 
                int in_offset = infos[i][j].in_offset;
                int out_offset = infos[i][j].out_offset;
                int level = i;//infos[i][j].level;
                int length = infos[i][j].length;
                int radix = infos[i][j].radix;
                int stride = infos[i][j].stride;
                if(length == 1){
                    for(int k = 0; k <  radix; k++){
                        data.scratch[out_offset + k] = in[in_offset + k * stride]; 
                    }
                }
                switch (infos[i][j].radix)
                {
                    case 2: butterfly_2(data.scratch, inverse ? data.iTwiddles[level] : data.fTwiddles[level], 0, length, out_offset); break;
                    case 4: butterfly_4(data.scratch, inverse ? data.iTwiddles[level] : data.fTwiddles[level], 0, length, out_offset); break;
                    default: std::cout << "error" << std::endl; assert(false);
                }
            }
        }
        assert(in.size() == data.scratch.size());
        std::swap(in, data.scratch);
    }
    //double t2 = omp_get_wtime();
    //printf("preprocess time %f, calc time %f\n", t1-t0, t2-t1);
}

#else
template<typename FieldT, bool smt>
void _recursive_FFT_inner(
    std::vector<FieldT>& in,
    std::vector<FieldT>& out,
    const std::vector<std::vector<FieldT>>& twiddles,
    const std::vector<fft_stage>& stages,
    unsigned int in_offset,
    unsigned int out_offset,
    unsigned int stride,
    unsigned int level,
    unsigned int num_threads)
{
    const unsigned int radix = stages[level].radix;
    const unsigned int stage_length = stages[level].length;

    if (num_threads > 1)
    {
        if (stage_length == 1)
        {
            for (unsigned int i = 0; i < radix; i++)
            {
                out[out_offset + i] = in[in_offset + i * stride];
            }
        }
        else
        {
#ifdef MULTICORE
            unsigned int num_threads_recursive = (num_threads >= radix) ? radix : num_threads;
            #pragma omp parallel for num_threads(num_threads_recursive)
#endif
            for (unsigned int i = 0; i < radix; i++)
            {
                unsigned int num_threads_in_recursion = (num_threads < radix) ? 1 : (num_threads + i) / radix;
                if (smt)
                {
#ifdef MULTICORE
                    omp_set_num_threads(num_threads_in_recursion * 2);
#endif
                }
                // std::cout << "Start thread on " << level << ": " << num_threads_in_recursion << std::endl;
                _recursive_FFT_inner<FieldT, smt>(in, out, twiddles, stages, in_offset + i*stride, out_offset + i*stage_length, stride*radix, level+1, num_threads_in_recursion);
            }
        }

        switch (radix)
        {
            case 2: butterfly_2_parallel(out, twiddles[level], stride, stage_length, out_offset, num_threads); break;
            case 4: butterfly_4_parallel(out, twiddles[level], stride, stage_length, out_offset, num_threads); break;
            default: std::cout << "error" << std::endl; assert(false);
        }
    }
    else
    {
        if (stage_length == 1)
        {
            for (unsigned int i = 0; i < radix; i++)
            {
                out[out_offset + i] = in[in_offset + i * stride];
            }
        }
        else
        {
            for (unsigned int i = 0; i < radix; i++)
            {
                _recursive_FFT_inner<FieldT, smt>(in, out, twiddles, stages, in_offset + i*stride, out_offset + i*stage_length, stride*radix, level+1, num_threads);
            }
        }

        /*if (smt)
        {
            switch (radix)
            {
                case 2: butterfly_2_parallel(out, twiddles[level], stride, stage_length, out_offset, 2); break;
                case 4: butterfly_4_parallel(out, twiddles[level], stride, stage_length, out_offset, 2); break;
                default: std::cout << "error" << std::endl; assert(false);
            }
        }
        else*/
        {
            switch (radix)
            {
                case 2: butterfly_2(out, twiddles[level], stride, stage_length, out_offset); break;
                case 4: butterfly_4(out, twiddles[level], stride, stage_length, out_offset); break;
                default: std::cout << "error" << std::endl; assert(false);
            }
        }
    }
}

template<typename FieldT>
void _recursive_FFT(fft_data<FieldT>& data, std::vector<FieldT>& in, bool inverse)
{
#ifdef MULTICORE
    size_t num_threads = omp_get_max_threads();
    if (data.smt)
    {
        num_threads /= 2;
    }
#else
    size_t num_threads = 1;
#endif
    if (data.smt)
    {
        _recursive_FFT_inner<FieldT, true>(in, data.scratch, inverse? data.iTwiddles : data.fTwiddles, data.stages, 0, 0, 1, 0, num_threads);
    }
    else
    {
        _recursive_FFT_inner<FieldT, false>(in, data.scratch, inverse? data.iTwiddles : data.fTwiddles, data.stages, 0, 0, 1, 0, num_threads);
    }
    assert(in.size() == data.scratch.size());
    std::swap(in, data.scratch);
}
#endif

template<typename FieldT>
void _multiply_by_coset_and_constant(unsigned int m, std::vector<FieldT> &a, const FieldT &g, const FieldT &c)
{
    auto ranges = libsnark::get_cpu_ranges(1, m);

    a[0] *= c;
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ranges.size(); ++j)
    {
        FieldT u = c * (g^ranges[j].first);
        for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
        {
            a[i] *= u;
            u *= g;
        }
    }

}

} // libfqfft

#endif // RECURSIVE_DOMAIN_AUX_TCC_

/** @file
 *****************************************************************************

 Implementation of interfaces for the "recursive" evaluation domain.

 See recursive_domain.hpp .

 *****************************************************************************
 * @author     This file is part of libfqfft, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef RECURSIVE_DOMAIN_TCC_
#define RECURSIVE_DOMAIN_TCC_

#include <libff/algebra/fields/field_utils.hpp>
#include <libff/common/double.hpp>
#include <libff/common/utils.hpp>

#include <libfqfft/evaluation_domain/domains/recursive_domain_aux.hpp>

namespace libfqfft {

template<typename FieldT>
recursive_domain<FieldT>::recursive_domain(const size_t m, const libsnark::Config& config) : evaluation_domain<FieldT>(m)
{
    if (m <= 1) throw InvalidSizeException("recursive(): expected m > 1");
    const size_t logm = libff::log2(m);

    if (!std::is_same<FieldT, libff::Double>::value)
    {
        if (logm > (FieldT::s)) throw DomainSizeException("recursive(): expected logm <= FieldT::s");
    }

    try { omega = libff::get_root_of_unity<FieldT>(m); }
    catch (const std::invalid_argument& e) { throw DomainSizeException(e.what()); }

    this->data.m = m;
    this->data.smt = config.smt;

    this->data.stages = get_stages(m, config.radixes);

    auto ranges = libsnark::get_cpu_ranges(0, m);

    this->data.scratch.resize(m+1);

    // Generate stage twiddles
    for (unsigned int inv = 0; inv < 2; inv++)
    {
        bool inverse = (inv == 0);
        const FieldT o = inverse ? omega.inverse() : omega;
        std::vector<std::vector<FieldT>>& stageTwiddles = inverse ? this->data.iTwiddles : this->data.fTwiddles;

        std::vector<FieldT>& twiddles = this->data.scratch;

        // Twiddles
        {
#ifdef MULTICORE
            #pragma omp parallel for
#endif
            for (size_t j = 0; j < ranges.size(); ++j)
            {
                const FieldT w_m = o;
                FieldT w = (w_m^ranges[j].first);
                for (unsigned int i = ranges[j].first; i < ranges[j].second; i++)
                {
                    twiddles[i] = w;
                    w *= w_m;
                }
            }
        }

        // Re-order twiddles for cache friendliness
        unsigned int n = this->data.stages.size();
        stageTwiddles.resize(n);
        for (unsigned int l = 0; l < n; l++)
        {
            const unsigned int radix = this->data.stages[l].radix;
            const unsigned int stage_length = this->data.stages[l].length;

            unsigned int numTwiddles = stage_length * (radix - 1);
            stageTwiddles[l].resize(numTwiddles + 1);

            // Set j
            stageTwiddles[l][numTwiddles] = twiddles[(twiddles.size() * 3) / 4];

            unsigned int stride = m / (stage_length * radix);
            std::vector<unsigned int> tws(radix - 1, 0);
            for (unsigned int i = 0; i < stage_length; i++)
            {
                for(unsigned int j = 0; j < radix-1; j++)
                {
                    stageTwiddles[l][i*(radix-1) + j] = twiddles[tws[j]];
                    tws[j] += (j+1)*stride;
                }
            }
        }
    }
}

template<typename FieldT>
void recursive_domain<FieldT>::FFT(std::vector<FieldT> &a)
{
    std::vector<std::vector<Info>> infos;
    _recursive_FFT(this->data, a, false, infos);
}

template<typename FieldT>
void recursive_domain<FieldT>::fft_internal(std::vector<FieldT> &a, std::vector<std::vector<Info>>& infos)
{
    iFFT_internal(a, infos, true);
}


template<typename FieldT>
void recursive_domain<FieldT>::iFFT(std::vector<FieldT> &a)
{
    //double t0 = omp_get_wtime();
    std::vector<std::vector<Info>> infos;
    iFFT_internal(a, infos);
    //double t1 = omp_get_wtime();

    const FieldT sconst = FieldT(this->m).inverse();
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < this->m; ++i)
    {
        a[i] *= sconst;
        //a[i].gpu_mul(sconst);
    }
    //double t2 = omp_get_wtime();
    //printf("internal time = %f, elementwise mul time = %f\n", t1-t0, t2-t1); 
}

template<typename FieldT>
void recursive_domain<FieldT>::iFFT_internal(std::vector<FieldT> &a, std::vector<std::vector<Info>>& infos, bool use_gpu)
{
    _recursive_FFT(this->data, a, true, infos, use_gpu);
}

template<typename FieldT>
void recursive_domain<FieldT>::cosetFFT(std::vector<FieldT> &a, const FieldT &g)
{
    _multiply_by_coset_and_constant(this->m, a, g);
    //FFT(a);
}

template<typename FieldT>
void recursive_domain<FieldT>::icosetFFT(std::vector<FieldT> &a, const FieldT &g)
{
    //std::vector<std::vector<Info>> infos;
    //iFFT_internal(a, infos);
    const FieldT sconst = FieldT(this->m).inverse();
    _multiply_by_coset_and_constant(this->m, a, g.inverse(), sconst);
}

template<typename FieldT>
std::vector<FieldT> recursive_domain<FieldT>::evaluate_all_lagrange_polynomials(const FieldT &t)
{
    return _basic_radix2_evaluate_all_lagrange_polynomials(this->m, t);
}

template<typename FieldT>
FieldT recursive_domain<FieldT>::get_domain_element(const size_t idx)
{
    return omega^idx;
}

template<typename FieldT>
FieldT recursive_domain<FieldT>::compute_vanishing_polynomial(const FieldT &t)
{
    return (t^this->m) - FieldT::one();
}

template<typename FieldT>
void recursive_domain<FieldT>::add_poly_Z(const FieldT &coeff, std::vector<FieldT> &H)
{
    if (H.size() != this->m+1) throw DomainSizeException("recursive: expected H.size() == this->m+1");

    H[this->m] += coeff;
    H[0] -= coeff;
}

template<typename FieldT>
void recursive_domain<FieldT>::divide_by_Z_on_coset(std::vector<FieldT> &P)
{
    const FieldT coset = FieldT::multiplicative_generator;
    const FieldT Z_inverse_at_coset = this->compute_vanishing_polynomial(coset).inverse();
#ifdef MULTICORE
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < this->m; ++i)
    {
        P[i] *= Z_inverse_at_coset;
    }
}

} // libfqfft

#endif // RECURSIVE_DOMAIN_TCC_

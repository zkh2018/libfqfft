/** @file
 *****************************************************************************

 Declaration of interfaces for the "recursive" evaluation domain.

 Rhe domain has size m = 2^k and consists of the m-th roots of unity.

 *****************************************************************************
 * @author     This file is part of libfqfft, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef RECURSIVE_DOMAIN_HPP_
#define RECURSIVE_DOMAIN_HPP_

#include <vector>

#include <libfqfft/evaluation_domain/evaluation_domain.hpp>
#include "recursive_domain_aux.hpp"
#include "prover_config.hpp"

namespace libfqfft {

template<typename FieldT>
class recursive_domain : public evaluation_domain<FieldT> {
public:

    FieldT omega;

    recursive_domain(const size_t m, const libsnark::Config& config = libsnark::Config());

    void FFT(std::vector<FieldT> &a);
    void iFFT(std::vector<FieldT> &a);
    void cosetFFT(std::vector<FieldT> &a, const FieldT &g);
    void icosetFFT(std::vector<FieldT> &a, const FieldT &g);
    std::vector<FieldT> evaluate_all_lagrange_polynomials(const FieldT &t);
    FieldT get_domain_element(const size_t idx);
    FieldT compute_vanishing_polynomial(const FieldT &t);
    void add_poly_Z(const FieldT &coeff, std::vector<FieldT> &H);
    void divide_by_Z_on_coset(std::vector<FieldT> &P);

public:

    void iFFT_internal(std::vector<FieldT> &a);

    fft_data<FieldT> data;
};

} // libfqfft

#include <libfqfft/evaluation_domain/domains/recursive_domain.tcc>

#endif // RECURSIVE_DOMAIN_HPP_

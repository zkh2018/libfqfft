/** @file
 *****************************************************************************

 Declaration of interfaces for auxiliary functions for the "basic radix-2" evaluation domain.

 These functions compute the radix-2 FFT (in single- or multi-thread mode) and,
 also compute Lagrange coefficients.

 *****************************************************************************
 * @author     This file is part of libfqfft, developed by SCIPR Lab
 *             and contributors (see AUTHORS).
 * @copyright  MIT license (see LICENSE file)
 *****************************************************************************/

#ifndef RECURSIVE_DOMAIN_AUX_HPP_
#define RECURSIVE_DOMAIN_AUX_HPP_

#include <vector>

namespace libfqfft {


/**
 * Compute the FFT of the vector a over the set S={omega^{0},...,omega^{m-1}}.
 */
template<typename FieldT>
void _recursive_FFT(fft_data<FieldT>& data, std::vector<FieldT>& in, bool inverse, std::vector<std::vector<Info>>& infos, bool use_gpu=false);

/**
 * Translate the vector a to a coset defined by g + extra constant multiplication.
 */
template<typename FieldT>
void _multiply_by_coset_and_constant(unsigned int m, std::vector<FieldT> &a, const FieldT &g, const FieldT &c = FieldT::one());


} // libfqfft

#include <libfqfft/evaluation_domain/domains/recursive_domain_aux.tcc>

#endif // RECURSIVE_DOMAIN_AUX_HPP_

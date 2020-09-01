#include <cmath>
#include <iostream>
#include "Kernels.h"
#include "Error.h"
#include "Dist.h"

using namespace CovidSim::TBD1;

void KernelLookup::setup(float longest_distance)
{
	size_t size = (size_t)size_ + 1;
	lookup_.resize(size);
	hi_res_.resize(size);
	delta_ = longest_distance / size_;
}

void KernelLookup::init(float norm, KernelStruct& kernel)
{
	float (KernelStruct::*fp)(float) const;

	if (kernel.type_ == 1)
		fp = &KernelStruct::exponential;
	else if (kernel.type_ == 2)
		fp = &KernelStruct::power;
	else if (kernel.type_ == 3)
		fp = &KernelStruct::gaussian;
	else if (kernel.type_ == 4)
		fp = &KernelStruct::step;
	else if (kernel.type_ == 5)
		fp = &KernelStruct::power_b;
	else if (kernel.type_ == 6)
		fp = &KernelStruct::power_us;
	else if (kernel.type_ == 7)
		fp = &KernelStruct::power_exp;
	else
		ERR_CRITICAL_FMT("Unknown kernel type %d.\n", kernel.type_);

#pragma omp parallel for schedule(static,500) default(none) \
		shared(kernel, fp, norm)
	for (int i = 0; i <= size_; i++)
	{
		lookup_[i] = (kernel.*fp)(i * delta_) / norm;
		hi_res_[i] = (kernel.*fp)(i * delta_ / expansion_factor_) / norm;
	}
}

/// \todo Move this to somewhere more appropriate
void KernelLookup::init(const KernelLookup& lookup, Cell **cell_lookup, int cell_lookup_size)
{
#pragma omp parallel for schedule(static,500) default(none) \
		shared(lookup, cell_lookup, cell_lookup_size)
	for (int i = 0; i < cell_lookup_size; i++)
	{
		Cell *l = cell_lookup[i];
		l->tot_prob = 0.0f;
		for (int j = 0; j < cell_lookup_size; j++)
		{
			Cell *m = cell_lookup[j];
			l->max_trans[j] = (float)lookup.num(dist2_cc_min(l, m));
			l->tot_prob += l->max_trans[j] * m->n;
		}
	}
}

//// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// ****
//// **** KERNEL DEFINITIONS

float KernelStruct::exponential(float r2) const
{
	return exp(-sqrt(r2) / scale_);
}

float KernelStruct::power(float r2) const
{
	float t = -shape_ * log(sqrt(r2) / scale_ + 1.0);
	return (t < -690.0) ? 0.0 : exp(t);
}

float KernelStruct::power_b(float r2) const
{
	float t = 0.5 * shape_ * log(r2 / (scale_ * scale_));
	return (t > 690.0) ? 0.0 : (1.0 / (exp(t) + 1.0));
}

float KernelStruct::power_us(float r2) const
{
	float t = log(sqrt(r2) / scale_ + 1.0);
	return (t < -690.0) ? 0.0 : (exp(-shape_ * t) + p3_ * exp(-p4_ * t)) / (1.0 + p3_);
}

float KernelStruct::gaussian(float r2) const
{
	return exp(-r2 / (scale_ * scale_));
}

float KernelStruct::step(float r2) const
{
	return (r2 > scale_ * scale_) ? 0.0 : 1.0;
}

float KernelStruct::power_exp(float r2) const
{
	float d = sqrt(r2);
	float t = -shape_ * log(d / scale_ + 1.0);
	return (t < -690.0) ? 0.0 : exp(t - pow(d / p3_, p4_));
}

float KernelLookup::num(float r2) const
{
	float t = r2 / delta_;
	if (t > size_)
	{
		fprintf(stderr, "** %g  %g  %g**\n", r2, delta_, t);
		ERR_CRITICAL("r too large in NumKernel\n");
	}

	float s = t * expansion_factor_;
	if (s < size_)
	{
		t = s - floor(s);
		t = (1.0 - t) * hi_res_[(int)s] + t * hi_res_[(int)(s + 1.0)];
	}
	else
	{
		s = t - floor(t);
		t = (1.0 - s) * lookup_[(int)t] + s * lookup_[(int)(t + 1.0)];
	}
	return t;
}

#pragma once

#include "Constants.h"

class InverseCdf
{
	float cdf_values_[CDF_RES + 1];

public:

	void set_neg_log(float start_value);

	void assign_exponent();

	void assign_exponent(float value);

	unsigned short int choose(float Mean, int tn, float timesteps_per_day);

	// Getter
	float* get_values()
	{
		return cdf_values_;
	}

	// Overloading [] operator to access elements in array style 
	float& operator[](int i)
	{
		return cdf_values_[i];
	}
};





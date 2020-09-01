#include <cstdlib>
#include <cmath>

#include "Constants.h"
#include "Dist.h"
#include "Param.h"

#include "Model.h"

float sinx[DEGREES_PER_TURN + 1], cosx[DEGREES_PER_TURN + 1], asin2sqx[1001];

//// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// ****
//// **** DISTANCE FUNCTIONS (return distance-squared, which is input for every Kernel function)
//// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// **** //// ****

float periodic_xy(float x, float y) {
	if (P.DoPeriodicBoundaries)
	{
		if (x > P.in_degrees_.width * 0.5) x = P.in_degrees_.width - x;
		if (y > P.in_degrees_.height * 0.5) y = P.in_degrees_.height - y;
	}
	return x * x + y * y;
}

float dist2UTM(float x1, float y1, float x2, float y2)
{
	float x, y, cy1, cy2, yt, xi, yi;

	x = fabs(x1 - x2) / 2;
	y = fabs(y1 - y2) / 2;
	xi = floor(x);
	yi = floor(y);
	x -= xi;
	y -= yi;
	x = (1 - x) * sinx[(int)xi] + x * sinx[((int)xi) + 1];
	y = (1 - y) * sinx[(int)yi] + y * sinx[((int)yi) + 1];
	yt = fabs(y1 + P.SpatialBoundingBox.bottom_left().y);
	yi = floor(yt);
	cy1 = yt - yi;
	cy1 = (1 - cy1) * cosx[((int)yi)] + cy1 * cosx[((int)yi) + 1];
	yt = fabs(y2 + P.SpatialBoundingBox.bottom_left().y);
	yi = floor(yt);
	cy2 = yt - yi;
	cy2 = (1 - cy2) * cosx[((int)yi)] + cy2 * cosx[((int)yi) + 1];
	x = fabs(1000 * (y * y + x * x * cy1 * cy2));
	xi = floor(x);
	x -= xi;
	y = (1 - x) * asin2sqx[((int)xi)] + x * asin2sqx[((int)xi) + 1];
	return 4 * EARTHRADIUS * EARTHRADIUS * y;
}
float dist2(Person* a, Person* b)
{
	float x, y;

	if (P.DoUTM_coords)
		return dist2UTM(Households[a->hh].loc.x, Households[a->hh].loc.y, Households[b->hh].loc.x, Households[b->hh].loc.y);
	else
	{
		x = fabs(Households[a->hh].loc.x - Households[b->hh].loc.x);
		y = fabs(Households[a->hh].loc.y - Households[b->hh].loc.y);
		return periodic_xy(x, y);
	}
}
float dist2_cc(Cell* a, Cell* b)
{
	float x, y;
	int l, m;

	l = (int)(a - Cells);
	m = (int)(b - Cells);
	if (P.DoUTM_coords)
		return dist2UTM(P.in_cells_.width * fabs((float)(l / P.nch)), P.in_cells_.height * fabs((float)(l % P.nch)),
			P.in_cells_.width * fabs((float)(m / P.nch)), P.in_cells_.height * fabs((float)(m % P.nch)));
	else
	{
		x = P.in_cells_.width * fabs((float)(l / P.nch - m / P.nch));
		y = P.in_cells_.height * fabs((float)(l % P.nch - m % P.nch));
		return periodic_xy(x, y);
	}
}
float dist2_cc_min(Cell* a, Cell* b)
{
	float x, y;
	int l, m, i, j;

	l = (int)(a - Cells);
	m = (int)(b - Cells);
	i = l; j = m;
	if (P.DoUTM_coords)
	{
		if (P.in_cells_.width * ((float)abs(m / P.nch - l / P.nch)) > PI)
		{
			if (m / P.nch > l / P.nch)
				j += P.nch;
			else if (m / P.nch < l / P.nch)
				i += P.nch;
		}
		else
		{
			if (m / P.nch > l / P.nch)
				i += P.nch;
			else if (m / P.nch < l / P.nch)
				j += P.nch;
		}
		if (m % P.nch > l % P.nch)
			i++;
		else if (m % P.nch < l % P.nch)
			j++;
		return dist2UTM(P.in_cells_.width * fabs((float)(i / P.nch)), P.in_cells_.height * fabs((float)(i % P.nch)),
			P.in_cells_.width * fabs((float)(j / P.nch)), P.in_cells_.height * fabs((float)(j % P.nch)));
	}
	else
	{
		if ((P.DoPeriodicBoundaries) && (P.in_cells_.width * ((float)abs(m / P.nch - l / P.nch)) > P.in_degrees_.width * 0.5))
		{
			if (m / P.nch > l / P.nch)
				j += P.nch;
			else if (m / P.nch < l / P.nch)
				i += P.nch;
		}
		else
		{
			if (m / P.nch > l / P.nch)
				i += P.nch;
			else if (m / P.nch < l / P.nch)
				j += P.nch;
		}
		if ((P.DoPeriodicBoundaries) && (P.in_degrees_.height * ((float)abs(m % P.nch - l % P.nch)) > P.in_degrees_.height * 0.5))
		{
			if (m % P.nch > l % P.nch)
				j++;
			else if (m % P.nch < l % P.nch)
				i++;
		}
		else
		{
			if (m % P.nch > l % P.nch)
				i++;
			else if (m % P.nch < l % P.nch)
				j++;
		}
		x = P.in_cells_.width * fabs((float)(i / P.nch - j / P.nch));
		y = P.in_cells_.height * fabs((float)(i % P.nch - j % P.nch));
		return periodic_xy(x, y);
	}
}
float dist2_mm(Microcell* a, Microcell* b)
{
	float x, y;
	int l, m;

	l = (int)(a - Mcells);
	m = (int)(b - Mcells);
	if (P.DoUTM_coords)
		return dist2UTM(P.in_microcells_.width * fabs((float)(l / P.total_microcells_high_)), P.in_microcells_.height * fabs((float)(l % P.total_microcells_high_)),
			P.in_microcells_.width * fabs((float)(m / P.total_microcells_high_)), P.in_microcells_.height * fabs((float)(m % P.total_microcells_high_)));
	else
	{
		x = P.in_microcells_.width * fabs((float)(l / P.total_microcells_high_ - m / P.total_microcells_high_));
		y = P.in_microcells_.height * fabs((float)(l % P.total_microcells_high_ - m % P.total_microcells_high_));
		return periodic_xy(x, y);
	}
}

float dist2_raw(float ax, float ay, float bx, float by)
{
	float x, y;

	if (P.DoUTM_coords)
		return dist2UTM(ax, ay, bx, by);
	else
	{
		x = fabs(ax - bx);
		y = fabs(ay - by);
		return periodic_xy(x, y);
	}
}

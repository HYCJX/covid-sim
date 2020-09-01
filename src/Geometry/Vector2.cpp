#include "Vector2.h"

using namespace Geometry;

Vector2f Geometry::operator*(const Vector2f &left, const Vector2d &right){
	return left * Vector2f(right);
}
Vector2f Geometry::operator*(const Vector2d &left, const Vector2f &right){
	return Vector2f(left) * right;
}

Vector2f Geometry::operator-(const Vector2f &left, const Vector2i &right){
	return left - Vector2f(right);
}
Vector2f Geometry::operator-(const Vector2i &left, const Vector2f &right){
	return Vector2f(left) - right;
}

// Package vector provides basic vector mathematics for Euclidean n-space
package vector

import (
	"fmt"
	"math"
	"sort"
)

// Vector is an element of a vector space in R^n
type Vector []float64

// DeeplyEqual tests whether two vectors are element-wise equal within the smallest possible tolerance
func DeeplyEqual(u Vector, v Vector) bool {
	return NearlyEqual(u, v, math.SmallestNonzeroFloat64)
}

// NearlyEqual tests whether two vectors are element-wise equal within the tolerance provided
func NearlyEqual(u Vector, v Vector, eps float64) bool {
	if len(u) != len(v) {
		return false
	}
	for i := range u {
		if !NearlyEqualValues(u[i], v[i], eps) {
			return false
		}
	}
	return true
}

// Close returns true if the distance between two vectors is less than or equal to the provided value
func Close(u Vector, v Vector, eps float64) bool {
	if Distance(u, v) <= eps {
		return true
	}
	return false
}

// Add defines vector addition
func Add(vecs ...Vector) Vector {
	set := Set(vecs)
	new, _ := set.Sum()
	return new
}

// Scale defines scalar multiplication on a vector
func Scale(v Vector, n float64) (new Vector) {
	new = make(Vector, len(v))
	for dim, el := range v {
		new[dim] = el * n
	}
	return new
}

// Distance returns the distance between two vectors
func Distance(u Vector, v Vector) (d float64) {
	return Add(u, Scale(v, -1)).Mag()
}

// Dot implements the InnerProduct fot real numbers
// the InnerProduct may eventually extend to include complex numbers
func Dot(u Vector, v Vector) (sum float64) {
	return InnerProduct(u, v)
}

// InnerProduct returns the inner product of two vectors
func InnerProduct(u Vector, v Vector) (sum float64) {
	// handle length mismatch by sorting the two vectors
	set := Set{u, v}
	sort.Sort(set)
	// assuming zero values for any 'missing' elements implies
	// it is sufficient to iterate over the smallest vector
	for i := range set[0] {
		sum += set[0][i] * set[1][i]
	}
	return sum
}

// Cross is the cross product of two vectors in three dimensions
func Cross(u Vector, v Vector) (Vector, error) {
	if len(u) != 3 || len(v) != 3 {
		return nil, &DimensionError{fmt.Errorf("Cross(%v, %v): vectors must be in 3 dimensions", u, v)}
	}
	cross := make(Vector, 3)
	cross[0] = u[1]*v[2] - u[2]*v[1]
	cross[1] = u[2]*v[0] - u[0]*v[2]
	cross[2] = u[0]*v[1] - u[1]*v[0]
	return cross, nil
}

// Len is the number of elements in a vector
func (v Vector) Len() int {
	return len(v)
}

// Mag calculates a vector's magnitude (2-norm)
func (v Vector) Mag() float64 {
	return math.Sqrt(InnerProduct(v, v))
}

// Unit returns a vector's unit vector representation
func (v Vector) Unit() Vector {
	mag := v.Mag()
	if mag == 0 {
		return nil
	}
	return Scale(v, 1/v.Mag())
}

// Set of vector pointers on which to perform a mathematical operation
// implements sort.Interface by vector length
type Set []Vector

// Len is the length of Set
func (set Set) Len() int {
	return len(set)
}

// Swap switches the position of two elements in a Set
func (set Set) Swap(i, j int) {
	set[i], set[j] = set[j], set[i]
}

// Less returns true if the i-th vector has fewer elements than the j-th vector in Set
func (set Set) Less(i, j int) bool {
	return len(set[i]) < len(set[j])
}

// Sum defines vector addition on a Set
// addition of vectors of differing lengths is possible by assuming zero values for missing elements
func (set Set) Sum() (sum Vector, err error) {
	// sort vectors from most to least number of elements
	sort.Sort(sort.Reverse(set))
	// since the vector of greatest length is the first element of the set,
	// element-wise addition can be perfomed with subsequent vectors
	sum = make(Vector, len(set[0]))
	copy(sum, set[0])
	for i := 1; i < len(set); i++ {
		v := set[i]
		for dim, el := range v {
			sum[dim] += el
		}
	}
	return sum, nil
}

// NearlyEqualValues returns true if the percent difference between two numbers is less than or equal to the provided value
func NearlyEqualValues(x, y, eps float64) bool {
	if x == y {
		return true
	}
	absX := math.Abs(x)
	absY := math.Abs(y)
	absDiff := math.Abs(x - y)
	if absDiff <= math.SmallestNonzeroFloat64 {
		return true
	}
	avgAbs := (absX + absY) / 2
	return absDiff/avgAbs <= eps
}

// DimensionError is used to identify when a vector operation that requires Vectors
// to have the same length is called with one or more vectors of differing lengths
type DimensionError struct {
	error
}

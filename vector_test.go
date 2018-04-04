package vector_test

import (
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/alexcooperse/vector"
)

func typeof(v interface{}) string {
	return fmt.Sprintf("%T", v)
}

var set = vector.Set{
	vector.Vector{},
	vector.Vector{1},
	vector.Vector{2, 3},
	vector.Vector{4, 5, 6},
	vector.Vector{7, 8},
	vector.Vector{9},
}

func TestDeeplyEqual(t *testing.T) {
	cases := []struct {
		u, v vector.Vector
		want bool
	}{
		{set[0], vector.Vector{}, true},
		{set[0], vector.Vector{0}, false},
		{set[1], vector.Vector{1}, true},
		{set[2], vector.Vector{2, 3}, true},
		{set[2], vector.Vector{3, 2}, false},
	}

	for _, c := range cases {
		got := vector.DeeplyEqual(c.u, c.v)
		if got != c.want {
			t.Errorf("DeeplyEqual(%v, %v) == %t, want %t", c.u, c.v, got, c.want)
		}
	}
}

func TestNearlyEqual(t *testing.T) {
	cases := []struct {
		u, v vector.Vector
		eps  float64
		want bool
	}{
		{set[1], vector.Vector{1.1}, .1, true},
		{set[1], vector.Vector{0.09}, .1, false},
	}

	for _, c := range cases {
		got := vector.NearlyEqual(c.u, c.v, c.eps)
		if got != c.want {
			t.Errorf("NearlyEqual(%v, %v, %v) == %v, want %v", c.u, c.v, c.eps, got, c.want)
		}
	}
}

func TestClose(t *testing.T) {
	cases := []struct {
		u, v vector.Vector
		eps  float64
		want bool
	}{
		{set[0], vector.Vector{}, 0, true},
		{set[0], vector.Vector{0}, 0, true},
		{set[1], vector.Vector{1.00000001}, 0.00000001, true},
		{set[1], vector.Vector{1.00000001}, 0.000000009, false},
		{set[2], vector.Vector{2, 3}, 1, true},
		{set[2], vector.Vector{3, 2}, 1, false},
	}

	for _, c := range cases {
		got := vector.Close(c.u, c.v, c.eps)
		if got != c.want {
			t.Errorf("Close(%v, %v) == %t, want %t", c.u, c.v, got, c.want)
		}
	}
}

func TestAdd(t *testing.T) {
	cases := []struct {
		s    []vector.Vector
		want vector.Vector
	}{
		{ // empty vector is a zero vector
			[]vector.Vector{
				set[2],
				vector.Vector{},
			},
			set[2],
		},
		{ // Vectors of all zeros can be zero vectors iff their length
			// is less or equal to that of the longest non-zero vector
			// in the set of vectors being added.
			[]vector.Vector{
				set[2],
				vector.Vector{0, 0},
			},
			set[2],
		},
		{ // Vectors of all zeros with length greater than longest non-zero
			// vector being added ARE NOT zero vectors.  Addition of such Vectors
			// effectively change the vector's dimension
			[]vector.Vector{
				set[2],
				vector.Vector{0, 0, 0},
			},
			vector.Vector{2, 3, 0},
		},
		{
			// negatives
			[]vector.Vector{
				vector.Vector{1, 2},
				vector.Vector{-1, -2},
			},
			vector.Vector{0, 0},
		},
	}

	for _, c := range cases {
		got := vector.Add(c.s...)
		if !vector.DeeplyEqual(got, c.want) {
			t.Errorf("Add(%v) == %v, want %v", c.s, got, c.want)
		}
	}
}

func TestScale(t *testing.T) {
	cases := []struct {
		v    vector.Vector
		n    float64
		want vector.Vector
	}{
		{set[0], 5, set[0]}, // zero vectors do not scale
		{set[2], 1, set[2]}, // multiplicative identity
		{set[3], -1, vector.Vector{-4, -5, -6}},
		{set[3], 2.5, vector.Vector{10, 12.5, 15}},
		{set[4], 0, vector.Vector{0, 0}},
		{set[5], -7, vector.Vector{-63}},
	}

	for _, c := range cases {
		got := vector.Scale(c.v, c.n)
		if !vector.DeeplyEqual(got, c.want) {
			t.Errorf("%v.Scale(%v) == %v, want %v", c.v, c.n, got, c.want)
		}
	}
}

func TestVectorProperties(t *testing.T) {
	cases := []struct {
		p        string
		lhs, rhs vector.Vector
	}{
		{
			"associativity",
			vector.Scale(vector.Scale(set[3], 7), 5),
			vector.Scale(set[3], 7*5),
		},
		{
			"scalar distributivity",
			vector.Scale(vector.Add(set[2], set[4]), 5),
			vector.Add(vector.Scale(set[2], 5), vector.Scale(set[4], 5)),
		},
		{
			"vector distributivity",
			vector.Scale(set[4], 5+7),
			vector.Add(vector.Scale(set[4], 5), vector.Scale(set[4], 7)),
		},
	}

	for _, c := range cases {
		if !vector.DeeplyEqual(c.lhs, c.rhs) {
			t.Errorf("%v: %v != %v", c.p, c.lhs, c.rhs)
		}
	}
}

func TestDistance(t *testing.T) {
	cases := []struct {
		u, v vector.Vector
		want float64
	}{
		{set[0], set[2], math.Sqrt(13)},
		{set[2], set[4], math.Sqrt(50)},
	}

	for _, c := range cases {
		got := vector.Distance(c.u, c.v)
		if got != c.want {
			t.Errorf("Distance(%v, %v) == %v, want %v", c.u, c.v, got, c.want)
		}
	}
}

func TestDotProduct(t *testing.T) {
	cases := []struct {
		u, v vector.Vector
		want float64
	}{
		{set[0], set[0], 0},
		{set[1], set[2], 2},
		{set[2], set[1], 2},
	}

	for _, c := range cases {
		got := vector.Dot(c.u, c.v)
		if got != c.want {
			t.Errorf("Dot(%v, %v) == %v, want %v", c.u, c.v, got, c.want)
		}
	}
}

func TestInnerProductProperties(t *testing.T) {
	cases := []struct {
		p        string
		lhs, rhs float64
	}{
		{
			"distributivity",
			vector.InnerProduct(set[2], vector.Add(set[3], set[4])),
			vector.InnerProduct(set[2], set[3]) + vector.InnerProduct(set[2], set[4]),
		},
		{
			"multiplicativity",
			vector.InnerProduct(vector.Scale(set[2], 7), set[4]),
			7 * vector.InnerProduct(set[2], set[4]),
		},
	}

	for _, c := range cases {
		if c.lhs != c.rhs {
			t.Errorf("%v: %v != %v", c.p, c.lhs, c.rhs)
		}
	}
}

func TestCross(t *testing.T) {
	cases := []struct {
		u, v, want vector.Vector
		err        interface{}
	}{
		{vector.Vector{1, 0, 0}, vector.Vector{0, 1, 0}, vector.Vector{0, 0, 1}, nil},
		{vector.Vector{0, 1, 0}, vector.Vector{0, 0, 1}, vector.Vector{1, 0, 0}, nil},
		{vector.Vector{0, 0, 1}, vector.Vector{1, 0, 0}, vector.Vector{0, 1, 0}, nil},
		{vector.Vector{1, 0, 0}, vector.Vector{0, 1}, nil, &vector.DimensionError{}},
		{vector.Vector{1, 0}, vector.Vector{0, 0, 1}, nil, &vector.DimensionError{}},
		{set[3], set[3], vector.Vector{0, 0, 0}, nil},
		{vector.Vector{2, 3, 0}, vector.Vector{7, 8, 0}, vector.Vector{0, 0, -5}, nil},
		{vector.Vector{2, 3, 0}, set[3], vector.Vector{18, -12, -2}, nil},
		{vector.Vector{7, 8, 0}, set[3], vector.Vector{48, -42, 3}, nil},
	}

	for _, c := range cases {
		got, err := vector.Cross(c.u, c.v)
		if !vector.DeeplyEqual(got, c.want) {
			t.Errorf("Cross(%v, %v) == %v, want %v", c.u, c.v, got, c.want)
		}
		if typeof(err) != typeof(c.err) {
			t.Errorf("Cross(%v, %v): typeof(err) == %v, want %v", c.u, c.v, typeof(err), typeof(c.err))
		}
	}
}

func TestVectorLen(t *testing.T) {
	cases := []struct {
		v    vector.Vector
		want int
	}{
		{set[0], 0},
		{set[1], 1},
		{set[2], 2},
		{set[3], 3},
		{set[4], 2},
		{set[5], 1},
	}

	for _, c := range cases {
		got := c.v.Len()
		if got != c.want {
			t.Errorf("%v.Len() == %v, want %v", c.v, got, c.want)
		}
	}
}

func TestMag(t *testing.T) {
	cases := []struct {
		v    vector.Vector
		want float64
	}{
		{set[0], 0},
		{set[1], 1},
		{set[2], math.Sqrt(13)},
		{set[3], math.Sqrt(77)},
		{set[4], math.Sqrt(113)},
		{set[5], 9},
	}

	for _, c := range cases {
		got := c.v.Mag()
		if got != c.want {
			t.Errorf("%v.Mag() == %v, want %v", c.v, got, c.want)
		}
	}
}

func TestUnit(t *testing.T) {
	cases := []struct {
		v, want vector.Vector
	}{
		{set[0], set[0]},
		{set[1], set[1]},
		{set[2], vector.Vector{2 / math.Sqrt(13), 3 / math.Sqrt(13)}},
		{set[3], vector.Vector{4 / math.Sqrt(77), 5 / math.Sqrt(77), 6 / math.Sqrt(77)}},
		{set[4], vector.Vector{7 / math.Sqrt(113), 8 / math.Sqrt(113)}},
		{set[5], vector.Vector{1}},
		{vector.Vector{0, 0}, nil},
	}

	for _, c := range cases {
		got := c.v.Unit()
		if !vector.DeeplyEqual(got, c.want) {
			t.Errorf("%v.Unit() == %v, want %v", c.v, got, c.want)
		}
	}
}

func TestInequalities(t *testing.T) {
	cases := []struct {
		f, n     string
		lhs, rhs float64
	}{
		{
			"Mag",
			"triangle",
			vector.Add(set[3], set[4]).Mag(),
			set[3].Mag() + set[4].Mag(),
		},
		{
			"Distance",
			"triangle",
			vector.Distance(set[2], set[3]),
			vector.Distance(set[2], set[4]) + vector.Distance(set[4], set[3]),
		},
		{
			"InnerProduct",
			"Cauchy-Schwarz",
			vector.InnerProduct(set[3], set[4]),
			set[3].Mag() * set[4].Mag(),
		},
	}

	for _, c := range cases {
		if c.lhs > c.rhs {
			t.Errorf("%v inequality for %v: %v > %v", c.n, c.f, c.lhs, c.rhs)
		}
	}
}

func TestLen(t *testing.T) {
	cases := []struct {
		s    vector.Set
		want int
	}{
		{nil, 0},
		{vector.Set{}, 0},
		{vector.Set{vector.Vector{0}}, 1},
		{set[0:2], 2},
	}

	for _, c := range cases {
		got := c.s.Len()
		if got != c.want {
			t.Errorf("%v.Len() == %v, want %v", c.s, got, c.want)
		}
	}
}

func TestSwap(t *testing.T) {
	cases := []struct {
		s    vector.Set
		i, j int
	}{
		{set, 2, 3},
		{set, 1, 4},
		{set, 0, 4},
	}

	for _, c := range cases {
		toSwap := make(vector.Set, len(c.s))
		copy(toSwap, c.s)
		toSwap.Swap(c.i, c.j)
		if !vector.DeeplyEqual(c.s[c.i], toSwap[c.j]) {
			t.Errorf("%v.Swap(%v, %v) -> %v; %v, want %v", c.s, c.i, c.j, toSwap, c.s[c.i], toSwap[c.j])
		}
	}
}

func TestLess(t *testing.T) {
	cases := []struct {
		s    vector.Set
		i, j int
		want bool
	}{
		{set, 3, 4, false},
		{set, 2, 4, false},
		{set, 1, 4, true},
	}

	for _, c := range cases {
		got := c.s.Less(c.i, c.j)
		if got != c.want {
			t.Errorf("[%v, %v].Less(%v, %v) == %v, want %v", c.s[c.i], c.s[c.j], c.i, c.j, got, c.want)
		}
	}
}

func TestReverse(t *testing.T) {
	cases := []struct {
		i, want int
	}{
		{0, 3},
		{1, 2},
		{2, 2},
		{3, 1},
		{4, 1},
		{5, 0},
	}

	sort.Sort(sort.Reverse(set))

	for _, c := range cases {
		if len(set[c.i]) != c.want {
			t.Errorf("sort.Sort(%v); len(%v) == %v, want %v", set, set[c.i], len(set[c.i]), c.want)
		}
	}
}

func TestSum(t *testing.T) {
	cases := []struct {
		s    vector.Set
		want vector.Vector
	}{
		{
			vector.Set{
				vector.Vector{1, 2, 3, 4, 5, 6},
				vector.Vector{7, 8, 9},
				vector.Vector{10},
			},
			vector.Vector{18, 10, 12, 4, 5, 6},
		},
		{
			vector.Set{
				vector.Vector{1, -2, 3},
				vector.Vector{4, 5, 6},
				vector.Vector{-7, 8, 9},
				vector.Vector{10, 11, -12},
				vector.Vector{13, 14, 15},
			},
			vector.Vector{21, 36, 21},
		},
		{
			set,
			vector.Vector{23, 16, 6},
		},
	}

	for _, c := range cases {
		got, _ := c.s.Sum()
		if !vector.DeeplyEqual(got, c.want) {
			t.Errorf("%v.Sum() == %v, want %v", set, got, c.want)
		}
	}
}

func TestNearlyEqualValues(t *testing.T) {
	cases := []struct {
		x, y, eps float64
		want      bool
	}{
		{1, 1.1, 0.1, true},
		{1, 1.1, 0.09, false},
		{1, 1, math.SmallestNonzeroFloat64, true},
		{math.SmallestNonzeroFloat64, 2 * math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, true},
	}

	for _, c := range cases {
		got := vector.NearlyEqualValues(c.x, c.y, c.eps)
		if c.want != got {
			t.Errorf("NearlyEqual(%v, %v, %v) == %v, want %v", c.x, c.y, c.eps, got, c.want)
		}
	}
}

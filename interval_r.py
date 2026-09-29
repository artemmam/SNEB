import math
import numpy as np

# ---------------------------------------------------------------------------
# Outward-rounding helpers
# ---------------------------------------------------------------------------
# Every arithmetic result is "pushed outward" by one ULP (unit in the last
# place) using math.nextafter (available in Python ≥ 3.9).
# This guarantees that even if the underlying float operation already rounded
# the true result, the stored interval endpoint still contains it.
# The error introduced is at most 1 ULP per operation — the standard approach
# used by virtually every IEEE-754-aware interval library.
# ---------------------------------------------------------------------------

_INF = math.inf


def _dn(x: float) -> float:
    """Round x downward by one ULP (toward -∞)."""
    return math.nextafter(x, -_INF)


def _up(x: float) -> float:
    """Round x upward by one ULP (toward +∞)."""
    return math.nextafter(x, _INF)


# ---------------------------------------------------------------------------
# Interval class
# ---------------------------------------------------------------------------

class Interval:

    def __init__(self, x):
        self.x = [_dn(x[0]), _up(x[1])]   # [lower, upper]  (stored as plain Python floats)

    def __repr__(self):
        return "[" + str(round(self.x[0], 3)) + ", " + str(round(self.x[1], 3)) + "]"

    def __round__(self, n=3):
        return Interval([np.round(self.x[0], 3), np.round(self.x[1], 3)])

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    def mid(self):
        """Middle point of the interval."""
        return 0.5 * (self.x[0] + self.x[1])

    def width(self):
        """Width of the interval."""
        return _up(self.x[1] - self.x[0])

    def abs_scale(self, a):
        """Enlarge interval by an absolute value a."""
        return Interval([_dn(self.x[0] - a), _up(self.x[1] + a)])

    def scale(self, factor):
        """Scale the interval around its midpoint by factor (in-place)."""
        m = 0.5 * (self.x[0] + self.x[1])
        r = 0.5 * (self.x[1] - self.x[0])
        self.x[0] = _dn(m - factor * r)
        self.x[1] = _up(m + factor * r)

    # ------------------------------------------------------------------
    # Set-theoretic predicates
    # ------------------------------------------------------------------

    def isIn(self, other):
        """True if self ⊆ other."""
        return (self.x[0] >= other.x[0]) and (self.x[1] <= other.x[1])

    def isNoIntersec(self, other):
        """True if self ∩ other = ∅."""
        return (self.x[0] > other.x[1]) or (self.x[1] < other.x[0])

    def intersec(self, other):
        """Intersection of two intervals, or None if they are disjoint."""
        if self.x[1] < other.x[0] or other.x[1] < self.x[0]:
            return None
        return Interval([max(self.x[0], other.x[0]), min(self.x[1], other.x[1])])

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, item):
        return self.x[item]

    def __setitem__(self, key, value):
        self.x[key] = value

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other):
        return self.x[0] == other.x[0] and self.x[1] == other.x[1]

    # ------------------------------------------------------------------
    # Arithmetic operators  (all with outward rounding)
    # ------------------------------------------------------------------

    def __neg__(self):
        return Interval([-self.x[1], -self.x[0]])

    def __add__(self, other):
        o = _to_interval(other)
        return Interval([_dn(self.x[0] + o.x[0]),
                         _up(self.x[1] + o.x[1])])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = _to_interval(other)
        return Interval([_dn(self.x[0] - o.x[1]),
                         _up(self.x[1] - o.x[0])])

    def __rsub__(self, other):
        return _to_interval(other).__sub__(self)

    def __mul__(self, other):
        o = _to_interval(other)
        a, b = self.x[0], self.x[1]
        c, d = o.x[0],   o.x[1]
        # _dn/_up applied to each product individually before taking min/max:
        # the products a*c etc. are already float-rounded, so min/max over them
        # could silently pick the wrong candidate. Pushing each one outward first
        # and then taking min/max guarantees the true extremum is enclosed.
        dn_products = [_dn(a*c), _dn(a*d), _dn(b*c), _dn(b*d)]
        up_products = [_up(a*c), _up(a*d), _up(b*c), _up(b*d)]
        return Interval([min(dn_products), max(up_products)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, n):
        a, b = self.x[0], self.x[1]
 
        if n == 0:
            return Interval([1.0, 1.0])
 
        if n > 0:
            # ---- positive exponent ----
            u, v = a ** n, b ** n
            if n % 2 == 0:
                # even: x^n is convex, minimum at 0 if interval straddles zero
                hi = max(_up(u), _up(v))
                lo = 0.0 if (a <= 0 <= b) else min(_dn(u), _dn(v))
            else:
                # odd: x^n is strictly monotone increasing
                lo, hi = _dn(u), _up(v)
            return Interval([lo, hi])
 
        else:
            # ---- negative exponent ----
            if n % 2 == 0:
                # even negative: x^n = 1/x^{|n|} — always positive, symmetric,
                # tends to +inf as x -> 0.
                if a <= 0 <= b:
                    # interval contains 0: function is unbounded
                    return Interval([0.0, _INF])
                else:
                    # strictly positive or strictly negative interval:
                    # function is monotone decreasing in |x|,
                    # so maximum is at the endpoint closest to 0.
                    u, v = a ** n, b ** n
                    hi = max(_up(u), _up(v))
                    lo = min(_dn(u), _dn(v))
                    return Interval([lo, hi])
            else:
                # odd negative: x^n = 1/x^{|n|} — strictly monotone decreasing,
                # with a discontinuity (pole) at x = 0.
                if a < 0 < b:
                    # interval straddles 0: result is (-inf, a^n] ∪ [b^n, +inf)
                    return ExtendedInterval([
                        Interval([-_INF, _up(a ** n)]),
                        Interval([_dn(b ** n), _INF]),
                    ])
                elif a == 0:
                    # left endpoint is 0: only right half [b^n, +inf)
                    return Interval([_dn(b ** n), _INF])
                elif b == 0:
                    # right endpoint is 0: only left half (-inf, a^n]
                    return Interval([-_INF, _up(a ** n)])
                else:
                    # no zero in interval: monotone decreasing,
                    # so lo = b^n, hi = a^n
                    lo, hi = _dn(b ** n), _up(a ** n)
                    return Interval([lo, hi])

    def __truediv__(self, other):
        o = _to_interval(other)
        a, b = self.x[0], self.x[1]
        c, d = o.x[0],   o.x[1]

        if not (_to_interval(0.0).isIn(o)):
            # 0 ∉ denominator — standard case
            # Same reasoning as __mul__: push each quotient outward first.
            dn_q = [_dn(a/c), _dn(a/d), _dn(b/c), _dn(b/d)]
            up_q = [_up(a/c), _up(a/d), _up(b/c), _up(b/d)]
            return Interval([min(dn_q), max(up_q)])

        # 0 ∈ denominator — extended division (returns Interval or ExtendedInterval)
        if d == 0 and b <= 0:
            c_ = [_dn(b / c), _INF]
        elif b < 0 and c < 0 and d > 0:
            c_ = [[-_INF, _up(b / d)], [_dn(b / c), _INF]]
        elif b <= 0 and c == 0:
            c_ = [-_INF, _up(b / d)]
        elif a >= 0 and d == 0:
            c_ = [-_INF, _up(a / c)]
        elif a > 0 and c < 0 and d > 0:
            c_ = [[-_INF, _up(a / c)], [_dn(a / d), _INF]]
        elif a >= 0 and c == 0:
            c_ = [_dn(a / d), _INF]
        elif a <= 0 <= b:
            c_ = [-_INF, _INF]
        else:
            c_ = [-_INF, _INF]

        if len(np.shape(c_)) != 2:
            return Interval(c_)
        else:
            return ExtendedInterval([Interval(el) for el in c_])

    def __rtruediv__(self, other):
        return _to_interval(other).__truediv__(self)

    def __floordiv__(self, other):
        o = _to_interval(other)
        a, b = self.x[0], self.x[1]
        c, d = o.x[0],   o.x[1]
        dn_q = [_dn(a//c), _dn(a//d), _dn(b//c), _dn(b//d)]
        up_q = [_up(a//c), _up(a//d), _up(b//c), _up(b//d)]
        return Interval([min(dn_q), max(up_q)])

    def __abs__(self):
        a, b = self.x[0], self.x[1]
        if b < 0:
            # entirely negative: |[a,b]| = [-b, -a], negation is exact
            return Interval([-b, -a])
        elif a < 0 < b:
            # straddles zero: minimum is always 0, maximum is max(|a|, |b|)
            return Interval([0.0, max(-a, b)])
        else:
            # entirely non-negative: identity, no arithmetic involved
            return Interval([a, b])

    # ------------------------------------------------------------------
    # Mathematical functions (as methods — outward rounding applied)
    # ------------------------------------------------------------------

    def sqrt(self):
        return Interval([_dn(math.sqrt(self.x[0])),
                         _up(math.sqrt(self.x[1]))])

    def exp(self):
        return Interval([_dn(math.exp(self.x[0])),
                         _up(math.exp(self.x[1]))])

    def log(self, base=math.e):
        if base > 1:
            return Interval([_dn(math.log(self.x[0], base)),
                             _up(math.log(self.x[1], base))])
        else:
            return Interval([_dn(math.log(self.x[1], base)),
                             _up(math.log(self.x[0], base))])

    def sin(self):
        """
        Interval sine: locates all extrema of sin inside [a, b]
        and applies outward rounding to the resulting bounds.
        """
        a, b = self.x[0], self.x[1]
        # math.sin(a/b) is already float-rounded in an unknown direction,
        # so push each endpoint outward before taking min/max.
        sa_dn, sa_up = _dn(math.sin(a)), _up(math.sin(a))
        sb_dn, sb_up = _dn(math.sin(b)), _up(math.sin(b))
        pi2  = 2 * math.pi
        pi05 = math.pi / 2
        # -1 is exact; if minimum is at an endpoint, take the smaller _dn value.
        lo = -1.0 if math.ceil((a + pi05) / pi2) <= math.floor((b + pi05) / pi2) \
                  else min(sa_dn, sb_dn)
        # +1 is exact; if maximum is at an endpoint, take the larger _up value.
        hi =  1.0 if math.ceil((a - pi05) / pi2) <= math.floor((b - pi05) / pi2) \
                  else max(sa_up, sb_up)
        return Interval([lo, hi])

    def cos(self):
        """
        Interval cosine: locates all extrema of cos inside [a, b]
        and applies outward rounding to the resulting bounds.
        """
        a, b = self.x[0], self.x[1]
        # Same reasoning as sin: push each endpoint outward before min/max.
        ca_dn, ca_up = _dn(math.cos(a)), _up(math.cos(a))
        cb_dn, cb_up = _dn(math.cos(b)), _up(math.cos(b))
        pi2 = 2 * math.pi
        # -1 is exact; if minimum is at an endpoint, take the smaller _dn value.
        lo = -1.0 if math.ceil((a - math.pi) / pi2) <= math.floor((b - math.pi) / pi2) \
                  else min(ca_dn, cb_dn)
        # +1 is exact; if maximum is at an endpoint, take the larger _up value.
        hi =  1.0 if math.ceil(a / pi2) <= math.floor(b / pi2) \
                  else max(ca_up, cb_up)
        return Interval([lo, hi])
    
    def arctan(self):
        """
        Интервальный арктангенс.
        Поскольку arctan(x) монотонно возрастает, 
        lo = atan(inf), hi = atan(sup).
        """
        print("arctan", self)
        # Используем внешнее округление для левой и правой границ
        if isinstance(self, ExtendedInterval):
            # Если возникло деление на ноль, возвращаем полный диапазон atan
            return Interval([_dn(-math.pi/2), _up(math.pi/2)])
        
        # Стандартный случай для обычного Interval
        lo = _dn(math.atan(self.x[0]))
        hi = _up(math.atan(self.x[1]))
        return Interval([lo, hi])
#     def atan2(self, other):
#         """
#         Интервальный atan2(self, other) -> atan2(Y, X), где self=Y, other=X.
#         Реализует функцию atan2 с внешним округлением.
#         """
#         y = self
#         x = _to_interval(other)
        
#         # Определяем квадранты для выбора диапазона результата
#         # atan2(y, x) возвращает значение в (-pi, pi]
        
#         # 1. Если 0 не содержится в X (стандартный случай)
#         if x.x[0] > 0:
#             # x > 0: atan2(y, x) = atan(y/x)
#             q = y / x
#             return Interval([_dn(math.atan(q.x[0])), _up(math.atan(q.x[1]))])
        
#         elif x.x[1] < 0:
#             # x < 0
#             if y.x[0] >= 0:
#                 # y >= 0: atan2(y, x) = atan(y/x) + pi
#                 q = y / x
#                 return Interval([_dn(math.atan(q.x[0]) + math.pi), 
#                                  _up(math.atan(q.x[1]) + math.pi)])
#             elif y.x[1] <= 0:
#                 # y <= 0: atan2(y, x) = atan(y/x) - pi
#                 q = y / x
#                 return Interval([_dn(math.atan(q.x[0]) - math.pi), 
#                                  _up(math.atan(q.x[1]) - math.pi)])
#             else:
#                 # y содержит 0: результат покрывает весь диапазон (-pi, pi)
#                 return Interval([_dn(-math.pi), _up(math.pi)])
                
#         else:
#             # 2. X содержит 0 (знаменатель может быть 0)
#             if y.x[0] > 0:
#                 # y > 0: результат [0, pi]
#                 return Interval([0.0, _up(math.pi)])
#             elif y.x[1] < 0:
#                 # y < 0: результат [-pi, 0]
#                 return Interval([_dn(-math.pi), 0.0])
#             else:
#                 # y содержит 0: atan2(0,0) не определен, 
#                 # результат — полный диапазон (-pi, pi]
#                 return Interval([_dn(-math.pi), _up(math.pi)])

# ---------------------------------------------------------------------------
# ExtendedInterval  (result of division when 0 ∈ denominator)
# ---------------------------------------------------------------------------

class ExtendedInterval(Interval):

    def __init__(self, x):
        # x is a list of two Interval objects
        self.x = list(x)

    def __repr__(self):
        return f"ExtendedInterval({self.x[0]}, {self.x[1]})"

    def name(self):
        return "Extended"

    def __sub__(self, other):
        o = _to_interval_extended(other)
        result = ExtendedInterval([Interval([0.0, 0.0]), Interval([0.0, 0.0])])
        result[0][0] = -_INF
        result[1][1] =  _INF
        result[0][1] = _up(self.x[0].x[1]  - o.x[1].x[0])
        result[1][0] = _dn(self.x[1].x[0]  - o.x[0].x[1])
        return result

    def __rsub__(self, other):
        return _to_interval_extended(other).__sub__(self)

    def isIn(self, other):
        return [(iv.x[0] >= other.x[0]) and (iv.x[1] <= other.x[1]) for iv in self.x]

    def isNoIntersec(self, other):
        return [(iv.x[0] >= other.x[1]) or (iv.x[1] <= other.x[0]) for iv in self.x]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _to_interval(expr) -> Interval:
    """Convert a scalar (int / float / numpy scalar) to a degenerate Interval."""
    if isinstance(expr, Interval):
        return expr
    v = float(expr)
    return Interval([v, v])


def _to_interval_extended(expr) -> ExtendedInterval:
    if isinstance(expr, ExtendedInterval):
        return expr
    v = float(expr)
    iv = Interval([v, v])
    return ExtendedInterval([iv, iv])


# Keep original names as public aliases so existing code still works
valueToInterval         = _to_interval
valueToIntervalExtended = _to_interval_extended


# ---------------------------------------------------------------------------
# Module-level functions  (outward rounding applied)
# ---------------------------------------------------------------------------

def sin(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.sin(float(x))
    return x.sin()


def cos(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.cos(float(x))
    return x.cos()


def exp(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.exp(float(x))
    return x.exp()


def log(x, base=math.e):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.log(float(x), base)
    return x.log(base)


def sqrt(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.sqrt(float(x))
    return x.sqrt()


def abs(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return math.fabs(float(x))
    return x.__abs__()

def atan2(y, x):
    """Интервальный atan2(y, x)."""
    y_iv = _to_interval(y)
    return y_iv.atan2(x)
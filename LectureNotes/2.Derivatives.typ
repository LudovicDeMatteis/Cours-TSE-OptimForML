= Derivatives and Gradients

== Reminder on derivatives
A function is said to be differentiable at a point $x_0$ if the following limit exists:
$ lim_(h -> 0) (f(x_0 + h) - f(x_0)) / h $
The value of this limit when it exists is called the derivative of $f$ at $x_0$ and is denoted $f'(x_0)$ or $(d f) / (d x) (x_0)$.

_Examples - _ Let's illustrate this idea on several classical functions
- $f(x) = 3 x^2 - x$
$
  f(x + h) - f(x) = 3 (x + h)^2 - x - h - 3 x^2 + x \
  = 3 (h^2 + 2 x h) - h
$
which gives
$ (f(x+h) - f(x)) / h = 3h + 6x - 1 $
the limit when $h->0$ as this expression is defined for all points $x in RR$ and gives the derivative
$ f'(x) = 6x - 1 $

- $f(x) = e^(-2x)$
$
  (f(x+h) - f(x)) / h = (e^(-2x -2h) - e^(-2x)) / h \
  = (e^(-2x) (e^(-2h) - 1)) / h \
  = -2 e^(-2x) ((e^(-2h) - 1)) / (-2h)
$
Moreover


== First order optimality conditions

== Derivative of classification cost

== The chain rule



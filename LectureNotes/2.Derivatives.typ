#import "include/imports.typ": *

= Derivatives and Gradients

== Reminder on derivatives
A function $f: RR -> RR$ is said to be differentiable at a point $x_0$ if the following limit exists:
$ lim_(h -> 0) (f(x_0 + h) - f(x_0)) / h $
The value of this limit when it exists is called the derivative of $f$ at $x_0$ and is denoted $f'(x_0)$ or $(d f) / (d x) (x_0)$.

_Examples - _ Let's illustrate this idea on several classical functions
+ $f(x) = 3 x^2 - x$
  $
    f(x + h) - f(x) & = 3 (x + h)^2 - x - h - 3 x^2 + x \
                    & = 3 (h^2 + 2 x h) - h
  $
  which gives
  $ (f(x+h) - f(x)) / h = 3h + 6x - 1 $
  the limit when $h->0$ as this expression is defined for all points $x in RR$ and gives the derivative
  $ f'(x) = 6x - 1 $
+ $f(x) = e^(-2x)$
  $
    (f(x+h) - f(x)) / h & = (e^(-2x -2h) - e^(-2x)) / h        \
                        & = (e^(-2x) (e^(-2h) - 1)) / h        \
                        & = -2 e^(-2x) ((e^(-2h) - 1)) / (-2h)
  $
Moreover, the limit when $h->0$ of $((e^(-2h) - 1)) / (-2h)$ is $1$ (it can be shown using the Taylor expansion of the exponential function). This gives
$ f'(x) = -2 e^(-2x) $

== On Gradient and Jacobian
The definition of a derivative can be extended to functions with multiple variables. Let $f: RR^d -> RR$ be a function of $d$ variables. The function $f$ is said to be differentiable at a point $x_0 in RR^d$ if there exists a vector $g in RR^d$ such that
$ f(x_0 + h) = f(x_0) + g^T h + o(||h||) $
where $o(||h||)$ is a function such that $o(||h||) / (||h||) -> 0$ when $||h|| -> 0$.
The vector $g$ is called the gradient of $f$ at point $x_0$ and is denoted $nabla f(x_0)$ or $(d f) / (d x) (x_0)$.
Note that the vector $g$ is unique (if it exists).

The gradient vector is related to the derivatives of the function by
$ nabla f(x) = ( (d f) / (d x_1) (x), (d f) / (d x_2) (x), ..., (d f) / (d x_d) (x) )^T $
where $x_i$ is the $i$-th component of vector $x$.

We also define the *Jacobian* of a function $f: RR^d -> RR^p$ as the matrix $J in RR^(p times d)$ writen as
$
  J & = mat(
        (d f_1) / (d x_1), (d f_1) / (d x_2), ..., (d f_1) / (d x_d);
        (d f_2) / (d x_1), (d f_2) / (d x_2), ..., (d f_2) / (d x_d);
        dots.v, dots.v, dots.down, dots.v;
        (d f_p) / (d x_1), (d f_p) / (d x_2), ..., (d f_p) / (d x_d);
      ) \
    & = mat(
        nabla f_1(x)^T;
        nabla f_2(x)^T;
        dots.v;
        nabla f_p (x)^T
      )
$
where $f_i$ is the $i$-th component of the vector-valued function $f$.

_Note: We observe that for a scalar-valued function, the Jacobian equal the transpose of the gradient._

== Derivative of classification cost

== The chain rule


== First order optimality conditions
Let's consider again an unconstrained minimization problem of the form of @inf_problem.
We can show that if $f$ is differentiable and $x^*$ is a local minimizer of $f$, then the following condition holds
$ nabla f(x^*) = 0 $

#proof[
  Let's consider a point $x^*$ which is a local minimizer of $f$. \
  By definition of a local minimizer, there exists a radius $r > 0$ such that for all $x in RR^d$ such that $||x - x^*|| <= r$, we have $f(x^*) <= f(x)$. \
  This means that for $h in RR^d$ such that $||h|| <= r$, we have
  #nonum($ f(x^*) <= f(x^* + h) = f(x^*) + nabla f(x^*)^T h + o(||h||) $)
  Simplifying by $f(x^*)$ and dividing by $||h||$ (which is non zero as $h$ is non zero), we obtain,
  #nonum($ nabla f(x^*)^T overline(h) >= 0 $)
  with $overline(h) = h / (||h||)$. \
  We can apply the same reasoning with $-h$ instead of $h$ and obtain $nabla f(x^*)^T (-bar(h)) >= 0$, which gives #nonum($nabla f(x^*)^T bar(h) <= 0$)
  Eventually, we have $nabla f(x^*)^T bar(h) = 0$ for all $bar(h)$ such that $||bar(h)|| = 1$, and thus #nonum($nabla f(x^*) = 0$)
  #align(right)[$qed$]
]
